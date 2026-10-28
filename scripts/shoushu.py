# -*- coding: utf-8 -*-
"""beatmap-kit · one-click pipeline.

Drop in a song, get one JSON: BPM, every beat, accents (onsets), loudness jumps,
vocal onsets, per-kana Japanese lyric timing and section structure.

Each stage runs in its own Python process, so Demucs / Whisper memory is
released before the next stage starts.

Examples:
  python shoushu.py song.mp3
  python shoushu.py song.mp3 0:21 0:56 --lyrics song.txt --title "..." --artist "..."
  python shoushu.py instrumental.flac --instrumental
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

KIT = Path(__file__).resolve().parents[1]
PY_MAIN = Path(os.environ.get("SHOUSHU_PY_MAIN", KIT / "venv" / "Scripts" / "python.exe"))
PY_ASR = Path(os.environ.get("SHOUSHU_PY_ASR", KIT / "venv_asr" / "Scripts" / "python.exe"))
STEPS = KIT / "scripts" / "steps"
MODELS = KIT / "models"

sys.path.insert(0, str(KIT / "scripts"))
from common import parse_time, slugify, split_artist_title, read_json  # noqa: E402


def base_env():
    env = os.environ.copy()
    env.update({
        "SHOUSHU_KIT": str(KIT),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(MODELS / "hf"),
        "TORCH_HOME": str(MODELS / "torch"),
        "XDG_CACHE_HOME": str(KIT / "cache"),
        "TMP": str(KIT / "tmp"),
        "TEMP": str(KIT / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def device_info(requested, env):
    if requested == "cpu":
        return "cpu", ""
    code = (
        "import torch\n"
        "print('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\n"
    )
    try:
        out = subprocess.check_output([str(PY_MAIN), "-c", code], env=env,
                                      stderr=subprocess.DEVNULL, text=True, timeout=60)
        lines = out.strip().splitlines()
        dev = lines[0] if lines else "cpu"
        name = lines[1] if len(lines) > 1 else ""
    except Exception as exc:
        print(f"! device check failed: {exc}")
        dev, name = "cpu", ""
    if requested == "cuda" and dev != "cuda":
        print("! CUDA requested but torch.cuda.is_available() is false -> CPU")
        dev = "cpu"
    return dev, name


def run_step(name, python, script, args, run, env, log_name):
    log = run / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), str(script)] + [str(a) for a in args]
    print(f"==> {name}")
    print("    " + " ".join(cmd), flush=True)
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(run), env=env, stdout=f,
                           stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"[FAIL] {name} ({dt:.1f}s). Last lines of {log}:", flush=True)
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("\n".join(tail))
        except Exception:
            pass
        raise SystemExit(p.returncode)
    print(f"    ok ({dt:.1f}s)  log={log.name}", flush=True)
    return dt


def ensure_models(env, run, whisper_dir, align_dir, needs_vocals):
    """Download the ASR + alignment weights on first run (they land in models/)."""
    missing = []
    if needs_vocals:
        if not (Path(whisper_dir) / "model.bin").exists():
            missing.append("whisper")
        if not (Path(align_dir) / "config.json").exists():
            missing.append("align")
    if not missing:
        return
    print("首次运行：下载模型（%s）→ %s" % (", ".join(missing), MODELS), flush=True)
    args = ["--whisper-dir", whisper_dir, "--align-dir", align_dir,
            "--only", ",".join(missing)]
    run_step("download models", PY_MAIN, KIT / "scripts" / "download_models.py",
             args, run, env, "00_download_models.log")


def rel_to_run(path, run):
    """Keep paths inside the JSON relative to the run folder (portable, no user dirs)."""
    try:
        return Path(path).resolve().relative_to(Path(run).resolve()).as_posix()
    except Exception:
        return str(path)


def run_demucs(run, env, device, shifts="0"):
    log = run / "logs" / "03_demucs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    outdir = run / "demucs"
    used = device
    t0 = time.time()

    def invoke(dev):
        cmd = [str(PY_MAIN), "-m", "demucs.separate", "-n", "htdemucs",
               "--two-stems=vocals", "-d", dev, "--shifts", str(shifts),
               "-o", str(outdir), str(run / "segment.wav")]
        print(f"==> Demucs ({dev})")
        print("    " + " ".join(cmd), flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("+ " + " ".join(cmd) + "\n")
            f.flush()
            return subprocess.run(cmd, cwd=str(run), env=env, stdout=f,
                                  stderr=subprocess.STDOUT, text=True)

    p = invoke(used)
    if p.returncode != 0 and used == "cuda":
        print("! Demucs CUDA failed; retrying on CPU", flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write("\n[retry on cpu]\n")
        used = "cpu"
        p = invoke("cpu")
    if p.returncode != 0:
        print(f"[FAIL] Demucs failed; see {log}")
        raise SystemExit(p.returncode)

    vocals = sorted(outdir.rglob("vocals.wav"))
    no_vocals = sorted(outdir.rglob("no_vocals.wav"))
    if not vocals:
        raise SystemExit("Demucs finished but vocals.wav was not found")
    info = {
        "tool": "Demucs htdemucs",
        "device_requested": device,
        "device": used,
        "shifts": int(shifts),
        "vocals": rel_to_run(vocals[0], run),
        "no_vocals": rel_to_run(no_vocals[0], run) if no_vocals else None,
        "seconds": round(time.time() - t0, 1),
    }
    (run / "demucs.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    ok ({info['seconds']}s)  device={used}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="beatmap-kit: one song -> one JSON (BPM / beats / onsets / lyrics / structure)")
    ap.add_argument("input", help="audio file: mp3 / wav / flac / m4a")
    ap.add_argument("start", nargs="?", default="", help="segment start, e.g. 0:21 (default: 0)")
    ap.add_argument("end", nargs="?", default="", help="segment end, e.g. 0:56 (default: end of song)")
    ap.add_argument("--lyrics", default="", help="lyrics file (optional, one line per lyric line)")
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--slug", default="", help="output name: beats-<slug>.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--demucs-shifts", default="0",
                    help="Demucs random shifts; 0 is deterministic (default)")
    ap.add_argument("--align-mode", default="auto", choices=["auto", "per-line", "global", "asr"])
    ap.add_argument("--instrumental", action="store_true",
                    help="instrumental track: skip Demucs / vocal onsets / ASR / lyrics")
    ap.add_argument("--structure-method", default="auto", choices=["auto", "allin1", "librosa"])
    ap.add_argument("--structure-k", type=int, default=0,
                    help="laplacian cluster count; 0 = auto")
    ap.add_argument("--whisper-model", default=str(MODELS / "whisper_medium"))
    ap.add_argument("--align-model", default=str(MODELS / "align_ja"))
    ap.add_argument("--run-dir", default="")
    args = ap.parse_args()

    if not PY_MAIN.exists() or not PY_ASR.exists():
        raise SystemExit(
            "Python environments not found under " + str(KIT) + "\n"
            "Run setup.bat first (it creates venv\\ and venv_asr\\).")

    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")
    if inp.suffix.lower() not in (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"):
        raise SystemExit(f"unsupported input type: {inp.suffix} (mp3 / wav / flac / m4a)")

    start = parse_time(args.start) if args.start else 0.0
    end = parse_time(args.end) if args.end else None
    if end is not None and end <= start:
        raise SystemExit("end must be after start")

    stem_artist, stem_title = split_artist_title(inp.stem)
    title = args.title or stem_title
    artist = args.artist or stem_artist
    slug = slugify(args.slug or title)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(args.run_dir).expanduser().resolve() if args.run_dir else KIT / "runs" / f"{stamp}-{slug}"
    if run.exists():
        run = run.with_name(run.name + "-" + str(int(time.time())))
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)

    lyrics = Path(args.lyrics).expanduser().resolve() if args.lyrics else None
    if args.instrumental and lyrics:
        print("! --instrumental: ignoring --lyrics")
        lyrics = None
    if lyrics:
        if not lyrics.exists():
            raise SystemExit(f"lyrics not found: {lyrics}")
        shutil.copyfile(lyrics, run / "lyrics.txt")
        lyrics = run / "lyrics.txt"

    env = base_env()
    device, gpu_name = device_info(args.device, env)
    print(f"beatmap-kit  {KIT}")
    print(f"输入: {inp.name}")
    print(f"片段: {start:.2f}-" + (f"{end:.2f}s" if end is not None else "全曲"))
    print(f"标题: {artist} - {title}   slug={slug}")
    print(f"设备: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"运行目录: {run}\n")

    ensure_models(env, run, args.whisper_model, args.align_model,
                  needs_vocals=not args.instrumental)

    timings = {}
    prepare_args = ["--input", inp, "--start", start, "--run", run,
                    "--title", title, "--artist", artist, "--slug", slug]
    if end is not None:
        prepare_args += ["--end", end]
    timings["prepare"] = run_step(
        "prepare audio", PY_MAIN, STEPS / "prepare_audio.py",
        prepare_args, run, env, "01_prepare.log")
    meta = read_json(run / "meta.json")

    timings["mix"] = run_step(
        "librosa mix analysis", PY_MAIN, STEPS / "analyze_mix.py",
        ["--wav", "segment.wav", "--out", "mix.json",
         "--start", meta["start_sec"], "--source", meta["source_audio"],
         "--title", title, "--artist", artist],
        run, env, "02_mix.log")

    if args.instrumental:
        print("纯音乐模式：跳过 Demucs / 人声 onset / ASR / WhisperX")
        demucs = {"tool": "skipped (instrumental)", "device": "skipped",
                  "vocals": None, "no_vocals": None}
    else:
        run_demucs(run, env, device, args.demucs_shifts)
        demucs = read_json(run / "demucs.json")

        timings["vocals"] = run_step(
            "vocal onsets", PY_MAIN, STEPS / "analyze_vocals.py",
            ["--wav", demucs["vocals"], "--out", "vocals.json",
             "--start", meta["start_sec"]],
            run, env, "04_vocals.log")

        timings["asr"] = run_step(
            "Japanese ASR", PY_ASR, STEPS / "asr_transcribe.py",
            ["--vocals", demucs["vocals"], "--out", "asr.json",
             "--model-dir", args.whisper_model],
            run, env, "05_asr.log")

        align_args = ["--vocals", demucs["vocals"], "--asr", "asr.json",
                      "--out", "alignment.json", "--start", meta["start_sec"],
                      "--model-dir", args.align_model, "--mode", args.align_mode]
        if lyrics:
            align_args += ["--lyrics", lyrics]
        timings["align"] = run_step(
            "WhisperX kana alignment", PY_ASR, STEPS / "align_kana.py",
            align_args, run, env, "06_align.log")

    out_json = run / f"beats-{slug}.json"
    build_args = ["--run", ".", "--out", out_json.name]
    if args.instrumental:
        build_args.append("--instrumental")
    timings["build"] = run_step(
        "merge JSON", PY_MAIN, STEPS / "build_json.py",
        build_args, run, env, "07_build.log")

    timings["structure"] = run_step(
        "structure + energy", PY_MAIN, STEPS / "analyze_structure.py",
        ["--json", out_json.name, "--source", meta["source_audio"],
         "--out", "extra.json", "--method", args.structure_method,
         "--k", args.structure_k, "--device", device],
        run, env, "08_structure.log")

    timings["merge_extra"] = run_step(
        "merge structure/energy", PY_MAIN, STEPS / "merge_extra.py",
        ["--json", out_json.name, "--extra", "extra.json", "--no-backup"],
        run, env, "09_merge.log")

    manifest = {
        "input": args.input, "start": start, "end": end,
        "title": title, "artist": artist, "slug": slug,
        "device_requested": args.device, "device": demucs.get("device"),
        "instrumental": bool(args.instrumental),
        "gpu": gpu_name, "lyrics": "lyrics.txt" if lyrics else None,
        "run": run.name, "json": out_json.name,
        "timings_sec": {k: round(v, 1) for k, v in timings.items()},
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    final = read_json(out_json)
    print("\n[OK] 完成")
    print(f"JSON:   {out_json}")
    print(f"COUNT:  {json.dumps(final['counts'], ensure_ascii=False)}")
    if "structure" in final:
        print("EXTRA:  structure={} segments, energy={} beats, density={} bars, jumps={}".format(
            len(final["structure"].get("segments", [])),
            len(final.get("energy", {}).get("per_beat", [])),
            len(final.get("density", {}).get("bars", [])),
            len(final.get("jumps", {}).get("events", [])),
        ))
    print(f"SUMMARY: {final['summary_zh']}")


if __name__ == "__main__":
    main()