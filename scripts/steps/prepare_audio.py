# -*- coding: utf-8 -*-
"""Step 1: song file -> exact segment WAV + meta.json.

Accepts mp3 / wav / flac / ogg directly (libsndfile).  m4a / aac need ffmpeg
on PATH; it is only used to transcode to wav first.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import parse_time, write_json  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 44100
SUPPORTED = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to_run(path: Path, run: Path) -> str:
    try:
        return path.resolve().relative_to(run.resolve()).as_posix()
    except Exception:
        return str(path)


def ensure_readable(path: Path, run: Path):
    """Return (audio_path, conversion_info). Transcode via ffmpeg if needed."""
    try:
        sf.info(str(path))
        return path, None
    except Exception as exc:
        print(f"libsndfile cannot read {path.name} ({exc}); trying ffmpeg", flush=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            f"{path.suffix} needs ffmpeg on PATH (install ffmpeg, or convert the file to mp3/wav/flac first).")
    out = run / "source.wav"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-vn",
           "-ac", "2", "-ar", str(SR), str(out)]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"ffmpeg failed ({r.returncode}): {(r.stderr or '').strip()[:400]}")
    return out, {"tool": "ffmpeg", "command": f'ffmpeg -i "{path.name}" source.wav',
                 "output": out.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", default="0")
    ap.add_argument("--end", default="", help="default: end of the song")
    ap.add_argument("--run", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--slug", default="")
    args = ap.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input).expanduser().resolve()
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")
    if inp.suffix.lower() not in SUPPORTED:
        raise SystemExit(f"unsupported input type: {inp.suffix} ({' / '.join(SUPPORTED)})")

    start = parse_time(args.start)
    source_audio, conversion = ensure_readable(inp, run)

    # duration without decoding the whole file
    info = sf.info(str(source_audio))
    source_duration = float(info.frames) / float(info.samplerate)
    end = parse_time(args.end) if args.end else source_duration
    if start >= source_duration:
        raise SystemExit(f"start {start}s is beyond audio duration {source_duration:.2f}s")
    end = min(end, source_duration)
    if end <= start:
        raise SystemExit(f"end must be after start: {start} -> {end}")
    if end - start < 0.05:
        raise SystemExit("segment too short")

    # decode. For normal songs this is small; >10 min uses offset to save RAM.
    if source_duration <= 600:
        y, sr = librosa.load(str(source_audio), sr=SR, mono=False)
        a = int(round(start * sr))
        b = int(round(end * sr))
        a = max(0, min(a, y.shape[1] - 1))
        b = max(a + 1, min(b, y.shape[1]))
        seg = y[:, a:b]
    else:
        # already the requested segment; keep start/end in the metadata
        seg, sr = librosa.load(str(source_audio), sr=SR, mono=False,
                               offset=start, duration=end - start)

    if seg.ndim == 1:
        seg = np.vstack([seg, seg])
    if seg.shape[0] > 2:
        seg = seg[:2]
    if seg.shape[0] == 1:
        seg = np.vstack([seg, seg])
    wav = run / "segment.wav"
    sf.write(str(wav), seg.T, sr, subtype="PCM_24")

    meta = {
        "input": inp.name,
        "input_type": "audio",
        "input_sha256": sha256(inp) if inp.stat().st_size <= 100 * 1024 * 1024 else None,
        "source_audio": rel_to_run(source_audio, run),
        "source_audio_sha256": (sha256(source_audio)
                                if source_audio.stat().st_size <= 100 * 1024 * 1024 else None),
        "source_duration_sec": round(source_duration, 2),
        "title": args.title,
        "artist": args.artist,
        "slug": args.slug,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(float(seg.shape[1]) / sr, 3),
        "sample_rate": sr,
        "channels": int(seg.shape[0]),
        "wav": "segment.wav",
        "conversion": conversion,
    }
    write_json(run / "meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("input", "source_audio", "start_sec", "end_sec",
                                          "duration_sec", "sample_rate", "channels")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()