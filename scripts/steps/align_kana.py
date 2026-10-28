# -*- coding: utf-8 -*-
"""Step 5b: WhisperX Japanese CTC alignment -> kana timeline + lyric lines.

Modes
  per-line : one lyric line per ASR segment (line count must match)
  global   : concatenate all lyric kana, align once over the whole segment
  asr      : no lyrics file; align the ASR transcript per segment
auto chooses per-line when counts match, global when lyrics exist, else asr.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KIT, parse_lyrics, r2, to_kana, write_json, read_json  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

TARGET_SR = 16000


def is_kana(ch):
    o = ord(ch)
    return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF) or ch == "ー"


def split_word(w, line_idx, start_offset=0.0):
    chars = [c for c in w.get("word", "") if is_kana(c)]
    if not chars or "start" not in w or "end" not in w:
        return []
    dur = float(w["end"]) - float(w["start"])
    step = dur / len(chars)
    out = []
    for j, ch in enumerate(chars):
        st = float(w["start"]) + j * step
        en = st + step
        out.append({
            "kana": ch,
            "t": round(st, 3),
            "end_t": round(en, 3),
            "dur": round(step, 3),
            "score": round(float(w.get("score", 0.0)), 3),
            "line": line_idx,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--mode", default="auto", choices=["auto", "per-line", "global", "asr"])
    ap.add_argument("--model-dir", default=str(KIT / "models" / "align_ja"))
    args = ap.parse_args()

    asr = read_json(args.asr)
    asr_segs = asr.get("segments", [])
    lyrics = parse_lyrics(args.lyrics) if args.lyrics else []

    mode = args.mode
    if mode == "auto":
        if lyrics and len(lyrics) == len(asr_segs):
            mode = "per-line"
        elif lyrics:
            mode = "global"
        else:
            mode = "asr"
    if mode == "per-line" and (not lyrics or len(lyrics) != len(asr_segs)):
        raise SystemExit(f"per-line needs equal counts: lyrics={len(lyrics)} asr={len(asr_segs)}")
    if mode == "global" and not lyrics:
        raise SystemExit("global mode needs --lyrics")

    y, sr = sf.read(args.vocals, always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    y16 = resample_poly(y, TARGET_SR, sr).astype(np.float32) if sr != TARGET_SR else y.astype(np.float32)
    duration = len(y16) / TARGET_SR

    # ---- decide alignment segments and line kana lengths --------------------
    line_lengths = []
    segs = []
    if mode == "per-line":
        for i, ((line, a)) in enumerate(zip(lyrics, asr_segs), 1):
            kana = line["kana_override"] or to_kana(line["text"])
            kana = "".join(c for c in kana if is_kana(c))
            if not kana:
                raise SystemExit(f"line {i} has no kana: {line['text']}")
            line_lengths.append(len(kana))
            segs.append({"start": float(a["start"]), "end": float(a["end"]),
                         "text": kana, "line": i, "display": line["text"],
                         "source": "override" if line["kana_override"] else "pykakasi"})
    elif mode == "global":
        kana_all = []
        for i, line in enumerate(lyrics, 1):
            kana = line["kana_override"] or to_kana(line["text"])
            kana = "".join(c for c in kana if is_kana(c))
            line_lengths.append(len(kana))
            kana_all.append(kana)
        text = "".join(kana_all)
        if not text:
            raise SystemExit("lyrics produced no kana")
        segs.append({"start": 0.0, "end": duration, "text": text,
                     "line": None, "display": "", "source": "lyrics"})
    else:
        for i, a in enumerate(asr_segs, 1):
            kana = a.get("kana") or to_kana(a.get("text", ""))
            kana = "".join(c for c in kana if is_kana(c))
            if not kana:
                continue
            line_lengths.append(len(kana))
            segs.append({"start": float(a["start"]), "end": float(a["end"]),
                         "text": kana, "line": i, "display": a.get("text", ""),
                         "source": "asr"})

    # ---- WhisperX -----------------------------------------------------------
    import whisperx.alignment as wa

    class _OneSentence:
        def span_tokenize(self, text):
            return [(0, len(text))]

    wa.nltk_load = lambda path: _OneSentence()

    print("loading alignment model ...", flush=True)
    t = time.time()
    model_a, metadata = wa.load_align_model(
        language_code="ja", device="cpu", model_name=args.model_dir)
    print(f"loaded in {time.time()-t:.1f}s; mode={mode}", flush=True)

    def do_align(transcript, label):
        t0 = time.time()
        res = wa.align(transcript, model_a, metadata, y16, "cpu",
                       return_char_alignments=True, print_progress=False)
        print(f"{label} aligned in {time.time()-t0:.1f}s", flush=True)
        return res

    kana_res = do_align(segs, "kana")

    flat = []
    for si, seg in enumerate(kana_res["segments"], 1):
        for w in seg.get("words", []):
            flat.extend(split_word(w, si))

    # assign line indices
    if mode == "global":
        idx = 0
        for li, n in enumerate(line_lengths, 1):
            for k in flat[idx:idx + n]:
                k["line"] = li
            idx += n
    for i, k in enumerate(flat, 1):
        k["i"] = i
        sc = k.get("score", 0.0)
        k["confidence"] = "high" if sc >= 0.70 else "medium" if sc >= 0.30 else "low"

    # ---- lyric lines from kana ---------------------------------------------
    lines = []
    if mode == "global":
        idx = 0
        for li, (line, n) in enumerate(zip(lyrics, line_lengths), 1):
            chunk = flat[idx:idx + n]
            idx += n
            lines.append(make_line(li, line["text"],
                                   line["kana_override"] or to_kana(line["text"]),
                                   chunk, args.start, "lyrics"))
    else:
        for si, s in enumerate(segs, 1):
            chunk = [k for k in flat if k["line"] == si]
            display = s["display"] or s["text"]
            lines.append(make_line(si, display, s["text"], chunk, args.start,
                                   "asr" if mode == "asr" else "lyrics"))

    # ---- char pass (original text) -----------------------------------------
    char_timeline = []
    try:
        if mode == "per-line":
            char_segs = [{"start": float(a["start"]), "end": float(a["end"]),
                          "text": line["text"]} for line, a in zip(lyrics, asr_segs)]
        elif mode == "global":
            char_segs = [{"start": 0.0, "end": duration,
                          "text": "".join(line["text"] for line in lyrics)}]
        else:
            char_segs = [{"start": float(a["start"]), "end": float(a["end"]),
                          "text": a.get("text", "")} for a in asr_segs]
        char_res = do_align(char_segs, "chars")
        for si, seg in enumerate(char_res["segments"], 1):
            for ch in seg.get("chars", []):
                c = ch.get("char", "")
                if not c.strip() or "start" not in ch or "end" not in ch:
                    continue
                char_timeline.append({
                    "char": c, "t": round(float(ch["start"]), 3),
                    "end_t": round(float(ch["end"]), 3),
                    "abs": r2(args.start + float(ch["start"])),
                    "end_abs": r2(args.start + float(ch["end"])),
                    "score": round(float(ch.get("score", 0.0)), 3),
                    "segment": si,
                })
        char_timeline.sort(key=lambda x: x["t"])
    except Exception as exc:
        print("char pass failed:", repr(exc), flush=True)

    # absolute times + 2 decimals on the kana timeline
    for k in flat:
        k["abs"] = r2(args.start + k["t"])
        k["end_abs"] = r2(args.start + k["end_t"])
        k["t"] = r2(k["t"])
        k["end_t"] = r2(k["end_t"])

    result = {
        "status": "ok",
        "mode": mode,
        "text_source": {
            "per-line": "lyrics lines aligned inside ASR segment windows",
            "global": "all lyrics kana aligned once over the whole segment",
            "asr": "ASR transcript aligned per segment",
        }[mode],
        "asr": asr,
        "alignment": {
            "engine": "WhisperX",
            "model": Path(args.model_dir).name,
            "device": "cpu",
            "kana_count": len(flat),
            "char_count": len(char_timeline),
            "segments": [{"start": s["start"], "end": s["end"],
                          "text": s.get("display", s["text"]), "kana": s["text"],
                          "source": s["source"]} for s in segs],
        },
        "kana_timeline": flat,
        "char_timeline": char_timeline,
        "lyrics_lines": lines,
        "notes": [
            "line t/end_t are taken from the first/last aligned kana, not the ASR segment boundary",
            "kana scores come from the CTC alignment; low scores should be reviewed",
            "wav2vec2 frame shift is ~20 ms; 0.01 s is output precision, not resolution",
        ],
    }
    write_json(args.out, result)
    print(f"wrote {args.out}: mode={mode} kana={len(flat)} lines={len(lines)} chars={len(char_timeline)}",
          flush=True)


def make_line(i, text, kana, chunk, start, source):
    kana = "".join(c for c in kana if is_kana(c))
    if chunk:
        t0 = min(k["t"] for k in chunk)
        t1 = max(k["end_t"] for k in chunk)
        scores = [k.get("score", 0.0) for k in chunk]
        mean = sum(scores) / len(scores) if scores else 0.0
        conf = "high" if mean >= 0.70 else "medium" if mean >= 0.30 else "low"
        first_i = min(k["i"] for k in chunk)
        last_i = max(k["i"] for k in chunk)
    else:
        t0 = t1 = 0.0
        mean, conf, first_i, last_i = 0.0, "low", None, None
    return {
        "i": i, "text": text, "kana": kana,
        "t": r2(t0), "abs": r2(start + t0),
        "end_t": r2(t1), "end_abs": r2(start + t1),
        "dur": r2(t1 - t0),
        "kana_count": len(chunk), "score_mean": round(mean, 3),
        "confidence": conf, "source": source,
        "kana_start_index": first_i, "kana_end_index": last_i,
    }


if __name__ == "__main__":
    main()