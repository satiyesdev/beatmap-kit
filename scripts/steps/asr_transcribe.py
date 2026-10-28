# -*- coding: utf-8 -*-
"""Step 5a: Japanese ASR on the vocal stem (faster-whisper, CPU int8)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import KIT, to_kana, write_json  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

TARGET_SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=str(KIT / "models" / "whisper_medium"))
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    y, sr = sf.read(args.vocals, always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    y16 = resample_poly(y, TARGET_SR, sr).astype(np.float32) if sr != TARGET_SR else y.astype(np.float32)

    from faster_whisper import WhisperModel
    t0 = time.time()
    print("loading faster-whisper ...", flush=True)
    model = WhisperModel(args.model_dir, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    segments_iter, info = model.transcribe(
        y16,
        language=args.language,
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.2,
        compression_ratio_threshold=2.8,
    )
    segs = []
    for s in segments_iter:
        text = s.text.strip()
        if not text:
            continue
        nsp = float(getattr(s, "no_speech_prob", 0.0))
        if nsp > 0.85:
            print(f"drop no_speech={nsp:.2f} {s.start:.2f}-{s.end:.2f} {text}", flush=True)
            continue
        segs.append({
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": text,
            "kana": to_kana(text),
            "avg_logprob": round(float(getattr(s, "avg_logprob", 0.0)), 3),
            "no_speech_prob": round(nsp, 3),
        })
    print(f"transcribed {len(segs)} segments in {time.time()-t0:.1f}s", flush=True)
    for s in segs:
        print(f"  {s['start']:6.2f}-{s['end']:6.2f}  {s['text']}", flush=True)

    result = {
        "engine": "faster-whisper",
        "model": Path(args.model_dir).name,
        "language": args.language,
        "segments": segs,
        "text": "".join(s["text"] for s in segs),
    }
    write_json(args.out, result)
    print(json.dumps({"segments": len(segs), "chars": len(result["text"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()