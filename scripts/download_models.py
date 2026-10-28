# -*- coding: utf-8 -*-
"""Download the model weights into models/ (first run only).

  whisper : Systran/faster-whisper-medium          -> models/whisper_medium
  align   : jonatasgrosman/wav2vec2-large-xlsr-53-japanese -> models/align_ja

Demucs (htdemucs) downloads itself into models/torch when the pipeline first
runs it, so nothing to do here.

Behind the GFW, set HF_ENDPOINT=https://hf-mirror.com (or a proxy) if the
download stalls.
"""
import argparse
import os
import sys
from pathlib import Path

REPOS = {
    "whisper": ("Systran/faster-whisper-medium", "whisper_medium"),
    "align": ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "align_ja"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--whisper-dir", default="")
    ap.add_argument("--align-dir", default="")
    ap.add_argument("--only", default="whisper,align",
                    help="comma separated: whisper,align")
    args = ap.parse_args()

    models = Path(args.models_dir)
    targets = {
        "whisper": Path(args.whisper_dir) if args.whisper_dir else models / REPOS["whisper"][1],
        "align": Path(args.align_dir) if args.align_dir else models / REPOS["align"][1],
    }
    wanted = [w.strip() for w in args.only.split(",") if w.strip()]

    from huggingface_hub import snapshot_download

    for key in wanted:
        if key not in REPOS:
            raise SystemExit(f"unknown model: {key}")
        repo, _ = REPOS[key]
        dest = targets[key]
        dest.mkdir(parents=True, exist_ok=True)
        print(f"==> {repo} -> {dest}", flush=True)
        snapshot_download(repo_id=repo, local_dir=str(dest))
        print(f"    ok ({sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) / 1e9:.2f} GB)",
              flush=True)

    print("models ready; Demucs htdemucs downloads itself on first separation run.")
    print("If the download is blocked, set HF_ENDPOINT=https://hf-mirror.com and retry.")


if __name__ == "__main__":
    main()