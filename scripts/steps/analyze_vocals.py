# -*- coding: utf-8 -*-
"""Step 4: onset / syllable-attack candidates on the Demucs vocal stem."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import r2, write_json  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

HOP = 256
SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=SR, mono=True)
    duration = len(y) / sr

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=sr, hop_length=HOP,
        units="frames", backtrack=True, delta=0.10, wait=6,
    )
    frames = np.asarray(frames, dtype=int)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    strength = oenv[frames]
    snorm = strength / strength.max() if len(strength) else np.array([])

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))

    onsets = []
    for i, (fr, t, s, sn) in enumerate(zip(frames, times, strength, snorm), 1):
        b = max(0, fr - int(round(0.06 * sr / HOP)))
        a = min(len(db), fr + int(round(0.06 * sr / HOP)))
        before = float(np.mean(db[b:max(b + 1, fr)]))
        after = float(np.mean(db[fr:max(fr + 1, a)]))
        onsets.append({
            "i": i, "t": r2(t), "abs": r2(args.start + t),
            "strength": r2(s), "strength_norm": round(float(sn), 3),
            "energy_rise_db": r2(after - before),
            "rms_db_before": r2(before), "rms_db_after": r2(after),
            "frame": int(fr),
        })

    lag = 4
    rise = np.maximum(0.0, db[lag:] - db[:-lag])
    rise = np.pad(rise, (0, lag), mode="edge")
    rise = uniform_filter1d(rise, size=3, mode="nearest")
    peaks, _ = find_peaks(rise, height=4.0,
                          distance=max(1, int(round(0.12 * sr / HOP))))
    energy = [{"i": i, "t": r2(p * HOP / sr), "abs": r2(args.start + p * HOP / sr),
               "rise_db": r2(rise[p]), "frame": int(p)}
              for i, p in enumerate(peaks, 1)]

    # rough activity regions; the Demucs stem often has low-level bleed, so this
    # is informational only
    fr_db = 20.0 * np.log10(np.maximum(
        librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0], 1e-8))
    active = fr_db > -40.0
    regions = []
    i = 0
    min_len = int(round(0.25 * sr / HOP))
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if j - i >= min_len:
                regions.append({"start": r2(i * HOP / sr), "end": r2((j - 1) * HOP / sr),
                                "abs_start": r2(args.start + i * HOP / sr),
                                "abs_end": r2(args.start + (j - 1) * HOP / sr)})
            i = j
        else:
            i += 1

    result = {
        "source": "Demucs htdemucs vocals stem",
        "method": (
            "librosa onset_strength (spectral flux) on the vocal stem, "
            "onset_detect(delta=0.10, wait=6, backtrack=True); "
            "energy_rise_db = 60 ms RMS after minus before the onset"
        ),
        "segment": {"start_sec": r2(args.start), "end_sec": r2(args.start + duration),
                    "duration_sec": r2(duration)},
        "counts": {"syllable_onsets": len(onsets), "energy_onsets": len(energy),
                   "vocal_regions": len(regions)},
        "syllable_onsets": onsets,
        "energy_onsets": energy,
        "vocal_regions": regions,
    }
    write_json(args.out, result)
    print(result["counts"])


if __name__ == "__main__":
    main()