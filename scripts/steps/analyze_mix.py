# -*- coding: utf-8 -*-
"""Step 2: librosa BPM / beats / onsets / drum onsets / loudness steps."""
import argparse
import json
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
    ap.add_argument("--source", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    args = ap.parse_args()

    y, sr = librosa.load(args.wav, sr=SR, mono=True)
    seg_dur = len(y) / sr

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo_arr, beat_frames = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=HOP,
        trim=False, tightness=400, units="frames", sparse=True,
    )
    tempo = float(np.ravel(tempo_arr)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    keep = (beat_times >= 0) & (beat_times <= seg_dur + 1e-9)
    beat_frames, beat_times = beat_frames[keep], beat_times[keep]
    beat_strength = oenv[beat_frames] if len(beat_frames) else np.array([])

    try:
        tc = librosa.feature.rhythm.tempo(
            onset_envelope=oenv, sr=sr, hop_length=HOP, aggregate=None, max_tempo=220,
        )
        tempo_candidates = sorted({round(float(v), 2) for v in np.ravel(tc)})[:8]
    except Exception as exc:
        tempo_candidates = [f"error: {exc}"]

    if len(beat_times) >= 4:
        bpm_from_beats = 60.0 / float(np.median(np.diff(beat_times)))
    else:
        bpm_from_beats = None

    full_bpm = None
    full_beat_count = None
    if args.source and Path(args.source).exists():
        try:
            yf, srf = librosa.load(args.source, sr=22050, mono=True, duration=900)
            if len(yf) / srf <= 900:
                of = librosa.onset.onset_strength(y=yf, sr=srf, hop_length=512)
                tf, bf = librosa.beat.beat_track(
                    onset_envelope=of, sr=srf, hop_length=512,
                    trim=False, tightness=400, units="frames", sparse=True,
                )
                full_bpm = r2(np.ravel(tf)[0])
                full_beat_count = int(len(bf))
        except Exception as exc:
            print("full-track tempo failed:", exc)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=sr, hop_length=HOP,
        units="frames", backtrack=True, delta=0.055, wait=2,
    )
    onset_frames = np.asarray(onset_frames, dtype=int)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
    onset_strength = oenv[onset_frames] if len(onset_frames) else np.array([])
    onset_norm = (onset_strength / onset_strength.max()) if len(onset_strength) else np.array([])

    _, y_perc = librosa.effects.hpss(y, margin=3.0)
    oenv_p = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=HOP)
    p_frames = librosa.onset.onset_detect(
        onset_envelope=oenv_p, sr=sr, hop_length=HOP,
        units="frames", backtrack=True, delta=0.075, wait=2,
    )
    p_frames = np.asarray(p_frames, dtype=int)
    p_times = librosa.frames_to_time(p_frames, sr=sr, hop_length=HOP)
    p_strength = oenv_p[p_frames] if len(p_frames) else np.array([])

    if len(onset_times):
        nearest = np.abs(onset_times[None, :] - beat_times[:, None]).argmin(axis=1)
        nearest_dist = np.abs(onset_times[nearest] - beat_times)
    else:
        nearest = np.array([], dtype=int)
        nearest_dist = np.array([])

    # ---- sustained loudness steps -------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    lev = uniform_filter1d(db, size=max(1, int(round(0.5 * sr / HOP))), mode="nearest")
    w = max(1, int(round(0.40 * sr / HOP)))
    step = np.zeros_like(lev)
    for i in range(len(lev)):
        a = lev[max(0, i - w):i]
        b = lev[i:min(len(lev), i + w)]
        if len(a) > 5 and len(b) > 5:
            step[i] = float(np.mean(b) - np.mean(a))
    peaks, _ = find_peaks(np.abs(step), height=2.5,
                          distance=max(1, int(round(1.0 * sr / HOP))))
    fast = uniform_filter1d(db, size=5, mode="nearest")
    changes = []
    for p in peaks:
        i = int(p)
        if i * HOP / sr < 0.6 or i * HOP / sr > seg_dur - 0.6:
            continue
        before0 = float(np.mean(lev[i - w:i]))
        after0 = float(np.mean(lev[i:i + w]))
        delta0 = after0 - before0
        if abs(delta0) < 2.5:
            continue
        mid = 0.5 * (before0 + after0)
        sign = 1.0 if delta0 > 0 else -1.0
        lo = max(2, i - int(round(0.60 * sr / HOP)))
        hi = min(len(fast) - 2, i + int(round(0.60 * sr / HOP)))
        side = (fast[lo:hi] - mid) * sign >= 0
        crossings = np.where(side[1:] & ~side[:-1])[0] + 1
        k = lo + int(crossings[np.abs(crossings - (i - lo)).argmin()]) if len(crossings) else i
        if k - w < 0 or k + w > len(lev):
            k = max(w, min(len(lev) - w - 1, k))
        b0 = max(0, k - int(round(1.0 * sr / HOP)))
        b1 = max(1, k - int(round(0.10 * sr / HOP)))
        a0 = min(len(lev) - 1, k + int(round(0.50 * sr / HOP)))
        a1 = min(len(lev), k + int(round(1.50 * sr / HOP)))
        before = float(np.mean(lev[b0:b1]))
        after = float(np.mean(lev[a0:a1]))
        delta = after - before
        if abs(delta) < 2.5:
            continue
        j = min(len(lev), k + w)
        while j < len(lev) and abs(lev[j] - before) > 1.0:
            j += 1
        held = (j - k) * HOP / sr
        changes.append({
            "i": len(changes) + 1,
            "t": r2(k * HOP / sr),
            "abs": r2(args.start + k * HOP / sr),
            "kind": "sustained_rise" if delta > 0 else "sustained_drop",
            "direction": "up" if delta > 0 else "down",
            "delta_db": r2(delta),
            "before_db": r2(before),
            "after_db": r2(after),
            "held_sec": r2(held),
            "held_to_segment_end": bool(j >= len(lev)),
            "window_sec": 0.40,
            "frame": int(k),
        })
    changes.sort(key=lambda c: c["t"])
    for i, c in enumerate(changes, 1):
        c["i"] = i

    prof_step = max(1, int(round(0.5 * sr / HOP)))
    profile = [{"t": r2(i * HOP / sr), "abs": r2(args.start + i * HOP / sr), "db": r2(lev[i])}
               for i in range(0, len(lev), prof_step)]

    beats = []
    for i, (fr, t, s) in enumerate(zip(beat_frames, beat_times, beat_strength), 1):
        item = {
            "i": i, "t": r2(t), "abs": r2(args.start + t),
            "strength": r2(s), "frame": int(fr),
        }
        if len(nearest):
            item["nearest_onset_t"] = r2(onset_times[nearest[i - 1]])
            item["nearest_onset_dist"] = r2(nearest_dist[i - 1])
        beats.append(item)

    onsets = []
    for i, (fr, t, s, sn) in enumerate(zip(onset_frames, onset_times, onset_strength, onset_norm), 1):
        nearest_beat = int(np.abs(beat_times - t).argmin()) + 1 if len(beat_times) else None
        dbeat = float(np.abs(beat_times - t).min()) if len(beat_times) else None
        onsets.append({
            "i": i, "t": r2(t), "abs": r2(args.start + t),
            "strength": r2(s), "strength_norm": round(float(sn), 3),
            "accent": bool(sn >= 0.60),
            "on_beat": bool(dbeat is not None and dbeat <= 0.05),
            "nearest_beat": nearest_beat, "frame": int(fr),
        })

    drums = [{"i": i, "t": r2(t), "abs": r2(args.start + t), "strength": r2(s), "frame": int(fr)}
             for i, (fr, t, s) in enumerate(zip(p_frames, p_times, p_strength), 1)]

    result = {
        "track": (args.artist + " - " + args.title).strip(" -") if args.title else "",
        "analysis": "librosa %s, mono 44.1 kHz, hop_length=256 (5.80 ms)" % librosa.__version__,
        "segment": {
            "start_sec": r2(args.start), "end_sec": r2(args.start + seg_dur),
            "duration_sec": r2(seg_dur), "start_label": fmt_label(args.start),
            "end_label": fmt_label(args.start + seg_dur),
            "sample_rate": sr, "samples": int(len(y)),
        },
        "tempo": {
            "bpm": r2(tempo), "bpm_from_beats": r2(bpm_from_beats) if bpm_from_beats else None,
            "beat_count": int(len(beats)), "full_track_bpm": full_bpm,
            "full_track_beat_count": full_beat_count,
            "tempogram_candidates": tempo_candidates,
        },
        "loudness_method": (
            "RMS (2048 samples, hop 256) -> 0.5 s smoothing; sustained step "
            ">= 2.5 dB comparing 0.40 s before/after, min 1.0 s apart; "
            "first/last 0.6 s ignored"
        ),
        "loudness_changes": changes,
        "loudness_profile_0p5s": profile,
        "beats": beats, "onsets": onsets, "drum_onsets": drums,
        "counts": {"beats": len(beats), "onsets": len(onsets),
                   "drum_onsets": len(drums), "loudness_changes": len(changes)},
    }
    write_json(args.out, result)
    print(json.dumps({"tempo": result["tempo"], "counts": result["counts"]},
                     ensure_ascii=False, indent=2))


def fmt_label(sec):
    m, s = divmod(float(sec), 60)
    return f"{int(m)}:{s:05.2f}"


if __name__ == "__main__":
    main()