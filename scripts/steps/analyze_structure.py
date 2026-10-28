# -*- coding: utf-8 -*-
"""Structure (allin1 or librosa laplacian) + per-beat energy / per-bar density / jumps.

Writes an "extra" dict with exactly: structure, energy, density, jumps.
It does NOT modify the input JSON; merge_extra.py merges with a backup.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json, r2, write_json  # noqa: E402

ALLIN1_NOTE = (
    "allin1 is optional (pip install allin1, plus NATTEN and madmom). "
    "On Windows it cannot be installed today: NATTEN ships no Windows wheels and "
    "madmom's source build fails, so librosa laplacian segmentation is used instead."
)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.sparse.csgraph import laplacian  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


def laplacian_segmentation(y, sr, k=None):
    """McFee & Ellis laplacian segmentation, adapted from the librosa example.

    librosa 1.0 removed ``librosa.segment.laplacian_segmentation``; this keeps
    the original CQT + recurrence + normalized Laplacian + KMeans algorithm.
    """
    bpo, noct = 36, 7
    C = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=bpo, n_bins=noct * bpo)),
        ref=np.max,
    )
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 8:
        raise RuntimeError("not enough beats for structure segmentation")

    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    df = librosa.segment.timelag_filter(median_filter)
    Rf = df(R, size=(1, 7))

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = float(np.median(path_distance))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    denom = float(np.sum((deg_path + deg_rec) ** 2))
    mu = float(deg_path.dot(deg_path + deg_rec) / denom) if denom else 0.5
    A = mu * Rf + (1.0 - mu) * R_path

    L = laplacian(A, normed=True)
    _, evecs = eigh(L)
    evecs = median_filter(evecs, size=(9, 1))
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

    duration = len(y) / sr
    if not k:
        k = max(4, min(9, int(round(duration / 38.0))))
    k = max(2, min(k, len(beats) - 1))
    X = evecs[:, :k] / np.maximum(Cnorm[:, k - 1:k], 1e-9)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    seg_ids = km.fit_predict(X)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
    bound_segs = [int(seg_ids[min(b, len(seg_ids) - 1)]) for b in bound_beats]
    bound_frames = beats[np.clip(bound_beats, 0, len(beats) - 1)]
    bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1] - 1)
    bound_times = librosa.frames_to_time(bound_frames, sr=sr)

    # ensure [0, duration] and keep one label per interval
    raw_times = [float(t) for t in bound_times]
    raw_labels = [int(x) for x in bound_segs]
    if not raw_times or raw_times[0] > 1e-6:
        raw_times = [0.0] + raw_times
        raw_labels = [(raw_labels[0] if raw_labels else 0)] + raw_labels
    else:
        raw_times[0] = 0.0
    if raw_times[-1] < duration - 1e-6:
        raw_times.append(duration)
        raw_labels.append(raw_labels[-1] if raw_labels else 0)
    times, labels = [raw_times[0]], []
    for i in range(1, len(raw_times)):
        if raw_times[i] > times[-1] + 1e-6:
            labels.append(raw_labels[min(i - 1, len(raw_labels) - 1)])
            times.append(raw_times[i])
    if times[-1] < duration - 1e-6:
        times.append(duration)
        labels.append(raw_labels[-1] if raw_labels else 0)
    return {
        "k": k,
        "tempo": float(np.ravel(tempo)[0]),
        "beat_count": len(beats),
        "times": times,
        "labels": labels,
        "cluster_count": len(set(seg_ids)),
    }


def stats_for_intervals(y, sr, times, labels, onset_times):
    out = []
    for i, lab in enumerate(labels):
        a, b = times[i], times[i + 1]
        ia = max(0, int(round(a * sr)))
        ib = min(len(y), int(round(b * sr)))
        if ib <= ia:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(y[ia:ib] ** 2)))
        db = 20.0 * math.log10(max(rms, 1e-8))
        ons = sum(1 for t in onset_times if a <= t < b)
        out.append({
            "i": i + 1,
            "cluster": int(lab),
            "start_abs": r2(a),
            "end_abs": r2(b),
            "duration_sec": r2(b - a),
            "rms_db": r2(db),
            "onset_count": int(ons),
            "onset_density_per_sec": r2(ons / max(b - a, 1e-6)),
        })
    return out


def _merge_two(a, b, keep_second=True):
    """Merge two adjacent segments; duration-weighted RMS.

    keep_second=True  -> keep b (for merging a tiny leading segment into the next)
    keep_second=False -> keep a (for merging a tiny trailing segment into the previous)
    """
    da = max(a["duration_sec"], 1e-6)
    db = max(b["duration_sec"], 1e-6)
    pa = 10.0 ** (a["rms_db"] / 10.0)
    pb = 10.0 ** (b["rms_db"] / 10.0)
    dur = a["duration_sec"] + b["duration_sec"]
    db_new = r2(10.0 * math.log10(max((pa * da + pb * db) / (da + db), 1e-12)))
    count = int(a["onset_count"] + b["onset_count"])
    if keep_second:
        b["start_abs"] = a["start_abs"]
        b["end_abs"] = b["start_abs"] + dur
        b["duration_sec"] = r2(dur)
        b["rms_db"] = db_new
        b["onset_count"] = count
        b["onset_density_per_sec"] = r2(count / max(dur, 1e-6))
        return b
    a["end_abs"] = b["end_abs"]
    a["duration_sec"] = r2(dur)
    a["rms_db"] = db_new
    a["onset_count"] = count
    a["onset_density_per_sec"] = r2(count / max(dur, 1e-6))
    return a


def merge_edges(segments):
    """Merge the tiny leading silence / trailing tail into their neighbours."""
    if len(segments) < 2:
        return segments
    if segments[0]["duration_sec"] < 2.0 and segments[0]["rms_db"] < -30.0:
        a = segments.pop(0)
        s = _merge_two(a, segments[0])
        s["label"] = "intro"
        s["label_source"] = "leading_silence_merge"
        s["label_confidence"] = "low"
    if len(segments) >= 2 and segments[-1]["duration_sec"] < 2.0:
        tiny = segments.pop()
        s = _merge_two(segments[-1], tiny, keep_second=False)
        segments[-1] = s
    for i, s in enumerate(segments, 1):
        s["i"] = i
    return segments


def combine_segments(a, b):
    """Combine adjacent segments, keeping the dominant cluster; labels are
    recomputed later by label_segments()."""
    dur = a["duration_sec"] + b["duration_sec"]
    pa = 10.0 ** (a["rms_db"] / 10.0)
    pb = 10.0 ** (b["rms_db"] / 10.0)
    da = max(a["duration_sec"], 1e-6)
    db = max(b["duration_sec"], 1e-6)
    return {
        "i": 0,
        "cluster": a["cluster"] if a["duration_sec"] >= b["duration_sec"] else b["cluster"],
        "start_abs": a["start_abs"],
        "end_abs": b["end_abs"],
        "duration_sec": r2(dur),
        "rms_db": r2(10.0 * math.log10(max((pa * da + pb * db) / (da + db), 1e-12))),
        "onset_count": int(a["onset_count"] + b["onset_count"]),
        "onset_density_per_sec": r2((a["onset_count"] + b["onset_count"]) / max(dur, 1e-6)),
    }


def merge_short(segments, duration, min_sec):
    """Merge sub-min_sec fragments into the most similar neighbour.

    A 3-minute loop-based track can otherwise yield dozens of 3-second
    'sections' that are really one chord loop.  This keeps section sizes
    useful for hand-drawn editing while preserving long sections.
    """
    segs = [dict(s) for s in segments]
    while len(segs) > 2:
        idx = min(range(len(segs)), key=lambda i: segs[i]["duration_sec"])
        if segs[idx]["duration_sec"] >= min_sec:
            break
        neighbours = [j for j in (idx - 1, idx + 1) if 0 <= j < len(segs)]
        if not neighbours:
            break
        same = [j for j in neighbours if segs[j]["cluster"] == segs[idx]["cluster"]]
        if same:
            j = same[0]
        else:
            j = min(neighbours, key=lambda k: abs(segs[k]["rms_db"] - segs[idx]["rms_db"]))
        if j < idx:
            new = combine_segments(segs[j], segs[idx])
            segs[j:idx + 1] = [new]
        else:
            new = combine_segments(segs[idx], segs[j])
            segs[idx:j + 1] = [new]
    for i, s in enumerate(segs, 1):
        s["i"] = i
    return segs


def label_segments(segments, duration):
    """Heuristic labels from position, energy, repetition and first occurrence."""
    if not segments:
        return segments
    clusters = {}
    for s in segments:
        clusters.setdefault(s["cluster"], []).append(s)
    stats = {}
    for cid, segs in clusters.items():
        stats[cid] = {
            "count": len(segs),
            "total": sum(s["duration_sec"] for s in segs),
            "mean_db": sum(s["rms_db"] for s in segs) / len(segs),
            "first_start": min(s["start_abs"] for s in segs),
        }
    ranked = sorted(stats, key=lambda c: stats[c]["mean_db"], reverse=True)
    repeated = [c for c in ranked if stats[c]["count"] >= 2]

    # A chorus usually appears after the opening verse, not at 0 s.
    late_repeated = [c for c in repeated if stats[c]["first_start"] >= 0.10 * duration]
    chorus_pool = late_repeated or repeated or ranked
    chorus_cid = max(chorus_pool, key=lambda c: stats[c]["mean_db"]) if chorus_pool else None

    # Multiple verse-like clusters are allowed (verse 1 / verse 2).
    verse_cids = set()
    for c in repeated:
        if c != chorus_cid and stats[c]["first_start"] < 0.12 * duration:
            verse_cids.add(c)
    remaining = [c for c in ranked if c != chorus_cid and c not in verse_cids]
    if not verse_cids and remaining:
        verse_cids.add(max(remaining, key=lambda c: stats[c]["total"]))

    # bridge: unique middle cluster with lower energy than chorus
    bridge_cid = None
    if chorus_cid is not None:
        candidates = [
            c for c in ranked
            if c not in ({chorus_cid} | verse_cids)
            and stats[c]["count"] == 1
            and stats[c]["mean_db"] < stats[chorus_cid]["mean_db"] - 1.0
            and 0.20 * duration < stats[c]["first_start"] < 0.85 * duration
        ]
        if candidates:
            bridge_cid = min(candidates, key=lambda c: stats[c]["mean_db"])

    median_db = sorted(s["rms_db"] for s in segments)[len(segments) // 2]
    for idx, s in enumerate(segments):
        s["cluster_mean_db"] = r2(stats[s["cluster"]]["mean_db"])
        s["cluster_count"] = stats[s["cluster"]]["count"]
        if idx == 0 and s["start_abs"] < 0.5 and (
                s["duration_sec"] < 20.0 or s["rms_db"] < median_db - 0.5):
            s["label"] = "intro"
            s["label_source"] = "opening_position"
            s["label_confidence"] = "low"
        elif idx == len(segments) - 1 and s["end_abs"] >= duration - 0.5 and (
                s["duration_sec"] < 30.0 or s["rms_db"] < median_db - 1.0):
            s["label"] = "outro"
            s["label_source"] = "ending_position"
            s["label_confidence"] = "low"
        elif s["cluster"] == chorus_cid:
            s["label"] = "chorus"
            s["label_source"] = "late_repeated_high_energy"
            s["label_confidence"] = "medium"
        elif s["cluster"] in verse_cids:
            s["label"] = "verse"
            s["label_source"] = "early_or_long_cluster"
            s["label_confidence"] = "medium"
        elif s["cluster"] == bridge_cid:
            s["label"] = "bridge"
            s["label_source"] = "unique_mid_low_energy"
            s["label_confidence"] = "low"
        else:
            s["label"] = "chorus" if s["rms_db"] >= median_db else "verse"
            s["label_source"] = "energy_fallback"
            s["label_confidence"] = "low"
    return segments


def run_allin1(source, device):
    import allin1
    res = allin1.analyze(str(source), device=device)
    raw = []
    for s in getattr(res, "segments", []):
        if isinstance(s, dict):
            raw.append({"start": float(s["start"]), "end": float(s["end"]),
                        "label": str(s["label"])})
        else:
            raw.append({"start": float(s.start), "end": float(s.end),
                        "label": str(s.label)})
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="auto", choices=["auto", "allin1", "librosa"])
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    j = read_json(args.json)
    source = Path(args.source)
    seg_start = float(j["segment"]["start_sec"])
    seg_end = float(j["segment"]["end_sec"])
    seg_dur = float(j["segment"]["duration_sec"])
    wav = Path(j["source"]["segment_wav"])
    if not wav.exists():
        wav = Path(args.json).parent / "segment.wav"
    if not wav.exists():
        raise SystemExit(f"segment wav not found: {wav}")

    # ---- energy / density / jumps (from the existing segment JSON) ---------
    yseg, sr = librosa.load(str(wav), sr=44100, mono=True)
    beats = j.get("beats", [])
    onsets = j.get("onsets", [])
    drums = j.get("drum_onsets", [])
    onset_times = [float(o["t"]) for o in onsets]
    drum_times = [float(o["t"]) for o in drums]

    per_beat = []
    for i, b in enumerate(beats):
        t0 = float(b["t"])
        t1 = float(beats[i + 1]["t"]) if i + 1 < len(beats) else seg_dur
        a = max(0, int(round(t0 * sr)))
        z = min(len(yseg), int(round(t1 * sr)))
        if z <= a:
            z = min(len(yseg), a + 1)
        rms = float(np.sqrt(np.mean(yseg[a:z] ** 2))) if z > a else 0.0
        per_beat.append({
            "i": int(b["i"]),
            "t": r2(t0),
            "abs": r2(seg_start + t0),
            "end_t": r2(t1),
            "end_abs": r2(seg_start + t1),
            "rms_db": r2(20.0 * math.log10(max(rms, 1e-8))),
            "rms_linear": round(rms, 6),
        })

    jumps = []
    for i in range(1, len(per_beat)):
        d = per_beat[i]["rms_db"] - per_beat[i - 1]["rms_db"]
        if abs(d) > 4.0:
            jumps.append({
                "from_beat": per_beat[i - 1]["i"],
                "to_beat": per_beat[i]["i"],
                "t": per_beat[i]["t"],
                "abs": per_beat[i]["abs"],
                "from_db": per_beat[i - 1]["rms_db"],
                "to_db": per_beat[i]["rms_db"],
                "delta_db": r2(d),
                "direction": "up" if d > 0 else "down",
            })

    bars = []
    for bi in range(0, len(beats), 4):
        b0 = beats[bi]
        b1 = beats[min(bi + 4, len(beats) - 1)]
        t0 = float(b0["t"])
        t1 = float(beats[bi + 4]["t"]) if bi + 4 < len(beats) else seg_dur
        count = sum(1 for x in onset_times if t0 <= x < t1)
        dcount = sum(1 for x in drum_times if t0 <= x < t1)
        bars.append({
            "bar": len(bars) + 1,
            "start_beat": int(b0["i"]),
            "end_beat": int(b1["i"]),
            "t": r2(t0),
            "abs": r2(seg_start + t0),
            "end_t": r2(t1),
            "end_abs": r2(seg_start + t1),
            "onset_count": int(count),
            "drum_onset_count": int(dcount),
            "onsets_per_sec": r2(count / max(t1 - t0, 1e-6)),
            "complete_4_beats": bool(bi + 3 < len(beats)),
        })
    mean_bar = sum(b["onset_count"] for b in bars) / len(bars) if bars else 0.0

    energy = {
        "method": "RMS dBFS over the beat-to-next-beat window",
        "unit": "dBFS",
        "beat_count": len(per_beat),
        "min_db": r2(min((x["rms_db"] for x in per_beat), default=0.0)),
        "max_db": r2(max((x["rms_db"] for x in per_beat), default=0.0)),
        "mean_db": r2(sum(x["rms_db"] for x in per_beat) / len(per_beat)) if per_beat else 0.0,
        "per_beat": per_beat,
    }
    density = {
        "method": "onsets per 4-beat bar (4/4 assumed; first beat = bar start)",
        "bar_count": len(bars),
        "mean_onsets_per_bar": r2(mean_bar),
        "bars": bars,
    }
    jumps_obj = {
        "threshold_db": 4.0,
        "definition": "abs(rms_db[beat i+1] - rms_db[beat i]) > 4.0",
        "count": len(jumps),
        "events": jumps,
    }

    # ---- structure ----------------------------------------------------------
    y, sr2 = librosa.load(str(source), sr=22050, mono=True)
    duration = len(y) / sr2
    onset_full = librosa.onset.onset_detect(
        y=y, sr=sr2, hop_length=512, units="time", backtrack=True,
        delta=0.055, wait=2,
    )
    onset_full = np.asarray(onset_full, dtype=float)

    method = args.method
    allin1_reason = ""
    have_allin1 = all(
        importlib.util.find_spec(m) is not None for m in ("allin1", "natten", "madmom")
    )
    if method in ("auto", "allin1"):
        if not have_allin1:
            allin1_reason = "allin1/natten/madmom not importable in this Python"
            if method == "allin1":
                raise SystemExit(allin1_reason)
            method = "librosa"
        else:
            try:
                raw = run_allin1(source, "cuda" if args.device != "cpu" else "cpu")
                times = [0.0] + [x["start"] for x in raw] + [duration]
                labels = [x["label"] for x in raw]
                # derive cluster-like ids from label text for consistency
                uniq = {lab: i for i, lab in enumerate(sorted({x["label"] for x in raw}))}
                segs = []
                for i, x in enumerate(raw):
                    a = x["start"]
                    b = x["end"]
                    ia, ib = int(a * sr2), int(b * sr2)
                    rms = float(np.sqrt(np.mean(y[ia:ib] ** 2))) if ib > ia else 0.0
                    ons = int(np.sum((onset_full >= a) & (onset_full < b)))
                    segs.append({
                        "i": i + 1, "label": x["label"], "cluster": uniq[x["label"]],
                        "start_abs": r2(a), "end_abs": r2(b), "duration_sec": r2(b - a),
                        "rms_db": r2(20.0 * math.log10(max(rms, 1e-8))),
                        "onset_count": ons,
                        "onset_density_per_sec": r2(ons / max(b - a, 1e-6)),
                        "label_source": "allin1", "label_confidence": "high",
                    })
                structure = {
                    "method": "allin1",
                    "scope": "full_track",
                    "duration_sec": r2(duration),
                    "segments": segs,
                    "allin1": {"available": True, "used": True},
                    "notes": ["allin1 1.1.0", f"device={args.device}"],
                }
                method = "allin1"
            except Exception as exc:
                allin1_reason = f"allin1 failed: {exc!r}"
                method = "librosa"

    if method == "librosa":
        t0 = time.time()
        seg = laplacian_segmentation(y, sr2, args.k or None)
        segs = stats_for_intervals(y, sr2, seg["times"], seg["labels"], onset_full)
        segs = merge_edges(segs)
        min_sec = max(8.0, duration / 16.0)
        segs = merge_short(segs, duration, min_sec)
        segs = label_segments(segs, duration)
        structure = {
            "method": "librosa_laplacian",
            "scope": "full_track",
            "duration_sec": r2(duration),
            "k": seg["k"],
            "min_segment_sec": r2(min_sec),
            "tempo_full": r2(seg["tempo"]),
            "beat_count_full": seg["beat_count"],
            "label_method": (
                "short fragments merged toward >= min_segment_sec; first short section= intro; "
                "last short quiet section= outro; late repeated high-energy cluster= chorus; "
                "early repeated / longest clusters= verse; unique mid low-energy cluster= bridge; "
                "remaining by energy vs segment median"
            ),
            "label_note": "Labels are heuristic from audio structure, not ground truth.",
            "allin1": {
                "available": bool(have_allin1),
                "used": False,
                "reason": ALLIN1_NOTE + (f" | last error: {allin1_reason}" if allin1_reason else ""),
            },
            "segments": segs,
            "notes": [
                "librosa 1.0 removed laplacian_segmentation; implemented from the official librosa example "
                "(CQT + recurrence + normalized Laplacian + KMeans).",
                f"allin1 unavailable: {ALLIN1_NOTE}" + (f" | {allin1_reason}" if allin1_reason else ""),
            ],
        }
        print(f"librosa structure in {time.time()-t0:.1f}s, k={seg['k']}, segments={len(segs)}")

    # clip structure to the analyzed segment for convenience
    overlap = []
    for s in structure["segments"]:
        a = max(float(s["start_abs"]), seg_start)
        b = min(float(s["end_abs"]), seg_end)
        if b - a > 0.05:
            overlap.append({
                "label": s["label"],
                "start_abs": r2(a), "end_abs": r2(b),
                "start_t": r2(a - seg_start), "end_t": r2(b - seg_start),
                "duration_sec": r2(b - a),
                "rms_db": s["rms_db"],
                "cluster": s["cluster"],
            })
    structure["segment_overlap"] = overlap
    structure["segment_window"] = {"start_abs": r2(seg_start), "end_abs": r2(seg_end)}

    extra = {
        "structure": structure,
        "energy": energy,
        "density": density,
        "jumps": jumps_obj,
    }
    write_json(args.out, extra)
    print(json.dumps({
        "method": structure["method"],
        "structure_segments": len(structure["segments"]),
        "beats": len(per_beat),
        "bars": len(bars),
        "jumps": len(jumps),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()