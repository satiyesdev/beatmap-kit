# -*- coding: utf-8 -*-
"""Validate a beats-*.json (structure, NaN, 2-decimal time fields)."""
import json
import math
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
TIME_KEYS = {"t", "abs", "end_t", "end_abs", "start", "end", "dur",
             "held_sec", "nearest_onset_t", "nearest_syllable_onset_t",
             "offset", "seconds", "start_sec", "end_sec", "duration_sec"}
bad_nan, bad_dp = [], []


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    bad_nan.append(path + "." + k)
                if k in TIME_KEYS and round(v, 2) != v:
                    bad_dp.append((path + "." + k, v))
            walk(v, path + "." + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk(v, path + f"[{i}]")


walk(j)
print("file", p, p.stat().st_size, "bytes")
print("schema", j.get("schema"), "title", j.get("title"))
print("counts", json.dumps(j.get("counts", {}), ensure_ascii=False))
print("nan", len(bad_nan), bad_nan[:3])
print("time fields not 2dp", len(bad_dp), bad_dp[:3])
kana = j["vocals"]["whisperx"]["kana_timeline"]
print("kana", len(kana), "sorted", all(a["t"] <= b["t"] for a, b in zip(kana, kana[1:])))
lines = j.get("lyrics_lines", [])
print("lines", len(lines))
for line in lines:
    print(f"  {line['i']:2d} {line['t']:6.2f}-{line['end_t']:6.2f} "
          f"abs={line['abs']:6.2f} {line['confidence']:6s} n={line['kana_count']:2d} "
          f"| {line['text']}")