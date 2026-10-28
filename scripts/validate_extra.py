# -*- coding: utf-8 -*-
"""Validate the four added fields (structure/energy/density/jumps)."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
for k in ("structure", "energy", "density", "jumps"):
    if k not in j:
        raise SystemExit(f"missing {k}")

s = j["structure"]
segs = s.get("segments", [])
assert segs and all(a["start_abs"] <= b["start_abs"] for a, b in zip(segs, segs[1:]))
assert all(x["end_abs"] > x["start_abs"] for x in segs)
assert s["segment_overlap"], "empty segment_overlap"
print("structure", s.get("method"), "k", s.get("k"), "segments", len(segs),
      "overlap", len(s["segment_overlap"]))
print("labels", [x["label"] for x in segs])
print("allin1", s.get("allin1"))

e = j["energy"]
per = e.get("per_beat", [])
assert len(per) == len(j["beats"]), (len(per), len(j["beats"]))
assert all(x["end_t"] >= x["t"] for x in per)
print("energy", len(per), "range", e.get("min_db"), e.get("max_db"))

d = j["density"]
bars = d.get("bars", [])
assert bars
print("density", len(bars), "mean", d.get("mean_onsets_per_bar"),
      "last complete", bars[-1].get("complete_4_beats"))

jp = j["jumps"]
for ev in jp.get("events", []):
    assert abs(ev["delta_db"]) > 4.0
    assert (ev["delta_db"] > 0) == (ev["direction"] == "up")
print("jumps", jp.get("count"), "threshold", jp.get("threshold_db"))
print("OK")