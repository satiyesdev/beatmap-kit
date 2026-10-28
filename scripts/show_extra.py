# -*- coding: utf-8 -*-
"""Print structure / energy / density / jumps from an extra.json."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
s = j.get("structure", {})
print("file", p)
print("method", s.get("method"), "scope", s.get("scope"), "k", s.get("k"),
      "segments", len(s.get("segments", [])))
for x in s.get("segments", []):
    print(f"  {x['i']:2d} {x['start_abs']:7.2f}-{x['end_abs']:7.2f} "
          f"{x.get('label',''):8s} db={x.get('rms_db')} "
          f"dens={x.get('onset_density_per_sec')} cluster={x.get('cluster')}")
print("overlap:")
for x in s.get("segment_overlap", []):
    print(f"  {x['start_t']:6.2f}-{x['end_t']:6.2f} {x['label']:8s} {x['rms_db']}")
e = j.get("energy", {})
print("energy", e.get("beat_count"), e.get("min_db"), e.get("max_db"), e.get("mean_db"))
d = j.get("density", {})
print("density", d.get("bar_count"), d.get("mean_onsets_per_bar"))
print("jumps", j.get("jumps", {}).get("count"))
for x in j.get("jumps", {}).get("events", []):
    print("  ", x)