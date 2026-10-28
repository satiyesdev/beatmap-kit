# -*- coding: utf-8 -*-
"""Inspect kana / onsets / lines in a time range (segment-relative seconds)."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
a, b = float(sys.argv[2]), float(sys.argv[3])
j = json.loads(p.read_text(encoding="utf-8"))
print(f"region {a:.2f}-{b:.2f}")
print("kana:")
for k in j["vocals"]["whisperx"]["kana_timeline"]:
    if a - 0.2 <= k["t"] <= b + 0.2:
        print(f"  {k['i']:3d} {k['t']:6.2f}-{k['end_t']:6.2f} {k['kana']} score={k['score']:.3f} line={k.get('line')}")
print("vocal onsets:")
for o in j["vocals"]["syllable_onsets"]:
    if a - 0.2 <= o["t"] <= b + 0.2:
        print(f"  {o['i']:3d} {o['t']:6.2f} strength={o['strength']:.2f} rise={o['energy_rise_db']:+.1f}")
print("lines:")
for line in j["lyrics_lines"]:
    if line["end_t"] >= a - 0.2 and line["t"] <= b + 0.2:
        print(f"  {line['i']} {line['t']:.2f}-{line['end_t']:.2f} {line['text']}")