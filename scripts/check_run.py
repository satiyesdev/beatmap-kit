# -*- coding: utf-8 -*-
"""Print a quick human-readable check of a beats-*.json file."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
j = json.loads(p.read_text(encoding="utf-8"))
print("file", p, p.stat().st_size, "bytes")
print("schema", j.get("schema"), "title", j.get("title"), "artist", j.get("artist"))
print("counts", json.dumps(j.get("counts", {}), ensure_ascii=False))
print("tempo", json.dumps(j.get("tempo", {}), ensure_ascii=False))
print("loudness", [(c["t"], c["direction"], c["delta_db"]) for c in j.get("loudness", {}).get("changes", [])])
print("lines:")
for line in j.get("lyrics_lines", []):
    print(f"  {line['i']:2d} {line['t']:6.2f}-{line['end_t']:6.2f} "
          f"abs={line['abs']:6.2f} {line['confidence']:6s} n={line['kana_count']:2d} "
          f"| {line['text']} | {line['kana']}")
kana = j.get("vocals", {}).get("whisperx", {}).get("kana_timeline", [])
print("kana", len(kana), "low",
      sum(1 for k in kana if k.get("confidence") == "low"),
      "medium", sum(1 for k in kana if k.get("confidence") == "medium"),
      "high", sum(1 for k in kana if k.get("confidence") == "high"))
print("summary", j.get("summary_zh"))