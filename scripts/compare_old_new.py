# -*- coding: utf-8 -*-
"""Verify that a merge only added/replaced allowed keys and changed nothing else."""
import json
import sys
from pathlib import Path

old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
new = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
allowed = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else {
    "structure", "energy", "density", "jumps"
}
missing = sorted(set(old) - set(new))
changed = []
for k, v in old.items():
    if k in allowed:
        continue
    if new.get(k) != v:
        changed.append(k)
print("old keys", len(old), "new keys", len(new))
print("missing old keys", missing)
print("changed old keys", changed)
print("new keys", sorted(set(new) - set(old)))
print("allowed", sorted(allowed))
if missing or changed:
    raise SystemExit(1)
print("OK: only allowed keys added/replaced")