# -*- coding: utf-8 -*-
"""Merge structure/energy/density/jumps into a beats JSON, with a local backup.

Only the four new keys are added or replaced.  Every pre-existing key is
verified byte-for-byte (as parsed JSON) before the file is written.
"""
import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

NEW_KEYS = ("structure", "energy", "density", "jumps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.json).expanduser().resolve()
    extra_path = Path(args.extra).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    extra = json.loads(extra_path.read_text(encoding="utf-8"))

    missing = [k for k in NEW_KEYS if k not in extra]
    if missing:
        raise SystemExit(f"extra is missing keys: {missing}")

    backup = None
    if not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = base_path.with_name(base_path.name + f".bak.{ts}")
        if backup.exists():
            backup = base_path.with_name(base_path.name + f".bak.{ts}-1")
        shutil.copy2(base_path, backup)
        print("backup:", backup)

    new = dict(base)
    for k in NEW_KEYS:
        new[k] = extra[k]

    # hard guarantee: old fields are untouched
    for k, v in base.items():
        if k not in NEW_KEYS and new[k] != v:
            raise SystemExit(f"refusing to write: old field changed: {k}")

    tmp = base_path.with_name(base_path.name + ".tmp")
    tmp.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, base_path)

    print("wrote:", base_path)
    print("old keys:", len(base), "new keys:", len(new))
    print("counts:", {
        "structure_segments": len(new["structure"].get("segments", [])),
        "energy_beats": len(new["energy"].get("per_beat", [])),
        "density_bars": len(new["density"].get("bars", [])),
        "jumps": len(new["jumps"].get("events", [])),
    })


if __name__ == "__main__":
    main()