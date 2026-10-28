# -*- coding: utf-8 -*-
"""Shared helpers for the 手书工具箱 steps (stdlib only at import time)."""
import json
import os
import re
import unicodedata
from pathlib import Path

KIT = Path(os.environ.get("SHOUSHU_KIT", Path(__file__).resolve().parents[1]))
MODELS = KIT / "models"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def r2(x):
    return round(float(x), 2)


def parse_time(value):
    """'0:21', '00:21.5', '1:02:03', '21', '21.5' -> seconds."""
    s = str(value).strip()
    if not s:
        raise ValueError("empty time")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"bad time: {value}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"bad time: {value}") from None
    if any(n < 0 for n in nums):
        raise ValueError(f"negative time: {value}")
    sec = 0.0
    for n in nums:
        sec = sec * 60.0 + n
    return sec


def fmt_sec(sec):
    sec = float(sec)
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:05.2f}"


def is_kana(ch: str) -> bool:
    o = ord(ch)
    return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF) or ch == "ー"


_kks = None


def to_kana(text: str) -> str:
    """Japanese text -> hiragana-only (punctuation/spaces/Latin dropped)."""
    global _kks
    if _kks is None:
        import pykakasi
        _kks = pykakasi.kakasi()
    out = []
    for item in _kks.convert(text):
        for ch in item.get("hira", ""):
            if is_kana(ch):
                out.append(ch)
    return "".join(out)


def parse_lyrics(path):
    """Read a lyrics file.

    One lyric line per line.  Optional per-line reading after the first '|':
        病める時も 健やかな時も | やめるときもすこやかなときも
    Blank lines and lines starting with '#' are ignored.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "|" in s:
            body, kana = s.split("|", 1)
            body = body.strip()
            kana = "".join(ch for ch in kana.strip() if is_kana(ch))
        else:
            body, kana = s, ""
        if not body:
            continue
        lines.append({"text": body, "kana_override": kana})
    return lines


def slugify(title: str, fallback="track"):
    """Filesystem-safe slug; keeps CJK characters, drops punctuation/spaces."""
    s = unicodedata.normalize("NFKC", title or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\u3040-\u30ff\u4e00-\u9fff.-]+", "", s, flags=re.UNICODE)
    s = s.strip("._-")
    if not s:
        s = fallback
    return s[:80]


def split_artist_title(stem: str):
    """'Dannie May - 未完成婚姻論' -> ('Dannie May', '未完成婚姻論')."""
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", stem.strip()