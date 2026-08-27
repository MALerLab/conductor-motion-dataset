"""Year-entry tab logic — scanning representative clips, filtering/sorting rows, saving year_manual.

Only touches year_manual in recordings_meta.csv (other columns/rows are
preserved, and writes are atomic).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .recmeta import normalize_year_input
from .review_ui import CROP_SUFFIX_RE, parse_stem


def sample_clips(clips_root: Path) -> dict[tuple[str, str], str]:
    """Scan data/clips/<conductor>/ -> {(conductor, video_id): representative clip filename}.

    Cropped derivatives (_cNN) are excluded; the first clip in sort order
    (usually (01)) is used as the representative."""
    out: dict[tuple[str, str], str] = {}
    for sub in sorted(p for p in clips_root.iterdir() if p.is_dir()):
        for clip in sorted(sub.glob("*.mp4")):
            if CROP_SUFFIX_RE.search(clip.stem):
                continue
            try:
                vid, _ = parse_stem(clip.stem)
            except ValueError:
                continue
            out.setdefault((sub.name, vid), clip.name)
    return out


def year_rows(meta_rows: list[dict], samples: dict[tuple[str, str], str],
              conductor: str | None = None, view: str = "clips") -> list[dict]:
    """Meta rows -> tab display rows. view: clips (only recordings with an accepted run)/all/empty (accepted + not yet entered).

    Sorted with not-yet-entered rows first, then by conductor and title."""
    out = []
    for r in meta_rows:
        c, vid = r["conductor"], r["video_id"]
        if conductor and c != conductor:
            continue
        clip = samples.get((c, vid))
        filled = bool(r.get("year_manual", "").strip())
        if view in ("clips", "empty") and clip is None:
            continue
        if view == "empty" and filled:
            continue
        out.append({"conductor": c, "video_id": vid, "title": r.get("title", ""),
                    "year_title": r.get("year_title", ""),
                    "year_manual": r.get("year_manual", ""), "clip": clip})
    out.sort(key=lambda r: (bool(r["year_manual"].strip()),
                            r["conductor"], r["title"]))
    return out


def set_year(csv_path: Path, video_id: str, value: str) -> str:
    """Update year_manual for the given recording — raises ValueError on bad format, KeyError if not found.

    Atomic read-modify-write (temp file -> replace). Returns the normalized value."""
    norm = normalize_year_input(value)
    if norm is None:
        raise ValueError(f"not a valid year format: {value!r} (e.g. 1985 or 1980s)")
    rows = list(csv.DictReader(csv_path.open()))
    fields = list(rows[0].keys()) if rows else []
    hit = next((r for r in rows if r["video_id"].strip() == video_id), None)
    if hit is None:
        raise KeyError(f"video_id not found: {video_id}")
    hit["year_manual"] = norm
    tmp = csv_path.with_name(f".tmp-{time.time_ns()}-{csv_path.name}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)
    return norm
