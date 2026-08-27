"""Core logic for the manual review web UI — queue loading, verdict recording,
Range parsing, crop export.

Shared by scripts/review_ui.py (HTTP server) and scripts/apply_review.py.
Verdict recording always re-reads the CSV, patches only the matching row, and
writes it back atomically (read-modify-write) — this minimizes the collision
window with other writers (e.g. the prefill script).

A single clip can have multiple crop segments: stored in the CSV `crops`
column as "1.5-8.0;12.0-20.1", and apply_review trims each segment to
`<stem>_c00.mp4, _c01.mp4 ...`.
"""
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

VERDICTS = ("accept", "reject", "hold", "")  # "" = verdict cleared

STEM_RE = re.compile(r"^(?P<vid>.+)\((?P<num>\d+)\)$")
CROP_SUFFIX_RE = re.compile(r"_c\d+$")


def parse_stem(stem: str) -> tuple[str, str]:
    """"<video_id>(NN)" -> (video_id, "NN"). Crop filenames (..._cNN) don't match this format."""
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(f"not a valid stem format: {stem!r}")
    return m.group("vid"), m.group("num").zfill(2)


def _p90_key(row: dict) -> float:
    """Triage sort philosophy: ascending p90, with zero-face rows (empty value) sent to the front via -1.0."""
    v = (row.get("rescore_p90") or "").strip()
    return float(v) if v else -1.0


ACCEPT_FIRST = False   # If True, start from the lowest reject-probability rows (accept candidates) —
                       # for minority-class data augmentation review (2026-08-26, review_ui --accept-first)


def _queue_key(row: dict) -> tuple:
    """If model_p is present, prioritize descending reject probability (clear-cut cases first);
    otherwise fall back to the original triage order (ascending p90, empty values first).
    When ACCEPT_FIRST is set, reverse this — start from accept candidates (low reject probability)."""
    mp = (row.get("model_p") or "").strip()
    if mp:
        p = float(mp)
        return (0, p if ACCEPT_FIRST else -p)
    return (1, _p90_key(row))


def load_queue(csv_path: Path, only_pending: bool = True) -> list[dict]:
    """Review queue — by default only rows with an empty verdict, ascending rescore_p90. Adds a stem column."""
    rows = list(csv.DictReader(csv_path.open()))
    if only_pending:
        rows = [r for r in rows if not (r.get("verdict") or "").strip()]
    rows.sort(key=_queue_key)
    for r in rows:
        r["stem"] = f"{r['video_id'].strip()}({str(r['clip_num']).strip().zfill(2)})"
    return rows


def format_crops(crops: list[tuple[float, float]]) -> str:
    """[(start,end),...] -> "s-e;s-e" (sorted by start time). Rejects inverted segments."""
    for s, e in crops:
        if e <= s:
            raise ValueError(f"inverted crop segment: {s}-{e}")
    return ";".join(f"{s}-{e}" for s, e in sorted(crops))


def parse_crops(s: str) -> list[tuple[float, float]]:
    out = []
    for part in (s or "").split(";"):
        if part.strip():
            a, _, b = part.partition("-")
            out.append((float(a), float(b)))
    return out


def crops_of(row: dict) -> list[tuple[float, float]]:
    """Read crop segments from a CSV row — [] if the column is missing or empty."""
    return parse_crops(row.get("crops") or "")


def record_verdict(csv_path: Path, video_id: str, clip_num: str, verdict: str,
                   crops: list[tuple[float, float]] | None = None) -> dict:
    """Update only that row's verdict (+crops) and save atomically. Returns the updated row.

    A verdict submitted without crops clears any leftover crops on the row (prevents stale
    crops from surviving a reversal).
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}: {verdict!r}")
    crops_s = format_crops(crops or [])

    rows = list(csv.DictReader(csv_path.open()))
    fields = list(rows[0].keys()) if rows else []
    if "crops" not in fields:
        fields.append("crops")
    num = str(clip_num).strip().zfill(2)
    hit = None
    for r in rows:
        r.setdefault("crops", "")
        if r["video_id"].strip() == video_id.strip() \
                and str(r["clip_num"]).strip().zfill(2) == num:
            r["verdict"] = verdict
            r["crops"] = crops_s
            hit = r
    if hit is None:
        raise KeyError(f"row not found: {video_id}({num}) in {csv_path}")

    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    return hit


# ------------------------------------------------------------- grid board
#
# Places verdicted clips (accept/hold/reject) and qualify's auto-ACCEPT clips
# on one screen, and lets the user drag clips to change their verdict. The
# relationship between state and file location:
#   CSV row + hold file      : verdict recorded only (not yet applied) — freely movable via record_verdict
#   CSV accept + clips file  : applied by apply_review — demoting moves the file back clips->hold
#   CSV accept + _cNN file   : crop already trimmed (original gone) — locked
#   CSV reject, no file      : applied and deleted — locked
#   clips file, no CSV row   : qualify's auto-ACCEPT — demoting moves to hold + creates a new row


def _find_row(rows: list[dict], video_id: str, clip_num: str) -> dict | None:
    num = str(clip_num).strip().zfill(2)
    for r in rows:
        if r["video_id"].strip() == video_id.strip() \
                and str(r["clip_num"]).strip().zfill(2) == num:
            return r
    return None


def upsert_row(csv_path: Path, video_id: str, clip_num: str, verdict: str,
               defaults: dict | None = None) -> dict:
    """Update the verdict if the row exists, otherwise add a new row from defaults (atomic write).

    Used when demoting an auto-ACCEPT clip that qualify never put into the hold CSV, to bring it
    into the review ledger. New columns (keys from defaults) are appended to the header.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}: {verdict!r}")
    rows = list(csv.DictReader(csv_path.open()))
    fields = list(rows[0].keys()) if rows else ["video_id", "clip_num", "verdict"]
    hit = _find_row(rows, video_id, clip_num)
    if hit is None:
        hit = {k: "" for k in fields}
        hit.update(defaults or {})
        hit["video_id"] = video_id
        hit["clip_num"] = str(clip_num).strip().zfill(2)
        rows.append(hit)
        for k in hit:
            if k not in fields:
                fields.append(k)
    hit["verdict"] = verdict
    for r in rows:
        for k in fields:
            r.setdefault(k, "")
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    return hit


def _locate(row: dict, stem: str, hold: Path, clips: Path) -> tuple[str | None, bool]:
    """The row's playable filename and lock status. (file, locked) — file is a dir-relative name."""
    if (hold / f"{stem}.mp4").exists():
        return f"{stem}.mp4", False
    if (clips / f"{stem}.mp4").exists():
        return f"{stem}.mp4", False
    crops = crops_of(row)
    if crops:
        c0 = crop_dsts(clips / f"{stem}.mp4", crops)[0]
        if c0.exists():
            return c0.name, True  # trim already applied — original is gone, cannot revert
    return None, True  # applied (deleted), or otherwise missing


def grid_groups(rows: list[dict], hold_dir: Path, clips_dir: Path) -> dict:
    """load_queue(only_pending=False) rows -> {accept|hold|reject: [tiles]}.

    clips/*.mp4 files not present in the CSV (excluding crop outputs) are folded in as
    auto-ACCEPT tiles.
    """
    out = {"accept": [], "hold": [], "reject": []}
    stems = set()
    for r in rows:
        stems.add(r["stem"])
        v = (r.get("verdict") or "").strip()
        if v not in out:
            continue  # unverdicted rows stay off the grid board (review tab only)
        file, locked = _locate(r, r["stem"], hold_dir, clips_dir)
        out[v].append({"stem": r["stem"], "video_id": r["video_id"].strip(),
                       "clip_num": str(r["clip_num"]).strip().zfill(2),
                       "dur_s": r.get("dur_s", ""), "crops": r.get("crops", ""),
                       "auto": False, "file": file, "locked": locked})
    if clips_dir.is_dir():
        for p in sorted(clips_dir.glob("*.mp4")):
            if CROP_SUFFIX_RE.search(p.stem) or p.stem in stems:
                continue
            try:
                vid, num = parse_stem(p.stem)
            except ValueError:
                continue  # leave non-conforming filenames untouched
            out["accept"].append({"stem": p.stem, "video_id": vid, "clip_num": num,
                                  "dur_s": "", "crops": "", "auto": True,
                                  "file": p.name, "locked": False})
    for v in out:
        out[v].sort(key=lambda t: t["stem"])
    return out


def grid_move(csv_path: Path, stem: str, to: str,
              hold_dir: Path, clips_dir: Path) -> dict:
    """Apply a grid-board drag move. Returns the updated CSV row.

    Moves the file first, then writes the CSV — if the process dies mid-way, the CSV stays on
    the safe side of lagging behind the file location (i.e. looking like it wasn't applied yet)
    rather than the reverse.
    """
    if to not in ("accept", "hold", "reject"):
        raise ValueError(f"move target must be accept/hold/reject: {to!r}")
    vid, num = parse_stem(stem)
    rows = list(csv.DictReader(csv_path.open()))
    row = _find_row(rows, vid, num)
    hold_f = hold_dir / f"{stem}.mp4"
    clips_f = clips_dir / f"{stem}.mp4"

    if row is None:  # qualify's auto-ACCEPT — only enters the CSV when demoted
        if not clips_f.exists():
            raise KeyError(f"auto-accepted clip not found: {clips_f}")
        if to == "accept":
            return {"video_id": vid, "clip_num": num, "verdict": "accept"}  # already in place
        hold_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(clips_f), str(hold_f))
        return upsert_row(csv_path, vid, num, to,
                          defaults={"hold_why": "auto_accept demoted (grid board)"})

    if hold_f.exists():  # not yet applied — just changing the verdict is enough
        crops = crops_of(row) if to == "accept" else None
        return record_verdict(csv_path, vid, num, to, crops=crops)

    if clips_f.exists():  # accept already applied — move the file back if demoting
        if to != "accept":
            hold_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(clips_f), str(hold_f))
        crops = crops_of(row) if to == "accept" else None
        return record_verdict(csv_path, vid, num, to, crops=crops)

    raise PermissionError(f"cannot move (locked): {stem} — crop already trimmed, or file was deleted")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """HTTP Range header -> (start, end) byte offsets (inclusive). None if invalid."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def find_ffmpeg() -> Path:
    """Prefer the ffmpeg bundled with the venv (next to the python binary), else fall back to PATH."""
    cand = Path(sys.executable).parent / "ffmpeg"
    if cand.exists():
        return cand
    found = shutil.which("ffmpeg")
    if not found:
        raise FileNotFoundError("ffmpeg not found (venv bin / PATH)")
    return Path(found)


def crop_dsts(dst: Path, crops: list[tuple[float, float]]) -> list[Path]:
    """Per-segment output paths — <stem>_c00.mp4 ... (the video_id parser recognizes this suffix)."""
    return [dst.with_stem(f"{dst.stem}_c{i:02d}") for i in range(len(crops))]


def export_clip(src: Path, dst: Path, crops: list[tuple[float, float]] | None = None) -> None:
    """Move an accepted clip into clips — a plain move if there are no crops, otherwise trim
    each segment with ffmpeg.

    Re-encoding (rather than stream-copy) is used for keyframe-boundary accuracy; clips are
    short enough that this isn't costly. The original is only deleted after every segment has
    been exported.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not crops:
        src.rename(dst)
        return
    for (s, e), out in zip(sorted(crops), crop_dsts(dst, crops)):
        subprocess.run([str(find_ffmpeg()), "-y", "-i", str(src),
                        "-ss", str(s), "-to", str(e),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-c:a", "aac", str(out)],
                       check=True, capture_output=True)
    src.unlink()
