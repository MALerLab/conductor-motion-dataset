"""Run-level identity review — row generation, inheritance, preview span, verdict recording.

Spec: docs/superpowers/specs/2026-08-07-run-review-pipeline-design.md.
The original skeleton.npy is immutable; every decision is recorded only in
run_verdicts.csv. Clip verdicts (accept/reject) are read from
<conductor>_decisions.csv purely for inheritance — that file is never modified.
"""
import csv
import datetime
import os
import re
import threading
from pathlib import Path

from .skeleton_qc import dispose_run

RUN_COLUMNS = ["conductor", "video_id", "clip_num", "run", "run_dir", "fps",
               "n_frames", "kept_frames", "longest_seg_s", "disposition",
               "source_verdict", "verdict", "preview", "judged_at"]

# accept=is the conductor and actually conducting / reject=someone else /
# nonconduct=is the conductor but not conducting (bowing, applause, turning
# a score page, etc.) / skeljump=mostly the conductor but the skeleton track
# jumps to a different person partway through — QC missed the identity
# switch, so this is **not discarded but flagged for threshold recalibration
# and re-splitting**. Mixing it with rejects would lose the recalibration
# sample, so it is tracked separately.
VERDICTS = ("accept", "reject", "nonconduct", "skeljump", "hold", "")

_STEM_RE = re.compile(r"^(?P<vid>[\w-]{11})\((?P<num>\d+)\)(?:_c\d+)?$")


def parse_clip_stem(stem: str) -> tuple[str, str]:
    """"<vid>(NN)[_cMM]" -> (video_id, clip number as a string of natural digits). Crop derivatives resolve to the original."""
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"not a valid clip stem: {stem!r}")
    return m.group("vid"), str(int(m.group("num")))


def run_rows(runs: list[dict], decisions: dict[tuple[str, str], str],
             min_seg_seconds: float = 2.0) -> list[dict]:
    """run scan results -> run_verdicts rows. Excludes runs from rejected clips; inherits accepts.

    hold / unjudged / no decision are left with an empty verdict, awaiting review.
    No qc means no_qc (excluded from the review queue).
    """
    out = []
    for r in runs:
        stem = r["meta"]["source_clip"].removesuffix(".mp4")
        vid, num = parse_clip_stem(stem)
        clip_verdict = decisions.get((vid, num), "")
        if clip_verdict == "reject":
            continue
        qc = r["qc"]
        if qc is None:
            disp, fps, nf, kept, longest = "no_qc", "", "", "", ""
        else:
            disp = dispose_run(qc, min_seg_seconds)
            fps = f"{qc['fps']:.3f}"
            nf, kept = qc["n_frames"], qc["kept_frames"]
            seg = max((b - a for a, b in qc["segments"]), default=0)
            longest = f"{seg / qc['fps']:.2f}"
        out.append({"conductor": r["conductor"], "video_id": vid, "clip_num": num,
                    "run": r["run"], "run_dir": r["run_dir"], "fps": fps,
                    "n_frames": nf, "kept_frames": kept, "longest_seg_s": longest,
                    "disposition": disp,
                    "source_verdict": clip_verdict if clip_verdict == "accept" else "",
                    "verdict": "", "preview": "", "judged_at": ""})
    return out


def preview_span(qc: dict, max_seconds: float = 3.0) -> tuple[int, int]:
    """Review preview span (frames relative to the run) — longest clean segment, centered, capped at 3 seconds.

    This only selects the review view; it does not trim the underlying data (non-destructive principle).
    """
    a, b = max(qc["segments"], key=lambda s: s[1] - s[0])
    limit = int(round(max_seconds * qc["fps"]))
    if b - a <= limit:
        return a, b
    start = a + (b - a - limit) // 2
    return start, start + limit


def interleave_by_recording(rows: list[dict]) -> list[dict]:
    """Interleave rows by recording so reviewing from the top still covers many recordings evenly.

    Sorting by video would let a reviewer burn through many judgments on just
    the first few videos, but the evaluation is **per-recording LORO**, so the
    number of distinct recordings is the effective sample size (Haitink had
    273 judgments covering only 14 recordings, as an example of the failure
    mode). Round-robin ordering secures many more distinct recordings for the
    same review effort.
    """
    buckets: dict = {}
    for r in rows:
        buckets.setdefault((r["conductor"], r["video_id"]), []).append(r)
    for v in buckets.values():
        v.sort(key=lambda r: r["run"])
    order = sorted(buckets)
    out: list[dict] = []
    i = 0
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                out.append(buckets[k].pop(0))
        i += 1
    return out


def merge_rows(new_rows: list[dict], old_rows: list[dict]) -> list[dict]:
    """Merge on re-scan — a human verdict is never erased.

    For rows present in the new scan: verdict keeps the old value if present,
    preview prefers the new value (freshly rendered). For old rows absent from
    the new scan: if a human verdict exists, keep the row entirely (prevents a
    partial scan or a lost-folder re-run from wiping out judgments); otherwise drop it.
    """
    old = {(r["conductor"], r["run"]): r for r in old_rows}
    seen = set()
    for r in new_rows:
        key = (r["conductor"], r["run"])
        seen.add(key)
        prev = old.get(key)
        if prev:
            r["verdict"] = prev.get("verdict", "") or r.get("verdict", "")
            r["preview"] = r.get("preview", "") or prev.get("preview", "")
            r["judged_at"] = prev.get("judged_at", "")
    orphans = [r for k, r in old.items()
               if k not in seen and (r.get("verdict") or "").strip()]
    return new_rows + orphans


def save_rows(csv_path: Path, rows: list[dict]) -> None:
    """Atomic write (unique tmp -> replace) — preserves unknown columns too (upsert_row pattern)."""
    extra = []
    for r in rows:
        extra += [k for k in r if k not in RUN_COLUMNS and k not in extra]
    fields = RUN_COLUMNS + extra
    tmp = csv_path.with_name(f".tmp-{os.getpid()}-{id(rows)}-{csv_path.name}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in fields} for r in rows)
    os.replace(tmp, csv_path)


_WRITE_LOCK = threading.Lock()


def record_run_verdict(csv_path: Path, conductor: str, run: str, verdict: str) -> None:
    """Update only that run row's verdict via an atomic rewrite (review_ui.upsert_row pattern).

    An in-process lock serializes read-modify-write so that concurrent POSTs
    on the ThreadingHTTPServer can't clobber each other's verdicts.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    with _WRITE_LOCK:
        rows = list(csv.DictReader(csv_path.open()))
        hit = [r for r in rows if r["conductor"] == conductor and r["run"] == run]
        if not hit:
            raise KeyError(f"run not found: {conductor}/{run}")
        hit[0]["verdict"] = verdict
        hit[0]["judged_at"] = (
            datetime.datetime.now().isoformat(timespec="seconds") if verdict else "")
        save_rows(csv_path, rows)


def undo_last(csv_path: Path, conductor: str = "") -> dict | None:
    """Revert the single most recent verdict back to unjudged — for fixing typos/mis-clicks.

    If judged_at is present, picks the max of those; otherwise (legacy
    records) picks the last row in queue order. Inherited accepts
    (source_verdict) are excluded since they aren't human judgments.
    Returns the reverted row, or None if there was nothing to revert.
    """
    with _WRITE_LOCK:
        rows = list(csv.DictReader(csv_path.open()))
        cand = [r for r in rows
                if r["verdict"].strip() and not r["source_verdict"].strip()
                and (not conductor or r["conductor"] == conductor)]
        if not cand:
            return None
        stamped = [r for r in cand if r.get("judged_at", "").strip()]
        pick = (max(stamped, key=lambda r: r["judged_at"]) if stamped
                else max(cand, key=lambda r: (r["conductor"], r["video_id"], r["run"])))
        undone = dict(pick)
        pick["verdict"] = ""
        pick["judged_at"] = ""
        save_rows(csv_path, rows)
        return undone
