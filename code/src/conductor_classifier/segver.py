"""Segment-identity verdict (segment_verdicts.csv) loader — filters out contaminated segments.

Background (DEVLOG 2026-08-13): the run-identity review preview shows **only the
middle 3 seconds of the longest clean segment**, so the remaining segments of a
multi-segment run (18% of total clean footage) have never been seen by human eyes.
An audit found that part of that footage is contaminated by a non-conductor person
(e.g. a performer) in a sample of 9/60 runs, so we keep a segment-level verdict CSV
and have the training loader exclude contaminated segments from window extraction.

CSV schema: conductor,run,seg_idx,verdict,source,judged_at
- seg_idx is the **segments index from qc.json** (in native fps, before rescaling).
- Only verdict 'contaminated' is excluded. 'clean' / unjudged pass through.
The original skeleton.npy/qc.json stay immutable — verdicts live only as CSV
metadata (non-destructive principle).
"""
import csv
from pathlib import Path


def load_video_exclude(path: str | Path) -> frozenset[tuple[str, str]]:
    """Set of (conductor, video_id) videos excluded from training. Empty set if the file is missing.

    The OOD set (rehearsals, etc.) only means something if it stays held-out, but
    if the review UI accepts a run because "the conductor is visible," it stays
    marked accept in run_verdicts and can get pulled into the next snapshot
    (observed 2026-08-26: 37 accepted runs turned up among OOD videos). Relying
    only on excluding them at snapshot-build time depends on human memory, so we
    add a second block at the training entry point.
    """
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(
        (r["conductor"], r["video_id"])
        for r in csv.DictReader(p.open())
        if r.get("video_id"))


def load_blacklist(path: str | Path) -> frozenset[tuple[str, str, int]]:
    """Set of (conductor, run, seg_idx) verdicts marked contaminated. Empty set if the file is missing."""
    p = Path(path)
    if not p.exists():
        return frozenset()
    return frozenset(
        (r["conductor"], r["run"], int(r["seg_idx"]))
        for r in csv.DictReader(p.open())
        if (r.get("verdict") or "").strip() == "contaminated"
    )
