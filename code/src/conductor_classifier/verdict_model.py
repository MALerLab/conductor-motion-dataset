"""Review-verdict learning model — pure logic (labels, source resolution, features,
calibration, CSV application).

The GPU/IO-heavy extraction (embeddings, pose, scenedetect) lives in verdict_extract.py; this
module holds only testable functions that use numpy/sklearn.

Since apply_review has already deleted the clip files for the 402 reject-labeled rows, source
resolution falls back in the order hold -> clips (whole clip) -> raw original + qualify jsonl
span.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np


def stem_of(row: dict) -> str:
    return f"{row['video_id'].strip()}({str(row['clip_num']).strip().zfill(2)})"


def load_labels(csv_path: Path) -> list[dict]:
    """From the decisions CSV, only accept/reject rows become training labels. y: reject=1."""
    out = []
    for r in csv.DictReader(csv_path.open()):
        v = (r.get("verdict") or "").strip().lower()
        if v not in ("accept", "reject"):
            continue
        out.append({
            "row": r,
            "stem": stem_of(r),
            "video_id": r["video_id"].strip(),
            "y": 1 if v == "reject" else 0,
            "has_crops": bool((r.get("crops") or "").strip()),
        })
    return out


def resolve_source(row: dict, conductor: str,
                   data_dir: Path) -> tuple[Path, tuple[float, float] | None] | None:
    """Resolve a clip's source — hold > clips (whole clip) > raw+qualify span. None if not found.

    Crop-export artifacts (_c00...) are only part of a clip, so they aren't used as a training
    source — if the whole-clip file doesn't exist, it falls back to the raw span (even a
    crop-accept is labeled at the level of the whole clip).
    """
    stem = stem_of(row)
    hold = data_dir / "hold" / conductor / f"{stem}.mp4"
    if hold.exists():
        return hold, None
    clip = data_dir / "clips" / conductor / f"{stem}.mp4"
    if clip.exists():
        return clip, None

    vid = row["video_id"].strip()
    num = str(row["clip_num"]).strip().zfill(2)
    qpath = data_dir / "qualify" / conductor / f"{vid}.jsonl"
    span = None
    if qpath.exists():
        for line in qpath.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("video_id") == vid and rec.get("clip_num") == num \
                    and "start_sec" in rec:
                span = (float(rec["start_sec"]), float(rec["end_sec"]))
    if span is None:
        return None
    raw_dir = data_dir / "raw" / conductor
    if raw_dir.is_dir():
        for p in sorted(raw_dir.iterdir()):
            if f"[{vid}]" in p.name:
                return p, span
    return None


def sample_times(duration_s: float, n: int = 12) -> list[float]:
    """Midpoints of n evenly-spaced segments — avoids the start/end boundaries (fades, titles)."""
    return [(i + 0.5) * duration_s / n for i in range(n)]


def pool_embeddings(embs: np.ndarray) -> np.ndarray:
    """(n,d) per-frame embeddings -> (2d,) mean+max pooling.

    Max pooling captures "is there at least one good conducting moment" — a safeguard that
    protects partially-good clips (which cropping could salvage) from automatic rejection.
    """
    return np.concatenate([embs.mean(axis=0), embs.max(axis=0)])


def motion_stats(gray: np.ndarray) -> dict:
    """(n,h,w) grayscale frames -> mean/p90 of the mean absolute difference between adjacent frames."""
    if len(gray) < 2:
        return {"motion_mean": 0.0, "motion_p90": 0.0}
    d = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))
    return {"motion_mean": float(d.mean()), "motion_p90": float(np.percentile(d, 90))}


def feature_vector(feat: dict) -> np.ndarray:
    """Feature dict -> 1D training vector (embedding pooling only).

    The 4 scalar features (pose_frac, cuts_per_min, motion_mean/p90) were confirmed to
    contribute nothing in the 2026-07-25 ablation (1,427 labels) — OOF AUC full 0.9148 vs
    emb-only 0.9150. Currently disabled via comment-out; consider removing outright later.
    """
    # scalars = [float(feat["pose_frac"]), float(feat["cuts_per_min"]),
    #            float(feat["motion_mean"]), float(feat["motion_p90"])]
    # return np.concatenate([np.asarray(feat["emb_pooled"], dtype=np.float32),
    #                        np.array(scalars, dtype=np.float32)])
    return np.asarray(feat["emb_pooled"], dtype=np.float32)


def make_model():
    """Standardization + logistic regression — a sound baseline at a scale of a few hundred labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def oof_probs(X, y, groups, k: int = 5) -> np.ndarray:
    """Out-of-fold reject probabilities using a video-level group split — prevents clips from
    the same video from appearing in both train and test simultaneously (which would inflate
    performance).

    Uses StratifiedGroupKFold to preserve the class ratio across folds as much as possible.
    Even so, with a small sample containing few groups (e.g. a --limit smoke test), one fold's
    training labels can end up entirely a single class — in that case, instead of training a
    model, that class value is filled in directly (the maximum-likelihood estimate available
    for that fold), returning a low-information probability instead of crashing, so the
    calibration step still works even on small samples.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    k = min(k, len(np.unique(groups)))
    out = np.zeros(len(y), dtype=float)
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=False).split(X, y, groups):
        y_tr = y[tr]
        if len(np.unique(y_tr)) < 2:
            out[te] = float(y_tr[0])
            continue
        m = make_model().fit(X[tr], y_tr)
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def calibrate_threshold(y, p, min_precision: float = 0.97,
                        min_support: int = 20) -> float | None:
    """The lowest threshold that satisfies the precision/support constraints (= maximum coverage).

    min_support: a lower bound on sample count, so that an accidental 100% precision on a
    handful of samples isn't adopted as the threshold.
    Returns None if no threshold satisfies the constraints — the caller must not auto-reject in that case.
    """
    y = np.asarray(y); p = np.asarray(p)
    best = None
    for t in np.unique(p)[::-1]:
        pred = p >= t
        if int(pred.sum()) < min_support:
            continue
        if float(y[pred].mean()) >= min_precision:
            best = float(t)
    return best


def crop_subgroup_report(labels: list[dict], p, threshold) -> dict:
    """Monitors wrongful-discard rate for the crop-accept (partially-good) subgroup — the
    protection item for the crop targets called out in the spec."""
    idx = [i for i, l in enumerate(labels) if l["y"] == 0 and l["has_crops"]]
    if threshold is None or not idx:
        return {"n": len(idx), "false_reject": 0, "rate": None}
    p = np.asarray(p)
    fr = int(sum(p[i] >= threshold for i in idx))
    return {"n": len(idx), "false_reject": fr, "rate": fr / len(idx)}


def apply_model_to_csv(decisions: Path, scores: dict[str, float],
                       threshold: float | None, tag: str) -> tuple[int, int]:
    """Records model_p on pending rows, and for rows at/above the threshold sets
    verdict=reject, model_auto=tag.

    Rows that already have a verdict (human or an existing prefill) are left untouched in every
    column. If threshold is None (calibration failed), only the score is recorded. Writes
    atomically (the review_ui pattern). Returns (number auto-rejected, total row count).
    """
    with decisions.open() as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        fields = list(rd.fieldnames or [])
    for col in ("model_p", "model_auto"):
        if col not in fields:
            fields.append(col)
    n_auto = 0
    for r in rows:
        r.setdefault("model_p", "")
        r.setdefault("model_auto", "")
        if (r.get("verdict") or "").strip():
            continue
        s = scores.get(stem_of(r))
        if s is None:
            continue
        r["model_p"] = f"{s:.4f}"
        if threshold is not None and s >= threshold:
            r["verdict"] = "reject"
            r["model_auto"] = tag
            n_auto += 1
    tmp = decisions.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, decisions)
    return n_auto, len(rows)
