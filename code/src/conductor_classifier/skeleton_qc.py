"""Raw-skeleton QC — detecting jumps (teleports/ID switches) and splitting into clean segments.

Runs on skeleton.npy (original pixel coordinates) **before** normalization —
it needs to catch spikes before normalize's low-pass filter (6Hz) smears them
into smooth contamination.

Everything is in units of shoulder-widths/second: independent of resolution
(pixel size) and fps, so a single threshold applies across all recordings.
Transitions we can't judge (insufficient joint confidence) are not counted as
spikes — this avoids an era bias where "can't see it" gets misjudged as
"jumped," which would over-trim only old low-quality recordings.
"""
import numpy as np

from . import keypoints as kp

_EPS = 1e-8
SPEED_TH = 25.0   # wrist speed cap (shoulder-widths/second) — vigorous conducting ~6-13, a teleport ~75+
JUMP_TH = 3.0     # shoulder-width rate-of-change cap (/second) — zoom ~0.5, a person switch ~15+
MIN_CONF = 0.3


def _shoulder_widths(seq: np.ndarray, min_conf: float = MIN_CONF) -> np.ndarray:
    """(T,) shoulder width (px). Frames where either shoulder's conf is below threshold are nan."""
    ls, rs = seq[:, kp.LEFT_SHOULDER], seq[:, kp.RIGHT_SHOULDER]
    w = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1)
    w[(ls[:, 2] < min_conf) | (rs[:, 2] < min_conf)] = np.nan
    return w


def _scale(seq: np.ndarray, min_conf: float = MIN_CONF) -> float:
    """Representative shoulder width (px) for the run — median over confident frames. nan if none."""
    w = _shoulder_widths(seq, min_conf)
    return float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan")


def median3(seq: np.ndarray) -> np.ndarray:
    """Apply a 3-frame temporal median filter to x,y — removes single-frame flicker (a jump-then-return).

    A sustained jump (teleport/ID switch) remains as a step and still trips
    spike detection. The conf channel and the two endpoint frames are left as-is.
    """
    if len(seq) < 3:
        return seq
    out = seq.astype(float).copy()
    stack = np.stack([seq[:-2, :, :2], seq[1:-1, :, :2], seq[2:, :, :2]])
    out[1:-1, :, :2] = np.median(stack, axis=0)
    return out


def wrist_speed(seq: np.ndarray, fps: float,
                min_conf: float = MIN_CONF) -> np.ndarray:
    """(T-1,) wrist speed per frame transition (shoulder-widths/second) — max over both wrists.

    Computed on median3-filtered coordinates so flicker is ignored and only
    sustained jumps register. A transition where one wrist is below conf
    threshold is computed from the other wrist alone; if both are below threshold, it's nan.
    """
    scale = _scale(seq, min_conf)
    out = np.full(max(0, len(seq) - 1), np.nan)
    if not np.isfinite(scale) or scale < _EPS or len(seq) < 2:
        return out
    med = median3(seq)
    for j in (kp.LEFT_WRIST, kp.RIGHT_WRIST):
        d = np.linalg.norm(np.diff(med[:, j, :2], axis=0), axis=1) / scale * fps
        ok = (seq[:-1, j, 2] >= min_conf) & (seq[1:, j, 2] >= min_conf)
        d[~ok] = np.nan
        out = np.where(np.isnan(out), d, np.fmax(out, d))
    return out


def shoulder_jump(seq: np.ndarray, fps: float,
                  min_conf: float = MIN_CONF) -> np.ndarray:
    """(T-1,) shoulder-width rate of change (/second) — a signal for a person switch (sudden scale change).

    Applied after median3, so single-frame shoulder flicker isn't counted.
    """
    scale = _scale(seq, min_conf)
    if not np.isfinite(scale) or scale < _EPS or len(seq) < 2:
        return np.full(max(0, len(seq) - 1), np.nan)
    return np.abs(np.diff(_shoulder_widths(median3(seq), min_conf))) / scale * fps


def find_spike_frames(seq: np.ndarray, fps: float, speed_th: float = SPEED_TH,
                      jump_th: float = JUMP_TH,
                      min_conf: float = MIN_CONF) -> np.ndarray:
    """Arrival-frame indices of jump transitions (ascending). A nan transition is never a spike."""
    sp = wrist_speed(seq, fps, min_conf)
    jp = shoulder_jump(seq, fps, min_conf)
    with np.errstate(invalid="ignore"):
        bad = (sp > speed_th) | (jp > jump_th)
    return np.flatnonzero(bad) + 1


def clean_segments(n_frames: int, spike_frames: np.ndarray, fps: float,
                   min_seconds: float = 1.0) -> list[tuple[int, int]]:
    """[start,end) segments cut at spikes, keeping only those >= min_seconds."""
    cuts = [0] + [int(s) for s in spike_frames] + [n_frames]
    min_frames = int(round(min_seconds * fps))
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_frames]


def pick_review_samples(rows: list[dict], speed_th: float = SPEED_TH,
                        n_worst: int = 20,
                        n_borderline: int = 20) -> tuple[list[dict], list[dict]]:
    """Visual-review samples: worst cases (descending max_speed) + threshold-borderline cases (ascending |speed-th|).

    Instead of labeling everything, only these two batches are reviewed via
    overlay to calibrate the threshold. Anything already in the worst batch is
    excluded from the borderline batch.
    """
    by_speed = sorted(rows, key=lambda r: -r["max_speed"])
    worst = by_speed[:n_worst]
    taken = {id(r) for r in worst}
    border = sorted((r for r in rows if id(r) not in taken),
                    key=lambda r: abs(r["max_speed"] - speed_th))[:n_borderline]
    return worst, border


def dispose_run(qc: dict, min_seg_seconds: float = 2.0) -> str:
    """Run disposition: 'keep' | 'discard' — only checks whether a clean segment long enough for a training window exists.

    Quality-style metrics (conf, spike rate) are not used — this avoids
    dropping older recordings disproportionately (era bias).
    """
    if not qc.get("segments"):
        return "discard"
    longest = max(b - a for a, b in qc["segments"])
    return "keep" if longest >= min_seg_seconds * qc["fps"] else "discard"


def qc_run(seq: np.ndarray, fps: float, speed_th: float = SPEED_TH,
           jump_th: float = JUMP_TH, min_conf: float = MIN_CONF,
           min_seconds: float = 1.0) -> dict:
    """QC result for a single run — a dict that can be saved as-is to qc.json."""
    n = len(seq)
    spikes = find_spike_frames(seq, fps, speed_th, jump_th, min_conf)
    segs = clean_segments(n, spikes, fps, min_seconds)
    kept = sum(b - a for a, b in segs)
    sp = wrist_speed(seq, fps, min_conf)
    jp = shoulder_jump(seq, fps, min_conf)
    wconf = seq[:, [kp.LEFT_WRIST, kp.RIGHT_WRIST], 2]
    return {
        "n_frames": n,
        "fps": float(fps),
        "spike_frames": [int(s) for s in spikes],
        "segments": [[int(a), int(b)] for a, b in segs],
        "kept_frames": int(kept),
        "drop_frac": float(1.0 - kept / n) if n else 1.0,
        "spike_rate": float(len(spikes) / (n / fps)) if n else 0.0,
        "max_speed": float(np.nanmax(sp)) if np.isfinite(sp).any() else 0.0,
        "max_jump": float(np.nanmax(jp)) if np.isfinite(jp).any() else 0.0,
        "low_conf_frac": float((wconf < min_conf).any(axis=1).mean()) if n else 1.0,
        "params": {"speed_th": speed_th, "jump_th": jump_th,
                   "min_conf": min_conf, "min_seconds": min_seconds},
    }
