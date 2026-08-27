from typing import Callable

import numpy as np

from . import keypoints as kp

_EPS = 1e-6


def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """Normalize a (T,K,3) skeleton to a shoulder-center origin and shoulder-width scale.

    Frames where shoulder width is near zero are masked with conf=0.
    """
    out = seq.astype(float).copy()
    ls = seq[:, kp.LEFT_SHOULDER, :2]
    rs = seq[:, kp.RIGHT_SHOULDER, :2]
    center = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=1)

    valid = width > _EPS
    out[..., :2] = out[..., :2] - center[:, None, :]
    safe_w = np.where(valid, width, 1.0)
    out[..., :2] = out[..., :2] / safe_w[:, None, None]
    out[~valid, :, 2] = 0.0
    return out


def extract_normalized_skeleton(
    start_frame: int,
    end_frame: int,
    extract: Callable[[int, int], np.ndarray],
) -> np.ndarray:
    """extract(start, end) -> raw (T,K,3) skeleton. Returns it after normalization."""
    raw = extract(start_frame, end_frame)
    return normalize_skeleton(raw)


def _moving_avg(a: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge padding. Returns the input unchanged (as float) if win<=1."""
    a = a.astype(float)
    if win <= 1:
        return a
    pad = win // 2
    padded = np.pad(a, pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")[: len(a)]


def conductor_bboxes(
    kpts_seq: np.ndarray,
    img_w: int,
    img_h: int,
    margin: float = 0.35,
    smooth_win: int = 15,
    min_conf: float = 0.3,
    upper_body_only: bool = True,
) -> np.ndarray:
    """(T,17,3) keypoint sequence -> per-frame conductor bbox (T,4) int [x0,y0,x1,y1].

    Coordinate convention: x0/y0 inclusive, x1/y1 exclusive (matches NumPy/OpenCV
    slicing frame[y0:y1, x0:x1]).
    The box is the min/max of joints with conf>=min_conf, expanded by the margin
    ratio, then clamped to image bounds. Frames with no confident joint reuse
    the box from the nearest valid frame. Center and size are smoothed with a
    moving average (smooth_win) to reduce crop jitter.

    When upper_body_only=True, the box is fit using only upper-body joints
    (COCO 0-12: face, shoulders, arms, hips). YOLO occasionally assigns spurious
    high confidence to the legs (13-16), which stretches the box down to the
    floor and loosens the crop zoom (costing hand/baton resolution), so
    conducting footage defaults to upper body only.
    """
    T = kpts_seq.shape[0]
    n_joints = 13 if upper_body_only else kpts_seq.shape[1]  # COCO 0-12 = upper body
    raw = np.zeros((T, 4), float)
    valid = np.zeros(T, bool)
    for t in range(T):
        k = kpts_seq[t, :n_joints]
        m = k[:, 2] >= min_conf
        if not m.any():
            continue
        xs, ys = k[m, 0], k[m, 1]
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        bw, bh = x1 - x0, y1 - y0
        raw[t] = [x0 - margin * bw, y0 - margin * bh,
                  x1 + margin * bw, y1 + margin * bh]
        valid[t] = True

    if not valid.any():
        raw[:] = [0, 0, img_w, img_h]
    else:
        idx = np.where(valid)[0]
        for t in range(T):
            if not valid[t]:
                raw[t] = raw[idx[np.argmin(np.abs(idx - t))]]

    cx = _moving_avg((raw[:, 0] + raw[:, 2]) / 2.0, smooth_win)
    cy = _moving_avg((raw[:, 1] + raw[:, 3]) / 2.0, smooth_win)
    w = _moving_avg(raw[:, 2] - raw[:, 0], smooth_win)
    h = _moving_avg(raw[:, 3] - raw[:, 1], smooth_win)
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, 0] = np.clip(out[:, 0], 0, img_w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, img_h - 1)
    out[:, 2] = np.clip(out[:, 2], 1, img_w)
    out[:, 3] = np.clip(out[:, 3], 1, img_h)
    return out.astype(int)


def add_crop_offset(kpts: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Add the crop's top-left offset to the x,y of crop-local (...,3) keypoints, recovering original-image coordinates."""
    out = kpts.astype(float).copy()
    out[..., 0] += x0
    out[..., 1] += y0
    return out


# --- Bone-length (body-shape) normalization ---------------------------------
# RTMPose WholeBody (133) upper-body + both-hands kinematic tree. ROOT=shoulder center.
# Each bone is rebuilt to a shared "canonical length" -> only angle (posture/motion)
# remains, and body shape (size/proportions) is removed. Order is topologically
# sorted so a parent always comes before its child.
ROOT = -1
_LH, _RH = 91, 112  # left/right wrist (hand root)
BONES = [
    (ROOT, kp.LEFT_SHOULDER), (ROOT, kp.RIGHT_SHOULDER),
    (kp.LEFT_SHOULDER, kp.LEFT_ELBOW), (kp.LEFT_ELBOW, kp.LEFT_WRIST),
    (kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW), (kp.RIGHT_ELBOW, kp.RIGHT_WRIST),
    (kp.LEFT_WRIST, _LH), (kp.RIGHT_WRIST, _RH),
    # Head/torso: shoulder center to nose and both hips. Head cueing and
    # upper-body lean/rotation are major axes of conducting motion, so they're
    # included in the tree. (Eyes/ears are redundant with the nose; knees/ankles
    # are excluded since the podium hides them and they're essentially never
    # tracked. Hips are extrapolated off-frame 28% of the time — filter by
    # conf before use; measured in DEVLOG 2026-08-09.)
    (ROOT, kp.NOSE), (ROOT, kp.LEFT_HIP), (ROOT, kp.RIGHT_HIP),
]
# Both hands, 21 points each: wrist + 5 fingers x 4 joints. Each finger's first
# joint attaches to the wrist.
for _root in (_LH, _RH):
    for _f in range(5):
        _base = _root + 1 + _f * 4
        BONES.append((_root, _base))
        for _j in range(3):
            BONES.append((_base + _j, _base + _j + 1))


def _shoulder_center(frame: np.ndarray) -> np.ndarray:
    return (frame[kp.LEFT_SHOULDER, :2] + frame[kp.RIGHT_SHOULDER, :2]) / 2.0


def bone_lengths(seq: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """(T,133,3) -> (T, len(BONES)) raw length of each bone. Low-confidence bones are NaN."""
    T = seq.shape[0]
    out = np.full((T, len(BONES)), np.nan)
    for t in range(T):
        sc = _shoulder_center(seq[t])
        for b, (p, c) in enumerate(BONES):
            pp = sc if p == ROOT else seq[t, p, :2]
            pconf = 1.0 if p == ROOT else seq[t, p, 2]
            if min(pconf, seq[t, c, 2]) >= min_conf:
                out[t, b] = float(np.linalg.norm(seq[t, c, :2] - pp))
    return out


def normalize_bone_lengths(
    seq: np.ndarray, canonical: np.ndarray, min_conf: float = 0.3,
    fore: np.ndarray | None = None,
) -> np.ndarray:
    """Keep bone direction (angle) but fix length to a shared canonical value, removing body shape.

    With the shoulder center as origin, walk the tree so each child =
    parent + unit direction (raw) x canonical[b]. Since canonical is the same
    across all sequences, static body shape (size, limb proportions)
    disappears and only angle/motion remains. Joints outside the tree
    (face, feet, etc.) are left empty with 0/conf0.

    If fore (T, len(BONES)) is supplied, each frame/bone is drawn using
    canonical x fore instead — this preserves foreshortening (tilt toward the
    camera) in the geometry, eliminating both the "flattening onto the plane"
    distortion of a tilted bone and the resulting amplified direction noise
    (the canonical body shape is projected at that same angle). If fore=None,
    falls back to the previous behavior (always full length).
    """
    T = seq.shape[0]
    out = np.zeros((T, seq.shape[1], 3), dtype=float)
    for t in range(T):
        sc = _shoulder_center(seq[t])
        newpos = {ROOT: np.zeros(2)}
        for b, (p, c) in enumerate(BONES):
            praw = sc if p == ROOT else seq[t, p, :2]
            d = seq[t, c, :2] - praw
            n = np.linalg.norm(d)
            unit = d / n if n > _EPS else np.zeros(2)
            scale = canonical[b] * (fore[t, b] if fore is not None else 1.0)
            nc = newpos[p] + unit * scale
            newpos[c] = nc
            out[t, c, :2] = nc
            out[t, c, 2] = seq[t, c, 2]
    return out
