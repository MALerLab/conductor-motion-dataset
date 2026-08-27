"""Joint-set selection + multi-branch input channel construction (GaitGraph2 multi-branch +
acceleration extension).

Builds a (C,T,V) tensor by selecting a joint subset from the normalized (T,133,3) skeleton.
Branches can be toggled on/off, which is exactly what's used for the hand-shortcut ablation
(body/hand/both).

Channels per branch:
  pos  3 = x, y, conf   (since normalization places the shoulder midpoint at the origin, the
                         coordinates themselves are already relative — the old "midpoint-relative"
                         2-channel was fully redundant with x,y and has been removed)
  vel  4 = lag1(dx,dy), lag2(dx,dy)
  acc  2 = first difference of velocity (dx,dy)   <- an extension not found in the literature
                                                     (captures the sharpness of a beat attack)
  bone 4 = parent-joint vector (dx,dy), foreshortening ratio r, delta^1 r
         (the parent of the nose and both shoulders is the **shoulder midpoint** — preserves
          left-right symmetry.
          Length/angle channels were removed: normalization fixes length to a constant so it
          carries no information, and angle was fully redundant with dx,dy.
          r = raw 2D bone length / run median — this is the cos(theta) proxy left after s*L
          cancels out of Taylor's (2000, CVIU 80) l = s*L*cos(theta). It shrinks as the bone
          tilts toward the camera = a reinjection of the depth-direction motion that
          normalization erased.
          Sign (front/back) is not provided since it's fundamentally ambiguous from a single
          viewpoint.)
"""
import numpy as np

from . import keypoints as kp

# The sets only contain joints that **survive normalization** (the bone tree matches
# pose.BONES). Hips/legs were excluded from the tree because 28% of frames extrapolate them
# outside the frame in conducting shots, and face landmarks (23-90) are also unused — the
# design goal is to identify by motion alone, not appearance.
_ARM = [kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER, kp.LEFT_ELBOW,
        kp.RIGHT_ELBOW, kp.LEFT_WRIST, kp.RIGHT_WRIST]        # 6 arm joints
_HIP = [kp.LEFT_HIP, kp.RIGHT_HIP]                            # upper-body tilt/rotation
_BODY_NF = _ARM + _HIP                                        # 8 (no head)
_BODY = [kp.NOSE] + _BODY_NF                                  # 9 = head(1) + arm(6) + hip(2)
_HAND = list(range(91, 133))  # RTMPose WholeBody: left hand 91-111, right hand 112-132

JOINT_SETS: dict[str, list[int]] = {
    "body": _BODY,                # 9 — the default "big picture" set, without hands
    "body_nf": _BODY_NF,          # 8 — excludes the nose (to measure head-info contribution)
    "arm": _ARM,                  # 6 — arms only
    "hand": _HAND,                # 42
    "both": _BODY + _HAND,        # 51 — includes hands (note the 42:9 node imbalance)
    "both_nf": _BODY_NF + _HAND,  # 50
}

ALL_BRANCHES = ("pos", "vel", "acc", "bone")
BRANCH_WIDTH = {"pos": 3, "vel": 4, "acc": 2, "bone": 4}

# Hand skeleton (21 points): wrist (0) -- 4 segments per finger. (parent, child) order,
# hand-local indices.
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def parents(joint_set: str) -> np.ndarray:
    """Parent-joint array indexed within the set (the root is its own parent)."""
    js = JOINT_SETS[joint_set]
    par = np.arange(len(js))
    pos = {j: i for i, j in enumerate(js)}

    def link(child, parent):
        if child in pos and parent in pos:
            par[pos[child]] = pos[parent]

    # upper-body anatomical connections (rooted at the shoulders)
    link(kp.NOSE, kp.LEFT_SHOULDER)
    link(kp.RIGHT_SHOULDER, kp.LEFT_SHOULDER)
    link(kp.LEFT_ELBOW, kp.LEFT_SHOULDER)
    link(kp.RIGHT_ELBOW, kp.RIGHT_SHOULDER)
    link(kp.LEFT_WRIST, kp.LEFT_ELBOW)
    link(kp.RIGHT_WRIST, kp.RIGHT_ELBOW)
    link(kp.LEFT_HIP, kp.LEFT_SHOULDER)
    link(kp.RIGHT_HIP, kp.RIGHT_SHOULDER)
    # hand skeleton (left hand starts at 91, right hand at 112) — wrist (0) is the parent,
    # with children pointing toward the fingertips
    for base in (91, 112):
        for p, c in _HAND_EDGES:
            link(base + c, base + p)
        # wrist junction: attach the hand root to the arm's wrist (only holds for "both";
        # "hand" alone treats it as the root)
        link(base, kp.LEFT_WRIST if base == 91 else kp.RIGHT_WRIST)
    return par


def localize_hands(seq: np.ndarray) -> np.ndarray:
    """Subtract each hand's own wrist (91/112) coordinate from its 21 points, leaving only
    **finger shape**.

    The hand joints' absolute coordinates carry the entire arm trajectory along with them, so
    even running the "hand" set effectively just observes arm movement. Properly testing the
    finger-shortcut hypothesis requires these arm-trajectory-removed local coordinates.
    """
    out = seq.copy()
    for base in (91, 112):
        root = seq[:, base:base + 1, :2]
        out[:, base:base + 21, :2] = seq[:, base:base + 21, :2] - root
    return out


def build_channels(seq: np.ndarray, joint_set: str,
                   use: tuple[str, ...] = ALL_BRANCHES,
                   drop_conf: bool = False) -> np.ndarray:
    """(T,133,3) normalized skeleton -> (C,T,V) channel tensor.

    seq is assumed to be skeleton_norm.npy that has already gone through bone normalization
    and low-pass filtering.
    If drop_conf=True, the confidence channel is filled with zeros — since conf is a direct
    proxy for footage quality, this is used as a control for era-confound diagnostics (the
    channel count is kept the same so structures remain comparable).
    """
    for b in use:
        if b not in BRANCH_WIDTH:
            raise ValueError(f"unknown branch: {b!r} (allowed: {ALL_BRANCHES})")
    js = JOINT_SETS[joint_set]          # unknown sets are filtered out via KeyError
    xy = seq[:, js, :2].astype(np.float32)      # (T,V,2)
    conf = seq[:, js, 2].astype(np.float32)     # (T,V)
    if drop_conf:
        conf = np.zeros_like(conf)
    T, V = conf.shape
    out: list[np.ndarray] = []

    if "pos" in use:
        out += [xy[..., 0], xy[..., 1], conf]

    if "vel" in use or "acc" in use:
        v1 = _diff(xy, 1)
        v2 = _diff(xy, 2)
    if "vel" in use:
        out += [v1[..., 0], v1[..., 1], v2[..., 0], v2[..., 1]]
    if "acc" in use:
        a1 = _diff(v1, 1)
        out += [a1[..., 0], a1[..., 1]]

    if "bone" in use:
        par = parents(joint_set)
        par_xy = xy[:, par, :].copy()
        # replace the parent of the nose and both shoulders with the shoulder midpoint —
        # restores left-right symmetry
        mid = (seq[:, kp.LEFT_SHOULDER, :2] + seq[:, kp.RIGHT_SHOULDER, :2]) / 2.0
        for j in (kp.NOSE, kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER):
            if j in js:
                par_xy[:, js.index(j), :] = mid.astype(np.float32)
        vec = xy - par_xy
        if seq.shape[-1] >= 4:
            r = seq[:, js, 3].astype(np.float32)
        else:                       # input with no depth info (tests / older format) -> neutral 1
            r = np.ones_like(conf)
        out += [vec[..., 0], vec[..., 1], r, _diff(r[..., None], 1)[..., 0]]

    return np.stack(out, axis=0).astype(np.float32)   # (C,T,V)


# left-right joint pairs: (left joint, right joint). The nose has no pair, so it's excluded
# from the coordination representation.
PAIR_JOINTS = [(kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER),
               (kp.LEFT_ELBOW, kp.RIGHT_ELBOW),
               (kp.LEFT_WRIST, kp.RIGHT_WRIST),
               (kp.LEFT_HIP, kp.RIGHT_HIP)] +               [(91 + k, 112 + k) for k in range(21)]


def build_pair_channels(seq: np.ndarray) -> np.ndarray:
    """(T,133,3) -> (11,T,25) — a representation that keeps **only left-right coordination**.

    Discards each arm's absolute trajectory and encodes only the relationship within each pair
    (left-right):
      m  2 = mirror-asymmetry vector pL - mirror(pR) = (xL+xR, yL-yR)
             (0 for a perfectly mirror-symmetric motion — this is where left-hand independence lives)
      d  2 = left-right relative vector pL - pR (spacing/arrangement between the hands)
      dist 1, conf 1 = |d|, min(confL,confR)
      vel 5 = lag1 difference of the 5 geometric channels above (temporal change in coordination)
    """
    L = [l for l, _ in PAIR_JOINTS]
    R = [r for _, r in PAIR_JOINTS]
    pl = seq[:, L, :2].astype(np.float32)
    pr = seq[:, R, :2].astype(np.float32)
    m = pl - np.stack([-pr[..., 0], pr[..., 1]], axis=-1)   # mirror asymmetry
    d = pl - pr
    dist = np.linalg.norm(d, axis=2)
    conf = np.minimum(seq[:, L, 2], seq[:, R, 2]).astype(np.float32)
    geo = [m[..., 0], m[..., 1], d[..., 0], d[..., 1], dist]
    out = geo + [conf] + [_diff(g[..., None], 1)[..., 0] for g in geo]
    return np.stack(out, axis=0).astype(np.float32)


def _diff(x: np.ndarray, lag: int) -> np.ndarray:
    """Forward difference along the time axis, x[t+lag]-x[t]. The trailing lag frames are
    zero-padded (length preserved)."""
    d = np.zeros_like(x)
    if lag < len(x):
        d[:-lag] = x[lag:] - x[:-lag]
    return d
