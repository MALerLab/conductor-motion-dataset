"""Joint-set selection + multi-branch input channel construction (GaitGraph2 multi-branch + acceleration extension).

Selects a joint subset from the normalized (T,133,3) skeleton and builds a (C,T,V) tensor.
Branches can be toggled on and off, so this is used as-is for the hand-shortcut
ablation experiments (body/hand/both).

Channels per branch:
  pos  3 = x, y, conf   (normalization places the shoulder midpoint at the origin, so the
                         coordinates themselves are relative — the earlier "midpoint-relative"
                         2 channels were fully redundant with x,y and were removed)
  vel  4 = lag1(dx,dy), lag2(dx,dy)
  acc  2 = first difference of velocity (dx,dy)   <- extension not in the literature (sharpness of beat ictus)
  bone 4 = parent-joint vector (dx,dy), foreshortening ratio r, delta^1 r
         (the parent of the nose and both shoulders is the **shoulder midpoint** — preserves
          left-right symmetry.
          Length and angle channels were removed: normalization fixes length to a constant,
          so length is uninformative, and angle was fully redundant with dx,dy.
          r = raw 2D bone length / run median — a cos(theta) proxy from Taylor (2000, CVIU 80),
          where s*L cancels out of l = s*L*cos(theta). It shrinks as the bone tilts toward
          the camera = re-injection of the depth-direction motion that normalization erased.
          The sign (forward/backward) is inherently ambiguous from a single viewpoint,
          so it is not provided.)
"""
import numpy as np

from . import keypoints as kp

# The sets contain only **joints that survive normalization** (bone tree = matches pose.BONES).
# Hips-down/legs were excluded from the tree because 28% of frames extrapolate them outside
# the frame in conducting shots, and facial landmarks (23-90) are also unused — by design,
# identification relies on motion only, not appearance.
_ARM = [kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER, kp.LEFT_ELBOW,
        kp.RIGHT_ELBOW, kp.LEFT_WRIST, kp.RIGHT_WRIST]        # arms 6
_HIP = [kp.LEFT_HIP, kp.RIGHT_HIP]                            # torso lean / rotation
_BODY_NF = _ARM + _HIP                                        # 8 (no head)
_BODY = [kp.NOSE] + _BODY_NF                                  # 9 = head 1 + arms 6 + hips 2
_HAND = list(range(91, 133))  # RTMPose WholeBody: left hand 91-111, right hand 112-132

JOINT_SETS: dict[str, list[int]] = {
    "body": _BODY,                # 9 — hand-free 'big picture' default configuration
    "body_nf": _BODY_NF,          # 8 — nose excluded (to measure head-information contribution)
    "arm": _ARM,                  # 6 — arms only
    "hand": _HAND,                # 42
    "both": _BODY + _HAND,        # 51 — with hands (beware node imbalance 42:9)
    "both_nf": _BODY_NF + _HAND,  # 50
}

ALL_BRANCHES = ("pos", "vel", "acc", "bone", "bonedir")
# bonedir = only the 2 direction channels of bone, with foreshortening (r, delta r) removed.
# Depth is already reflected in the coordinates themselves by the fore normalization
# (norm_mode=bone_fore), so this is the minimal configuration that does not feed it in
# again as a channel — adopted 2026-08-27. Information-wise, r is recoverable from the
# coordinates (correlation 0.99), and keeping the channels gained only +1.5%p.
BRANCH_WIDTH = {"pos": 3, "vel": 4, "acc": 2, "bone": 4, "bonedir": 2}

# Hand skeleton (21 points): wrist (0) — 4 segments per finger. (parent, child) order, hand-internal indices.
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def parents(joint_set: str) -> np.ndarray:
    """Parent-joint array in set-internal indices (roots point to themselves)."""
    js = JOINT_SETS[joint_set]
    par = np.arange(len(js))
    pos = {j: i for i, j in enumerate(js)}

    def link(child, parent):
        if child in pos and parent in pos:
            par[pos[child]] = pos[parent]

    # Upper-body anatomical links (shoulders as the axis)
    link(kp.NOSE, kp.LEFT_SHOULDER)
    link(kp.RIGHT_SHOULDER, kp.LEFT_SHOULDER)
    link(kp.LEFT_ELBOW, kp.LEFT_SHOULDER)
    link(kp.RIGHT_ELBOW, kp.RIGHT_SHOULDER)
    link(kp.LEFT_WRIST, kp.LEFT_ELBOW)
    link(kp.RIGHT_WRIST, kp.RIGHT_ELBOW)
    link(kp.LEFT_HIP, kp.LEFT_SHOULDER)
    link(kp.RIGHT_HIP, kp.RIGHT_SHOULDER)
    # Hand skeleton (left hand starts at 91, right hand at 112) — wrist (0) is the parent,
    # fingertip direction is the child
    for base in (91, 112):
        for p, c in _HAND_EDGES:
            link(base + c, base + p)
        # Wrist junction: attach the hand root to the arm wrist (holds only in "both";
        # in standalone "hand" it remains a root)
        link(base, kp.LEFT_WRIST if base == 91 else kp.RIGHT_WRIST)
    return par


def localize_hands(seq: np.ndarray) -> np.ndarray:
    """Subtract each hand's wrist (91/112) coordinates from its 21 points, leaving **finger shape only**.

    The absolute coordinates of the hand joints carry the entire arm trajectory, so even
    running the 'hand' set effectively sees arm motion. To properly test the finger-shortcut
    hypothesis, these local coordinates with the arm trajectory removed must be used.
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

    seq is assumed to be skeleton_norm.npy after bone normalization and low-pass filtering.
    With drop_conf=True the confidence channel is zero-filled — conf is a direct proxy for
    video quality, so this is used as a control for diagnosing era confounds (the channel
    count is kept so architectures remain comparable).
    """
    for b in use:
        if b not in BRANCH_WIDTH:
            raise ValueError(f"Unknown branch: {b!r} (available: {ALL_BRANCHES})")
    js = JOINT_SETS[joint_set]          # unknown sets are rejected via KeyError
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

    if "bone" in use or "bonedir" in use:
        par = parents(joint_set)
        par_xy = xy[:, par, :].copy()
        # Replace the parent of the nose and both shoulders with the shoulder midpoint —
        # restores left-right symmetry
        mid = (seq[:, kp.LEFT_SHOULDER, :2] + seq[:, kp.RIGHT_SHOULDER, :2]) / 2.0
        for j in (kp.NOSE, kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER):
            if j in js:
                par_xy[:, js.index(j), :] = mid.astype(np.float32)
        vec = xy - par_xy
        if "bonedir" in use:
            out += [vec[..., 0], vec[..., 1]]
        if "bone" not in use:
            return np.stack(out, axis=0).astype(np.float32)
        if seq.shape[-1] >= 4:
            r = seq[:, js, 3].astype(np.float32)
        else:                       # input without depth information (tests / old versions) -> neutral 1
            r = np.ones_like(conf)
        out += [vec[..., 0], vec[..., 1], r, _diff(r[..., None], 1)[..., 0]]

    return np.stack(out, axis=0).astype(np.float32)   # (C,T,V)


# Left-right corresponding pairs: (left joint, right joint). The nose has no pair and is
# excluded from the coordination representation.
PAIR_JOINTS = [(kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER),
               (kp.LEFT_ELBOW, kp.RIGHT_ELBOW),
               (kp.LEFT_WRIST, kp.RIGHT_WRIST),
               (kp.LEFT_HIP, kp.RIGHT_HIP)] +               [(91 + k, 112 + k) for k in range(21)]


def build_pair_channels(seq: np.ndarray) -> np.ndarray:
    """(T,133,3) -> (11,T,25) — a representation keeping **left-right coordination only**.

    The absolute trajectory of each arm is discarded; only the relation of corresponding
    (left-right) pairs is encoded:
      m  2 = mirror-asymmetry vector pL - mirror(pR) = (xL+xR, yL-yR)
             (zero for perfectly mirror-symmetric motion — left-hand independence lives here)
      d  2 = left-right relative vector pL - pR (spacing / placement between hands)
      dist 1, conf 1 = |d|, min(confL, confR)
      vel 5 = lag1 differences of the 5 geometric channels above (temporal change of coordination)
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
    """Forward difference along time, x[t+lag]-x[t]. The trailing lag frames are zero-padded (length preserved)."""
    d = np.zeros_like(x)
    if lag < len(x):
        d[:-lag] = x[lag:] - x[:-lag]
    return d
