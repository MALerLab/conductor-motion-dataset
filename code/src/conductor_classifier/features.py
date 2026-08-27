"""View-robust conducting-motion features.

A 2D skeleton is a viewpoint projection, so *absolute angle* and *absolute
horizontal extent* depend on the camera angle (and are therefore meaningless).
So we only use signals that survive a change of viewpoint:
  - left-right coordination (timing relationship between the two hands — view-invariant)
  - regularity of the vertical-axis beat (gravity ~= screen vertical; rate-agnostic)
Every feature is a correlation/ratio/regularity measure, so it's insensitive
to scale and tempo (absolute amplitude and absolute rate are not used — in
experiment 4 they were confirmed to be neither a confound nor a signal).

First, each frame's shoulder line is rotated to horizontal (body-frame
alignment) to remove camera roll.
Input: bone-normalized (T,133,3). Output: a fixed-length feature vector per window.
"""
import numpy as np

from . import keypoints as kp

_EPS = 1e-8
FEATURE_NAMES = [
    "corr_LR", "xcorr_max_LR", "phase_lag_frac",
    "vrange_ratio_L", "pathlen_ratio_L", "active_ratio_L",
    "autocorr_peak_L", "autocorr_peak_R",
    "spec_conc_L", "spec_conc_R",
    "speed_cv_L", "speed_cv_R",
]


def align_body_frame(window: np.ndarray) -> np.ndarray:
    """Rotate the shoulder line (R-L) to horizontal each frame -> removes camera roll (body-frame)."""
    out = window.astype(float).copy()
    for t in range(window.shape[0]):
        v = window[t, kp.RIGHT_SHOULDER, :2] - window[t, kp.LEFT_SHOULDER, :2]
        ang = np.arctan2(v[1], v[0])           # shoulder-line angle
        c, s = np.cos(-ang), np.sin(-ang)       # rotate by -ang
        R = np.array([[c, -s], [s, c]])
        out[t, :, :2] = window[t, :, :2] @ R.T
    return out


def _autocorr_peak(sig: np.ndarray) -> float:
    """Max of the normalized autocorrelation over lag>=3 (regularity, 0-1). Rate-independent."""
    s = sig - sig.mean()
    if s.std() < _EPS:
        return 0.0
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac = ac / ac[0]
    return float(ac[3:].max()) if len(ac) > 3 else 0.0


def _spectral_concentration(sig: np.ndarray) -> float:
    """Fraction of spectral energy in the dominant frequency band (peak +/-1) (regularity)."""
    s = sig - sig.mean()
    if s.std() < _EPS:
        return 0.0
    p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    p = p[1:]                                   # exclude DC
    if p.sum() < _EPS or len(p) < 3:
        return 0.0
    k = int(np.argmax(p))
    band = p[max(0, k - 1): k + 2].sum()
    return float(band / p.sum())


def _speed_cv(traj: np.ndarray) -> float:
    """Speed coefficient of variation, std/mean (evenness of motion). Scale-independent."""
    sp = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return float(sp.std() / (sp.mean() + _EPS))


def _dominant_period(sig: np.ndarray) -> float:
    s = sig - sig.mean()
    if s.std() < _EPS:
        return float(len(s))
    p = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    fr = np.fft.rfftfreq(len(s), d=1.0)
    p[0] = 0.0
    f = fr[int(np.argmax(p))]
    return 1.0 / f if f > _EPS else float(len(s))


RICH_FEATURE_NAMES = [
    f"{name}_{side}" for side in ("L", "R")
    for name in ("elbow_angle_mean", "elbow_angle_std", "turn_rate",
                 "pause_frac", "vert_ratio", "speed_skew", "wrist_height_mean")
]


def _elbow_angle(sh: np.ndarray, el: np.ndarray, wr: np.ndarray) -> np.ndarray:
    """Per-frame elbow flexion angle (rad): the angle between (shoulder-elbow) and (wrist-elbow)."""
    a, b = sh - el, wr - el
    na = np.linalg.norm(a, axis=1) + _EPS
    nb = np.linalg.norm(b, axis=1) + _EPS
    cos = np.clip((a * b).sum(axis=1) / (na * nb), -1.0, 1.0)
    return np.arccos(cos)


def _turn_rate(traj: np.ndarray) -> float:
    """Mean per-step turning angle (rad) of the wrist's velocity vector. Straight line ~= 0, circular motion = 2*pi/period.

    Low-speed steps (below 25% of mean speed) are excluded since their direction is noise."""
    v = np.diff(traj, axis=0)
    sp = np.linalg.norm(v, axis=1)
    valid = sp > 0.25 * (sp.mean() + _EPS)
    pair = valid[:-1] & valid[1:]
    if not pair.any():
        return 0.0
    v0, v1 = v[:-1][pair], v[1:][pair]
    cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]   # 2D cross product (np.cross for 2D is deprecated)
    ang = np.arctan2(np.abs(cross), (v0 * v1).sum(axis=1))
    return float(ang.mean())


def _pause_frac(traj: np.ndarray) -> float:
    """Pause fraction: fraction of steps with speed < 0.2*median. A fully static trajectory gives 1."""
    sp = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    med = float(np.median(sp))
    if med < _EPS:
        return 1.0
    return float((sp < 0.2 * med).mean())


def _skew(sig: np.ndarray) -> float:
    s = sig.std()
    if s < _EPS:
        return 0.0
    z = (sig - sig.mean()) / s
    return float((z ** 3).mean())


def rich_motion_features(window: np.ndarray) -> np.ndarray:
    """Joint-angle, trajectory, and dynamics features (len(RICH_FEATURE_NAMES),). All scale-independent.

    Complements view_robust_features: 12 features were at chance on the A-set
    LORO evaluation (research/2026-07-08-data-audit.md) -> this adds posture
    (elbow angle, hand height) and trajectory style (curvature, pausing,
    verticality, speed-distribution asymmetry)."""
    w = align_body_frame(window)
    mid_y = (w[:, kp.LEFT_SHOULDER, 1] + w[:, kp.RIGHT_SHOULDER, 1]) / 2.0
    sh_w = np.abs(w[:, kp.RIGHT_SHOULDER, 0] - w[:, kp.LEFT_SHOULDER, 0]).mean() + _EPS

    out = []
    for sh_i, el_i, wr_i in ((kp.LEFT_SHOULDER, kp.LEFT_ELBOW, kp.LEFT_WRIST),
                             (kp.RIGHT_SHOULDER, kp.RIGHT_ELBOW, kp.RIGHT_WRIST)):
        p = w[:, wr_i, :2]
        ang = _elbow_angle(w[:, sh_i, :2], w[:, el_i, :2], p)
        v = np.diff(p, axis=0)
        sp = np.linalg.norm(v, axis=1)
        vert = float(v[:, 1].std() / (v[:, 0].std() + v[:, 1].std() + _EPS))
        out += [float(ang.mean()), float(ang.std()), _turn_rate(p), _pause_frac(p),
                vert, _skew(sp), float(((p[:, 1] - mid_y) / sh_w).mean())]
    return np.array(out)


def view_robust_features(window: np.ndarray) -> np.ndarray:
    """bone-normalized (T,133,3) window -> view-robust features (len(FEATURE_NAMES),)."""
    w = align_body_frame(window)
    yL, yR = w[:, kp.LEFT_WRIST, 1], w[:, kp.RIGHT_WRIST, 1]
    pL, pR = w[:, kp.LEFT_WRIST, :2], w[:, kp.RIGHT_WRIST, :2]

    # left-right coordination
    if yL.std() < _EPS or yR.std() < _EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(yL, yR)[0, 1])
    a, b = yL - yL.mean(), yR - yR.mean()
    denom = np.std(a) * np.std(b) * len(a) + _EPS
    xc = np.correlate(a, b, "full") / denom
    lag = abs(int(np.arange(-len(a) + 1, len(a))[int(np.argmax(xc))]))
    period = _dominant_period(yR)
    phase_lag_frac = min(lag / (period + _EPS), 1.0)
    xcorr_max = float(np.max(xc))

    # amplitude/momentum asymmetry (ratio -> scale-independent)
    vrL, vrR = yL.max() - yL.min(), yR.max() - yR.min()
    vrange_ratio_L = vrL / (vrL + vrR + _EPS)
    plL = np.linalg.norm(np.diff(pL, axis=0), axis=1).sum()
    plR = np.linalg.norm(np.diff(pR, axis=0), axis=1).sum()
    pathlen_ratio_L = plL / (plL + plR + _EPS)
    spL = np.linalg.norm(np.diff(pL, axis=0), axis=1)
    spR = np.linalg.norm(np.diff(pR, axis=0), axis=1)
    # Exact ties with the median are common (bimanual synchrony, quantized
    # keypoints). A strict > flips ties on 1-ULP rounding under rescaling,
    # breaking scale invariance -> use a relative margin so ties and rounding
    # residue are excluded consistently.
    med = float(np.median(spR))
    active_ratio_L = float((spL > med * (1 + 1e-9) + _EPS).mean())

    return np.array([
        corr, xcorr_max, phase_lag_frac,
        vrange_ratio_L, pathlen_ratio_L, active_ratio_L,
        _autocorr_peak(yL), _autocorr_peak(yR),
        _spectral_concentration(yL), _spectral_concentration(yR),
        _speed_cv(pL), _speed_cv(pR),
    ])
