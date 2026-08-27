"""Time-axis normalization (numpy only) to reduce era (recording quality) and frame-rate confounds.

A skeleton sequence (T,K,3) [x,y,conf] is
  1) resampled to a common fps (resample_to_fps) — unifying the sample interval,
  2) low-pass filtered to a common bandwidth (lowpass_temporal) — unifying motion
     smoothness toward the "lower" end,
before being passed on to windowing/classification. keypoint_jitter is a
recording-quality proxy (used to diagnose era leakage). To avoid fabricating
information, the principle is to never upsample fps — always downsample to
the common minimum.
"""
import numpy as np


def resample_to_fps(seq: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample (T,K,3) from src_fps to dst_fps via linear interpolation on the time axis. Endpoints are preserved."""
    T = seq.shape[0]
    if T < 2 or src_fps == dst_fps:
        return seq.astype(float, copy=True)
    new_T = int(round((T - 1) * (dst_fps / src_fps))) + 1
    src_t = np.arange(T, dtype=float)
    dst_t = np.linspace(0.0, T - 1, new_T)
    out = np.empty((new_T, seq.shape[1], seq.shape[2]), dtype=float)
    for k in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            out[:, k, c] = np.interp(dst_t, src_t, seq[:, k, c])
    return out


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(int(round(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def lowpass_temporal(seq: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Gaussian low-pass the x,y channels (cutoff cutoff_hz). conf passes through unchanged.

    A lower cutoff_hz smooths more strongly, unifying smoothness toward the
    "lower" end. At the boundaries the kernel is truncated and renormalized
    (no reflect padding, so no edge effects).
    """
    T = seq.shape[0]
    if T < 2:
        return seq.astype(float, copy=True)
    sigma = fps / (2.0 * np.pi * cutoff_hz)
    kernel = _gaussian_kernel(sigma)
    r = len(kernel) // 2
    out = seq.astype(float, copy=True)
    for k in range(seq.shape[1]):
        for c in range(2):  # x, y only
            sig = seq[:, k, c]
            sm = np.empty(T)
            for i in range(T):
                lo, hi = max(0, i - r), min(T, i + r + 1)
                w = kernel[lo - (i - r): hi - (i - r)]
                sm[i] = np.dot(sig[lo:hi], w) / w.sum()
            out[:, k, c] = sm
    return out


def keypoint_jitter(seq: np.ndarray, min_conf: float = 0.3) -> float:
    """Recording-quality proxy: median inter-frame acceleration magnitude over confident joints.

    Acceleration = second-order difference (x,y). Samples (frame, joint) with
    conf < min_conf are excluded. Returns 0.0 for a static sequence or when no
    samples remain.
    """
    T = seq.shape[0]
    if T < 3:
        return 0.0
    xy = seq[:, :, :2]
    accel = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]          # (T-2, K, 2)
    mag = np.linalg.norm(accel, axis=2)                # (T-2, K)
    conf_ok = seq[2:, :, 2] >= min_conf                # align with accel frames
    vals = mag[conf_ok]
    return float(np.median(vals)) if vals.size else 0.0
