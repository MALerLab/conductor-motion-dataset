"""Extract fixed-length windows from clean segments -- index computation only, source untouched.

This is the third tier (the "view") of the non-destructive three-tier design:
windows are never saved to disk, only sliced from memory at train/eval time.
Window length T stays a hyperparameter, and drawing a different start point
each training step is itself a tempo/phase augmentation (per the literature:
random temporal crop + global average pooling = phase invariance).
"""
import numpy as np


def rescale_segments(segments: list[tuple[int, int]], src_fps: float,
                     dst_fps: float) -> list[tuple[int, int]]:
    """Rescale segment frame indices to dst_fps. Length filtering is the caller's responsibility."""
    if src_fps == dst_fps:
        return [(int(a), int(b)) for a, b in segments]
    k = dst_fps / src_fps
    return [(int(round(a * k)), int(round(b * k))) for a, b in segments]


def window_starts(segments: list[tuple[int, int]], T: int, n_eval: int,
                  min_stride_frac: float = 0.5, max_total: int = 0,
                  margin: int = 0) -> list[int]:
    """Evenly-spaced start points for evaluation -- deterministic (reproducible).

    min_stride_frac: forces a minimum spacing between adjacent windows, as a
      fraction of T. Since the median clean-segment length is 3.6s, drawing
      8 windows of a 2s window would produce 89% overlap -- effectively
      counting nearly the same data multiple times (inflating the effective
      sample size of window-level metrics).
    max_total: if nonzero, caps the total number of windows -- prevents long
      recordings from dominating the metric (per-recording window counts
      ranged from 8 to 698). Thins them out at even spacing.
    margin: pull back by this many frames from both ends of the segment --
      used for sensitivity experiments on boundary-frame artifacts (e.g. cut-
      transition residue). If the segment is too short to give the full
      margin, it's reduced as much as possible (margin=0 is fully backward
      compatible).
    """
    out: list[int] = []
    stride = max(1, int(round(T * min_stride_frac)))
    for a, b in segments:
        if margin:
            m = min(margin, max(0, b - a - T) // 2)
            a, b = a + m, b - m
        span = b - a - T
        if span <= 0:
            out.append(a)
            continue
        k = max(1, min(n_eval, span // stride + 1))
        if k == 1:
            out.append(a)
            continue
        step = span / (k - 1)
        out += [a + int(round(i * step)) for i in range(k)]
    if max_total and len(out) > max_total:
        idx = np.linspace(0, len(out) - 1, max_total)
        out = [out[int(round(i))] for i in idx]
    return out


def window_spans(segments: list[tuple[int, int]], T: int, n_eval: int,
                 min_stride_frac: float = 0.5, margin: int = 0
                 ) -> list[tuple[int, int]]:
    """Evaluation (start, actual_frame_count) pairs -- used to keep segments shorter than the window alive via padding.

    Uses the same start-point rule as window_starts, but a segment shorter
    than T has its true length returned as-is (the caller pads the remaining
    tail). The length floor (what fraction of the window the actual frames
    must cover) is the responsibility of the segment-filtering stage.
    """
    out: list[tuple[int, int]] = []
    stride = max(1, int(round(T * min_stride_frac)))
    for a, b in segments:
        if margin:
            m = min(margin, max(0, b - a - T) // 2)
            a, b = a + m, b - m
        span = b - a - T
        if span <= 0:                       # segment is shorter than the window -> needs padding
            out.append((a, b - a))
            continue
        k = max(1, min(n_eval, span // stride + 1))
        if k == 1:
            out.append((a, T))
            continue
        step = span / (k - 1)
        out += [(a + int(round(i * step)), T) for i in range(k)]
    return out


def sample_span(segments: list[tuple[int, int]], T: int,
                rng: np.random.Generator) -> tuple[int, int]:
    """Training (start, actual_frame_count) pair -- the padding-aware counterpart to sample_start.

    Segment selection weighting matches sample_start (proportional to window
    count). A segment shorter than the window has only one possible start
    point, and its actual frame count equals its length.
    """
    spans = [max(1, (b - a) - T + 1) for a, b in segments]
    tot = float(sum(spans))
    i = int(rng.choice(len(segments), p=[s / tot for s in spans]))
    a, b = segments[i]
    hi = (b - a) - T
    if hi <= 0:
        return a, b - a
    return a + int(rng.integers(0, hi + 1)), T


def sample_start(segments: list[tuple[int, int]], T: int,
                 rng: np.random.Generator) -> int:
    """Random training start point -- pick a segment weighted by length, then sample uniformly within it."""
    spans = [max(1, (b - a) - T + 1) for a, b in segments]
    tot = float(sum(spans))
    i = int(rng.choice(len(segments), p=[s / tot for s in spans]))
    a, b = segments[i]
    hi = (b - a) - T
    return a if hi <= 0 else a + int(rng.integers(0, hi + 1))


def crop(seq: np.ndarray, start: int, T: int) -> np.ndarray:
    """seq[start:start+T] -- if there isn't enough data, fill via mirror padding along the time axis."""
    end = start + T
    if start >= 0 and end <= len(seq):
        return seq[start:end]
    idx = _mirror_index(np.arange(start, end), len(seq))
    return seq[idx]


def _mirror_index(idx: np.ndarray, n: int) -> np.ndarray:
    """Fold out-of-range indices back in by reflection (mirroring). If n=1, everything is 0."""
    if n <= 1:
        return np.zeros_like(idx)
    period = 2 * (n - 1)
    m = np.mod(idx, period)
    return np.where(m < n, m, period - m)
