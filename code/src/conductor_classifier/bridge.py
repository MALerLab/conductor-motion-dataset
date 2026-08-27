"""Bridge test — uses a third conductor to separate 'era confound' from 'real motion signal'.

Bernstein (1959-1989) and Dudamel (2007-2014) don't overlap in era, so the two
of them alone can't tell whether the model is reading conducting style or
recording era. Running a conductor whose career spans both eras (Haitink,
1959-2021) through the B-vs-D model gives us a test:

  * If predictions **track the recording year** -> the model is reading era (confound confirmed)
  * If predictions are independent of year -> the era-confound hypothesis is rejected (evidence it reads motion)

Haitink's identity label isn't needed — only the **recording year**.
"""
import collections
import math

import numpy as np


def aggregate_scores(items: list[tuple[str, float]]) -> dict[str, float]:
    """(recording_id, score) list -> mean score per recording (majority vote over windows)."""
    acc = collections.defaultdict(list)
    for rec, p in items:
        acc[rec].append(float(p))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _rank(xs: list[float]) -> np.ndarray:
    """Ties get the average rank."""
    a = np.asarray(xs, dtype=float)
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation coefficient. Returns nan if fewer than 2 samples or if either side is constant."""
    if len(x) < 2 or len(x) != len(y):
        return math.nan
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def era_auc(scores: dict[str, float], years: dict[str, int],
            cutoff: int = 2000) -> float:
    """How well the score separates 'old recording' from 'new recording' (AUC).

    1.0 = the score alone perfectly predicts era (= the model is reading era),
    0.5 = independent of era. Recordings without a year are excluded; if only
    one era is present, returns nan.
    """
    old = [s for k, s in scores.items() if k in years and years[k] < cutoff]
    new = [s for k, s in scores.items() if k in years and years[k] >= cutoff]
    if not old or not new:
        return math.nan
    wins = sum((n > o) + 0.5 * (n == o) for n in new for o in old)
    return float(wins / (len(old) * len(new)))
