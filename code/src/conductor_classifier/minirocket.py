"""A numpy reimplementation of the MiniRocket transform (Dempster, Schmidt & Webb, KDD 2021).

Why reimplement it: installing aeon/sktime downgrades scikit-learn, which breaks the existing
verdict model (joblib). The algorithm is simple enough (84 fixed kernels + PPV) that
implementing it directly is cheaper than adding the dependency. This does not replicate the
paper's exact dilation/bias schedule — it follows the core design (fixed kernel bank, dilation
multiplexing, training-set quantile bias, PPV pooling).

Why it's used in this project:
  - There are no learned kernels -> minimal overfitting risk at a scale of ~80 recordings.
  - PPV (proportion of positive values) discards "where it appeared" and keeps only "how often
    it matched" -> structurally phase-invariant. This makes the beat/measure-alignment debate
    moot. (docs/research/2026-07-28-phase-alignment-literature.md)

Input is a real-valued array of shape (n_samples, n_channels, n_timesteps).
"""
from itertools import combinations

import numpy as np

KERNEL_LENGTH = 9
_HIGH = 2.0   # the large-weight value (3 of them)
_LOW = -1.0   # the remaining 6 — sums to 0, so it's invariant to a constant offset


def kernel_bank() -> np.ndarray:
    """All kernels of length 9, values in {-1,2}, with exactly 3 positions set to 2 -> (84, 9)."""
    bank = np.full((84, KERNEL_LENGTH), _LOW, dtype=float)
    for i, idx in enumerate(combinations(range(KERNEL_LENGTH), 3)):
        bank[i, list(idx)] = _HIGH
    return bank


def dilations_for(length: int, max_dilations: int = 32,
                  min_output_frac: float = 0.5) -> np.ndarray:
    """Set of dilations (exponentially spaced) sized to the time-series length.

    If dilations are used all the way up to the hard ceiling ((length-1)//8, i.e. just enough
    for the kernel to fit), the output length at the maximum dilation shrinks to only about 9
    positions, making PPV noisy and **sensitive to phase shift** (the original paper mitigates
    this with padding). Here, only dilations that leave at least min_output_frac of the
    original output length are used — since phase invariance is the whole reason these features
    exist in this project, it's prioritized over covering the full frequency range.
    """
    hard = max(1, (length - 1) // (KERNEL_LENGTH - 1))
    max_d = max(1, min(hard, int(length * (1.0 - min_output_frac)) // (KERNEL_LENGTH - 1)))
    n = max(1, min(max_dilations, int(np.log2(max_d)) + 1))
    ds = np.unique(np.floor(2.0 ** np.linspace(0, np.log2(max_d), n)).astype(int))
    return ds[ds >= 1]


def _high_indices() -> np.ndarray:
    """Per-kernel positions with weight 2, shape (84, 3) — used to accelerate the convolution."""
    return np.array(list(combinations(range(KERNEL_LENGTH), 3)), dtype=int)


class MiniRocket:
    """Fixed-kernel convolution + PPV feature transformer.

    fit only determines the bias (quantiles) from the training set, and transform reuses that
    bias — so test-set statistics never leak into the features.
    """

    def __init__(self, n_features: int = 10000, seed: int = 0,
                 max_dilations: int = 32, batch_size: int = 512,
                 bias_sample: int = 256):
        self.n_features = int(n_features)
        self.seed = int(seed)
        self.max_dilations = int(max_dilations)
        self.batch_size = int(batch_size)
        self.bias_sample = int(bias_sample)
        self._biases: np.ndarray | None = None

    # ------------------------------------------------------------------ internal
    def _conv_all(self, X: np.ndarray, dilation: int) -> np.ndarray:
        """(n, C, T) -> (n, 84, L) convolution output. Channels are summed over each kernel's subset.

        Since weights are only {-1,2}, conv can be written as
        3*(sum of the 3 positions weighted 2) - (sum over all 9 positions). Only 9 shifts per
        dilation are needed to cheaply obtain all 84 kernels.
        """
        n, C, T = X.shape
        span = (KERNEL_LENGTH - 1) * dilation
        L = T - span
        # signal pre-summed over each kernel's channel subset, shape (n, 84, T)
        sig = np.einsum("nct,kc->nkt", X, self._chan_mask)
        shifts = np.stack([sig[:, :, i * dilation: i * dilation + L]
                           for i in range(KERNEL_LENGTH)], axis=0)  # (9, n, 84, L)
        total = shifts.sum(axis=0)
        hi = self._hi_idx  # (84, 3)
        picked = np.take_along_axis(
            shifts, hi.T[:, None, :, None].repeat(shifts.shape[1], axis=1)
                      .repeat(L, axis=3), axis=0).sum(axis=0)
        return _HIGH * picked + _LOW * (total - picked)

    def _plan(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n, C, T = X.shape
        self._dilations = dilations_for(T, self.max_dilations)
        n_combo = 84 * len(self._dilations)
        self._n_bias = max(1, int(round(self.n_features / n_combo)))
        self.n_features_ = n_combo * self._n_bias
        self._hi_idx = _high_indices()
        # per-kernel channel subset (multivariate). If there's only 1 channel, use it entirely.
        mask = np.zeros((84, C), dtype=float)
        for k in range(84):
            size = 1 if C == 1 else int(rng.integers(1, C + 1))
            mask[k, rng.choice(C, size=size, replace=False)] = 1.0
        self._chan_mask = mask
        self._quantiles = np.linspace(0.0, 1.0, self._n_bias + 2)[1:-1]

    # ------------------------------------------------------------------- API
    def fit(self, X: np.ndarray) -> "MiniRocket":
        """Only fits the bias (quantiles). If there are many samples, only bias_sample of them
        are examined, evenly spaced.

        The original paper also draws the bias from a single random training sample per kernel
        — the full set isn't needed for quantile estimation, and keeping all of it only costs
        memory.
        """
        X = np.asarray(X, dtype=float)
        self._plan(X)
        if len(X) > self.bias_sample:
            pick = np.linspace(0, len(X) - 1, self.bias_sample).round().astype(int)
            Xb = X[np.unique(pick)]
        else:
            Xb = X
        self.n_bias_windows_ = len(Xb)
        biases = []
        for d in self._dilations:
            out = self._conv_all(Xb, int(d))                   # (m, 84, L)
            flat = out.transpose(1, 0, 2).reshape(84, -1)      # (84, m*L)
            biases.append(np.quantile(flat, self._quantiles, axis=1).T)  # (84, B)
        self._biases = np.stack(biases, axis=0)                # (D, 84, B)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._biases is None:
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), self.n_features_), dtype=float)
        for s in range(0, len(X), self.batch_size):
            out[s:s + self.batch_size] = self._transform_batch(X[s:s + self.batch_size])
        return out

    def _transform_batch(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for di, d in enumerate(self._dilations):
            conv = self._conv_all(X, int(d))                   # (n, 84, L)
            b = self._biases[di][None, :, :, None]             # (1, 84, B, 1)
            ppv = (conv[:, :, None, :] > b).mean(axis=3)       # (n, 84, B)
            feats.append(ppv.reshape(len(X), -1))
        return np.hstack(feats)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
