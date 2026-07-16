"""Non-parametric RCR: user-defined mu computation.

The user subclasses `NonParametric` and overrides `mu_func` (unweighted)
and/or `mu_func_w` (weighted). The rejection loops call into these when
`RCR.set_mu_type(MuType.NONPARAMETRIC)` is in effect.

Ported from cpp/src/NonParametric.{h,cpp}. The C++ is an abstract class
with virtual `muFunc`; in Python we use the same shape with subclass
overrides.
"""
from __future__ import annotations

import numpy as np


class NonParametric:
    """Base class for user-supplied non-parametric mu computation.

    Subclass and override `mu_func` (and/or `mu_func_w`) to control which
    points contribute to the central-value estimate at each iteration of
    the RCR rejection loop.

    The `indices` attribute records the post-`mu_func` index list; the
    rejection loop reads it back after each call.
    """

    def __init__(self):
        self.indices: np.ndarray = np.empty(0, dtype=np.int64)

    def mu_func(self, flags: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, true_y) given current flags and the raw y array.

        Default behavior mirrors the C++ no-op base case: keep all flagged
        points. Subclasses override for non-trivial filtering.
        """
        idx = np.where(flags)[0].astype(np.int64)
        self.indices = idx
        return idx, y[idx].copy()

    def mu_func_w(
        self, flags: np.ndarray, w: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted twin of `mu_func`. Returns (indices, true_w, true_y)."""
        idx = np.where(flags)[0].astype(np.int64)
        self.indices = idx
        return idx, w[idx].copy(), y[idx].copy()
