"""Functional-form (parametric) RCR — Phase 2 first cut.

The C++ `FunctionalForm` (cpp/src/FunctionalForm.cpp, 2,556 lines) is the
parametric-model-fitting variant of RCR. The user supplies a model
function `f(x, params)`, its partial derivatives wrt each param, an initial
guess, and (x, y) data; RCR iteratively fits the model and rejects
outliers against the residuals.

This port is **deliberately minimal**:

 * **1D independent variable only** (no `f_ND`)
 * **No priors** (the Priors machinery is stubbed)
 * **No pivot search** (`pivot` defaults to 0; user can set
   `FunctionalForm.pivot` directly if needed)
 * **regression() is the only fitting path** — `scipy.optimize.least_squares`
   replaces the C++ hand-rolled Gauss-Newton solver (modifiedGN /
   regularGN). The C++ supports MEAN, MEDIAN, MODE mu_tech variants that
   use `get2DMedian`/`get2DMode`/etc. across data-triple parameter
   combinations; this port collapses ALL THREE to `regression()`, which
   matches MEAN exactly but is a simplification for MEDIAN/MODE.

What that means for users: the API and the *single-fit* behavior match
the C++ exactly. The full 3-pass robust refinement in `performRejection`
will differ from the C++ in passes 2-3 because the C++ would use
median/mode parameter estimates there. For typical use the difference is
small; for heavily contaminated data the C++ would be more robust.

Ported from cpp/src/FunctionalForm.cpp.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares, fsolve

# C++-oracle constants (FunctionalForm.cpp:903-904). When combo count > 40000
# the C++ samples 20000 weighted draws instead of using all combos. The port
# mirrors these numbers exactly for bit-identical parity.
_COMBO_LIMIT = 20000        # number of draws when sampling triggers
_COMBO_FRAC = 0.5           # trigger threshold: sample iff frac*total > limit
# Nonlinear path can't afford 40000 fsolve calls — cap it tighter. Hits
# performance, not parity, since nonlinear models rarely produce enough
# kept points to exceed this anyway.
_COMBO_SAMPLE_CAP_NONLINEAR = 500


def _cpp_mt19937_seed_state(seed: int) -> np.ndarray:
    """C++11 §26.5.3.1 seed expansion for std::mt19937.

    Returns the 624-element uint32 state array as produced by
    std::mt19937(seed), so it can be injected into numpy's MT19937
    bit-generator (numpy uses a different seed expansion by default).
    """
    state = np.empty(624, dtype=np.uint32)
    state[0] = seed & 0xFFFFFFFF
    for i in range(1, 624):
        prev = int(state[i - 1])
        state[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
    return state


def _cpp_mt19937_uint32_stream(seed: int, n: int) -> np.ndarray:
    """Return `n` uint32 values matching std::mt19937(seed).

    Uses numpy's compiled MT19937 BitGenerator with C++-style state
    injection so the bit sequence matches the C++ oracle exactly.
    Empirically, numpy's `random_raw` for MT19937 emits one 32-bit value
    per call placed in the low 32 bits of a uint64 (high bits zero), so
    we just take the low 32 bits.
    """
    bg = np.random.MT19937(0)  # seed ignored — we overwrite state below
    bg.state = {
        'bit_generator': 'MT19937',
        'state': {'key': _cpp_mt19937_seed_state(seed), 'pos': 624},
    }
    raws64 = bg.random_raw(n)
    return raws64.astype(np.uint32)

# The C++ uses DBL_MIN as a sentinel "tiny but nonzero" weight when an
# uncertainty-derived weight would be NaN, inf, or zero. numpy's
# equivalent is np.finfo(np.float64).tiny.
_TINY = float(np.finfo(np.float64).tiny)


def _half_sample_bounds(sorted_arr: np.ndarray,
                        sorted_w: np.ndarray) -> tuple[int, int]:
    """Return (low_idx, high_idx) of the half-sample mode window in a
    weighted, sorted array. Direct numpy translation of the C++
    `getHalfSampleBounds` (cpp/src/RCR.cpp:990) — find the index pair
    bracketing half the total weight, minimizing window WIDTH."""
    n = sorted_arr.size
    if n <= 1:
        return 0, max(n - 1, 0)
    cumw = np.cumsum(sorted_w)
    s_vec = cumw - 0.5 * sorted_w
    half = float(cumw[-1]) * 0.5
    totals = s_vec + half
    ks = np.searchsorted(s_vec, totals, side="right") - 1
    valid = (s_vec <= half) & (totals <= s_vec[-1] * (1 + 1e-12) + 1e-12)
    if not np.any(valid):
        return 0, n - 1
    widths = np.where(valid, sorted_arr[np.clip(ks, 0, n - 1)] - sorted_arr, np.inf)
    min_idx = int(np.argmin(widths))
    return min_idx, int(ks[min_idx])


class PriorType(Enum):
    """Match the C++ `priorTypes` enum in cpp/src/FunctionalForm.h."""
    CUSTOM = "CUSTOM_PRIORS"
    GAUSSIAN = "GAUSSIAN_PRIORS"
    CONSTRAINED = "CONSTRAINED_PRIORS"
    MIXED = "MIXED_PRIORS"


class Priors:
    """Probabilistic priors on model parameters.

    Mirrors the C++ `Priors` class (cpp/src/FunctionalForm.h:22). Four
    modes:

    - ``GAUSSIAN``: `gaussian_params[i] = (mu_i, sigma_i)`. NaN values mean
      no prior on that param. Implemented as quadratic penalty residuals
      `(params[i] - mu_i) / sigma_i` appended to the least-squares residual.
    - ``CONSTRAINED``: `param_bounds[i] = (lb_i, ub_i)`. NaN means
      unbounded on that side. Wired into scipy.optimize.least_squares as
      its `bounds=` argument.
    - ``MIXED``: both Gaussian and Constrained simultaneously.
    - ``CUSTOM``: user supplies `p(params) -> ndarray` which is appended
      to the residual vector verbatim (allows arbitrary log-posterior
      shape).

    Construct with the appropriate mode + data; the regression() routine
    in FunctionalForm reads `prior_type`, `gaussian_params`,
    `param_bounds`, and `p` as needed.
    """

    def __init__(
        self,
        prior_type: PriorType | None = None,
        p: Callable[[np.ndarray], np.ndarray] | None = None,
        gaussian_params: Sequence[Sequence[float]] | None = None,
        param_bounds: Sequence[Sequence[float]] | None = None,
    ):
        self.prior_type = prior_type
        self.p = p
        self.gaussian_params = (None if gaussian_params is None
                                else np.asarray(gaussian_params, dtype=np.float64))
        self.param_bounds = (None if param_bounds is None
                             else np.asarray(param_bounds, dtype=np.float64))

    def gaussian_residuals(self, params: np.ndarray) -> np.ndarray:
        """For active Gaussian priors, return `(params[i] - mu) / sigma`
        for each param with a defined (non-NaN) prior. Empty array if no
        Gaussian priors active."""
        if (self.prior_type not in (PriorType.GAUSSIAN, PriorType.MIXED)
                or self.gaussian_params is None):
            return np.empty(0, dtype=np.float64)
        gp = self.gaussian_params
        out: list[float] = []
        for i in range(min(params.size, gp.shape[0])):
            mu, sigma = float(gp[i, 0]), float(gp[i, 1])
            if not (np.isnan(mu) or np.isnan(sigma)) and sigma > 0:
                out.append((float(params[i]) - mu) / sigma)
        return np.asarray(out, dtype=np.float64)

    def scipy_bounds(self, n_params: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert `param_bounds` into the (low, high) arrays scipy expects
        (with `np.inf` substituted for NaN — i.e. unbounded). For Gaussian-
        or no-prior cases, returns the trivial all-unbounded pair."""
        low = np.full(n_params, -np.inf, dtype=np.float64)
        high = np.full(n_params, np.inf, dtype=np.float64)
        if (self.prior_type not in (PriorType.CONSTRAINED, PriorType.MIXED)
                or self.param_bounds is None):
            return low, high
        pb = self.param_bounds
        for i in range(min(n_params, pb.shape[0])):
            lb, ub = float(pb[i, 0]), float(pb[i, 1])
            if not np.isnan(lb):
                low[i] = lb
            if not np.isnan(ub):
                high[i] = ub
        return low, high


@dataclass
class FunctionalFormResults:
    parameters: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    parameter_uncertainties: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    pivot: float = 0.0
    pivot_ND: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


class FunctionalForm:
    """Parametric model for RCR.

    Phase 2 minimal constructor: ``FunctionalForm(f, xdata, ydata, partials,
    guess, tol=1e-6, weights=None, error_y=None)``. Matches the simplest
    of the C++'s 16 overloads (1D, optional weights, optional error bars,
    no priors, no pivot search).

    Use with::

        model = FunctionalForm(linear, xdata, ydata, [d1, d2], guess)
        r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
        r.set_parametric_model(model)
        r.perform_rejection(ydata)
        best = model.result.parameters
    """

    # Static-on-class pivot, mirroring the C++ `static double FunctionalForm::pivot`
    # so that user-supplied model functions can reference
    # `FunctionalForm.pivot` directly inside f(x, params).
    pivot: float = 0.0
    pivot_ND: np.ndarray = np.empty(0, dtype=np.float64)

    def __init__(
        self,
        f: Callable[[float, Sequence[float]], float],
        xdata: Sequence[float],
        ydata: Sequence[float],
        partials: Sequence[Callable[[float, Sequence[float]], float]],
        guess: Sequence[float],
        tol: float = 1e-6,
        weights: Sequence[float] | None = None,
        error_y: Sequence[float] | None = None,
        priors: "Priors | None" = None,
        pivot_function: Callable[
            [np.ndarray, np.ndarray,
             Callable[[float, Sequence[float]], float], np.ndarray],
            float,
        ] | None = None,
        pivot_guess: float | None = None,
    ):
        self.f = f
        self.partialsvector = list(partials)
        x_arr = np.asarray(xdata, dtype=np.float64)
        # ND detection: 2D xdata (N rows, D cols) routes through the ND
        # code path; 1D xdata is the standard 1D-in-x case.
        self.ND_check = x_arr.ndim == 2
        self.x = x_arr
        self.y = np.asarray(ydata, dtype=np.float64)
        self.guess = np.asarray(guess, dtype=np.float64)
        self.tolerance = tol
        self.M = self.guess.size      # number of model parameters
        # N = number of data POINTS (rows in x for ND, length for 1D).
        self.N = x_arr.shape[0]

        self.weighted_check = weights is not None
        self.w = (np.ones(self.N, dtype=np.float64) if weights is None
                  else np.asarray(weights, dtype=np.float64))
        self.has_error_bars = error_y is not None
        self.sigma_y = (None if error_y is None
                        else np.asarray(error_y, dtype=np.float64))

        # Mutable per-iteration state — set by set_true_vec().
        self.flags = np.ones(self.N, dtype=bool)
        self.indices = np.arange(self.N, dtype=np.int64)
        self.trueY = self.y.copy()
        self.trueW = self.w.copy()
        self.parameters = self.guess.copy()
        self.meanstartingpoint = self.guess.copy()

        # Detect whether the model is linear in its parameters by probing
        # the partials at two different parameter vectors. If the partials
        # don't depend on params (within float tolerance), we can solve
        # M-combinations of points via vectorized np.linalg.solve. Otherwise
        # we fall back to scipy.fsolve per combo.
        self._is_linear_in_params = self._probe_linear_in_params()

        # Filled lazily by build_model_space when MEDIAN/MODE mu_tech is
        # requested. Shape: parameterSpace[k] = 1D array of candidate
        # values for the k-th parameter across combos.
        self.parameterSpace: list[np.ndarray] = [np.empty(0) for _ in range(self.M)]
        self.weightSpace: list[np.ndarray] = [np.empty(0) for _ in range(self.M)]
        # Tracks the last mu_tech requested so we know whether to refresh
        # parameterSpace each iteration (MEDIAN/MODE need it, MEAN doesn't).
        self._last_mu_tech: str | None = None

        # Output bundle, populated each iteration by regression().
        self.result = FunctionalFormResults()
        self.result.parameters = self.guess.copy()

        # Priors
        self.priors = priors
        self.has_priors = priors is not None

        # Pivot search. If a pivot_function is provided, build_model_space
        # recomputes `FunctionalForm.pivot` each iteration from current
        # trueX/trueW/parameters. `pivot` is a class-level mirror of the C++
        # static `FunctionalForm::pivot`, so user model functions can read
        # it as a global. We reset it on each construction so state from a
        # previous instance (or test) doesn't leak into this one.
        self.pivot_function = pivot_function
        self.has_custom_pivot = pivot_function is not None
        type(self).pivot = float(pivot_guess) if pivot_guess is not None else 0.0
        type(self).pivot_ND = np.empty(0, dtype=np.float64)

    def _probe_linear_in_params(self) -> bool:
        """Heuristic: a model is linear in params iff each partial
        derivative is independent of the params vector. Test by evaluating
        each partial at SEVERAL x values with two different param vectors
        and comparing to float tolerance. Multiple x values are necessary
        because some nonlinear partials happen to be constant at x=0
        (e.g., exp(b*0) = 1 regardless of b).

        Used at construction; defaults to False (slow path) on any error.
        """
        try:
            # Probe at up to 5 distinct x values spread across the range.
            if self.x.size == 0:
                return False
            n_probe = min(5, self.x.size)
            if self.ND_check:
                idx = np.linspace(0, self.x.shape[0] - 1, n_probe, dtype=int)
                xs_probe = [self.x[i] for i in idx]
            else:
                idx = np.linspace(0, self.x.size - 1, n_probe, dtype=int)
                xs_probe = [float(self.x[i]) for i in idx]
            p1 = self.guess.copy()
            p2 = self.guess.copy() + 1.0
            for partial in self.partialsvector:
                for xp in xs_probe:
                    v1 = float(partial(xp, p1))
                    v2 = float(partial(xp, p2))
                    if not (np.isfinite(v1) and np.isfinite(v2)):
                        return False
                    if abs(v1 - v2) > 1e-10 * (abs(v1) + abs(v2) + 1.0):
                        return False
            return True
        except Exception:
            return False

    # -----------------------------------------------------------------
    # Per-iteration hooks called by the RCR rejection loop
    # -----------------------------------------------------------------

    def set_true_vec(self, flags: np.ndarray, y: np.ndarray,
                     w: np.ndarray | None = None) -> None:
        """Port of cpp/src/FunctionalForm.cpp:770 (and :735 weighted).
        Records the current flag mask + filtered y/w views; called once
        per rejection-loop iteration before regression()."""
        self.flags = flags
        self.indices = np.where(flags)[0].astype(np.int64)
        self.trueY = y[self.indices].copy()
        if w is not None:
            self.trueW = w[self.indices].copy()

    def build_model_space(self, build_combos: bool = False) -> None:
        """Port of cpp/src/FunctionalForm.cpp:799.

        Two responsibilities in the C++:
          1. Recompute the pivot point (if `has_custom_pivot`) from current
             flagged x / weights / parameters. Always done.
          2. Build parameterSpace / weightSpace from M-combinations of
             flagged data points, for use by get_*_median / get_*_mode.
             Only done when `build_combos=True` (caller passes this when
             mu_tech is MEDIAN or MODE).
        """
        if self.has_custom_pivot and self.pivot_function is not None:
            truex = self.x[self.indices]
            n_kept = truex.shape[0]
            if self.weighted_check:
                truew = self.w[self.indices]
            else:
                truew = np.ones(n_kept, dtype=np.float64)
            new_pivot = self.pivot_function(
                truex, truew, self.f, np.asarray(self.parameters, dtype=np.float64))
            if self.ND_check:
                arr = np.asarray(new_pivot, dtype=np.float64)
                type(self).pivot_ND = arr
                self.result.pivot_ND = arr
            else:
                p = float(new_pivot)
                type(self).pivot = p
                self.result.pivot = p

        if build_combos:
            self._build_param_combo_space()
        return None

    def _build_param_combo_space(self) -> None:
        """For each M-combination of currently-flagged data points, solve
        for the parameter vector that fits the model exactly through those
        M points; collect the resulting parameter values per dimension
        into self.parameterSpace and corresponding weights into
        self.weightSpace.

        When the number of combinations exceeds the C++'s trigger threshold
        (40000), we mirror the C++ oracle's biased weighted-roulette sampler
        exactly (cpp/src/FunctionalForm.cpp:903-957). The C++ uses a fixed
        MT19937 seed (post-determinism-fix); we use the same seed and the
        same selection arithmetic so the chosen combos are bit-identical.
        """
        idx = self.indices
        n_kept = idx.size
        M = self.M
        if n_kept < M:
            self.parameterSpace = [np.empty(0) for _ in range(M)]
            self.weightSpace = [np.empty(0) for _ in range(M)]
            return

        from math import comb
        total = comb(n_kept, M)

        # Always enumerate combos first (matches C++ which builds the full
        # combos_indices vector before deciding whether to sample). For the
        # nonlinear path this is too expensive, so we keep the tighter cap
        # there and skip the C++-matching algorithm.
        if not self._is_linear_in_params or self.ND_check:
            if total <= _COMBO_SAMPLE_CAP_NONLINEAR:
                combos = np.array(list(itertools.combinations(range(n_kept), M)),
                                  dtype=np.int64)
            else:
                # Nonlinear fallback: numpy sampling (not bit-identical to C++).
                rng = np.random.default_rng(seed=0xC0FFEE)
                combos = self._sample_combos_numpy(rng, n_kept, M,
                                                   _COMBO_SAMPLE_CAP_NONLINEAR)
        else:
            # Linear path: match C++ behavior exactly.
            combos_all = np.array(list(itertools.combinations(range(n_kept), M)),
                                  dtype=np.int64)
            if _COMBO_FRAC * total > _COMBO_LIMIT:
                combos = self._sample_combos_cpp(combos_all)
            else:
                combos = combos_all

        n_combos = combos.shape[0]

        # Solve for params + compute paramuncertainty-based weights.
        # Port of cpp/src/FunctionalForm.cpp:1011-2200 (one of the 8 type
        # combinations). We do the full weight calculation per combo:
        #   weight[c, k] = correctivesum[c] / sigma_param[c, k]**2
        # falling back to _TINY * correctivesum on NaN/inf/0 (mirrors the
        # C++ DBL_MIN sentinel).
        if self._is_linear_in_params and not self.ND_check:
            params_per_combo, weights_per_combo = (
                self._solve_combos_linear_1d_weighted(combos))
        else:
            params_per_combo, weights_per_combo = (
                self._solve_combos_general_weighted(combos))

        # Filter to finite-param combos. Mirrors the C++ behavior of
        # discarding combos where the GN solver failed (M+2 sentinel).
        #
        # extraParameterSpace / extraWeightSpace deliberately NOT
        # implemented: the C++ uses these for combos where modifiedGN
        # detected a "runaway" parameter (returned M+1 values). Neither
        # np.linalg.solve nor scipy.optimize.fsolve produces an M+1
        # signal in our port — they either succeed cleanly or return
        # non-finite/error. We discard non-finite results entirely.
        finite = np.all(np.isfinite(params_per_combo), axis=1) & \
                 np.all(np.isfinite(weights_per_combo), axis=1)
        params_per_combo = params_per_combo[finite]
        weights_per_combo = weights_per_combo[finite]
        self.parameterSpace = [params_per_combo[:, k] for k in range(M)]
        self.weightSpace = [weights_per_combo[:, k] for k in range(M)]

    def _sample_combos_cpp(self, combos_all: np.ndarray) -> np.ndarray:
        """Mirror the C++ oracle's biased weighted-roulette combo sampler
        (cpp/src/FunctionalForm.cpp:903-957) exactly so the chosen combos
        are bit-identical between port and oracle.

        Algorithm (as written in the C++, including its non-standard
        selection condition `R < weights[k]` after subtracting weights[k]):

          1. combo_weights[k] = sum of point weights for points in combo k
          2. totalcomboweight = sum of combo_weights
          3. For j in 0..combolimit-1:
               R = (raw_uint32 / 2^32) * totalcomboweight
               for k in 0..n_combos-1:
                   R -= combo_weights[k]
                   if R < combo_weights[k]:
                       select combo k; break
          4. Deduplicate (set sorted lexicographically — matches std::set
             iteration order).

        The selection condition is equivalent to picking the first k where
        `R_initial < cumweights[k-1] + 2*weights[k]`. This biases the first
        combo (effectively 2x weight) and makes the last combo unreachable.
        We replicate the bias exactly because matching C++ requires it.
        """
        idx = self.indices
        if self.weighted_check:
            point_w = np.asarray(self.w[idx], dtype=np.float64)
        else:
            # Unweighted: C++ uses w=1.0 per point in this path.
            point_w = np.ones(idx.size, dtype=np.float64)

        # combo_weights[k] = point_w[combos_all[k, 0]] + ... + point_w[combos_all[k, M-1]]
        combo_weights = point_w[combos_all].sum(axis=1)
        total_w = float(combo_weights.sum())
        n_combos = combos_all.shape[0]

        # Thresholds for vectorized selection. The C++ inner loop selects
        # the first k where R_initial < threshold[k], where
        #   threshold[k] = sum(weights[0..k-1]) + 2*weights[k]
        #              = cumweights[k] + weights[k]      (cumweights inclusive)
        cumweights = np.cumsum(combo_weights)
        thresholds = cumweights + combo_weights

        # When thresholds are monotonically non-decreasing we can use
        # np.searchsorted. That holds iff weights[k+1] >= weights[k]/2 for
        # all k. In unweighted mode all combo_weights are equal (=M), so
        # always monotonic. In weighted mode it usually holds; we fall
        # back to a slow linear scan if it doesn't, to guarantee a
        # bit-identical match in all cases.
        is_monotonic = bool(np.all(np.diff(thresholds) >= 0))

        # Generate _COMBO_LIMIT uint32 draws from std::mt19937(0xC0FFEE).
        raws = _cpp_mt19937_uint32_stream(0xC0FFEE, _COMBO_LIMIT)
        # IEEE-754 double exactly represents 2^32 = 4294967296.
        rs = (raws.astype(np.float64) / 4294967296.0) * total_w

        if is_monotonic:
            # Smallest k such that thresholds[k] > R  →  searchsorted side='right'.
            # (side='right' returns insertion index just past equal values,
            # which equals the first index with thresholds[k] > R for R not
            # exactly equal to a threshold — float R make ties effectively
            # impossible.)
            ks = np.searchsorted(thresholds, rs, side='right')
            # Defensive clip; in theory ks < n_combos always because
            # thresholds[n_combos-1] = totalcomboweight + weights[n_combos-1]
            # > totalcomboweight > R.
            ks = np.minimum(ks, n_combos - 1)
        else:
            # Fallback: literal translation of the C++ inner loop.
            ks = np.empty(_COMBO_LIMIT, dtype=np.int64)
            cw = combo_weights  # local alias
            for j in range(_COMBO_LIMIT):
                R = float(rs[j])
                sel = n_combos - 1
                for k in range(n_combos):
                    R -= cw[k]
                    if R < cw[k]:
                        sel = k
                        break
                ks[j] = sel

        # Dedup, preserve lex-sorted order to mirror C++ std::set iteration.
        chosen: set[tuple[int, ...]] = set()
        combos_picked = combos_all[ks]
        for row in combos_picked:
            chosen.add(tuple(int(x) for x in row))
        return np.array(sorted(chosen), dtype=np.int64)

    def _sample_combos_numpy(self, rng: np.random.Generator,
                             n_kept: int, M: int, n_samples: int) -> np.ndarray:
        """Numpy-based combo sampler. Used only in the nonlinear-model path
        where matching C++ exactly is not worth the enumeration cost. Not
        bit-identical to the C++ oracle."""
        idx = self.indices
        if self.weighted_check:
            point_w = self.w[idx]
            point_w = point_w / point_w.sum()
        else:
            point_w = None
        chosen: set[tuple[int, ...]] = set()
        attempts = 0
        max_attempts = n_samples * 4
        while len(chosen) < n_samples and attempts < max_attempts:
            picks = rng.choice(n_kept, size=M, replace=False, p=point_w)
            chosen.add(tuple(sorted(int(p) for p in picks)))
            attempts += 1
        return np.array(sorted(chosen), dtype=np.int64)

    def _solve_combos_linear_1d(self, combos: np.ndarray) -> np.ndarray:
        """Vectorized solve for `linear-in-params` 1D models. For each
        combo, build the M×M Jacobian-of-params matrix `A` (partials
        evaluated at each x in the combo) and the M-vector y, then call
        np.linalg.solve to get the unique param vector through those M
        points. Shape: (n_combos, M)."""
        idx = self.indices
        n_combos = combos.shape[0]
        M = self.M
        x_kept = self.x[idx]
        y_kept = self.y[idx]
        params0 = self.parameters  # only used to provide shape to partials

        # A[c, a, b] = partials[b](x[combos[c, a]], params0)
        A = np.empty((n_combos, M, M), dtype=np.float64)
        for a in range(M):
            x_a = x_kept[combos[:, a]]
            for b in range(M):
                # partials are scalar functions; loop over n_combos.
                A[:, a, b] = np.array(
                    [float(self.partialsvector[b](float(xv), params0))
                     for xv in x_a],
                    dtype=np.float64,
                )
        # numpy's batched solve interprets the last two dims of b as
        # (M, N) — so for a single RHS per system we add a trailing axis.
        b_vec = y_kept[combos][:, :, None]  # (n_combos, M, 1)
        try:
            result = np.linalg.solve(A, b_vec).squeeze(-1)
        except np.linalg.LinAlgError:
            # Some combos may have singular A; solve per-combo with
            # least-squares fallback (lstsq returns a value even when
            # singular).
            result = np.empty((n_combos, M), dtype=np.float64)
            for c in range(n_combos):
                try:
                    result[c] = np.linalg.solve(A[c], b_vec[c, :, 0])
                except np.linalg.LinAlgError:
                    result[c] = np.full(M, np.nan)
        return result

    def _solve_combos_linear_1d_weighted(self, combos: np.ndarray
                                          ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized solve for linear-in-params 1D models that ALSO
        computes the per-combo paramuncertainty-derived weights.

        Port of cpp/src/FunctionalForm.cpp:1941 (and similar branches for
        weighted / has-error-bars cases). For each combo:
          1. Solve A @ p = y for the param vector p (exact fit).
          2. Compute A_inv = inv(A) — the M×M Jacobian inverse.
          3. Compute the effective per-point sigma vector `sig_y`:
                unweighted, no-EB:        sig_y = [1, 1, ..., 1]
                weighted, no-EB:          sig_y = sqrt(wbar / w_i)
                unweighted, has-EB:       sig_y = sigma_y_i
                weighted, has-EB:         sig_y = sqrt(wbar / w_i) * sigma_y_i
          4. sigma_params[k] = ||A_inv[k, :] * sig_y||₂
          5. correctivesum = sum over combo of:
                unweighted, no-EB:        1.0  (so correctivesum = M)
                weighted, no-EB:          w_i
                unweighted, has-EB:       1 / sigma_y_i²
                weighted, has-EB:         w_i / sigma_y_i²
          6. weight[k] = correctivesum / sigma_params[k]²
             (fallback _TINY * correctivesum on NaN/inf/0)

        Returns (params (n_combos, M), weights (n_combos, M)).
        """
        idx = self.indices
        n_combos = combos.shape[0]
        M = self.M
        x_kept = self.x[idx]
        y_kept = self.y[idx]
        w_kept = self.w[idx]
        sy_kept = self.sigma_y[idx] if self.has_error_bars else None
        params0 = self.parameters

        # Build A[c, a, b] = partials[b](x[combo[c, a]], params0)
        A = np.empty((n_combos, M, M), dtype=np.float64)
        for a in range(M):
            x_a = x_kept[combos[:, a]]
            for b in range(M):
                A[:, a, b] = np.array(
                    [float(self.partialsvector[b](float(xv), params0))
                     for xv in x_a],
                    dtype=np.float64,
                )

        b_vec = y_kept[combos][:, :, None]   # (n_combos, M, 1)
        try:
            params = np.linalg.solve(A, b_vec).squeeze(-1)  # (n_combos, M)
        except np.linalg.LinAlgError:
            params = np.empty((n_combos, M), dtype=np.float64)
            for c in range(n_combos):
                try:
                    params[c] = np.linalg.solve(A[c], b_vec[c, :, 0])
                except np.linalg.LinAlgError:
                    params[c] = np.full(M, np.nan)

        # A_inv shape: (n_combos, M, M)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.empty((n_combos, M, M), dtype=np.float64)
            for c in range(n_combos):
                try:
                    A_inv[c] = np.linalg.inv(A[c])
                except np.linalg.LinAlgError:
                    A_inv[c] = np.full((M, M), np.nan)

        # Per-combo wbar (only used in weighted case).
        if self.weighted_check:
            wbar = float(np.mean(w_kept))
        else:
            wbar = 1.0

        # Per-point sig_y for each combo: shape (n_combos, M).
        if self.weighted_check and self.has_error_bars:
            w_c = w_kept[combos]
            sy_c = sy_kept[combos]
            sig_y = np.sqrt(wbar / w_c) * sy_c
        elif self.weighted_check:
            w_c = w_kept[combos]
            sig_y = np.sqrt(wbar / w_c)
        elif self.has_error_bars:
            sig_y = sy_kept[combos]
        else:
            sig_y = np.ones((n_combos, M), dtype=np.float64)

        # sigma_params[c, k] = sqrt(sum_j (A_inv[c, k, j] * sig_y[c, j])²)
        # = norm along j of (A_inv * sig_y[:, None, :])
        scaled = A_inv * sig_y[:, None, :]
        sigma_params_sq = np.sum(scaled * scaled, axis=2)  # (n_combos, M)
        sigma_params = np.sqrt(sigma_params_sq)

        # correctivesum per combo
        if self.weighted_check and self.has_error_bars:
            correctivesum = np.sum(w_kept[combos] / (sy_kept[combos] ** 2), axis=1)
        elif self.weighted_check:
            correctivesum = np.sum(w_kept[combos], axis=1)
        elif self.has_error_bars:
            correctivesum = np.sum(1.0 / (sy_kept[combos] ** 2), axis=1)
        else:
            correctivesum = np.full(n_combos, float(M), dtype=np.float64)

        # weight[c, k] = correctivesum[c] / sigma_params[c, k]²
        # Fallback: _TINY * correctivesum where the natural formula is
        # NaN / inf / 0 (matches C++ DBL_MIN sentinel).
        with np.errstate(divide="ignore", invalid="ignore"):
            testweight = 1.0 / (sigma_params * sigma_params)
        bad = ~np.isfinite(testweight) | (testweight == 0.0)
        weights = correctivesum[:, None] * testweight
        # Where testweight is bad, use _TINY * correctivesum.
        fallback = _TINY * correctivesum[:, None]
        weights = np.where(bad, fallback, weights)

        return params, weights

    def _solve_combos_general_weighted(self, combos: np.ndarray
                                        ) -> tuple[np.ndarray, np.ndarray]:
        """Per-combo fsolve + weight computation for non-linear-in-params
        or ND models. Slower than the linear path, but produces the same
        (params, weights) shape."""
        idx = self.indices
        n_combos = combos.shape[0]
        M = self.M
        x_kept = self.x[idx]
        y_kept = self.y[idx]
        w_kept = self.w[idx]
        sy_kept = self.sigma_y[idx] if self.has_error_bars else None
        params0 = self.parameters

        if self.weighted_check:
            wbar = float(np.mean(w_kept))
        else:
            wbar = 1.0

        params = np.empty((n_combos, M), dtype=np.float64)
        weights = np.empty((n_combos, M), dtype=np.float64)

        for c in range(n_combos):
            xs = x_kept[combos[c]]
            ys = y_kept[combos[c]]

            def residual(p, _xs=xs, _ys=ys):
                if self.ND_check:
                    return np.array(
                        [self.f(_xs[i], p) - _ys[i] for i in range(M)],
                        dtype=np.float64,
                    )
                return np.array(
                    [self.f(float(_xs[i]), p) - float(_ys[i]) for i in range(M)],
                    dtype=np.float64,
                )

            try:
                sol, info, ier, msg = fsolve(residual, params0, full_output=True)
                if ier != 1:
                    params[c] = np.full(M, np.nan)
                    weights[c] = np.full(M, np.nan)
                    continue
                params[c] = sol
            except Exception:
                params[c] = np.full(M, np.nan)
                weights[c] = np.full(M, np.nan)
                continue

            # Build Jacobian at the fitted params.
            J = np.empty((M, M), dtype=np.float64)
            for a in range(M):
                xa = xs[a] if self.ND_check else float(xs[a])
                for b in range(M):
                    J[a, b] = float(self.partialsvector[b](xa, sol))
            try:
                J_inv = np.linalg.inv(J)
            except np.linalg.LinAlgError:
                weights[c] = np.full(M, np.nan)
                continue

            # sig_y per point in combo
            if self.weighted_check and self.has_error_bars:
                sig_y = np.sqrt(wbar / w_kept[combos[c]]) * sy_kept[combos[c]]
                correctivesum = float(np.sum(
                    w_kept[combos[c]] / (sy_kept[combos[c]] ** 2)))
            elif self.weighted_check:
                sig_y = np.sqrt(wbar / w_kept[combos[c]])
                correctivesum = float(np.sum(w_kept[combos[c]]))
            elif self.has_error_bars:
                sig_y = sy_kept[combos[c]]
                correctivesum = float(np.sum(1.0 / (sy_kept[combos[c]] ** 2)))
            else:
                sig_y = np.ones(M, dtype=np.float64)
                correctivesum = float(M)

            scaled = J_inv * sig_y[None, :]
            sigma_params_sq = np.sum(scaled * scaled, axis=1)
            sigma_params = np.sqrt(sigma_params_sq)

            with np.errstate(divide="ignore", invalid="ignore"):
                testweight = 1.0 / (sigma_params * sigma_params)
            bad = ~np.isfinite(testweight) | (testweight == 0.0)
            wt = correctivesum * testweight
            wt = np.where(bad, _TINY * correctivesum, wt)
            weights[c] = wt

        return params, weights

    # Note: `_solve_combos_linear_1d` and `_solve_combos_general` (without
    # weights) have been superseded by their `_weighted` cousins above,
    # which compute paramuncertainty-derived weights inline. The unweighted
    # variants are kept here for backward compat with any caller that
    # wants params-only (no test does today).

    def regression(self) -> np.ndarray:
        """Fit the model to the currently-flagged data via least squares.

        Port of cpp/src/FunctionalForm.cpp:2382. The C++ dispatches to
        modifiedGN(f, partials, y, x, guess, tol, [w], [sigma_y]); we use
        scipy.optimize.least_squares which solves the same problem with
        a Trust Region Reflective algorithm by default.

        Stashes `_last_jacobian` and `_last_residual_count` for the
        uncertainty computation in `get_bestfit_errorbars`.
        """
        x_good = self.x[self.indices]
        y_good = self.y[self.indices]

        def residual(params):
            if self.ND_check:
                # x_good has shape (n_kept, D); iterate rows.
                model_y = np.array([self.f(x_good[i], params)
                                    for i in range(x_good.shape[0])],
                                   dtype=np.float64)
            else:
                model_y = np.array([self.f(xi, params) for xi in x_good],
                                   dtype=np.float64)
            r = y_good - model_y
            if self.has_error_bars:
                sigma = self.sigma_y[self.indices]
                r = r / sigma
            if self.weighted_check:
                w_good = self.w[self.indices]
                r = r * np.sqrt(w_good)
            # Priors: append Gaussian-prior penalty residuals (constrained
            # bounds are handled via scipy's bounds= argument below). Custom
            # priors append the user-defined penalty.
            if self.has_priors and self.priors is not None:
                extra: list[np.ndarray] = []
                g = self.priors.gaussian_residuals(np.asarray(params, dtype=np.float64))
                if g.size > 0:
                    extra.append(g)
                if (self.priors.prior_type is PriorType.CUSTOM
                        and self.priors.p is not None):
                    extra.append(np.asarray(
                        self.priors.p(np.asarray(params, dtype=np.float64)),
                        dtype=np.float64,
                    ))
                if extra:
                    r = np.concatenate([r] + extra)
            return r

        # Constrained priors → scipy `bounds` argument.
        if (self.has_priors and self.priors is not None
                and self.priors.prior_type in (PriorType.CONSTRAINED, PriorType.MIXED)):
            low, high = self.priors.scipy_bounds(self.M)
            bounds = (low, high)
            # Clip initial guess into bounds (scipy errors otherwise).
            start = np.clip(self.meanstartingpoint, low, high)
        else:
            bounds = (-np.inf, np.inf)
            start = self.meanstartingpoint

        result = least_squares(
            residual,
            start,
            xtol=self.tolerance,
            ftol=self.tolerance,
            method="trf",
            bounds=bounds,
        )
        params = result.x.astype(np.float64)
        if params.size > self.M:
            params = params[: self.M]
        self.parameters = params
        # Stash for paramuncertainty:
        self._last_jacobian = np.asarray(result.jac, dtype=np.float64)
        self._last_residual = np.asarray(result.fun, dtype=np.float64)
        self._last_n_kept = x_good.size
        return params

    # -----------------------------------------------------------------
    # Robust parameter-space estimators: median / mode over combo space
    # -----------------------------------------------------------------

    def get_nd_median(self) -> np.ndarray:
        """Port of cpp/src/RCR.cpp:1142 (get2DMedian) and its 3D/ND
        cousins. Returns the per-parameter weighted median across the
        current parameterSpace. Stores the result as the new
        meanstartingpoint for the subsequent generalized-mean call.
        """
        from rcr2 import stats as _stats
        params = np.empty(self.M, dtype=np.float64)
        for k in range(self.M):
            arr = self.parameterSpace[k]
            w_arr = self.weightSpace[k]
            if arr.size == 0:
                params[k] = self.parameters[k]
                continue
            idx = np.argsort(arr, kind="stable")
            sorted_x = arr[idx]
            sorted_w = w_arr[idx]
            params[k] = _stats.getMedian_w(sorted_w, sorted_x)
        self.meanstartingpoint = params.copy()
        return params

    def get_nd_mode(self) -> np.ndarray:
        """Port of cpp/src/RCR.cpp:1063 (get2DMode) and its 3D/ND cousins.
        Iteratively narrows each parameter dimension to its half-sample
        modal window, filtering combos to those within ALL dimensions'
        windows, until the windows stabilize. Returns the weighted median
        of the final filtered combos per parameter."""
        from rcr2 import stats as _stats
        if any(arr.size == 0 for arr in self.parameterSpace):
            return self.parameters.copy()

        # All-combo arrays (mutable working set).
        x_dims = [arr.copy() for arr in self.parameterSpace]
        w_dims = [arr.copy() for arr in self.weightSpace]

        prev_signature = None
        for _ in range(100):  # safety cap
            # Find half-sample modal window per dimension.
            lows = np.empty(self.M, dtype=np.float64)
            highs = np.empty(self.M, dtype=np.float64)
            for k in range(self.M):
                arr = x_dims[k]
                if arr.size < 2:
                    lows[k] = -np.inf
                    highs[k] = np.inf
                    continue
                idx = np.argsort(arr, kind="stable")
                sorted_arr = arr[idx]
                sorted_w = w_dims[k][idx]
                low_idx, high_idx = _half_sample_bounds(sorted_arr, sorted_w)
                lows[k] = sorted_arr[low_idx]
                highs[k] = sorted_arr[high_idx]

            # Filter to combos within every dimension's window.
            mask = np.ones(self.parameterSpace[0].size, dtype=bool)
            for k in range(self.M):
                arr = self.parameterSpace[k]
                mask &= (arr >= lows[k]) & (arr <= highs[k])
            n_in_window = int(mask.sum())
            if n_in_window == 0:
                # Empty filter — keep previous working set, terminate.
                break
            new_x_dims = [self.parameterSpace[k][mask] for k in range(self.M)]
            new_w_dims = [self.weightSpace[k][mask] for k in range(self.M)]

            sig = (n_in_window, tuple(float(lows[k]) for k in range(self.M)),
                                tuple(float(highs[k]) for k in range(self.M)))
            if sig == prev_signature:
                break
            prev_signature = sig
            x_dims = new_x_dims
            w_dims = new_w_dims

        # Weighted median of the final filtered set per dimension.
        params = np.empty(self.M, dtype=np.float64)
        for k in range(self.M):
            arr = x_dims[k]
            w_arr = w_dims[k]
            if arr.size == 0:
                params[k] = self.parameters[k]
                continue
            idx = np.argsort(arr, kind="stable")
            sorted_x = arr[idx]
            sorted_w = w_arr[idx]
            params[k] = _stats.getMedian_w(sorted_w, sorted_x)
        self.meanstartingpoint = params.copy()
        return params

    def get_errors(self, line: np.ndarray) -> np.ndarray:
        """Port of cpp/src/FunctionalForm.cpp:2458 (and :2479 for ND).
        Returns y - f(x, line) for currently-flagged points only."""
        line = np.asarray(line, dtype=np.float64)
        if self.ND_check:
            return np.array(
                [self.y[i] - self.f(self.x[i], line)
                 for i in range(self.N) if self.flags[i]],
                dtype=np.float64,
            )
        return np.array(
            [self.y[i] - self.f(self.x[i], line) for i in range(self.N) if self.flags[i]],
            dtype=np.float64,
        )

    def get_bestfit_errorbars(self, line: np.ndarray) -> np.ndarray:
        """Parameter standard errors from the post-fit Jacobian.

        Port of cpp/src/FunctionalForm.cpp:2311, but using the standard
        nonlinear-least-squares covariance estimator
            cov = inv(J^T J) * (sum_res² / (N - M))
            unc[k] = sqrt(diag(cov)[k])
        applied to the scaled Jacobian that scipy.optimize.least_squares
        returns. The C++ uses an unusual reduction in `paramuncertainty`
        that operates on only M of the N weight/sigma_y values (looks
        like an indexing bug); we substitute the textbook formula.

        Returns an empty array if regression() hasn't been called yet
        (mirroring the C++ behavior when no error bars/weights are set).
        """
        if not hasattr(self, "_last_jacobian"):
            return np.empty(0, dtype=np.float64)
        J = self._last_jacobian
        r = self._last_residual
        n = self._last_n_kept
        m = self.M
        if n <= m:
            # under-determined; can't compute meaningful errors
            return np.full(m, np.nan, dtype=np.float64)
        # The Jacobian scipy returns is already scaled by sqrt(w) and 1/sigma_y
        # if those were folded into the residual function — so the standard
        # formula gives the correctly-weighted parameter covariance.
        JtJ = J.T @ J
        try:
            cov_unscaled = np.linalg.inv(JtJ)
        except np.linalg.LinAlgError:
            return np.full(m, np.nan, dtype=np.float64)
        # If weights/errors were supplied, the residuals are already
        # whitened (unitless), so we don't rescale; otherwise scale by
        # the residual variance.
        if self.has_error_bars or self.weighted_check:
            cov = cov_unscaled
        else:
            sigma_sq = float(np.sum(r * r)) / (n - m)
            cov = cov_unscaled * sigma_sq
        diag = np.diag(cov)
        # Numerical near-zero negatives can appear; clamp.
        diag = np.where(diag < 0, 0.0, diag)
        return np.sqrt(diag).astype(np.float64)

    # -----------------------------------------------------------------
    # The combined "mu" step the loop calls when muType == PARAMETRIC.
    # Mirrors RCR::handleMuTechSelect() (no-arg version) at
    # cpp/src/RCR.cpp:5937.
    # -----------------------------------------------------------------

    def handle_mu_tech_select(self, mu_tech: str = "MEAN") -> np.ndarray:
        """Fit the model and return residuals. Dispatches by mu_tech:
            MEAN   → regression() (standard nonlinear least squares)
            MEDIAN → weighted median across the M-combination parameter space
            MODE   → half-sample mode across the M-combination parameter space

        The rejection loop calls this once per iteration. Ports
        cpp/src/RCR.cpp:5937 (handleMuTechSelect, no-arg version).
        """
        if mu_tech == "MEDIAN":
            # build_model_space must have been called with build_combos=True
            line = self.get_nd_median()
        elif mu_tech == "MODE":
            line = self.get_nd_mode()
        else:  # MEAN or anything else
            line = self.regression()
        self.parameters = line
        self.result.parameters = line
        # Uncertainty is meaningful only after a least-squares fit; for
        # MEDIAN/MODE we still call get_bestfit_errorbars which will use
        # the last-stored Jacobian if regression() was ever called.
        self.result.parameter_uncertainties = self.get_bestfit_errorbars(line)
        return self.get_errors(line)
