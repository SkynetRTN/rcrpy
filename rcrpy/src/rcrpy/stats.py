"""Statistics primitives ported from cpp/src/RCR.cpp (anonymous namespace).

Phase 1 scope: just the helpers needed for the lower-sigma iterative loop
running LS_MODE_68 (and the 3-pass refinement that performRejection does).

Function names follow the C++ originals (camelCase) where parity matters,
so a reader can grep both source trees with the same names.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------

def _cpp_seqsum(arr: np.ndarray) -> float:
    """Sequential left-to-right summation matching C++'s naive `+=` loop.

    `np.sum` uses pairwise summation (numerically more accurate but a
    different bit pattern than C++). `np.cumsum(...)[-1]` is implemented
    as a sequential C loop, so it produces bit-identical results to
    C++'s ``double s = 0; for (i) s += arr[i];`` pattern. Required for
    bit-identical parity through the double-line sigma estimator
    (fitDL_w/mFinder_w) where summation drift cascades into ES_MODE_DL
    + parametric rejection decisions.
    """
    if arr.size == 0:
        return 0.0
    return float(np.cumsum(arr.astype(np.float64, copy=False))[-1])


def erfcCustom(x: float) -> float:
    """Closed-form Chebyshev-like approximation of erfc used by RCR. Ported
    from cpp/src/RCR.cpp:196. Do NOT replace with scipy.special.erfc — the
    rejection boundary depends on this exact approximation.

    Guards against Python's overflow-to-exception on enormous `x` (which
    can occur when parametric residuals on one side of mu collapse to a
    near-zero sigma and the rejection ratio explodes). C++ silently goes
    to 1/inf == 0; Python raises. We catch and return 0.0, matching the
    mathematical limit erfc(x) → 0 as x → ∞.
    """
    x = x / math.sqrt(2.0)
    inner = (
        1.0
        + x * (0.0705230784
        + x * (0.0422820123
        + x * (0.0092705272
        + x * (0.0001520143
        + x * (0.0002765672
        + 0.0000430638 * x)))))
    )
    try:
        return 1.0 / inner ** 16
    except OverflowError:
        return 0.0


# DBL_MIN (~2.2e-308). Hoisted to module scope: np.finfo(float).tiny constructs
# a finfo object, and isEqual is called ~466k times per cornifer pipeline — so
# rebuilding it per call was ~0.4s of pure waste (485k finfo.__new__). The value
# is identical, so this is bit-neutral.
_DBL_MIN = float(np.finfo(np.float64).tiny)


def isEqual(x: float, y: float,
            max_relative_error: float = 1e-8,
            max_absolute_error: float | None = None) -> bool:
    """Port of cpp/src/RCR.cpp:206. Default abs tolerance is DBL_MIN, which is
    effectively zero — only the relative error path matters."""
    if max_absolute_error is None:
        max_absolute_error = _DBL_MIN  # ~DBL_MIN, computed once at import
    if abs(x - y) < max_absolute_error:
        return True
    if abs(y) > abs(x):
        rel = abs((x - y) / y)
    else:
        rel = abs((x - y) / x) if x != 0 else float("inf")
    return rel <= max_relative_error


def isEqual_vec_scalar(a: np.ndarray, b: float,
                       max_relative_error: float = 1e-8) -> np.ndarray:
    """Vectorized `isEqual(a[i], b)` for an array `a` and scalar `b`.
    Same semantics as the scalar `isEqual`: an entry is "equal to b" iff
        |a[i] - b| < DBL_MIN  OR
        |a[i] - b| / max(|a[i]|, |b|)  <=  rel_tol.
    Returns a bool ndarray the same shape as a.
    """
    a = np.asarray(a, dtype=np.float64)
    tiny = _DBL_MIN  # module constant; avoids per-call np.finfo construction
    diff = np.abs(a - b)
    abs_a = np.abs(a)
    abs_b = abs(b)
    larger = np.maximum(abs_a, abs_b)
    abs_close = diff < tiny
    # rel_diff is meaningful only when larger > 0; in the both-zero case
    # abs_close already returns True. Use np.where to dodge 0/0 warnings.
    safe = np.where(larger > 0, larger, 1.0)
    rel = diff / safe
    return abs_close | (rel <= max_relative_error)


def getDiff(mu: float, datum: float) -> float:
    return abs(datum - mu)


# ---------------------------------------------------------------------------
# Mu calculations (mean / median / half-sample-mode)
# ---------------------------------------------------------------------------

def getMean(y: np.ndarray) -> float:
    return float(np.sum(y) / y.size)


def getMean_w(w: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(w * y) / np.sum(w))


def getMedian(y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:505 — getMedian(std::vector<double> &y).

    The C++ assumes y is already sorted ascending. We preserve the exact
    linear-interpolation formula it uses (NOT numpy.median, even though for
    the standard cases they agree, because we want byte-identical output
    on parity tests).
    """
    n = y.size
    if n <= 1:
        return float(y[0])
    high = n // 2
    low = high - 1
    total_sum = n
    if n % 2 == 0:
        running_sum = n / 2.0 + 0.5
    else:
        running_sum = n / 2.0
    return float(y[low] + (0.5 * total_sum - running_sum + 1.0) * (y[high] - y[low]))


def getMedian_w(w: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:474. Weighted median; assumes y is sorted
    ascending (and w is the weight at each y in that order)."""
    n = y.size
    if n <= 1:
        return float(y[0])
    total_sum = float(np.sum(w))
    sum_counter = 0
    running_sum = w[sum_counter] * 0.5
    while running_sum < 0.5 * total_sum:
        sum_counter += 1
        running_sum += w[sum_counter - 1] * 0.5 + w[sum_counter] * 0.5
    if sum_counter == 0:
        return float(y[0])
    denom = w[sum_counter - 1] * 0.5 + w[sum_counter] * 0.5
    return float(
        y[sum_counter - 1]
        + (0.5 * total_sum - (running_sum - denom)) / denom * (y[sum_counter] - y[sum_counter - 1])
    )


def _binarySearch(search_up: bool, minimum_index: int, to_find: float, to_search: np.ndarray) -> int:
    """Port of cpp/src/RCR.cpp:362. Custom binary search used by halfSampleMode."""
    if search_up:
        low, high = minimum_index, to_search.size
    else:
        low, high = 0, minimum_index
    low_in, high_in = -1, -1
    mid = 0
    while low != low_in or high != high_in:
        low_in, high_in = low, high
        mid = int(low + (high - low) / 2.0)
        if isEqual(to_find, float(to_search[mid])):
            low = mid
            high = mid
        elif to_find > to_search[mid]:
            low = mid
        elif to_find < to_search[mid]:
            high = mid
    return low if search_up else high


def halfSampleMode_w(w: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:531 — getMode(trueCount, w, y), weighted.

    The C++ builds the cumulative half-weight vector
        s_vec[i] = 0.5*w[lower] + w[lower+1] + ... + w[i-1] + 0.5*w[i]
    then, for every i, considers TWO candidate windows:
      * branch 1 (forward):  anchor the LOW end at i, extend up to the last k
        with s_vec[k] <= s_vec[i] + halfWeightSum.  Window [i, k].
      * branch 2 (backward): anchor the HIGH end at i, extend down to the
        first k with s_vec[k] >= s_vec[i] - halfWeightSum.  Window [k, i].
    It picks the minimum-width window (by |y| span), expanding across isEqual
    ties via min(finalLower)/max(finalUpper).

    IMPORTANT: both branches are required. A forward-only sweep silently drops
    the upper-anchored windows (e.g. [mid, top] on odd-size ranges), which
    makes the weighted mode disagree with the unweighted mode even at uniform
    weights and breaks parity with the C++ on LS_MODE_68 / LS_MODE_DL /
    ES_MODE_DL weighted runs. Both branches are vectorized here; s_vec is
    monotonically non-decreasing (positive weights), so searchsorted replaces
    the C++ linear k-search.
    """
    n = y.size
    lower_limit, upper_limit = 0, n - 1
    lower_limit_in, upper_limit_in = -1, -1
    REL_TOL = 1e-8

    while lower_limit != lower_limit_in or upper_limit != upper_limit_in:
        lower_limit_in = lower_limit
        upper_limit_in = upper_limit
        size = upper_limit - lower_limit + 1
        if size <= 1:
            break

        y_win = y[lower_limit:upper_limit + 1]
        w_win = w[lower_limit:upper_limit + 1]
        idx = np.arange(size)

        cumw = np.cumsum(w_win)
        s_vec = cumw - 0.5 * w_win
        hws = float(cumw[-1]) * 0.5

        # Branch 1 (forward): valid where s_vec[i] <= hws (isEqual-tolerant).
        # k1[i] = last index with s_vec[k] <= s_vec[i] + hws (tolerant). The
        # +tol nudge includes C++ isEqual matches just above the threshold.
        totals1 = s_vec + hws
        k1 = np.searchsorted(s_vec, totals1 + REL_TOL * np.abs(totals1),
                             side="right") - 1
        k1 = np.clip(k1, 0, size - 1)
        valid1 = s_vec <= hws + REL_TOL * abs(hws)
        dist1 = np.where(valid1, np.abs(y_win[k1] - y_win), np.inf)

        # Branch 2 (backward): valid where s_vec[i] >= hws (isEqual-tolerant).
        # k2[i] = first index with s_vec[k] >= s_vec[i] - hws (tolerant).
        totals2 = s_vec - hws
        k2 = np.searchsorted(s_vec, totals2 - REL_TOL * np.abs(totals2),
                             side="left")
        k2 = np.clip(k2, 0, size - 1)
        valid2 = s_vec >= hws - REL_TOL * abs(hws)
        dist2 = np.where(valid2, np.abs(y_win - y_win[k2]), np.inf)

        # Combine both branches' candidate windows. lo/hi are the window
        # endpoints; branch 1 is [i, k1], branch 2 is [k2, i].
        all_dist = np.concatenate([dist1, dist2])
        all_lo = np.concatenate([idx, k2])
        all_hi = np.concatenate([k1, idx])

        min_dist = float(np.min(all_dist))
        if not np.isfinite(min_dist):
            break  # degenerate; shouldn't occur for positive weights

        if min_dist == 0.0:
            tied = (all_dist == 0.0)
        else:
            tied = np.abs(all_dist - min_dist) <= REL_TOL * abs(min_dist)

        # Expand the chosen range across ties: min lower / max upper endpoint
        # (mirrors the C++ min(finalLower)/max(finalUpper) accumulation).
        final_lower = int(np.min(all_lo[tied])) + lower_limit
        final_upper = int(np.max(all_hi[tied])) + lower_limit
        lower_limit, upper_limit = final_lower, final_upper

    window_y = y[lower_limit:upper_limit + 1]
    window_w = w[lower_limit:upper_limit + 1]
    return getMedian_w(window_w, window_y)


def inverf(x: float) -> float:
    """Port of cpp/src/RCR.cpp:187. RCR's custom inverse-erf approximation.
    Do NOT replace with scipy.special.erfinv — see notes on erfcCustom.

    NOTE: kept as a scalar function on purpose. The C++ uses ``pow(t, 2)`` here;
    the scalar Python ``t ** 2`` reproduces that bit-for-bit, but NumPy has no
    way to apply libm ``pow`` elementwise (it lowers exponent-2 to a multiply),
    so a vectorized form drifts ~1 ULP from the oracle and can flip a rejection
    boundary. Speed for this hotspot comes from the optional Numba JIT
    (rcrpy._fast), which keeps the scalar form and is therefore bit-exact."""
    PI = 3.1415926535897932384626434
    inverf_mult = (8.0 * (PI - 3.0)) / (3 * PI * (4.0 - PI))
    sqrt2 = math.sqrt(2.0)
    x_log = math.log(1 - x * x)
    inner = (2.0 / (PI * inverf_mult) + x_log * 0.5) ** 2 - x_log / inverf_mult
    return sqrt2 * math.sqrt(-2.0 / (PI * inverf_mult) - x_log * 0.5 + math.sqrt(inner))


def countAmountLessThanOne(x: np.ndarray) -> int:
    """Port of cpp/src/RCR.cpp:342. Counts leading entries of x that are < 1.0,
    assuming x is monotonically nondecreasing (true for getXVec output)."""
    n = x.size
    if n == 1:
        return 0 if x[0] >= 1.0 else 1
    count = 0
    while count < n and x[count] < 1.0:
        count += 1
    return count


def getXVec_w(size: int, w: np.ndarray) -> np.ndarray:
    """Port of cpp/src/RCR.cpp:414. Weighted x-vector for single/double-line
    fits — inverse-erf of cumulative weight fractions.

    Scalar loop (NOT NumPy-vectorized): inverf() can't be vectorized bit-exactly
    (see its docstring). The optional Numba JIT in rcrpy._fast compiles this loop
    to machine code while preserving the exact scalar arithmetic.
    """
    w_sum = float(np.sum(w))
    s_sum = 0.682689 * float(w[0])
    inv_w_sum = 1.0 / w_sum
    x = np.empty(size, dtype=np.float64)
    x[0] = inverf(s_sum * inv_w_sum)
    for i in range(1, size):
        s_sum += 0.317311 * float(w[i - 1]) + 0.682689 * float(w[i])
        x[i] = inverf(s_sum * inv_w_sum)
    return x


def getOriginFixedRegressionLine_w(start: int, end: int, w: np.ndarray,
                                   x: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:709. Slope of best-fit line through the
    origin: sum(w*x*y) / sum(w*x*x) over [start, end). Uses sequential
    summation matching C++'s `+=` accumulation order — required so the
    `single_line_fit` value (which becomes the sigma fallback in fitDL_w)
    is bit-identical to the C++."""
    w_s = w[start:end]
    x_s = x[start:end]
    y_s = y[start:end]
    wx = w_s * x_s
    num = _cpp_seqsum(wx * y_s)
    den = _cpp_seqsum(wx * x_s)
    return float(num / den)


def fitSL_w(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:968 — single-line fit, weighted."""
    return getOriginFixedRegressionLine_w(0, countAmountLessThanOne(x), w, x, y)


def mFinder_w(low: int, high: int, last_x_under_one: int, increment: int,
              w: np.ndarray, x: np.ndarray, y: np.ndarray) -> int:
    """Port of cpp/src/RCR.cpp:730. Each-m accumulators use sequential
    summation matching the C++'s naive `+=` order; required for
    bit-identical parity in the double-line sigma estimator (see
    fitDL_w's docstring for the full chain). The outer m-loop stays
    Python because each iteration depends on the running best-m via
    m_low/m_high.
    """
    DBL_MAX = float("inf")
    stop = False
    best_m = -1
    m_low = -1
    m_high = -1
    min_error = DBL_MAX
    while not stop:
        m = low
        while m < high:
            x_at_m = float(x[m])

            # Per-index term arrays in C++ left-then-right order. `a` and
            # `e` accumulate across both loops (cpp/src/RCR.cpp:747-762).
            w_l = w[:m + 1]
            x_l = x[:m + 1]
            y_l = y[:m + 1]
            wx_l = w_l * x_l
            a_terms_left = wx_l * x_l
            e_terms_left = wx_l * y_l

            end_r = last_x_under_one + 1
            w_r = w[m + 1:end_r]
            x_r = x[m + 1:end_r]
            y_r = y[m + 1:end_r]
            diff_r = x_r - x_at_m
            xatm_w_r = x_at_m * w_r
            a_terms_right = (x_at_m * x_at_m) * w_r
            b_terms = xatm_w_r * diff_r
            d_terms = (w_r * diff_r) * diff_r
            e_terms_right = xatm_w_r * y_r
            f_terms = (w_r * y_r) * diff_r

            a = _cpp_seqsum(np.concatenate([a_terms_left, a_terms_right]))
            e = _cpp_seqsum(np.concatenate([e_terms_left, e_terms_right]))
            b = _cpp_seqsum(b_terms)
            d = _cpp_seqsum(d_terms)
            f = _cpp_seqsum(f_terms)
            c = b
            if a != 0 and (d - c * b) != 0:
                tau = (f - e * c / a) / (d - c * b / a)
                sigma = (e - tau * b) / a

                factors_l = sigma * x_l - y_l
                err_terms_left = (w_l * factors_l) * factors_l
                factors_r = sigma * x_at_m + tau * diff_r - y_r
                err_terms_right = (w_r * factors_r) * factors_r
                error = _cpp_seqsum(
                    np.concatenate([err_terms_left, err_terms_right]))

                if error < min_error:
                    min_error = error
                    best_m = m
                    m_low = max(best_m - increment - 1, 1)
                    m_high = min(best_m + increment + 1, last_x_under_one)
            m += increment
        if increment > 1:
            increment = max(int(math.floor((m_high - m_low) / 6.36)), 1)
        else:
            stop = True
        low = m_low
        high = m_high
        min_error = DBL_MAX
    if low == high:
        return low
    return best_m


def fitDL_w(counter: float, w: np.ndarray, x: np.ndarray, y: np.ndarray,
            get_fn_w) -> float:
    """Port of cpp/src/RCR.cpp:5785. Weighted double-line sigma estimator.

    Uses sequential (left-to-right) summation via _cpp_seqsum to match
    the C++'s naive `+=` accumulation order bit-for-bit. The previous
    implementation used np.sum (pairwise) which drifted ~1e-15 per call
    and cascaded into 8-20% parameter divergence on ES_MODE_DL +
    parametric (where two independent sigma_below/sigma_above tracks
    amplify the drift). `a` and `e` accumulate across BOTH the left and
    right loops in C++; we concatenate the term arrays so a single
    cumsum produces the same order. Same for double_line_error.
    """
    amount_x_under_one = countAmountLessThanOne(x)
    single_line_fit = getOriginFixedRegressionLine_w(0, amount_x_under_one, w, x, y)
    if x.size < 4:
        return single_line_fit
    m = mFinder_w(1, amount_x_under_one - 1, amount_x_under_one - 1,
                  max(int(y.size / 6.36), 1), w, x, y)
    x_at_m = float(x[m])

    # Per-index term arrays. Multiplication order must match the C++
    # exactly: `wxProd = w[i]*x[i]; a += wxProd * x[i]` → `(w*x) * x`.
    # Left piece (i in [0, m+1)):
    w_l = w[:m + 1]
    x_l = x[:m + 1]
    y_l = y[:m + 1]
    wx_l = w_l * x_l
    a_terms_left = wx_l * x_l
    e_terms_left = wx_l * y_l

    # Right piece (i in [m+1, amount_x_under_one)):
    w_r = w[m + 1:amount_x_under_one]
    x_r = x[m + 1:amount_x_under_one]
    y_r = y[m + 1:amount_x_under_one]
    diff_r = x_r - x_at_m
    xatm_w_r = x_at_m * w_r
    a_terms_right = (x_at_m * x_at_m) * w_r
    b_terms = xatm_w_r * diff_r
    d_terms = (w_r * diff_r) * diff_r
    e_terms_right = xatm_w_r * y_r
    f_terms = (w_r * y_r) * diff_r

    # Sequential cumsum matching C++ `+=` order. `a` and `e` accumulate
    # across BOTH loops in C++ (lines 5797-5811), so concatenate before
    # summing.
    a = _cpp_seqsum(np.concatenate([a_terms_left, a_terms_right]))
    e = _cpp_seqsum(np.concatenate([e_terms_left, e_terms_right]))
    b = _cpp_seqsum(b_terms)
    d = _cpp_seqsum(d_terms)
    f = _cpp_seqsum(f_terms)
    c = b

    tau = (f - e * c / a) / (d - c * b / a)
    sigma = (e - tau * b) / a

    # Double-line error also accumulates across both loops in C++.
    factors_l = sigma * x_l - y_l
    err_terms_left = (w_l * factors_l) * factors_l
    factors_r = sigma * x_at_m + tau * diff_r - y_r
    err_terms_right = (w_r * factors_r) * factors_r
    double_line_error = _cpp_seqsum(
        np.concatenate([err_terms_left, err_terms_right]))

    # Single-line error: single loop in C++ (5828-5832).
    w_full = w[:amount_x_under_one]
    x_full = x[:amount_x_under_one]
    y_full = y[:amount_x_under_one]
    err_full = single_line_fit * x_full - y_full
    single_line_error = _cpp_seqsum(w_full * err_full * err_full)

    # C++ computes (singleLineError - doubleLineError)/doubleLineError with
    # IEEE float division (cpp/src/RCR.cpp:5834). On perfectly-linear /
    # near-constant data the double-line fit is exact, so doubleLineError
    # collapses to 0 and the C++ gets +inf / -inf / NaN -- all of which make
    # the `delta_chi_squared < getFN(...)` test below false, leaving sigma at
    # the double-line value. Python raises ZeroDivisionError on the scalar
    # divide, so reproduce the IEEE result explicitly.
    num = single_line_error - double_line_error
    if double_line_error != 0.0:
        delta_chi_squared = num / double_line_error
    elif num > 0.0:
        delta_chi_squared = math.inf
    elif num < 0.0:
        delta_chi_squared = -math.inf
    else:
        delta_chi_squared = math.nan

    if sigma < 0:
        sigma = 1e-10
    if delta_chi_squared < get_fn_w(int(counter), x, w):
        sigma = single_line_fit
    return sigma


def getFNRatio(x: np.ndarray, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:167. Coefficient of variation of weights
    among entries with x[i] < 1."""
    counter = 0
    n = x.size
    while counter < n and x[counter] < 1.0:
        counter += 1
    if counter == 0:
        return math.nan   # no x<1: C++ computes 0/0 -> NaN (no spread)
    mean = float(np.sum(w[:counter])) / counter
    st_dev = 0.0
    for i in range(counter):
        st_dev += (float(w[i]) - mean) ** 2
    # counter == 1 -> sqrt(0/0) = NaN in C++ (single point, no CV). IEEE-safe.
    st_dev = _cpp_sqrt_div(st_dev, counter - 1)
    return math.nan if mean == 0.0 else st_dev / mean


def getCFRatio(w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:151. Coefficient of variation of weights
    (sample stdev / mean), used inside weighted CFs."""
    n = w.size
    mean = float(np.sum(w)) / n
    diffs = w - mean
    # n == 1 -> sqrt(0/0) = NaN in C++ (single point, no CV). IEEE-safe divide.
    st_dev = _cpp_sqrt_div(float(np.sum(diffs * diffs)), n - 1)
    return math.nan if mean == 0.0 else st_dev / mean


def halfSampleMode(y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:622 — getMode(trueCount, y), unweighted.

    Vectorized rewrite. The C++ inner loop over `i` computes, for each
    candidate window start, the distance `y[i + half_window] - y[i]` and
    tracks the minimum (expanding the chosen range across `isEqual` ties).
    All of that compresses to one numpy subtraction + argmin/where with
    a relative-tolerance tie mask.

    The unweighted s_vec is the arithmetic sequence [0.5, 1.5, ..., size-0.5],
    which makes the binary-search-and-distance computation collapse to
    `windows[i] = y[lower + i + half_window] - y[lower + i]` for
    `i in [0, size - half_window)`. Both C++ branches (forward and backward)
    iterate the same set of windows, so vectorizing the forward sweep alone
    is sufficient.
    """
    n = y.size
    lower_limit, upper_limit = 0, n - 1
    lower_limit_in, upper_limit_in = -1, -1
    REL_TOL = 1e-8

    while lower_limit != lower_limit_in or upper_limit != upper_limit_in:
        lower_limit_in = lower_limit
        upper_limit_in = upper_limit
        size = upper_limit - lower_limit + 1
        half_window = size // 2

        # Edge case: tiny windows. half_window == 0 happens only for size==1,
        # which the outer convergence check would have already exited on.
        if half_window == 0:
            break

        # windows[i] = y[lower + i + half_window] - y[lower + i]
        # length = size - half_window
        # For an even-`size` window, this is the standard HSM construction.
        # For odd `size`, the C++ midpoint-i has BOTH branches fire with the
        # same distance — equivalent to "midpoint gets counted twice", which
        # doesn't change the argmin result.
        y_win = y[lower_limit:upper_limit + 1]
        windows = y_win[half_window:] - y_win[:size - half_window]

        min_dist = float(np.min(windows))

        # Tie expansion: the C++ uses isEqual (relative tol 1e-8) and tracks
        # min(finalLower) / max(finalUpper) over tied windows.
        if min_dist == 0.0:
            tied = (windows == 0.0)
        else:
            tied = np.abs(windows - min_dist) <= REL_TOL * abs(min_dist)
        tied_idx = np.where(tied)[0]
        final_lower = int(tied_idx[0]) + lower_limit
        final_upper = int(tied_idx[-1]) + half_window + lower_limit

        lower_limit = final_lower
        upper_limit = final_upper

    window = y[lower_limit:upper_limit + 1]
    return getMedian(window)


# ---------------------------------------------------------------------------
# Sigma calculations
# ---------------------------------------------------------------------------

def _cpp_sqrt_div(top: float, denom: float) -> float:
    """``sqrt(top / denom)`` with C++ ``pow``/``sqrt`` IEEE semantics (no
    exception). When the kept set has no spread — e.g. a single point (N=1),
    where the residual variance vanishes — ``denom`` goes to 0 and the C++
    computes ``sqrt(0/0) = NaN`` (or ``sqrt(x/0) = +inf``). Python instead
    raises ZeroDivisionError on the divide and ValueError on ``sqrt`` of a
    negative, so reproduce the IEEE result: NaN for the indeterminate/negative
    cases, +inf for ``x>0`` over 0. Downstream sigma handling then treats it as
    'no spread -> nothing to reject' (kept set unchanged), matching the oracle.
    Same C++-IEEE-vs-Python-raise guard as erfcCustom / _reject_ratio /
    rejection._pow10pow10."""
    if denom > 0.0:
        frac = top / denom
    elif denom == 0.0:
        frac = math.inf if top > 0.0 else math.nan
    else:  # denom < 0
        frac = top / denom
    if frac != frac or frac < 0.0:   # NaN or negative -> C++ sqrt -> NaN
        return math.nan
    return math.sqrt(frac)           # sqrt(+inf) -> +inf


def getStDev(delta: float, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:891. RMS-style estimator: sqrt(sum(y^2) / (n - delta))."""
    n = y.size
    top = float(np.sum(y * y))
    return _cpp_sqrt_div(top, n - delta)


def getStDev_w(delta: float, w: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:878."""
    top = float(np.sum(w * y * y))
    w_sum = float(np.sum(w))
    w_sum_sq = float(np.sum(w * w))
    return _cpp_sqrt_div(top, w_sum - delta * w_sum_sq / w_sum)


def get68th(y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:934 — unweighted, sorts internally."""
    y_sorted = np.sort(y)
    n = y_sorted.size
    if n <= 1:
        return float(y_sorted[0])
    total_sum = float(n)
    sum_counter = 0
    running_sum = 1.0 * 0.682689
    while running_sum < 0.682689 * total_sum:
        sum_counter += 1
        running_sum += 1.0 * 0.317311 + 1.0 * 0.682689
    if sum_counter == 0:
        return float(y_sorted[0])
    denom = 1.0 * 0.317311 + 1.0 * 0.682689
    return float(
        y_sorted[sum_counter - 1]
        + (0.682689 * total_sum - (running_sum - denom)) / denom
        * (y_sorted[sum_counter] - y_sorted[sum_counter - 1])
    )


def get68th_w(w: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:901 — weighted; sorts y ascending, taking w along."""
    idx = np.argsort(y, kind="stable")
    y_sorted = y[idx]
    w_sorted = w[idx]
    n = y_sorted.size
    if n <= 1:
        return float(y_sorted[0])
    total_sum = float(np.sum(w_sorted))
    sum_counter = 0
    running_sum = float(w_sorted[sum_counter] * 0.682689)
    while running_sum < 0.682689 * total_sum:
        sum_counter += 1
        running_sum += float(w_sorted[sum_counter - 1] * 0.317311 + w_sorted[sum_counter] * 0.682689)
    if sum_counter == 0:
        return float(y_sorted[0])
    denom = float(w_sorted[sum_counter - 1] * 0.317311 + w_sorted[sum_counter] * 0.682689)
    return float(
        y_sorted[sum_counter - 1]
        + (0.682689 * total_sum - (running_sum - denom)) / denom
        * (y_sorted[sum_counter] - y_sorted[sum_counter - 1])
    )


# ---------------------------------------------------------------------------
# Auxiliary checks
# ---------------------------------------------------------------------------

def distinctValuesCheck(flags: np.ndarray, y: np.ndarray) -> bool:
    """Port of cpp/src/RCR.cpp:219. True if there are at least 3 distinct
    values among the True-flagged entries of y. Uses isEqual for the
    distinctness test (NOT exact float equality)."""
    n = y.size
    i = 0
    while i < n and not flags[i]:
        i += 1
    if i >= n:
        return False
    a = float(y[i])
    b = a
    i += 1
    while i < n:
        if flags[i] and not isEqual(float(y[i]), a):
            b = float(y[i])
            break
        i += 1
    i += 1
    while i < n:
        if flags[i] and not isEqual(float(y[i]), a) and not isEqual(float(y[i]), b):
            return True
        i += 1
    return False


def distinctValuesCheckParam(param_count: int, flags: np.ndarray, y: np.ndarray) -> bool:
    """Port of cpp/src/RCR.cpp:168 — the PARAMETRIC distinctness guard (3-arg overload).

    Used for a parametric model instead of :func:`distinctValuesCheck`: True once MORE than
    ``param_count`` distinct flagged values are seen. ``y`` is the compacted residual vector
    ``trueY`` (size = kept-count) while ``flags`` is the FULL-size flag array — the C++ iterates
    ``index`` over ``y`` (compacted) but reads ``flags[index]`` (full), so after early rejections it
    samples a mis-aligned sub-window and stops the peel EARLIER than the 2-arg check would. That
    mis-alignment is the C++ behaviour and is reproduced here for bit-parity (it is why the C++ keeps
    2 more marginal scans on the noise-line fit than the 2-arg check does)."""
    distincts: list[float] = []
    for index in range(y.size):
        if flags[index]:
            distinct = True
            for dj in distincts:
                if isEqual(dj, float(y[index])):
                    distinct = False
            if distinct:
                distincts.append(float(y[index]))
        if len(distincts) > param_count:
            return True
    return False
