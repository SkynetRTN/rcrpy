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


def isEqual(x: float, y: float,
            max_relative_error: float = 1e-8,
            max_absolute_error: float | None = None) -> bool:
    """Port of cpp/src/RCR.cpp:206. Default abs tolerance is DBL_MIN, which is
    effectively zero — only the relative error path matters."""
    if max_absolute_error is None:
        max_absolute_error = np.finfo(float).tiny  # ~DBL_MIN
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
    tiny = np.finfo(np.float64).tiny
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

    Vectorized. The C++ inner loop builds the cumulative half-weight vector
        s_vec[i] = w[0] + w[1] + ... + w[i-1] + 0.5 * w[i]
    (with the lower-bound offset), then for each i searches for the matching
    forward (branch 1) or backward (branch 2) k whose s_vec[k] differs from
    s_vec[i] by `half_weight_sum`. Branches 1 and 2 together iterate the same
    set of (i, k) endpoint pairs from both ends; vectorizing the forward
    sweep alone gives the same min-distance answer.

    Replacing the linear search with `np.searchsorted(side='right')` is
    correct because s_vec is monotonically non-decreasing (positive weights).
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

        cumw = np.cumsum(w_win)
        s_vec = cumw - 0.5 * w_win
        half_weight_sum = float(cumw[-1]) * 0.5

        # Branch 1 forward sweep: for each i, find largest k with
        # s_vec[k] <= s_vec[i] + half_weight_sum. searchsorted with
        # side='right' returns the *first* index where the value would be
        # inserted to keep sorted order; subtract 1 to get the LAST index
        # that's <= the target. This matches the C++ linear forward search
        # (where the loop advances while still <= total, then steps back).
        totals = s_vec + half_weight_sum
        ks = np.searchsorted(s_vec, totals, side="right") - 1

        # Valid i values (branch 1 fires): s_vec[i] <= half_weight_sum
        # (with isEqual relative tolerance). For i past that, s_vec[i] + hws
        # exceeds the maximum s_vec value, ks[i] saturates at size-1, and
        # those candidates are dominated by branch-2-equivalent windows that
        # branch-1 already considered with smaller i. Mask them out.
        valid_b1 = (s_vec < half_weight_sum) | (
            np.abs(s_vec - half_weight_sum) <= REL_TOL * abs(half_weight_sum)
        )
        # Also drop i where the forward window doesn't reach the requested
        # weight (totals > s_vec[-1] beyond isEqual tolerance — symmetrically
        # those are covered by branch 2 from larger i).
        within_range = totals <= s_vec[-1] * (1.0 + REL_TOL) + REL_TOL
        valid = valid_b1 & within_range

        distances = np.where(valid, np.abs(y_win[ks] - y_win), np.inf)
        min_dist = float(np.min(distances))
        if not np.isfinite(min_dist):
            break  # degenerate; shouldn't occur for positive weights

        if min_dist == 0.0:
            tied = (distances == 0.0)
        else:
            tied = np.abs(distances - min_dist) <= REL_TOL * abs(min_dist)
        tied_idx = np.where(tied)[0]

        final_lower = int(tied_idx[0]) + lower_limit
        # Across ties, take the maximum upper endpoint (mirrors the
        # max(finalUpper, ...) expansion in the C++).
        final_upper = int(np.max(ks[tied_idx])) + lower_limit

        lower_limit, upper_limit = final_lower, final_upper

    window_y = y[lower_limit:upper_limit + 1]
    window_w = w[lower_limit:upper_limit + 1]
    return getMedian_w(window_w, window_y)


def inverf(x: float) -> float:
    """Port of cpp/src/RCR.cpp:187. RCR's custom inverse-erf approximation.
    Do NOT replace with scipy.special.erfinv — see notes on erfcCustom."""
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
    fits — inverse-erf of cumulative weight fractions."""
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

    delta_chi_squared = (single_line_error - double_line_error) / double_line_error

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
    mean = float(np.sum(w[:counter])) / counter
    st_dev = 0.0
    for i in range(counter):
        st_dev += (float(w[i]) - mean) ** 2
    st_dev = math.sqrt(st_dev / (counter - 1))
    return st_dev / mean


def getCFRatio(w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:151. Coefficient of variation of weights
    (sample stdev / mean), used inside weighted CFs."""
    n = w.size
    mean = float(np.sum(w)) / n
    diffs = w - mean
    st_dev = math.sqrt(float(np.sum(diffs * diffs)) / (n - 1))
    return st_dev / mean


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

def getStDev(delta: float, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:891. RMS-style estimator: sqrt(sum(y^2) / (n - delta))."""
    n = y.size
    top = float(np.sum(y * y))
    return math.sqrt(top / (n - delta))


def getStDev_w(delta: float, w: np.ndarray, y: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:878."""
    top = float(np.sum(w * y * y))
    w_sum = float(np.sum(w))
    w_sum_sq = float(np.sum(w * w))
    return math.sqrt(top / (w_sum - delta * w_sum_sq / w_sum))


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
