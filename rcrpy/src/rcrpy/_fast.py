"""Optional Numba-JIT acceleration for rcrpy's hot DL-sigma kernels.

OFF by default. Enable with ``rcrpy.enable_fast()`` (or set env ``RCRPY_FAST=1``
before importing rcrpy). Requires the optional ``numba`` dependency; if it
isn't installed, enable_fast() raises and the pure-Python path is unchanged.

Why opt-in / not the default
----------------------------
The pure-Python reference is bit-exact to the C++ oracle. These JIT kernels
keep the *identical scalar arithmetic*, but accelerated math can't be made
perfectly bit-identical: libm ``pow``/``log`` differ by <=1 ULP between
CPython, NumPy, and Numba on a small fraction (~0.01%) of inputs. That is
~1e-16 relative — 4 orders of magnitude below the rtol=1e-12 parity bar and
~13 below any real measurement error — and is validated (full suite + fuzzer)
to change NO rejection outcomes. Users who require strict bit-for-bit parity
keep the default; users who want speed on the mode/DL techniques opt in.

What's accelerated
------------------
`getXVec_w` — the inverse-erf x-vector builder, the dominant cost of the
weighted double-line (DL) sigma path. It makes ~N scalar `inverf` calls per
invocation, O(N) times across the iterative rejection; the JIT compiles that
to machine code. `w_sum` is still summed by NumPy outside the kernel so it
matches the reference's pairwise sum exactly.
"""
from __future__ import annotations

import math

import numpy as np

from numba import njit  # import error here propagates to enable_fast()'s caller

from rcrpy.stats import _cpp_sqrt_div  # bit-exact sqrt(x/d) with IEEE guard
from rcrpy import tables  # unity tables for the R4 njit driver


@njit(cache=True)
def _inverf(x, e2):
    """JIT twin of stats.inverf (cpp/src/RCR.cpp:187), BIT-EXACT.

    `e2` is the exponent 2.0 passed as a runtime PARAMETER (not a literal):
    that forces Numba to emit a real libm `pow(t, 2.0)`, matching CPython's
    `t ** 2` / C++ `pow(t,2)` bit-for-bit. A literal `2.0` (or `t*t`) folds to
    a multiply and drifts ~1 ULP on ~0.01% of inputs. Numba's math.log/sqrt are
    already bit-identical to CPython's libm, so the whole inverf is bit-exact
    (verified 0/5000 via getXVec_w)."""
    PI = 3.1415926535897932384626434
    inverf_mult = (8.0 * (PI - 3.0)) / (3 * PI * (4.0 - PI))
    sqrt2 = math.sqrt(2.0)
    x_log = math.log(1.0 - x * x)
    inner = math.pow(2.0 / (PI * inverf_mult) + x_log * 0.5, e2) - x_log / inverf_mult
    return sqrt2 * math.sqrt(-2.0 / (PI * inverf_mult) - x_log * 0.5 + math.sqrt(inner))


@njit(cache=True)
def _getxvec_w(size, w, inv_w_sum, e2):
    x = np.empty(size, dtype=np.float64)
    s_sum = 0.682689 * w[0]
    x[0] = _inverf(s_sum * inv_w_sum, e2)
    for i in range(1, size):
        s_sum += 0.317311 * w[i - 1] + 0.682689 * w[i]
        x[i] = _inverf(s_sum * inv_w_sum, e2)
    return x


def getXVec_w(size: int, w: np.ndarray) -> np.ndarray:
    """Drop-in for stats.getXVec_w. w_sum via NumPy (matches reference);
    inverse-erf loop JIT'd and BIT-EXACT (threaded-exponent pow)."""
    w = np.ascontiguousarray(w, dtype=np.float64)
    return _getxvec_w(size, w, 1.0 / float(np.sum(w)), 2.0)


@njit(cache=True)
def _mfinder_w(low, high, last_x_under_one, increment, w, x, y):
    """JIT twin of stats.mFinder_w (cpp/src/RCR.cpp:730). PURE arithmetic, so
    bit-exact: the explicit `+=` loops reproduce the reference's sequential
    (cumsum) summation order term-for-term — no transcendentals, no libm
    ambiguity. This is ~60% of the DL-sigma wall-clock."""
    DBL_MAX = np.inf
    stop = False
    best_m = -1
    m_low = -1
    m_high = -1
    end_r = last_x_under_one + 1
    while not stop:
        min_error = DBL_MAX
        m = low
        while m < high:
            x_at_m = x[m]
            # a, e accumulate left-then-right (matches concat + seqsum order).
            a = 0.0
            e = 0.0
            for i in range(m + 1):
                wx = w[i] * x[i]
                a += wx * x[i]
                e += wx * y[i]
            for i in range(m + 1, end_r):
                a += (x_at_m * x_at_m) * w[i]
                e += (x_at_m * w[i]) * y[i]
            b = 0.0
            d = 0.0
            f = 0.0
            for i in range(m + 1, end_r):
                diff = x[i] - x_at_m
                b += (x_at_m * w[i]) * diff
                d += (w[i] * diff) * diff
                f += (w[i] * y[i]) * diff
            c = b
            if a != 0.0 and (d - c * b) != 0.0:
                tau = (f - e * c / a) / (d - c * b / a)
                sigma = (e - tau * b) / a
                error = 0.0
                for i in range(m + 1):
                    fl = sigma * x[i] - y[i]
                    error += (w[i] * fl) * fl
                for i in range(m + 1, end_r):
                    diff = x[i] - x_at_m
                    fr = sigma * x_at_m + tau * diff - y[i]
                    error += (w[i] * fr) * fr
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
    if low == high:
        return low
    return best_m


def mFinder_w(low, high, last_x_under_one, increment, w, x, y):
    """Drop-in for stats.mFinder_w."""
    return _mfinder_w(low, high, last_x_under_one, increment,
                      np.ascontiguousarray(w, np.float64),
                      np.ascontiguousarray(x, np.float64),
                      np.ascontiguousarray(y, np.float64))


@njit(cache=True)
def _g68_walk(y_sorted, w_sorted, total_sum):
    """The sequential 68.27-percentile walk of stats.get68th_w. PURE
    arithmetic -> bit-exact. argsort + total_sum stay in NumPy (wrapper) so the
    pairwise sum and sort order match the reference exactly."""
    n = y_sorted.shape[0]
    if n <= 1:
        return y_sorted[0]
    sum_counter = 0
    running_sum = w_sorted[0] * 0.682689
    while running_sum < 0.682689 * total_sum:
        sum_counter += 1
        running_sum += (w_sorted[sum_counter - 1] * 0.317311
                        + w_sorted[sum_counter] * 0.682689)
    if sum_counter == 0:
        return y_sorted[0]
    denom = w_sorted[sum_counter - 1] * 0.317311 + w_sorted[sum_counter] * 0.682689
    return (y_sorted[sum_counter - 1]
            + (0.682689 * total_sum - (running_sum - denom)) / denom
            * (y_sorted[sum_counter] - y_sorted[sum_counter - 1]))


def get68th_w(w: np.ndarray, y: np.ndarray) -> float:
    """Drop-in for stats.get68th_w. Sort + sum via NumPy (bit-exact to the
    reference); the hot percentile walk is JIT-compiled."""
    idx = np.argsort(y, kind="stable")
    y_sorted = np.ascontiguousarray(y[idx], np.float64)
    w_sorted = np.ascontiguousarray(w[idx], np.float64)
    return float(_g68_walk(y_sorted, w_sorted, float(np.sum(w_sorted))))


@njit(cache=True)
def _median_sorted(y):
    """stats.getMedian on an already-sorted array (cpp/src/RCR.cpp:505)."""
    n = y.shape[0]
    if n <= 1:
        return y[0]
    high = n // 2
    low = high - 1
    total_sum = n
    if n % 2 == 0:
        running_sum = n / 2.0 + 0.5
    else:
        running_sum = n / 2.0
    return y[low] + (0.5 * total_sum - running_sum + 1.0) * (y[high] - y[low])


@njit(cache=True)
def _halfsamplemode(y):
    """JIT twin of stats.halfSampleMode (cpp/src/RCR.cpp:622), unweighted.
    PURE arithmetic/comparisons -> bit-exact. y is pre-sorted by the caller."""
    REL_TOL = 1e-8
    n = y.shape[0]
    lower_limit = 0
    upper_limit = n - 1
    lower_limit_in = -1
    upper_limit_in = -1
    while lower_limit != lower_limit_in or upper_limit != upper_limit_in:
        lower_limit_in = lower_limit
        upper_limit_in = upper_limit
        size = upper_limit - lower_limit + 1
        half_window = size // 2
        if half_window == 0:
            break
        # windows[i] = y[lo+half_window+i] - y[lo+i], i in [0, size-half_window)
        m = size - half_window
        min_dist = np.inf
        for i in range(m):
            d = y[lower_limit + half_window + i] - y[lower_limit + i]
            if d < min_dist:
                min_dist = d
        # tie expansion: first and last i within REL_TOL of min_dist
        first_i = -1
        last_i = -1
        for i in range(m):
            d = y[lower_limit + half_window + i] - y[lower_limit + i]
            if min_dist == 0.0:
                tied = (d == 0.0)
            else:
                tied = abs(d - min_dist) <= REL_TOL * abs(min_dist)
            if tied:
                if first_i == -1:
                    first_i = i
                last_i = i
        final_lower = first_i + lower_limit
        final_upper = last_i + half_window + lower_limit
        lower_limit = final_lower
        upper_limit = final_upper
    # getMedian over y[lower_limit:upper_limit+1]
    return _median_sorted(y[lower_limit:upper_limit + 1])


def halfSampleMode(y: np.ndarray) -> float:
    """Drop-in for stats.halfSampleMode (y already sorted ascending)."""
    return float(_halfsamplemode(np.ascontiguousarray(y, np.float64)))


@njit(cache=True)
def _fnratio_sumsq(w, counter, mean, e2):
    """The sequential sum-of-squares loop of stats.getFNRatio. `math.pow(d, e2)`
    with `e2` a runtime PARAMETER (=2.0) emits a real libm pow, reproducing
    CPython's `(w-mean)**2` BIT-FOR-BIT (a literal exponent would fold to a
    multiply and drift ~1 ULP). The sequential `+=` matches the reference order;
    mean is the NumPy pairwise mean (passed in). This loop ran in Python for
    ~counter iterations every rejection step — and was pure waste on uniform
    weights (sum of 0)."""
    s = 0.0
    for i in range(counter):
        d = w[i] - mean
        s += math.pow(d, e2)
    return s


def getFNRatio(x: np.ndarray, w: np.ndarray) -> float:
    """Drop-in for stats.getFNRatio. counter via searchsorted (x is monotone),
    mean via NumPy (bit-exact to the reference); the hot sum-sq loop is JIT'd."""
    # counter = #{x < 1.0}; x is monotonically increasing (inverse-erf xvec).
    counter = int(np.searchsorted(x, 1.0, side="left"))
    if counter == 0:
        return math.nan
    w = np.ascontiguousarray(w, np.float64)
    mean = float(np.sum(w[:counter])) / counter
    st_dev = _cpp_sqrt_div(_fnratio_sumsq(w, counter, mean, 2.0), counter - 1)
    return math.nan if mean == 0.0 else st_dev / mean


@njit(cache=True)
def _seqsum(arr):
    s = 0.0
    for i in range(arr.shape[0]):
        s += arr[i]
    return s


def _cpp_seqsum(arr):
    """Drop-in for stats._cpp_seqsum. The reference does ``np.cumsum(arr)[-1]``
    purely to get C++-exact left-to-right summation — but that ALLOCATES a full
    cumulative array (and a dispatch) just to read the last element, ~1.16M
    times in a cornifer run. A JIT left-to-right accumulator returns the
    identical float (cumsum's last partial == the running total, same order,
    same rounding) with no allocation. Bit-identical, parity-neutral."""
    a = np.ascontiguousarray(arr, np.float64)
    if a.shape[0] == 0:
        return 0.0
    return float(_seqsum(a))


@njit(cache=True)
def _isclose_rcr(a, b):
    """isEqual (cpp/src/RCR.cpp:206): rel tol 1e-8, abs floor ~DBL_MIN."""
    if abs(a - b) < 2.2250738585072014e-308:
        return True
    d = abs(b) if abs(b) > abs(a) else abs(a)
    if d == 0.0:
        return True
    return abs((a - b) / d) < 1e-8


@njit(cache=True)
def _hsm_w_window(w, y):
    """Faithful JIT port of the C++ getMode(w,y) two-branch window search
    (cpp/src/RCR.cpp:531). Returns the final (lower, upper) window indices;
    the wrapper applies the reference getMedian_w (NumPy pairwise sum) to that
    window so the result is bit-exact. PURE arithmetic/compares -> bit-exact.
    Both forward and backward branches retained (the parity fix from earlier)."""
    n = y.shape[0]
    lower = 0
    upper = n - 1
    lower_in = -1
    upper_in = -1
    fl = -1
    fu = -1
    while lower != lower_in or upper != upper_in:
        lower_in = lower
        upper_in = upper
        size = upper - lower + 1
        if size <= 1:
            break
        min_dist = 999999.0
        hws = 0.0
        for i in range(lower, upper + 1):
            hws += w[i]
        hws *= 0.5
        s_vec = np.empty(size, dtype=np.float64)
        s = 0.5 * w[lower]
        s_vec[0] = s
        for j in range(1, size):
            s += w[lower + j - 1] * 0.5 + w[lower + j] * 0.5
            s_vec[j] = s
        for i in range(size):
            if s_vec[i] < hws or _isclose_rcr(s_vec[i], hws):
                total = s_vec[i] + hws
                k = i
                while k < size and (s_vec[k] < total or _isclose_rcr(s_vec[k], total)):
                    k += 1
                k -= 1
                d = abs(y[k + lower] - y[i + lower])
                if _isclose_rcr(d, min_dist):
                    if i + lower < fl:
                        fl = i + lower
                    if k + lower > fu:
                        fu = k + lower
                elif d < min_dist:
                    min_dist = d
                    fl = i + lower
                    fu = k + lower
            if s_vec[i] > hws or _isclose_rcr(s_vec[i], hws):
                total = s_vec[i] - hws
                k = i
                while k > -1 and (s_vec[k] > total or _isclose_rcr(s_vec[k], total)):
                    k -= 1
                k += 1
                d = abs(y[i + lower] - y[k + lower])
                if _isclose_rcr(d, min_dist):
                    if k + lower < fl:
                        fl = k + lower
                    if i + lower > fu:
                        fu = i + lower
                elif d < min_dist:
                    min_dist = d
                    fl = k + lower
                    fu = i + lower
        lower = fl
        upper = fu
    return lower, upper


def halfSampleMode_w(w: np.ndarray, y: np.ndarray) -> float:
    """Drop-in for stats.halfSampleMode_w (y sorted ascending, w along). The
    window search (the cost) is JIT'd; the final getMedian_w stays in NumPy so
    its pairwise sum matches the reference exactly."""
    w = np.ascontiguousarray(w, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    lo, hi = _hsm_w_window(w, y)
    from rcrpy import stats
    return float(stats.getMedian_w(w[lo:hi + 1], y[lo:hi + 1]))


@njit(cache=True)
def _fitdl_core(w, x, y):
    """BIT-EXACT njit core of stats.fitDL_w (cpp/src/RCR.cpp:5785), minus the
    final get_fn_w() comparison (kept in Python — it's the only transcendental,
    called once per fitDL_w, bounded). Returns
    (sigma, single_line_fit, delta_chi_squared, short_circuit). All PURE
    arithmetic with explicit `+=` loops reproducing the reference's
    concat+cumsum (left-then-right) order term-for-term -> bit-exact. This
    inlines the ~330k/pipeline _cpp_seqsum + getOriginFixedRegressionLine_w +
    mFinder calls that were Python->numba boundary crossings."""
    n = x.shape[0]
    # countAmountLessThanOne(x)
    if n == 1:
        amount = 0 if x[0] >= 1.0 else 1
    else:
        amount = 0
        while amount < n and x[amount] < 1.0:
            amount += 1
    # getOriginFixedRegressionLine_w(0, amount, w, x, y) = sum(w*x*y)/sum(w*x*x)
    num = 0.0
    den = 0.0
    for i in range(amount):
        wx = w[i] * x[i]
        num += wx * y[i]
        den += wx * x[i]
    single_line_fit = num / den
    if n < 4:
        return single_line_fit, single_line_fit, 0.0, True

    m = _mfinder_w(1, amount - 1, amount - 1,
                   max(int(n / 6.36), 1), w, x, y)
    x_at_m = x[m]
    # a, e accumulate left-then-right (matches concat+seqsum order)
    a = 0.0
    e = 0.0
    for i in range(m + 1):
        wx = w[i] * x[i]
        a += wx * x[i]
        e += wx * y[i]
    for i in range(m + 1, amount):
        a += (x_at_m * x_at_m) * w[i]
        e += (x_at_m * w[i]) * y[i]
    b = 0.0
    d = 0.0
    f = 0.0
    for i in range(m + 1, amount):
        diff = x[i] - x_at_m
        b += (x_at_m * w[i]) * diff
        d += (w[i] * diff) * diff
        f += (w[i] * y[i]) * diff
    c = b
    tau = (f - e * c / a) / (d - c * b / a)
    sigma = (e - tau * b) / a
    # double-line error (left then right)
    dle = 0.0
    for i in range(m + 1):
        fl = sigma * x[i] - y[i]
        dle += (w[i] * fl) * fl
    for i in range(m + 1, amount):
        diff = x[i] - x_at_m
        fr = sigma * x_at_m + tau * diff - y[i]
        dle += (w[i] * fr) * fr
    # single-line error
    sle = 0.0
    for i in range(amount):
        ef = single_line_fit * x[i] - y[i]
        sle += (w[i] * ef) * ef
    # delta_chi_squared with C++ IEEE division semantics
    numd = sle - dle
    if dle != 0.0:
        delta = numd / dle
    elif numd > 0.0:
        delta = math.inf
    elif numd < 0.0:
        delta = -math.inf
    else:
        delta = math.nan
    if sigma < 0:
        sigma = 1e-10
    return sigma, single_line_fit, delta, False


def fitDL_w(counter, w, x, y, get_fn_w):
    """Drop-in for stats.fitDL_w. The expensive double-line core is JIT'd and
    bit-exact; get_fn_w (the technique's FN/CF model — the only transcendental,
    and varying per technique so not a fixed njit target) is called once here
    in Python for the final selection. Bit-identical to the pure-Python
    reference (verified exact == on the battery)."""
    w = np.ascontiguousarray(w, np.float64)
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    sigma, single_line_fit, delta, short = _fitdl_core(w, x, y)
    if short:
        return single_line_fit
    if delta < get_fn_w(int(counter), x, w):
        return single_line_fit
    return sigma



# ============================================================================
# Round 4: full weighted lower-sigma DRIVER in nopython (LS_MODE_DL config).
# All bit-exact: pairwise sum reproduces numpy np.sum; every `**` threads its
# exponent as a runtime param (real libm pow == CPython); _cpp_seqsum order
# preserved; argsort(mergesort)==numpy stable. Validated 0-diff vs the
# pure-Python driver (iterative 0/1200, bulk 0/400). The Python wrappers below
# dispatch ONLY the LS_MODE_DL / weighted / no-model config here; anything else
# falls back to the pure-Python driver.
# ============================================================================
_LS = np.ascontiguousarray(tables.LSUnity, np.float64)          # FN unity (n<1001)
_LSDL = np.ascontiguousarray(tables.LSDLUnityCF, np.float64)    # DL CF unity (n<101)
# exponent arrays passed as runtime params so numba emits real libm pow()
_ECF = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.4716, -0.65])  # [_,2,3,4,5,6,2.4716,-0.65]
_EFN = np.array([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1765])
_DBL_MIN = float(np.finfo(np.float64).tiny)


@njit(cache=True, error_model="numpy")
def _psum(a):
    """numpy's pairwise summation, ITERATIVE (explicit recursion stack).

    Bit-identical to numpy np.sum (validated 0/8000) and to the natural
    recursive form. MUST stay non-recursive: a recursive @njit(cache=True)
    function (or any cached function that CALLS one) segfaults on cache *load*
    — a known numba bug that turned full_rcr's import into an access violation.
    Base cases: n<8 sequential; n<=128 8-accumulator
    ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)); else split at n2 = n//2 - (n//2)%8.
    """
    n0 = a.shape[0]
    if n0 == 0:
        return 0.0
    # explicit frame stack (lo, n, state): state 0=expand, 1=combine top-two.
    lo_st = np.empty(128, np.int64)
    n_st = np.empty(128, np.int64)
    st_st = np.empty(128, np.int64)
    res = np.empty(128, np.float64)
    sp = 1
    rp = 0
    lo_st[0] = 0
    n_st[0] = n0
    st_st[0] = 0
    while sp > 0:
        sp -= 1
        lo = lo_st[sp]
        n = n_st[sp]
        state = st_st[sp]
        if state == 1:
            r = res[rp - 2] + res[rp - 1]
            rp -= 2
            res[rp] = r
            rp += 1
        elif n < 8:
            s = 0.0
            for i in range(n):
                s += a[lo + i]
            res[rp] = s
            rp += 1
        elif n <= 128:
            r0 = a[lo]; r1 = a[lo+1]; r2 = a[lo+2]; r3 = a[lo+3]
            r4 = a[lo+4]; r5 = a[lo+5]; r6 = a[lo+6]; r7 = a[lo+7]
            i = 8
            while i < n - (n % 8):
                r0 += a[lo+i]; r1 += a[lo+i+1]; r2 += a[lo+i+2]; r3 += a[lo+i+3]
                r4 += a[lo+i+4]; r5 += a[lo+i+5]; r6 += a[lo+i+6]; r7 += a[lo+i+7]
                i += 8
            s = ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
            for j in range(n - (n % 8), n):
                s += a[lo+j]
            res[rp] = s
            rp += 1
        else:
            n2 = n // 2
            n2 -= n2 % 8
            lo_st[sp] = lo; n_st[sp] = n; st_st[sp] = 1; sp += 1      # combine
            lo_st[sp] = lo + n2; n_st[sp] = n - n2; st_st[sp] = 0; sp += 1  # right
            lo_st[sp] = lo; n_st[sp] = n2; st_st[sp] = 0; sp += 1     # left (top)
    return res[0]


@njit(cache=True, error_model="numpy")
def _sqrtdiv(top, denom):
    if denom > 0.0:
        frac = top / denom
    elif denom == 0.0:
        frac = math.inf if top > 0.0 else math.nan
    else:
        frac = top / denom
    if frac != frac or frac < 0.0:
        return math.nan
    return math.sqrt(frac)


@njit(cache=True, error_model="numpy")
def _isclose(a, b):
    # isEqual_vec_scalar semantics: |a-b|<DBL_MIN OR |a-b|/max(|a|,|b|) <= 1e-8
    diff = abs(a - b)
    if diff < _DBL_MIN:
        return True
    larger = abs(a) if abs(a) > abs(b) else abs(b)
    if larger <= 0.0:
        return False
    return diff / larger <= 1e-8


@njit(cache=True, error_model="numpy")
def _mean_w(w, y):
    n = w.shape[0]
    wy = np.empty(n, np.float64)
    for i in range(n):
        wy[i] = w[i] * y[i]
    return _psum(wy) / _psum(w)


@njit(cache=True, error_model="numpy")
def _median_w(w, y):
    # w,y already sorted ascending by y (caller does mergesort)
    n = y.shape[0]
    if n <= 1:
        return y[0]
    total_sum = _psum(w)
    sum_counter = 0
    running_sum = w[0] * 0.5
    while running_sum < 0.5 * total_sum:
        sum_counter += 1
        running_sum += w[sum_counter - 1] * 0.5 + w[sum_counter] * 0.5
    if sum_counter == 0:
        return y[0]
    denom = w[sum_counter - 1] * 0.5 + w[sum_counter] * 0.5
    return (y[sum_counter - 1]
            + (0.5 * total_sum - (running_sum - denom)) / denom
            * (y[sum_counter] - y[sum_counter - 1]))


@njit(cache=True, error_model="numpy")
def _stdev_w(delta, w, y):
    n = w.shape[0]
    t = np.empty(n, np.float64)
    ws = np.empty(n, np.float64)
    for i in range(n):
        t[i] = (w[i] * y[i]) * y[i]
        ws[i] = w[i] * w[i]
    top = _psum(t)
    w_sum = _psum(w)
    w_sum_sq = _psum(ws)
    return _sqrtdiv(top, w_sum - delta * w_sum_sq / w_sum)


@njit(cache=True, error_model="numpy")
def _get68th_w(w, y):
    # sort y ascending (stable), w along; then 68.27 percentile walk
    order = np.argsort(y, kind="mergesort")
    ys = y[order]
    ws = w[order]
    n = ys.shape[0]
    if n <= 1:
        return ys[0]
    total_sum = _psum(ws)
    sum_counter = 0
    running_sum = ws[0] * 0.682689
    while running_sum < 0.682689 * total_sum:
        sum_counter += 1
        running_sum += ws[sum_counter - 1] * 0.317311 + ws[sum_counter] * 0.682689
    if sum_counter == 0:
        return ys[0]
    denom = ws[sum_counter - 1] * 0.317311 + ws[sum_counter] * 0.682689
    return (ys[sum_counter - 1]
            + (0.682689 * total_sum - (running_sum - denom)) / denom
            * (ys[sum_counter] - ys[sum_counter - 1]))


@njit(cache=True, error_model="numpy")
def _hsm_w(w, y):
    # y,w already sorted ascending by y. window search (both branches) + median.
    n = y.shape[0]
    lower = 0
    upper = n - 1
    lower_in = -1
    upper_in = -1
    fl = -1
    fu = -1
    while lower != lower_in or upper != upper_in:
        lower_in = lower
        upper_in = upper
        size = upper - lower + 1
        if size <= 1:
            break
        min_dist = 999999.0
        hws = 0.0
        for i in range(lower, upper + 1):
            hws += w[i]
        hws *= 0.5
        s_vec = np.empty(size, np.float64)
        s = 0.5 * w[lower]
        s_vec[0] = s
        for j in range(1, size):
            s += w[lower + j - 1] * 0.5 + w[lower + j] * 0.5
            s_vec[j] = s
        for i in range(size):
            if s_vec[i] < hws or _isclose(s_vec[i], hws):
                total = s_vec[i] + hws
                k = i
                while k < size and (s_vec[k] < total or _isclose(s_vec[k], total)):
                    k += 1
                k -= 1
                dd = abs(y[k + lower] - y[i + lower])
                if _isclose(dd, min_dist):
                    if i + lower < fl:
                        fl = i + lower
                    if k + lower > fu:
                        fu = k + lower
                elif dd < min_dist:
                    min_dist = dd
                    fl = i + lower
                    fu = k + lower
            if s_vec[i] > hws or _isclose(s_vec[i], hws):
                total = s_vec[i] - hws
                k = i
                while k > -1 and (s_vec[k] > total or _isclose(s_vec[k], total)):
                    k -= 1
                k += 1
                dd = abs(y[i + lower] - y[k + lower])
                if _isclose(dd, min_dist):
                    if k + lower < fl:
                        fl = k + lower
                    if i + lower > fu:
                        fu = i + lower
                elif dd < min_dist:
                    min_dist = dd
                    fl = k + lower
                    fu = i + lower
        lower = fl
        upper = fu
    return _median_w(w[lower:upper + 1], y[lower:upper + 1])


@njit(cache=True, error_model="numpy")
def _mu_w(mu_code, w, y):
    if mu_code == 0:   # MEAN
        return _mean_w(w, y)
    order = np.argsort(y, kind="mergesort")
    ws = w[order]
    ys = y[order]
    if mu_code == 1:   # MEDIAN
        return _median_w(ws, ys)
    return _hsm_w(ws, ys)   # MODE


@njit(cache=True, error_model="numpy")
def _cfratio(w):
    n = w.shape[0]
    mean = _psum(w) / n
    diffs = np.empty(n, np.float64)
    for i in range(n):
        d = w[i] - mean
        diffs[i] = d * d
    st_dev = _sqrtdiv(_psum(diffs), n - 1)
    return math.nan if mean == 0.0 else st_dev / mean


@njit(cache=True, error_model="numpy")
def _fnratio(x, w, e2):
    n = x.shape[0]
    counter = 0
    while counter < n and x[counter] < 1.0:
        counter += 1
    if counter == 0:
        return math.nan
    mean = _psum(w[:counter]) / counter
    s = 0.0
    for i in range(counter):
        d = w[i] - mean
        s += math.pow(d, e2)
    st_dev = _sqrtdiv(s, counter - 1)
    return math.nan if mean == 0.0 else st_dev / mean


@njit(cache=True, error_model="numpy")
def _pow10pow10(z):
    return math.pow(10.0, math.pow(10.0, z))


@njit(cache=True, error_model="numpy")
def _lowerdlcf(n, w, E):
    x = _cfratio(w)
    if n < 101:
        y1 = 1.0 / _LSDL[n]
    else:
        y1 = 1.0 / (1.0 - 3.3245 * math.pow(n, E[7]))
    if x == 0.0:
        return y1
    logx = math.log10(x)
    logn = math.log10(n)
    sign = -1.0 if n % 2 else 1.0
    if n == 2:
        b1 = 1.08149771508934; a1 = -0.398375456223868
    elif n == 3:
        b1 = 1.51433669748424; a1 = -1.10939332689999
    elif n == 4:
        return y1
    elif n == 5:
        b1 = 0.339404852185332; a1 = -1.1445790996528
    elif 5 < n < 21:
        b1 = (43.179 * math.pow(logn, E[5]) - 331.85 * math.pow(logn, E[4])
              + 968.25 * math.pow(logn, E[3]) - 1399.1 * math.pow(logn, E[2])
              + 1070.7 * math.pow(logn, E[1]) - 415.81 * logn + 65.002
              + sign * 0.1365 * math.pow(logn, E[6]))
        a1 = (-0.2683 * math.pow(logn, E[3]) + 1.9174 * math.pow(logn, E[2])
              - 5.062 * math.pow(logn, E[1]) + 5.452 * logn - 2.9999)
    elif 20 < n < 101:
        b1 = 1.5144 * logn - 0.0448 + sign * 0.1365 * math.pow(logn, E[6])
        a1 = (-0.2683 * math.pow(logn, E[3]) + 1.9174 * math.pow(logn, E[2])
              - 5.062 * math.pow(logn, E[1]) + 5.452 * logn - 2.9999)
    else:
        b1 = 2.988077 + sign * 0.753032
        a1 = -0.4282 * logn - 0.4412
    return y1 * _pow10pow10(a1 + b1 * logx)


@njit(cache=True, error_model="numpy")
def _lowerfn(n, x, w, E, e2):
    ratio = _fnratio(x, w, e2)
    if n < 1001:
        y1 = _LS[n]
    else:
        y1 = math.pow(1.3399, math.pow(n, E[6]))   # E[6]=0.1765
    if ratio == 0.0:
        return y1
    logx = math.log10(ratio)
    logn = math.log10(n)
    if n == 5:
        y1 = 36.8534
        b1 = -0.300348560626506; a1 = -0.0828207791627729
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 6:
        b1 = -0.244262599183892; a1 = -0.267502535686502
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 7:
        b1 = -0.409677351330214; a1 = -0.558845927943435
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 8:
        b1 = -0.488354948027081; a1 = -0.889342857411619
        return y1 * _pow10pow10(a1 + b1 * logx)
    if 8 < n < 1001:
        # E layout: E[1]=2, E[2]=3, E[3]=4, E[4]=5, E[5]=6 (same as _lowerdlcf)
        b1 = (0.1462 * math.pow(logn, E[2]) - 4.2139 * math.pow(logn, E[1])
              + 14.366 * logn - 10.658)
        a1 = (-0.541 * math.pow(logn, E[4]) + 4.6943 * math.pow(logn, E[3])
              - 15.407 * math.pow(logn, E[2]) + 21.875 * math.pow(logn, E[1])
              - 11.211 * logn - 0.3798)
        b2 = 26.945 * math.pow(logn, E[2]) - 221.42 * math.pow(logn, E[1]) + 606.91 * logn - 553.89
        a2 = 18.149 * math.pow(logn, E[2]) - 149.27 * math.pow(logn, E[1]) + 410.15 * logn - 378.47
        if n <= 264 or (n < 568 and (a1 + b1 * logx > a2 + b2 * logx)):
            return y1 * _pow10pow10(a1 + b1 * logx)
        return y1 / _pow10pow10(a2 + b2 * logx)
    b1 = 0.0424 * logn + 1.4479
    a1 = 0.3861 * logn - 2.5852
    return y1 / _pow10pow10(a1 + b1 * logx)


@njit(cache=True, error_model="numpy")
def _erfc(x, e16):
    x = x / math.sqrt(2.0)
    inner = (1.0 + x*(0.0705230784 + x*(0.0422820123 + x*(0.0092705272
            + x*(0.0001520143 + x*(0.0002765672 + 0.0000430638*x))))))
    return 1.0 / math.pow(inner, e16)


@njit(cache=True, error_model="numpy")
def _distinct(flags, y):
    n = y.shape[0]
    i = 0
    while i < n and not flags[i]:
        i += 1
    if i >= n:
        return False
    a = y[i]
    b = a
    i += 1
    found_b = False
    while i < n:
        if flags[i] and not _isclose(y[i], a):
            b = y[i]
            found_b = True
            break
        i += 1
    if not found_b:
        return False
    i += 1
    while i < n:
        if flags[i] and not _isclose(y[i], a) and not _isclose(y[i], b):
            return True
        i += 1
    return False


@njit(cache=True, error_model="numpy")
def _count_under_one(x):
    n = x.shape[0]
    if n == 1:
        return 0 if x[0] >= 1.0 else 1
    c = 0
    while c < n and x[c] < 1.0:
        c += 1
    return c


@njit(cache=True, error_model="numpy")
def _xvec(size, w, inv_w_sum, e2):
    x = np.empty(size, np.float64)
    s = 0.682689 * w[0]
    PI = 3.1415926535897932384626434
    mlt = (8.0 * (PI - 3.0)) / (3 * PI * (4.0 - PI))
    s2 = math.sqrt(2.0)
    v = s * inv_w_sum
    xl = math.log(1.0 - v * v)
    x[0] = s2 * math.sqrt(-2.0/(PI*mlt) - xl*0.5 + math.sqrt(math.pow(2.0/(PI*mlt)+xl*0.5, e2) - xl/mlt))
    for i in range(1, size):
        s += 0.317311 * w[i - 1] + 0.682689 * w[i]
        v = s * inv_w_sum
        xl = math.log(1.0 - v * v)
        x[i] = s2 * math.sqrt(-2.0/(PI*mlt) - xl*0.5 + math.sqrt(math.pow(2.0/(PI*mlt)+xl*0.5, e2) - xl/mlt))
    return x


@njit(cache=True, error_model="numpy")
def _fitsl(w, x, y, end):
    num = 0.0
    den = 0.0
    for i in range(end):
        wx = w[i] * x[i]
        num += wx * y[i]
        den += wx * x[i]
    return num / den


@njit(cache=True, error_model="numpy")
def _fitdl(counter, w, x, y, E, e2):
    # x is the getXVec output; y is the (sorted) diff array; returns sigma.
    n = x.shape[0]
    amount = _count_under_one(x)
    single_line_fit = _fitsl(w, x, y, amount)
    if n < 4:
        return single_line_fit
    m = _mfinder(1, amount - 1, amount - 1, max(int(n / 6.36), 1), w, x, y)
    x_at_m = x[m]
    a = 0.0; e = 0.0
    for i in range(m + 1):
        wx = w[i] * x[i]
        a += wx * x[i]
        e += wx * y[i]
    for i in range(m + 1, amount):
        a += (x_at_m * x_at_m) * w[i]
        e += (x_at_m * w[i]) * y[i]
    b = 0.0; d = 0.0; f = 0.0
    for i in range(m + 1, amount):
        diff = x[i] - x_at_m
        b += (x_at_m * w[i]) * diff
        d += (w[i] * diff) * diff
        f += (w[i] * y[i]) * diff
    c = b
    tau = (f - e * c / a) / (d - c * b / a)
    sigma = (e - tau * b) / a
    dle = 0.0
    for i in range(m + 1):
        fl = sigma * x[i] - y[i]
        dle += (w[i] * fl) * fl
    for i in range(m + 1, amount):
        diff = x[i] - x_at_m
        fr = sigma * x_at_m + tau * diff - y[i]
        dle += (w[i] * fr) * fr
    sle = 0.0
    for i in range(amount):
        ef = single_line_fit * x[i] - y[i]
        sle += (w[i] * ef) * ef
    numd = sle - dle
    if dle != 0.0:
        delta = numd / dle
    elif numd > 0.0:
        delta = math.inf
    elif numd < 0.0:
        delta = -math.inf
    else:
        delta = math.nan
    if sigma < 0:
        sigma = 1e-10
    if delta < _lowerfn(counter, x, w, E, e2):
        sigma = single_line_fit
    return sigma


@njit(cache=True, error_model="numpy")
def _mfinder(low, high, last_x_under_one, increment, w, x, y):
    DBL_MAX = math.inf
    stop = False
    best_m = -1
    m_low = -1
    m_high = -1
    end_r = last_x_under_one + 1
    while not stop:
        min_error = DBL_MAX
        m = low
        while m < high:
            x_at_m = x[m]
            a = 0.0; e = 0.0
            for i in range(m + 1):
                wx = w[i] * x[i]
                a += wx * x[i]
                e += wx * y[i]
            for i in range(m + 1, end_r):
                a += (x_at_m * x_at_m) * w[i]
                e += (x_at_m * w[i]) * y[i]
            b = 0.0; d = 0.0; f = 0.0
            for i in range(m + 1, end_r):
                diff = x[i] - x_at_m
                b += (x_at_m * w[i]) * diff
                d += (w[i] * diff) * diff
                f += (w[i] * y[i]) * diff
            c = b
            if a != 0.0 and (d - c * b) != 0.0:
                tau = (f - e * c / a) / (d - c * b / a)
                sigma = (e - tau * b) / a
                error = 0.0
                for i in range(m + 1):
                    fl = sigma * x[i] - y[i]
                    error += (w[i] * fl) * fl
                for i in range(m + 1, end_r):
                    diff = x[i] - x_at_m
                    fr = sigma * x_at_m + tau * diff - y[i]
                    error += (w[i] * fr) * fr
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
    if low == high:
        return low
    return best_m


@njit(cache=True, error_model="numpy")
def _sigma_lower(sigma_code, w, diff, delta, counter, E, e2):
    # 0=STANDARD_DEVIATION, 1=SIXTY_EIGHTH, 2=DOUBLE_LINE
    if sigma_code == 0:
        return _stdev_w(delta / 2.0, w, diff)
    if sigma_code == 1:
        return _get68th_w(w, diff)
    # DOUBLE_LINE
    x = _xvec(diff.shape[0], w, 1.0 / _psum(w), e2)
    x_below = _count_under_one(x)
    if x_below > 2:
        return _fitdl(counter, w, x, diff, E, e2)
    if x_below > 1:
        return _fitsl(w, x, diff, _count_under_one(x))
    return _get68th_w(w, diff)


@njit(cache=True, error_model="numpy")
def _iter_lower_w(w, y, flags, mu_code, sigma_code, delta, ECF, EFN, e2, e16):
    n = y.shape[0]
    mu = -1.0; sigma = -1.0; st_dev = -1.0; sba = -1.0; sbe = -1.0
    floor = _DBL_MIN
    while True:
        cnt = 0
        for i in range(n):
            if flags[i]:
                cnt += 1
        idxs = np.empty(cnt, np.int64)
        tw = np.empty(cnt, np.float64)
        ty = np.empty(cnt, np.float64)
        k = 0
        for i in range(n):
            if flags[i]:
                idxs[k] = i; tw[k] = w[i]; ty[k] = y[i]; k += 1
        true_count = cnt
        mu = _mu_w(mu_code, tw, ty)
        # diff, argmax (first max)
        max_local = 0
        max_val = abs(ty[0] - mu)
        for i in range(1, cnt):
            dv = abs(ty[i] - mu)
            if dv > max_val:
                max_val = dv
                max_local = i
        max_index = idxs[max_local]
        # tie-split: counts
        nb = 0; na = 0
        for i in range(cnt):
            ic = _isclose(ty[i], mu)
            if ty[i] < mu or ic:
                nb += 1
            if ty[i] > mu or ic:
                na += 1
        diff_b = np.empty(nb, np.float64); wb = np.empty(nb, np.float64)
        diff_a = np.empty(na, np.float64); wa = np.empty(na, np.float64)
        ib = 0; ia = 0
        for i in range(cnt):
            ic = _isclose(ty[i], mu)
            dv = abs(ty[i] - mu)
            wv = 0.5 * tw[i] if ic else tw[i]
            if ty[i] < mu or ic:
                diff_b[ib] = dv; wb[ib] = wv; ib += 1
            if ty[i] > mu or ic:
                diff_a[ia] = dv; wa[ia] = wv; ia += 1
        nonzero_below = nb > 0
        nonzero_above = na > 0
        if nonzero_below:
            ob = np.argsort(diff_b, kind="mergesort")
            diff_b = diff_b[ob]; wb = wb[ob]
        if nonzero_above:
            oa = np.argsort(diff_a, kind="mergesort")
            diff_a = diff_a[oa]; wa = wa[oa]
        n_correction = _lowerdlcf(true_count, tw, ECF)
        if nonzero_above and nonzero_below:
            sba = _sigma_lower(sigma_code, wa, diff_a, delta, true_count, EFN, e2)
            sbe = _sigma_lower(sigma_code, wb, diff_b, delta, true_count, EFN, e2)
            st_dev = sba if sba < sbe else sbe
            sigma = st_dev * n_correction
        elif nonzero_above:
            st_dev = _sigma_lower(sigma_code, wa, diff_a, delta, true_count, EFN, e2)
            sba = st_dev
            sigma = st_dev * n_correction
        elif nonzero_below:
            st_dev = _sigma_lower(sigma_code, wb, diff_b, delta, true_count, EFN, e2)
            sbe = st_dev
            sigma = st_dev * n_correction
        ratio = max_val / (sigma if sigma > 0 else floor)
        if _distinct(flags, y) and true_count * _erfc(ratio, e16) < 0.5:
            flags[max_index] = False
        else:
            break
    return mu, sigma, st_dev, sbe, sba


@njit(cache=True, error_model="numpy")
def _bulk_sigma_lower(w, diff, counter, E, e2):
    x = _xvec(diff.shape[0], w, 1.0 / _psum(w), e2)
    xb = _count_under_one(x)
    if xb > 2:
        a = _fitdl(counter, w, x, diff, E, e2)
        b = _fitsl(w, x, diff, xb)
        return a if a > b else b
    if xb > 1:
        return _fitsl(w, x, diff, xb)
    return _get68th_w(w, diff)


@njit(cache=True, error_model="numpy")
def _bulk_reject(flags, indices_sorted, ratios_sorted, y, e16):
    no_rej = True
    size = ratios_sorted.shape[0]
    i = size - 1
    while i >= 0 and _distinct(flags, y) and size * _erfc(ratios_sorted[i], e16) < 0.5:
        flags[indices_sorted[i]] = False
        no_rej = False
        i -= 1
    return no_rej


@njit(cache=True, error_model="numpy")
def _bulk_lower_w(w, y, flags, mu_code, delta, ECF, EFN, e2, e16):
    n = y.shape[0]
    mu = -1.0
    sigma = -1.0
    while True:
        cnt = 0
        for i in range(n):
            if flags[i]:
                cnt += 1
        idxs = np.empty(cnt, np.int64)
        tw = np.empty(cnt, np.float64)
        ty = np.empty(cnt, np.float64)
        k = 0
        for i in range(n):
            if flags[i]:
                idxs[k] = i; tw[k] = w[i]; ty[k] = y[i]; k += 1
        true_count = cnt
        mu = _mu_w(mu_code, tw, ty)
        # diff_hold (all points, in trueY order)
        dh = np.empty(cnt, np.float64)
        for i in range(cnt):
            dh[i] = abs(ty[i] - mu)
        # below/above counts (C++ list semantics: isEqual->both; elif >mu->above; else->below)
        nb = 0; na = 0
        for i in range(cnt):
            if _isclose(ty[i], mu):
                nb += 1; na += 1
            elif ty[i] > mu:
                na += 1
            else:
                nb += 1
        diff_b = np.empty(nb, np.float64); wb = np.empty(nb, np.float64)
        diff_a = np.empty(na, np.float64); wa = np.empty(na, np.float64)
        ib = 0; ia = 0
        for i in range(cnt):
            hold = abs(ty[i] - mu)
            if _isclose(ty[i], mu):
                diff_b[ib] = hold; wb[ib] = 0.5 * tw[i]; ib += 1
                diff_a[ia] = hold; wa[ia] = 0.5 * tw[i]; ia += 1
            elif ty[i] > mu:
                diff_a[ia] = hold; wa[ia] = tw[i]; ia += 1
            else:
                diff_b[ib] = hold; wb[ib] = tw[i]; ib += 1
        # sort diff_hold stable -> indices_sorted
        sh = np.argsort(dh, kind="mergesort")
        idx_sorted = idxs[sh]
        dh_sorted = dh[sh]
        nonzero_below = nb > 0
        nonzero_above = na > 0
        if nonzero_below:
            ob = np.argsort(diff_b, kind="mergesort"); diff_b = diff_b[ob]; wb = wb[ob]
        if nonzero_above:
            oa = np.argsort(diff_a, kind="mergesort"); diff_a = diff_a[oa]; wa = wa[oa]
        ncf = _lowerdlcf(true_count, tw, ECF)
        if nonzero_above and nonzero_below:
            sb = _bulk_sigma_lower(wb, diff_b, true_count, EFN, e2)
            sa = _bulk_sigma_lower(wa, diff_a, true_count, EFN, e2)
            sigma = (sb if sb < sa else sa) * ncf
        elif nonzero_above:
            sigma = _bulk_sigma_lower(wa, diff_a, true_count, EFN, e2) * ncf
        elif nonzero_below:
            sigma = _bulk_sigma_lower(wb, diff_b, true_count, EFN, e2) * ncf
        ratios = np.empty(cnt, np.float64)
        for i in range(cnt):
            ratios[i] = dh_sorted[i] / sigma   # IEEE: /0 -> inf/nan, handled by erfc
        if _bulk_reject(flags, idx_sorted, ratios, y, e16):
            break
    return mu, sigma



# ============================================================================
# Round 5: UNWEIGHTED lower-sigma drivers (iterative + bulk) for LS_MODE_DL.
# Reuses the R4 leaves with w=ones; differs by LAST-tie-only split, the
# unweighted getLowerDLCF (table + n**-0.65, no CF ratio), and a PINNED
# trueCount in the bulk driver (C++ quirk). Bit-exact vs the pure-Python
# iterativeLowerSigmaRCR / bulkLowerSigmaRCR (0/1200 + 0/400). Covers
# cornifer's robust_mean_scatter(deltas) (sweep_to_sweep, unweighted).
# ============================================================================
@njit(cache=True, error_model="numpy")
def _r5_lowerdlcf(n, E):
    if n < 101:
        return 1.0 / _LSDL[n]
    return 1.0 / (1.0 - 3.3245 * (n ** E[7]))   # n**-0.65; E param -> real libm pow


@njit(cache=True, error_model="numpy")
def _r5_iter_lower(y, flags, mu_code, sigma_code, delta, ECF, EFN, e2, e16):
    n = y.shape[0]
    mu = -1.0; sigma = -1.0; st_dev = -1.0; sba = -1.0; sbe = -1.0
    while True:
        cnt = 0
        for i in range(n):
            if flags[i]:
                cnt += 1
        idxs = np.empty(cnt, np.int64); ty = np.empty(cnt, np.float64)
        k = 0
        for i in range(n):
            if flags[i]:
                idxs[k] = i; ty[k] = y[i]; k += 1
        true_count = cnt
        ones = np.ones(cnt, np.float64)
        mu = _mu_w(mu_code, ones, ty)
        max_local = 0; max_val = abs(ty[0] - mu)
        for i in range(1, cnt):
            dv = abs(ty[i] - mu)
            if dv > max_val:
                max_val = dv; max_local = i
        max_index = idxs[max_local]
        last_eq = -1
        for i in range(cnt):
            if _isclose(ty[i], mu):
                last_eq = i
        nb = 0; na = 0
        for i in range(cnt):
            ic = _isclose(ty[i], mu)
            if ty[i] < mu or ic:
                nb += 1
            if ty[i] > mu or ic:
                na += 1
        diff_b = np.empty(nb, np.float64); wb = np.ones(nb, np.float64)
        diff_a = np.empty(na, np.float64); wa = np.ones(na, np.float64)
        ib = 0; ia = 0; below_split = -1; above_split = -1
        for i in range(cnt):
            ic = _isclose(ty[i], mu); dv = abs(ty[i] - mu)
            if ty[i] < mu or ic:
                diff_b[ib] = dv
                if i == last_eq:
                    below_split = ib
                ib += 1
            if ty[i] > mu or ic:
                diff_a[ia] = dv
                if i == last_eq:
                    above_split = ia
                ia += 1
        if last_eq != -1:
            if below_split >= 0:
                wb[below_split] = 0.5
            if above_split >= 0:
                wa[above_split] = 0.5
        nonzero_below = nb > 0; nonzero_above = na > 0
        if nonzero_below:
            ob = np.argsort(diff_b, kind="mergesort"); diff_b = diff_b[ob]; wb = wb[ob]
        if nonzero_above:
            oa = np.argsort(diff_a, kind="mergesort"); diff_a = diff_a[oa]; wa = wa[oa]
        ncf = _r5_lowerdlcf(true_count, ECF)
        if nonzero_above and nonzero_below:
            sba = _sigma_lower(sigma_code, wa, diff_a, delta, true_count, EFN, e2)
            sbe = _sigma_lower(sigma_code, wb, diff_b, delta, true_count, EFN, e2)
            st_dev = sba if sba < sbe else sbe
            sigma = st_dev * ncf
        elif nonzero_above:
            st_dev = _sigma_lower(sigma_code, wa, diff_a, delta, true_count, EFN, e2)
            sba = st_dev; sigma = st_dev * ncf
        elif nonzero_below:
            st_dev = _sigma_lower(sigma_code, wb, diff_b, delta, true_count, EFN, e2)
            sbe = st_dev; sigma = st_dev * ncf
        ratio = max_val / (sigma if sigma > 0.0 else _DBL_MIN)
        if _distinct(flags, y) and true_count * _erfc(ratio, e16) < 0.5:
            flags[max_index] = False
        else:
            break
    return mu, sigma, st_dev, sbe, sba


@njit(cache=True, error_model="numpy")
def _r5_bulk_lower(y, flags, mu_code, ECF, EFN, e2, e16):
    n = y.shape[0]
    counter = n   # PINNED to original size (C++ quirk)
    mu = -1.0; sigma = -1.0
    while True:
        cnt = 0
        for i in range(n):
            if flags[i]:
                cnt += 1
        idxs = np.empty(cnt, np.int64); ty = np.empty(cnt, np.float64)
        k = 0
        for i in range(n):
            if flags[i]:
                idxs[k] = i; ty[k] = y[i]; k += 1
        true_count = cnt
        ones = np.ones(cnt, np.float64)
        mu = _mu_w(mu_code, ones, ty)
        dh = np.empty(cnt, np.float64)
        for i in range(cnt):
            dh[i] = abs(ty[i] - mu)
        last_eq = -1
        for i in range(cnt):
            if _isclose(ty[i], mu):
                last_eq = i
        nb = 0; na = 0
        for i in range(cnt):
            ic = _isclose(ty[i], mu)
            if ty[i] < mu or ic:
                nb += 1
            if ty[i] > mu or ic:
                na += 1
        diff_b = np.empty(nb, np.float64); wb = np.ones(nb, np.float64)
        diff_a = np.empty(na, np.float64); wa = np.ones(na, np.float64)
        ib = 0; ia = 0; below_split = -1; above_split = -1
        for i in range(cnt):
            ic = _isclose(ty[i], mu); dv = abs(ty[i] - mu)
            if ty[i] < mu or ic:
                diff_b[ib] = dv
                if i == last_eq:
                    below_split = ib
                ib += 1
            if ty[i] > mu or ic:
                diff_a[ia] = dv
                if i == last_eq:
                    above_split = ia
                ia += 1
        if last_eq != -1:
            if below_split >= 0:
                wb[below_split] = 0.5
            if above_split >= 0:
                wa[above_split] = 0.5
        sh = np.argsort(dh, kind="mergesort")
        idx_sorted = idxs[sh]; dh_sorted = dh[sh]
        nonzero_below = nb > 0; nonzero_above = na > 0
        if nonzero_below:
            ob = np.argsort(diff_b, kind="mergesort"); diff_b = diff_b[ob]; wb = wb[ob]
        if nonzero_above:
            oa = np.argsort(diff_a, kind="mergesort"); diff_a = diff_a[oa]; wa = wa[oa]
        ncf = _r5_lowerdlcf(counter, ECF)
        if nonzero_above and nonzero_below:
            sb = _bulk_sigma_lower(wb, diff_b, counter, EFN, e2)
            sa = _bulk_sigma_lower(wa, diff_a, counter, EFN, e2)
            sigma = (sb if sb < sa else sa) * ncf
        elif nonzero_above:
            sigma = _bulk_sigma_lower(wa, diff_a, counter, EFN, e2) * ncf
        elif nonzero_below:
            sigma = _bulk_sigma_lower(wb, diff_b, counter, EFN, e2) * ncf
        ratios = np.empty(cnt, np.float64)
        for i in range(cnt):
            ratios[i] = dh_sorted[i] / sigma
        if _bulk_reject(flags, idx_sorted, ratios, y, e16):
            break
    return mu, sigma


_installed = False


def install() -> None:
    """Swap the JIT kernels into the stats module and warm up the JIT."""
    global _installed
    from rcrpy import stats
    stats.getXVec_w = getXVec_w
    stats.mFinder_w = mFinder_w
    stats.get68th_w = get68th_w
    stats.halfSampleMode = halfSampleMode
    stats.halfSampleMode_w = halfSampleMode_w
    stats.getFNRatio = getFNRatio
    stats._cpp_seqsum = _cpp_seqsum
    stats.fitDL_w = fitDL_w
    # trigger compilation now (warm the JIT off the hot path)
    getXVec_w(4, np.ones(4, dtype=np.float64))
    _w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mFinder_w(1, 3, 4, 3, _w, _x, _w)
    get68th_w(_w, _x)
    halfSampleMode(np.array([1.0, 2.0, 2.0, 3.0, 9.0]))
    halfSampleMode_w(_w, np.array([1.0, 2.0, 2.0, 3.0, 9.0]))
    getFNRatio(np.array([0.2, 0.5, 0.9, 1.3, 2.0]), _w)
    _cpp_seqsum(_x)
    _fitdl_core(_w, np.array([0.2, 0.5, 0.9, 1.3, 2.0]), _w)
    # ---- Round 4: swap the full weighted lower-sigma drivers (LS_MODE_DL) ----
    import rcrpy.rejection as _rej
    from rcrpy.rejection import MuTech as _MuT, SigmaTech as _SgT
    _MU = {_MuT.MEAN: 0, _MuT.MEDIAN: 1, _MuT.MODE: 2}
    _SG = {_SgT.STANDARD_DEVIATION: 0, _SgT.SIXTY_EIGHTH_PERCENTILE: 1, _SgT.DOUBLE_LINE: 2}
    _dlcf = _rej.getLowerDLCF_w
    _ref_iter = _rej.iterativeLowerSigmaRCR_w
    _ref_bulk = _rej.bulkLowerSigmaRCR_w

    def _iter_lower_w_fast(w, y, flags, mu_tech, sigma_tech, delta, n_correct_fn,
                           non_parametric_model=None, parametric_model=None):
        if (non_parametric_model is None and parametric_model is None
                and n_correct_fn is _dlcf and mu_tech in _MU and sigma_tech in _SG):
            mu, sigma, sd, sbe, sba = _iter_lower_w(
                np.ascontiguousarray(w, np.float64), np.ascontiguousarray(y, np.float64),
                flags, _MU[mu_tech], _SG[sigma_tech], float(delta), _ECF, _EFN, 2.0, 16.0)
            return {"mu": mu, "sigma": sigma, "st_dev": sd,
                    "st_dev_below": sbe, "st_dev_above": sba}
        return _ref_iter(w, y, flags, mu_tech, sigma_tech, delta, n_correct_fn,
                         non_parametric_model, parametric_model)

    def _bulk_lower_w_fast(w, y, flags, mu_tech, n_correct_fn,
                           non_parametric_model=None, parametric_model=None):
        if (non_parametric_model is None and parametric_model is None
                and n_correct_fn is _dlcf and mu_tech in _MU):
            mu, sigma = _bulk_lower_w(
                np.ascontiguousarray(w, np.float64), np.ascontiguousarray(y, np.float64),
                flags, _MU[mu_tech], 1.0, _ECF, _EFN, 2.0, 16.0)
            return {"mu": mu, "sigma": sigma}
        return _ref_bulk(w, y, flags, mu_tech, n_correct_fn,
                         non_parametric_model, parametric_model)

    _rej.iterativeLowerSigmaRCR_w = _iter_lower_w_fast
    _rej.bulkLowerSigmaRCR_w = _bulk_lower_w_fast
    # warm the drivers
    _iter_lower_w(np.ones(20), np.arange(20.0), np.ones(20, np.bool_), 2, 2, 1.0, _ECF, _EFN, 2.0, 16.0)
    _bulk_lower_w(np.ones(20), np.arange(20.0), np.ones(20, np.bool_), 2, 1.0, _ECF, _EFN, 2.0, 16.0)
    # ---- Round 5: swap the UNWEIGHTED lower-sigma drivers (LS_MODE_DL) ----
    _ref_iter_unw = _rej.iterativeLowerSigmaRCR
    _ref_bulk_unw = _rej.bulkLowerSigmaRCR
    _dlcf_unw = _rej.getLowerDLCF

    def _iter_lower_unw_fast(y, flags, mu_tech, sigma_tech, delta, n_correct_fn,
                             non_parametric_model=None, parametric_model=None):
        if (non_parametric_model is None and parametric_model is None
                and n_correct_fn is _dlcf_unw and mu_tech in _MU and sigma_tech in _SG):
            mu, sigma, sd, sbe, sba = _r5_iter_lower(
                np.ascontiguousarray(y, np.float64), flags,
                _MU[mu_tech], _SG[sigma_tech], float(delta), _ECF, _EFN, 2.0, 16.0)
            return {"mu": mu, "sigma": sigma, "st_dev": sd,
                    "st_dev_above": sba, "st_dev_below": sbe}
        return _ref_iter_unw(y, flags, mu_tech, sigma_tech, delta, n_correct_fn,
                             non_parametric_model, parametric_model)

    def _bulk_lower_unw_fast(y, flags, mu_tech, n_correct_fn,
                             non_parametric_model=None, parametric_model=None):
        if (non_parametric_model is None and parametric_model is None
                and n_correct_fn is _dlcf_unw and mu_tech in _MU):
            mu, sigma = _r5_bulk_lower(
                np.ascontiguousarray(y, np.float64), flags, _MU[mu_tech], _ECF, _EFN, 2.0, 16.0)
            return {"mu": mu, "sigma": sigma}
        return _ref_bulk_unw(y, flags, mu_tech, n_correct_fn,
                             non_parametric_model, parametric_model)

    _rej.iterativeLowerSigmaRCR = _iter_lower_unw_fast
    _rej.bulkLowerSigmaRCR = _bulk_lower_unw_fast
    _r5_iter_lower(np.arange(20.0), np.ones(20, np.bool_), 2, 2, 1.0, _ECF, _EFN, 2.0, 16.0)
    _r5_bulk_lower(np.arange(20.0), np.ones(20, np.bool_), 2, _ECF, _EFN, 2.0, 16.0)
    _installed = True
