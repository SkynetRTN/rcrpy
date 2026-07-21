"""Iterative Chauvenet rejection loops + the 3-pass orchestration.

Phase 1 scope: unweighted lower-sigma path only (matches LS_MODE_68 +
LS_MODE_DL with sigmaTech overridden each pass). Single-sigma, each-sigma,
bulk, and weighted variants are deferred.

Ported from cpp/src/RCR.cpp:
  - performRejection (line 5017)
  - iterativeLowerSigmaRCR (line 6408)
  - nCorrect / getLower68CF (line 5244)
  - reject (line 5744)
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np

from rcrpy import stats, tables


def _pow10pow10(z: float) -> float:
    """``10**(10**z)`` with C++ ``pow`` IEEE overflow semantics (overflow ->
    +inf) instead of Python's OverflowError.

    The weighted CF/FN models below (getLowerFN_w, getLower68CF_w, ...) all use
    this double-exponential. For large ``z`` the C++ ``pow(10, pow(10, z))``
    silently overflows to +inf, and the ``y1 * (...)`` / ``y1 / (...)`` forms
    then go to +inf / 0 — which the downstream ``delta_chi_squared < FN``
    selection in stats.fitDL_w handles. Python instead raises OverflowError on
    the scalar ``10 ** ...``, so reproduce the IEEE result. Same C++-IEEE-vs-
    Python-raise guard already used in stats.erfcCustom and _reject_ratio.
    """
    try:
        inner = 10.0 ** z
    except OverflowError:
        return math.inf
    try:
        return 10.0 ** inner
    except OverflowError:
        return math.inf


# C++ enum equivalents used internally by the loop. The public-facing
# RejectionTech (in api.py) drives `alignTechniques`.

class MuTech(Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MODE = "MODE"


class SigmaTech(Enum):
    STANDARD_DEVIATION = "STANDARD_DEVIATION"
    SIXTY_EIGHTH_PERCENTILE = "SIXTY_EIGHTH_PERCENTILE"
    DOUBLE_LINE = "DOUBLE_LINE"


class SigmaChoice(Enum):
    SINGLE = "SINGLE"
    LOWER = "LOWER"
    EACH = "EACH"


# Floor used in each-sigma rejection loops when a side's sigma collapses
# to 0 or negative. Mirrors the C++'s implicit "x/0 = inf → erfc(inf) = 0
# → reject" behavior: with sigma replaced by ~2.2e-308, the rejection
# ratio is astronomically large and the worst-residual point on that side
# gets rejected, identical to what the C++ does.
_SIGMA_FLOOR = float(np.finfo(np.float64).tiny)


def _reject_ratio(max_val: float, sigma: float) -> float:
    """Compute the rejection ratio ``max_val / sigma`` with the degenerate-
    sigma guard the C++ gets for free from IEEE float division.

    When the kept set has no spread left to reject on — perfectly linear or
    near-constant data, so sigma collapses to 0 — the C++ evaluates
    ``max / 0`` as +inf (or NaN when max is 0 too) and the short-circuit in
    reject() (erfcCustom(inf) == 0, or distinctValuesCheck == false) stops
    the loop with nothing rejected. Python raises ZeroDivisionError on the
    scalar divide instead, so floor sigma to the smallest positive double —
    the same _SIGMA_FLOOR idiom the each-sigma loops already use — which
    overflows the ratio to +inf and yields the identical reject decision.

    Net effect on a line fit: with no outliers to reject, RCR keeps every
    point and the parametric result reduces to the plain (weighted) least-
    squares line, instead of crashing.
    """
    return max_val / (sigma if sigma > 0 else _SIGMA_FLOOR)


# ---------------------------------------------------------------------------
# Forward declarations (the FN helpers reference each other via the dispatch
# inside fitDL; the actual bodies are below).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Correction factor — n-dependent multiplier that turns stDev into sigma.
# ---------------------------------------------------------------------------

def getLower68CF(n: int) -> float:
    """Port of cpp/src/RCR.cpp:5244 — unweighted CF for LS_MODE_68."""
    if n < 101:
        return 1.0 / float(tables.LS68UnityCF[n])
    return 1.0 / (1.0 - 2.3525 * n ** -0.627)


def getLower68CF_w(n: int, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5186 — weighted CF for LS_MODE_68. Piecewise
    polynomial in (log10 n, log10 getCFRatio(w))."""
    import math as _math
    x = stats.getCFRatio(w)
    if n < 101:
        y1 = 1.0 / float(tables.LS68UnityCF[n])
    else:
        y1 = 1.0 / (1.0 - 2.3525 * n ** -0.627)
    if x == 0:
        return y1
    logx = _math.log10(x)
    logn = _math.log10(n)
    sign = -1 if n % 2 else 1  # pow(-1, n)
    if n == 2:
        b1, a1 = 1.08149771508934, -0.398375456223868
    elif n == 3:
        b1, a1 = 1.06994341034958, -1.14618991625901
    elif n == 4:
        return y1
    elif n == 5:
        b1, a1 = 0.45972894692595, -1.11957357644441
    elif 5 < n < 101:
        b1 = (-1.4528 * logn ** 3 + 5.3519 * logn ** 2 - 5.33 * logn + 2.2902
              + sign * 0.1879 * logn ** 0.9521)
        a1 = -1.1937 * logn ** 4 + 6.5268 * logn ** 3 - 13.308 * logn ** 2 + 11.432 * logn - 4.4769
    else:
        b1 = 1.4154 + sign * 0.363528
        a1 = -0.5408 * logn - 0.6482
    return y1 * _pow10pow10(a1 + b1 * logx)


def getLowerDLCF(n: int) -> float:
    """Port of cpp/src/RCR.cpp:5320 — unweighted CF for LS_MODE_DL."""
    if n < 101:
        return 1.0 / float(tables.LSDLUnityCF[n])
    return 1.0 / (1.0 - 3.3245 * n ** -0.65)


def getLowerDLCF_w(n: int, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5255 — weighted CF for LS_MODE_DL."""
    import math as _math
    x = stats.getCFRatio(w)
    if n < 101:
        y1 = 1.0 / float(tables.LSDLUnityCF[n])
    else:
        y1 = 1.0 / (1.0 - 3.3245 * n ** -0.65)
    if x == 0:
        return y1
    logx = _math.log10(x)
    logn = _math.log10(n)
    sign = -1 if n % 2 else 1
    if n == 2:
        b1, a1 = 1.08149771508934, -0.398375456223868
    elif n == 3:
        b1, a1 = 1.51433669748424, -1.10939332689999
    elif n == 4:
        return y1
    elif n == 5:
        b1, a1 = 0.339404852185332, -1.1445790996528
    elif 5 < n < 21:
        b1 = (43.179 * logn ** 6 - 331.85 * logn ** 5 + 968.25 * logn ** 4
              - 1399.1 * logn ** 3 + 1070.7 * logn ** 2 - 415.81 * logn + 65.002
              + sign * 0.1365 * logn ** 2.4716)
        a1 = -0.2683 * logn ** 4 + 1.9174 * logn ** 3 - 5.062 * logn ** 2 + 5.452 * logn - 2.9999
    elif 20 < n < 101:
        b1 = 1.5144 * logn - 0.0448 + sign * 0.1365 * logn ** 2.4716
        a1 = -0.2683 * logn ** 4 + 1.9174 * logn ** 3 - 5.062 * logn ** 2 + 5.452 * logn - 2.9999
    else:
        b1 = 2.988077 + sign * 0.753032
        a1 = -0.4282 * logn - 0.4412
    return y1 * _pow10pow10(a1 + b1 * logx)


def getSingleDLCF(n: int) -> float:
    """Port of cpp/src/RCR.cpp:5386 — unweighted CF for SS_MEDIAN_DL."""
    if n < 101:
        return 1.0 / float(tables.SSDLUnityCF[n])
    return 1.0 / (1.0 - 3.578 * n ** -0.942)


def getSingleDLCF_w(n: int, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5331 — weighted CF for SS_MEDIAN_DL."""
    import math as _math
    x = stats.getCFRatio(w)
    if n < 101:
        y1 = 1.0 / float(tables.SSDLUnityCF[n])
    else:
        y1 = 1.0 / (1.0 - 3.578 * n ** -0.942)
    if x == 0:
        return y1
    logx = _math.log10(x)
    logn = _math.log10(n)
    if n == 2:
        b1, a1 = 0.273907084639124, -3.15279135630884
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 3:
        b1, a1 = 0.448654915529039, -1.19134294551807
        return y1 / _pow10pow10(a1 + b1 * logx)
    if n == 4:
        b1, a1 = 3.38253309393705, -1.05087405984868
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 5:
        b1, a1 = 0.118507989164207, -1.41453721585464
        return y1 / _pow10pow10(a1 + b1 * logx)
    b1 = 0.1196 * logn + 4.5073
    a1 = -0.7914 * logn + 0.0243
    return y1 * _pow10pow10(a1 + b1 * logx)


def getSingleFN(n: int, x: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5511 — unweighted FN model for sigmaChoice=SINGLE."""
    if n < 1001:
        return float(tables.SSUnity[n])
    return 39.2519 * n ** -0.7969 + 1.8688


def getSingleFN_w(n: int, x: np.ndarray, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5470 — weighted FN model for sigmaChoice=SINGLE."""
    import math as _math
    ratio = stats.getFNRatio(x, w)
    if n < 1001:
        y1 = float(tables.SSUnity[n])
    else:
        y1 = getSingleFN(n, x)
    if ratio == 0:
        return y1
    logx = _math.log10(ratio)
    logn = _math.log10(n)
    if 3 < n < 8:
        a1 = float(tables.SSConstants[0, n])
        b1 = float(tables.SSConstants[1, n])
        return y1 * _pow10pow10(a1 + b1 * logx)
    if 7 < n < 1001:
        b1 = (-0.3556 * logn ** 6 + 3.7036 * logn ** 5 - 14.932 * logn ** 4
              + 29.176 * logn ** 3 - 28.81 * logn ** 2 + 14.397 * logn - 2.6451)
        sign = -1 if n % 2 else 1
        a1 = (0.2313 * logn ** 6 - 3.02 * logn ** 5 + 15.997 * logn ** 4
              - 43.713 * logn ** 3 + 64.629 * logn ** 2 - 49.976 * logn + 15.484
              + sign * 0.1513 * n ** -0.471)
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n > 1000:
        return y1
    return -999999.0


def getEachDLCF(n: int) -> float:
    """Port of cpp/src/RCR.cpp:5459 — unweighted CF for ES_MODE_DL."""
    if n < 101:
        return 1.0 / float(tables.ESDLUnityCF[n])
    return 1.0 / (1.0 - 3.1666 * n ** -0.833)


def getEachDLCF_w(n: int, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5397 — weighted CF for ES_MODE_DL."""
    import math as _math
    x = stats.getCFRatio(w)
    if n < 101:
        y1 = 1.0 / float(tables.ESDLUnityCF[n])
    else:
        y1 = 1.0 / (1.0 - 3.1666 * n ** -0.833)
    if x == 0:
        return y1
    logx = _math.log10(x)
    logn = _math.log10(n)
    if n == 2:
        b1, a1 = 0.733632602759432, -2.59506757852134
    elif n == 3:
        b1, a1 = 0.816017988131836, -0.854637214866955
    elif n == 4:
        b1, a1 = 1.16048439814909, -0.954253810365265
    elif 4 < n < 20:
        b1 = 4.0458 * logn ** 2 - 6.4354 * logn + 2.7667
        a1 = -1.3993 * logn ** 3 + 6.5746 * logn ** 2 - 9.8844 * logn + 2.8572
    elif 19 < n < 101:
        b1 = 1.7394 * logn - 1.0435
        a1 = -1.3993 * logn ** 3 + 6.5746 * logn ** 2 - 9.8844 * logn + 2.8572
    else:
        b1 = 1.4123 * logn - 0.3893
        a1 = -0.5989 * logn - 0.6097
    return y1 * _pow10pow10(a1 + b1 * logx)


def getEachFN(n: int, x: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5660 — unweighted FN model for sigmaChoice=EACH."""
    if n < 1001:
        return float(tables.ESUnity[n])
    return 1.2591 ** (n ** 0.2052)


def getEachFN_w(n: int, x: np.ndarray, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5600 — weighted FN model for sigmaChoice=EACH."""
    import math as _math
    ratio = stats.getFNRatio(x, w)
    if n < 1001:
        y1 = float(tables.ESUnity[n])
    else:
        y1 = getEachFN(n, x)
    if ratio == 0:
        return y1
    logx = _math.log10(ratio)
    logn = _math.log10(n)
    if n == 5:
        b1, a1 = 2.13417275654528, -0.466431459550531
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 6:
        b1, a1 = 1.0196951775215, -0.312373723591738
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 7:
        b1, a1 = 0.579724747519776, -0.750190463040497
        return y1 * _pow10pow10(a1 + b1 * logx)
    if 7 < n < 1001:
        b1 = (5.8718 * logn ** 4 - 47.049 * logn ** 3 + 131.12 * logn ** 2
              - 150.24 * logn + 61.727)
        a1 = (3.1767 * logn ** 6 - 34.561 * logn ** 5 + 152.16 * logn ** 4
              - 347.96 * logn ** 3 + 435.59 * logn ** 2 - 282.57 * logn + 73.696)
        b2 = -2.7584 * logn ** 2 + 17.078 * logn - 24.602
        a2 = -1.8953 * logn ** 2 + 11.745 * logn - 19.36
        if n < 191 or (n < 306 and (a1 + b1 * logx > a2 + b2 * logx)):
            return y1 * _pow10pow10(a1 + b1 * logx)
        return y1 / _pow10pow10(a2 + b2 * logx)
    b1, a1 = 1.8064, -1.1827
    return y1 / _pow10pow10(a1 + b1 * logx)


def getLowerFN(n: int, x: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5589 — unweighted FN model for sigmaChoice=LOWER."""
    if n < 1001:
        return float(tables.LSUnity[n])
    return 1.3399 ** (n ** 0.1765)


def getLowerFN_w(n: int, x: np.ndarray, w: np.ndarray) -> float:
    """Port of cpp/src/RCR.cpp:5522 — weighted FN model for sigmaChoice=LOWER."""
    import math as _math
    ratio = stats.getFNRatio(x, w)
    if n < 1001:
        y1 = float(tables.LSUnity[n])
    else:
        y1 = getLowerFN(n, x)
    if ratio == 0:
        return y1
    logx = _math.log10(ratio)
    logn = _math.log10(n)
    if n == 5:
        y1 = 36.8534
        b1, a1 = -0.300348560626506, -0.0828207791627729
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 6:
        b1, a1 = -0.244262599183892, -0.267502535686502
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 7:
        b1, a1 = -0.409677351330214, -0.558845927943435
        return y1 * _pow10pow10(a1 + b1 * logx)
    if n == 8:
        b1, a1 = -0.488354948027081, -0.889342857411619
        return y1 * _pow10pow10(a1 + b1 * logx)
    if 8 < n < 1001:
        b1 = 0.1462 * logn ** 3.0 - 4.2139 * logn ** 2.0 + 14.366 * logn - 10.658
        a1 = (-0.541 * logn ** 5.0 + 4.6943 * logn ** 4.0 - 15.407 * logn ** 3.0
              + 21.875 * logn ** 2.0 - 11.211 * logn - 0.3798)
        b2 = 26.945 * logn ** 3.0 - 221.42 * logn ** 2.0 + 606.91 * logn - 553.89
        a2 = 18.149 * logn ** 3.0 - 149.27 * logn ** 2.0 + 410.15 * logn - 378.47
        if n <= 264 or (n < 568 and (a1 + b1 * logx > a2 + b2 * logx)):
            return y1 * _pow10pow10(a1 + b1 * logx)
        return y1 / _pow10pow10(a2 + b2 * logx)
    # n >= 1001
    b1 = 0.0424 * logn + 1.4479
    a1 = 0.3861 * logn - 2.5852
    return y1 / _pow10pow10(a1 + b1 * logx)


# ---------------------------------------------------------------------------
# Mu and sigma dispatch — mirrors handleMuTechSelect / handleSigmaTechSelect.
# ---------------------------------------------------------------------------

def _mu(mu_tech: MuTech, trueY: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (mu, possibly-sorted trueY). MEDIAN and MODE both sort in-place
    in the C++, which the caller's diff-loop then ignores (it uses original
    trueY order). We return both for clarity.
    """
    if mu_tech is MuTech.MEAN:
        return stats.getMean(trueY), trueY
    if mu_tech is MuTech.MEDIAN:
        s = np.sort(trueY)
        return stats.getMedian(s), s
    # MODE
    s = np.sort(trueY)
    return stats.halfSampleMode(s), s


class _RCRDegenerate(Exception):
    """A rejection iteration found NO candidate points — an empty kept/residual
    set. Reached e.g. on a NonParametric muFunc where every point fails the
    model test (adjacent/edge/equal-angle neighbour lines), or on all-rejected
    input. Without points there is no mu/sigma to compute (getMedian would index
    y[0] on a size-0 array). This is raised internally and caught at the
    performRejection_LS / performBulkRejection_LS entry points, which return a
    nan result — the documented "rejection cannot proceed" outcome. It is NOT
    part of the public contract: callers see nan, never this exception. (No
    non-degenerate input produces an empty set, so parity is unaffected.)"""


def _degenerate_state(sigma_choice: "SigmaChoice") -> dict:
    """The per-pass mu/sigma state for a degenerate (no-candidate) rejection:
    all nan, with the key set matching the chosen sigma family."""
    if sigma_choice is SigmaChoice.EACH:
        return {"mu": np.nan, "sigma_below": np.nan, "sigma_above": np.nan,
                "st_dev_above": np.nan, "st_dev_below": np.nan}
    return {"mu": np.nan, "sigma": np.nan, "st_dev": np.nan,
            "st_dev_above": np.nan, "st_dev_below": np.nan}


def _select_candidates(
    flags: np.ndarray, y: np.ndarray, mu_tech: "MuTech",
    non_parametric_model=None, parametric_model=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Unweighted (indices, trueY, mu) selection for the iterative/bulk
    loop bodies. Handles all three mu_types (VALUE/NONPARAMETRIC/PARAMETRIC)
    in one place to keep the loops short."""
    if parametric_model is not None:
        parametric_model.set_true_vec(flags, y)
        mu_name = mu_tech.value if hasattr(mu_tech, "value") else str(mu_tech)
        needs_combos = mu_name in ("MEDIAN", "MODE")
        parametric_model.build_model_space(build_combos=needs_combos)
        trueY = parametric_model.handle_mu_tech_select(mu_tech=mu_name)
        if trueY.size == 0:
            raise _RCRDegenerate
        return parametric_model.indices, trueY, 0.0
    if non_parametric_model is not None:
        indices, trueY = non_parametric_model.mu_func(flags, y)
        if trueY.size == 0:
            raise _RCRDegenerate
        mu, _ = _mu(mu_tech, trueY)
        return indices, trueY, mu
    indices = np.where(flags)[0]
    trueY = y[indices]
    if trueY.size == 0:
        raise _RCRDegenerate
    mu, _ = _mu(mu_tech, trueY)
    return indices, trueY, mu


def _select_candidates_w(
    flags: np.ndarray, w: np.ndarray, y: np.ndarray, mu_tech: "MuTech",
    non_parametric_model=None, parametric_model=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Weighted (indices, trueW, trueY, mu) selection."""
    if parametric_model is not None:
        parametric_model.set_true_vec(flags, y, w=w)
        mu_name = mu_tech.value if hasattr(mu_tech, "value") else str(mu_tech)
        needs_combos = mu_name in ("MEDIAN", "MODE")
        parametric_model.build_model_space(build_combos=needs_combos)
        trueY = parametric_model.handle_mu_tech_select(mu_tech=mu_name)
        if trueY.size == 0:
            raise _RCRDegenerate
        return parametric_model.indices, parametric_model.trueW, trueY, 0.0
    if non_parametric_model is not None:
        indices, trueW, trueY = non_parametric_model.mu_func_w(flags, w, y)
        if trueY.size == 0:
            raise _RCRDegenerate
        return indices, trueW, trueY, _mu_w(mu_tech, trueW, trueY)
    indices = np.where(flags)[0]
    if indices.size == 0:
        raise _RCRDegenerate
    return indices, w[indices], y[indices], _mu_w(mu_tech, w[indices], y[indices])


def _mu_w(mu_tech: MuTech, trueW: np.ndarray, trueY: np.ndarray) -> float:
    """Weighted mu dispatch — handleMuTechSelect(w, y) at cpp/src/RCR.cpp:5907.
    For MEDIAN and MODE the C++ does sort(w, y) (sort y ascending, w along)
    before getMedian/getMode. We do the same."""
    if mu_tech is MuTech.MEAN:
        return stats.getMean_w(trueW, trueY)
    idx = np.argsort(trueY, kind="stable")
    w_s = trueW[idx]
    y_s = trueY[idx]
    if mu_tech is MuTech.MEDIAN:
        return stats.getMedian_w(w_s, y_s)
    # MODE
    return stats.halfSampleMode_w(w_s, y_s)


def _sigma_lower(sigma_tech: SigmaTech, w: np.ndarray, diff: np.ndarray,
                 delta: float, counter: int) -> float:
    """Port of handleSigmaTechSelect for sigmaChoice=LOWER. The delta passed
    in is the user-set delta (default 1.0); for LOWER, getStDev uses delta/2.

    For DOUBLE_LINE we always use the weighted helpers — multiplying by w=1.0
    is bit-exact under IEEE 754, so the unweighted loop (which passes w=ones)
    produces the same result as the C++ unweighted handleSigmaTechSelect.
    """
    if sigma_tech is SigmaTech.STANDARD_DEVIATION:
        return stats.getStDev_w(delta / 2.0, w, diff)
    if sigma_tech is SigmaTech.SIXTY_EIGHTH_PERCENTILE:
        return stats.get68th_w(w, diff)
    if sigma_tech is SigmaTech.DOUBLE_LINE:
        x = stats.getXVec_w(diff.size, w)
        x_below = stats.countAmountLessThanOne(x)
        if x_below > 2:
            return stats.fitDL_w(counter, w, x, diff, getLowerFN_w)
        if x_below > 1:
            return stats.fitSL_w(w, x, diff)
        return stats.get68th_w(w, diff)
    raise ValueError(f"unsupported sigma_tech {sigma_tech!r}")


def _sigma_each(sigma_tech: SigmaTech, w: np.ndarray, diff: np.ndarray,
                delta: float, counter: int) -> float:
    """Sigma dispatch for sigmaChoice=EACH. Same as _sigma_lower except
    DOUBLE_LINE falls back via getEachFN_w."""
    if sigma_tech is SigmaTech.STANDARD_DEVIATION:
        return stats.getStDev_w(delta / 2.0, w, diff)
    if sigma_tech is SigmaTech.SIXTY_EIGHTH_PERCENTILE:
        return stats.get68th_w(w, diff)
    if sigma_tech is SigmaTech.DOUBLE_LINE:
        x = stats.getXVec_w(diff.size, w)
        x_below = stats.countAmountLessThanOne(x)
        if x_below > 2:
            return stats.fitDL_w(counter, w, x, diff, getEachFN_w)
        if x_below > 1:
            return stats.fitSL_w(w, x, diff)
        return stats.get68th_w(w, diff)
    raise ValueError(f"unsupported sigma_tech {sigma_tech!r}")


def _sigma_single(sigma_tech: SigmaTech, w: np.ndarray, diff: np.ndarray,
                  delta: float, counter: int) -> float:
    """Sigma dispatch for sigmaChoice=SINGLE (used by iterativeSingleSigmaRCR).
    Differs from _sigma_lower: getStDev uses the FULL delta (not delta/2),
    and DOUBLE_LINE's fitDL falls back via getSingleFN instead of getLowerFN.
    """
    if sigma_tech is SigmaTech.STANDARD_DEVIATION:
        return stats.getStDev_w(delta, w, diff)
    if sigma_tech is SigmaTech.SIXTY_EIGHTH_PERCENTILE:
        return stats.get68th_w(w, diff)
    if sigma_tech is SigmaTech.DOUBLE_LINE:
        x = stats.getXVec_w(diff.size, w)
        x_below = stats.countAmountLessThanOne(x)
        if x_below > 2:
            return stats.fitDL_w(counter, w, x, diff, getSingleFN_w)
        if x_below > 1:
            return stats.fitSL_w(w, x, diff)
        return stats.get68th_w(w, diff)
    raise ValueError(f"unsupported sigma_tech {sigma_tech!r}")


# ---------------------------------------------------------------------------
# Rejection test (Chauvenet criterion).
# ---------------------------------------------------------------------------

def _reject(true_count: int, max_index: int, ratio: float,
            flags: np.ndarray, y: np.ndarray,
            true_y: np.ndarray | None = None,
            parametric_model=None) -> bool:
    """Port of cpp/src/RCR.cpp:5744. Returns True if rejection should STOP
    (no point rejected this iteration); False after marking flags[max_index]
    = False.

    Parametric models use the 3-arg :func:`stats.distinctValuesCheckParam` on the RESIDUAL vector
    ``true_y`` (C++ RCR.cpp:5223/5785 pass ``trueY``), which stops the peel earlier than the 2-arg
    check on the original ``y`` — that guard is what keeps the 2 marginal noise-fit scans the C++
    keeps. Non-parametric keeps the 2-arg check on ``y`` (already bit-parity)."""
    if parametric_model is not None:
        distinct = stats.distinctValuesCheckParam(int(parametric_model.M), flags, true_y)
    else:
        distinct = stats.distinctValuesCheck(flags, y)
    if distinct and true_count * stats.erfcCustom(ratio) < 0.5:
        flags[max_index] = False
        return False
    return True


# ---------------------------------------------------------------------------
# The iterative lower-sigma loop.
# ---------------------------------------------------------------------------

def iterativeLowerSigmaRCR(
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:6408 — iterativeLowerSigmaRCR(y), unweighted.

    Mutates `flags` in place. Returns the per-iteration sigma/stdev/mu state
    that the caller can stash on RCRResults.

    n_correct_fn(n) returns the correction factor for the user's rejection
    technique; passed in so the loop doesn't have to know about RejectionTech.
    """
    mu = -1.0
    sigma = -1.0
    st_dev = -1.0
    st_dev_above = -1.0
    st_dev_below = -1.0

    stop = False
    while not stop:
        indices, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        # Vectorized port of the C++ per-iteration walk (cpp/src/RCR.cpp:6461).
        # Computes |y-mu|, locates the worst residual, and splits the kept
        # set into below/above-mu lists. Ties at mu go to BOTH lists; only
        # the LAST tied point's position is reweighted to 0.5 (mirrors a
        # quirk in the C++ that is intentionally preserved for parity —
        # see rcrpy-porting-gotchas memory).
        diff = np.abs(trueY - mu)
        max_local = int(np.argmax(diff))
        max_val = float(diff[max_local])
        max_index = int(indices[max_local])

        is_eq = stats.isEqual_vec_scalar(trueY, mu)
        # Order-preserving partition: keep_below covers (y<mu OR y==mu),
        # keep_above covers (y>mu OR y==mu). The C++ appends ties to BOTH
        # lists in trueY order; np.where preserves that ordering.
        keep_below = (trueY < mu) | is_eq
        keep_above = (trueY > mu) | is_eq
        diff_below = diff[keep_below]
        diff_above = diff[keep_above]
        w_below = np.ones(diff_below.size, dtype=np.float64)
        w_above = np.ones(diff_above.size, dtype=np.float64)
        split = bool(is_eq.any())
        if split:
            last_eq = int(np.where(is_eq)[0][-1])
            # Position of `last_eq` within keep_below (zero-indexed): the
            # number of True entries in keep_below at indices <= last_eq,
            # minus one.
            below_split_idx = int(np.sum(keep_below[: last_eq + 1])) - 1
            above_split_idx = int(np.sum(keep_above[: last_eq + 1])) - 1
            w_below[below_split_idx] = 0.5
            w_above[above_split_idx] = 0.5

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0

        # Sort each (w, diff) pair by diff ascending.
        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below = w_below[idx]
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above = w_above[idx]

        n_correction = n_correct_fn(true_count)

        if nonzero_above and nonzero_below:
            st_dev_above = _sigma_lower(sigma_tech, w_above, diff_above, delta, true_count)
            st_dev_below = _sigma_lower(sigma_tech, w_below, diff_below, delta, true_count)
            st_dev = min(st_dev_above, st_dev_below)
            sigma = st_dev * n_correction
        elif nonzero_above:
            st_dev = _sigma_lower(sigma_tech, w_above, diff_above, delta, true_count)
            st_dev_above = st_dev
            sigma = st_dev * n_correction
        elif nonzero_below:
            st_dev = _sigma_lower(sigma_tech, w_below, diff_below, delta, true_count)
            st_dev_below = st_dev
            sigma = st_dev * n_correction

        stop = _reject(true_count, max_index, _reject_ratio(max_val, sigma), flags, y,
                       true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma": sigma,
        "st_dev": st_dev,
        "st_dev_above": st_dev_above,
        "st_dev_below": st_dev_below,
    }


# ---------------------------------------------------------------------------
# 3-pass refinement performed by performRejection.
# ---------------------------------------------------------------------------

def iterativeLowerSigmaRCR_w(
    w: np.ndarray,
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Weighted twin of iterativeLowerSigmaRCR. Port of cpp/src/RCR.cpp:6261.

    Note: unlike the unweighted port, the tie-at-mu split here is NOT buggy
    — the C++ pushes 0.5*trueW[i] into both wBelow and wAbove for EVERY
    tied point (vs. the unweighted version which only weights the LAST
    tie). We mirror that.
    """
    mu = -1.0
    sigma = -1.0
    st_dev = -1.0
    st_dev_above = -1.0
    st_dev_below = -1.0

    stop = False
    while not stop:
        indices, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        # Vectorized weighted twin of the unweighted lower-sigma walk.
        # Ties at mu get 0.5*trueW[i] in BOTH lists (this is the "no LAST-
        # only bug" weighted variant — see cpp/src/RCR.cpp:6331+).
        diff = np.abs(trueY - mu)
        max_local = int(np.argmax(diff))
        max_val = float(diff[max_local])
        max_index = int(indices[max_local])

        is_eq = stats.isEqual_vec_scalar(trueY, mu)
        keep_below = (trueY < mu) | is_eq
        keep_above = (trueY > mu) | is_eq
        diff_below = diff[keep_below]
        diff_above = diff[keep_above]
        # Weights: full trueW for below_only / above_only, 0.5*trueW for ties.
        w_below_full = np.where(is_eq, 0.5 * trueW, trueW)
        w_above_full = np.where(is_eq, 0.5 * trueW, trueW)
        w_below_arr = w_below_full[keep_below]
        w_above_arr = w_above_full[keep_above]

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0

        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below_arr = w_below_arr[idx]
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above_arr = w_above_arr[idx]

        n_correction = n_correct_fn(true_count, trueW)

        if nonzero_above and nonzero_below:
            st_dev_above = _sigma_lower(sigma_tech, w_above_arr, diff_above, delta, true_count)
            st_dev_below = _sigma_lower(sigma_tech, w_below_arr, diff_below, delta, true_count)
            st_dev = min(st_dev_above, st_dev_below)
            sigma = st_dev * n_correction
        elif nonzero_above:
            st_dev = _sigma_lower(sigma_tech, w_above_arr, diff_above, delta, true_count)
            st_dev_above = st_dev
            sigma = st_dev * n_correction
        elif nonzero_below:
            st_dev = _sigma_lower(sigma_tech, w_below_arr, diff_below, delta, true_count)
            st_dev_below = st_dev
            sigma = st_dev * n_correction

        stop = _reject(true_count, max_index, _reject_ratio(max_val, sigma), flags, y,
                       true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma": sigma,
        "st_dev": st_dev,
        "st_dev_above": st_dev_above,
        "st_dev_below": st_dev_below,
    }


def iterativeEachSigmaRCR(
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:6707 — iterativeEachSigmaRCR(y), unweighted.

    Each-sigma differs from lower-sigma:
      - sigmaBelow and sigmaAbove are kept separate (no min).
      - The rejection-candidate `max` is in units of *its own side's sigma*
        (not max-of-diff / single-sigma).
      - Only mu / stDevBelow / stDevAbove / sigmaBelow / sigmaAbove get set
        on the result; sigma and stDev are left at their previous value.
    """
    mu = -1.0
    sigma_below = -1.0
    sigma_above = -1.0
    st_dev_above = -1.0
    st_dev_below = -1.0

    stop = False
    while not stop:
        indices, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        # Vectorized each-sigma walk. Mirrors the unweighted lower-sigma
        # variant's tie handling (LAST tie position gets 0.5 weight),
        # except each-sigma doesn't track max here — that happens later
        # in the original-y side-aware max-ratio scan.
        diff = np.abs(trueY - mu)
        is_eq = stats.isEqual_vec_scalar(trueY, mu)
        keep_below = (trueY < mu) | is_eq
        keep_above = (trueY > mu) | is_eq
        diff_below = diff[keep_below]
        diff_above = diff[keep_above]
        w_below = np.ones(diff_below.size, dtype=np.float64)
        w_above = np.ones(diff_above.size, dtype=np.float64)
        split = bool(is_eq.any())
        if split:
            last_eq = int(np.where(is_eq)[0][-1])
            below_split_idx = int(np.sum(keep_below[: last_eq + 1])) - 1
            above_split_idx = int(np.sum(keep_above[: last_eq + 1])) - 1
            w_below[below_split_idx] = 0.5
            w_above[above_split_idx] = 0.5

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0
        n_correction = n_correct_fn(true_count)

        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below = w_below[idx]
            st_dev_below = _sigma_each(sigma_tech, w_below, diff_below, delta, true_count)
            sigma_below = st_dev_below * n_correction
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above = w_above[idx]
            st_dev_above = _sigma_each(sigma_tech, w_above, diff_above, delta, true_count)
            sigma_above = st_dev_above * n_correction

        # Find max of |y-mu|/sigma over the original y, side-dependent.
        #
        # Two degenerate cases need handling distinctly:
        #   - EMPTY side (sigma_X stays at -1.0 sentinel because there were
        #     no points on that side this iteration): don't consider any
        #     candidates from this side.
        #   - DEGENERATE side (sigma_X collapsed to ≤ 0 even though the
        #     side had points — e.g., all residuals on that side were
        #     exactly equal): the C++ does FP `hold / 0 = +inf` and uses
        #     the rejection criterion erfc(inf) = 0 to reject the worst
        #     point. We mirror that with a tiny sigma floor so the ratio
        #     becomes effectively `inf` and the same point gets selected.
        #
        # `nonzero_below` / `nonzero_above` distinguish the two cases.
        max_val = -99999.0
        max_index = -1
        for i in range(y.size):
            if not flags[i]:
                continue
            yi = float(y[i])
            hold = abs(yi - mu)
            if yi < mu and nonzero_below:
                s = sigma_below if sigma_below > 0 else _SIGMA_FLOOR
                ratio = hold / s
                if ratio > max_val:
                    max_val = ratio
                    max_index = i
            if yi > mu and nonzero_above:
                s = sigma_above if sigma_above > 0 else _SIGMA_FLOOR
                ratio = hold / s
                if ratio > max_val:
                    max_val = ratio
                    max_index = i

        if max_index < 0:
            # No valid rejection candidate (e.g., all kept residuals
            # exactly at mu). The C++ here is technically undefined
            # behavior — `maxIndex` is uninitialized. We choose the safe
            # interpretation: terminate the loop.
            stop = True
        else:
            stop = _reject(true_count, max_index, max_val, flags, y,
                           true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma_below": sigma_below,
        "sigma_above": sigma_above,
        "st_dev_above": st_dev_above,
        "st_dev_below": st_dev_below,
    }


def iterativeEachSigmaRCR_w(
    w: np.ndarray,
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Weighted twin of iterativeEachSigmaRCR. Port of cpp/src/RCR.cpp:6557.
    Tie-at-mu distributes 0.5*trueW[i] to both sides (no LAST-only bug)."""
    mu = -1.0
    sigma_below = -1.0
    sigma_above = -1.0
    st_dev_above = -1.0
    st_dev_below = -1.0

    stop = False
    while not stop:
        indices, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        # Vectorized weighted each-sigma walk. Ties at mu get 0.5*trueW[i]
        # in BOTH lists (no LAST-only quirk for the weighted variant —
        # cpp/src/RCR.cpp:6557+).
        diff = np.abs(trueY - mu)
        is_eq = stats.isEqual_vec_scalar(trueY, mu)
        keep_below = (trueY < mu) | is_eq
        keep_above = (trueY > mu) | is_eq
        diff_below = diff[keep_below]
        diff_above = diff[keep_above]
        w_below_full = np.where(is_eq, 0.5 * trueW, trueW)
        w_above_full = np.where(is_eq, 0.5 * trueW, trueW)
        w_below_arr = w_below_full[keep_below]
        w_above_arr = w_above_full[keep_above]

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0
        n_correction = n_correct_fn(true_count, trueW)

        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below_arr = w_below_arr[idx]
            st_dev_below = _sigma_each(sigma_tech, w_below_arr, diff_below, delta, true_count)
            sigma_below = st_dev_below * n_correction
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above_arr = w_above_arr[idx]
            st_dev_above = _sigma_each(sigma_tech, w_above_arr, diff_above, delta, true_count)
            sigma_above = st_dev_above * n_correction

        # See iterativeEachSigmaRCR for the rationale on the two-sentinel
        # split (empty side vs degenerate side).
        max_val = -99999.0
        max_index = -1
        for i in range(y.size):
            if not flags[i]:
                continue
            yi = float(y[i])
            hold = abs(yi - mu)
            if yi < mu and nonzero_below:
                s = sigma_below if sigma_below > 0 else _SIGMA_FLOOR
                ratio = hold / s
                if ratio > max_val:
                    max_val = ratio
                    max_index = i
            if yi > mu and nonzero_above:
                s = sigma_above if sigma_above > 0 else _SIGMA_FLOOR
                ratio = hold / s
                if ratio > max_val:
                    max_val = ratio
                    max_index = i

        if max_index < 0:
            stop = True
        else:
            stop = _reject(true_count, max_index, max_val, flags, y,
                           true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma_below": sigma_below,
        "sigma_above": sigma_above,
        "st_dev_above": st_dev_above,
        "st_dev_below": st_dev_below,
    }


def iterativeSingleSigmaRCR(
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:6168 — iterativeSingleSigmaRCR(y), unweighted.

    Single-sigma: one diff array (just |y - mu|), one sigma, no below/above
    split. Used by SS_MEDIAN_DL.
    """
    mu = -1.0
    sigma = -1.0
    st_dev = -1.0
    stop = False
    while not stop:
        indices, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff = np.abs(trueY - mu)
        max_local = int(np.argmax(diff))
        max_val = float(diff[max_local])
        max_index = int(indices[max_local])

        diff_sorted = np.sort(diff, kind="stable")
        w_ones = np.ones(diff_sorted.size, dtype=np.float64)
        st_dev = _sigma_single(sigma_tech, w_ones, diff_sorted, delta, true_count)
        sigma = st_dev * n_correct_fn(true_count)

        stop = _reject(true_count, max_index, _reject_ratio(max_val, sigma), flags, y,
                       true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma": sigma,
        "st_dev": st_dev,
        "st_dev_above": -1.0,
        "st_dev_below": -1.0,
    }


def iterativeSingleSigmaRCR_w(
    w: np.ndarray,
    y: np.ndarray,
    flags: np.ndarray,
    mu_tech: MuTech,
    sigma_tech: SigmaTech,
    delta: float,
    n_correct_fn,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:6066 — weighted single-sigma loop.

    The C++ does `sort(trueW, diff)` — sorts diff ascending, takes trueW
    along. Mirror that with argsort(diff)."""
    mu = -1.0
    sigma = -1.0
    st_dev = -1.0
    stop = False
    while not stop:
        indices, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff = np.abs(trueY - mu)
        max_local = int(np.argmax(diff))
        max_val = float(diff[max_local])
        max_index = int(indices[max_local])

        idx = np.argsort(diff, kind="stable")
        diff_sorted = diff[idx]
        w_sorted = trueW[idx]
        st_dev = _sigma_single(sigma_tech, w_sorted, diff_sorted, delta, true_count)
        sigma = st_dev * n_correct_fn(true_count, trueW)

        stop = _reject(true_count, max_index, _reject_ratio(max_val, sigma), flags, y,
                       true_y=trueY, parametric_model=parametric_model)

    return {
        "mu": mu,
        "sigma": sigma,
        "st_dev": st_dev,
        "st_dev_above": -1.0,
        "st_dev_below": -1.0,
    }


def _bulk_sigma_lower(w: np.ndarray, diff: np.ndarray, counter: int) -> float:
    """Port of handleBulkSigmaTechSelect (LOWER) at cpp/src/RCR.cpp:6029.
    Bulk always uses DL/SL/68th cascade regardless of sigmaTech."""
    x = stats.getXVec_w(diff.size, w)
    x_below = stats.countAmountLessThanOne(x)
    if x_below > 2:
        return max(stats.fitDL_w(counter, w, x, diff, getLowerFN_w),
                   stats.fitSL_w(w, x, diff))
    if x_below > 1:
        return stats.fitSL_w(w, x, diff)
    return stats.get68th_w(w, diff)


def _bulk_reject(flags: np.ndarray, indices_sorted: np.ndarray,
                 ratios_sorted: np.ndarray, y: np.ndarray,
                 true_y: np.ndarray | None = None,
                 parametric_model=None) -> bool:
    """Port of cpp/src/RCR.cpp:5768. Returns True if no point was rejected
    this pass (signal to stop). `ratios_sorted` is |y-mu|/sigma in ascending
    order, and `indices_sorted` is the original-y index at each position.

    Parametric models gate the peel with :func:`stats.distinctValuesCheckParam` on the RESIDUAL
    vector ``true_y`` (C++ bulkReject → RCR.cpp:5245 passes ``trueY``), which stops earlier than the
    2-arg check on the original ``y``. Non-parametric keeps the 2-arg check on ``y``."""
    param = parametric_model is not None
    m = int(parametric_model.M) if param else 0
    no_points_rejected = True
    size = ratios_sorted.size
    i = size - 1
    while i >= 0 \
            and (stats.distinctValuesCheckParam(m, flags, true_y) if param
                 else stats.distinctValuesCheck(flags, y)) \
            and size * stats.erfcCustom(float(ratios_sorted[i])) < 0.5:
        flags[int(indices_sorted[i])] = False
        no_points_rejected = False
        i -= 1
    return no_points_rejected


def bulkLowerSigmaRCR(y: np.ndarray, flags: np.ndarray, mu_tech: MuTech,
                      n_correct_fn,
                      non_parametric_model=None,
                      parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:7173 — bulkLowerSigmaRCR(y), unweighted.

    Bulk rejection: one mu/sigma estimate per iteration, then reject ALL
    points failing the Chauvenet criterion in one pass.

    QUIRK: the C++ initializes `trueCount = y.size()` once OUTSIDE the
    while-loop and never updates it. Both handleBulkSigmaTechSelect and
    nCorrect therefore use the ORIGINAL sample size as `counter`, not the
    post-rejection kept-count. We mirror that for parity.
    """
    counter = int(y.size)  # stays pinned to original size, per C++ quirk
    mu = -1.0
    sigma = -1.0
    stop = False
    while not stop:
        kept, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff_below_list: list[float] = []
        diff_above_list: list[float] = []
        diff_hold: list[float] = []
        below_split_idx = -1
        above_split_idx = -1
        split = False
        for i in range(true_count):
            yi = float(trueY[i])
            hold = stats.getDiff(mu, yi)
            diff_hold.append(hold)
            if stats.isEqual(yi, mu):
                diff_below_list.append(hold)
                diff_above_list.append(hold)
                split = True
                below_split_idx = len(diff_below_list) - 1
                above_split_idx = len(diff_above_list) - 1
            elif yi > mu:
                diff_above_list.append(hold)
            else:
                diff_below_list.append(hold)

        diff_below = np.array(diff_below_list, dtype=np.float64)
        diff_above = np.array(diff_above_list, dtype=np.float64)
        diff_hold_arr = np.array(diff_hold, dtype=np.float64)
        w_below = np.ones(diff_below.size, dtype=np.float64)
        w_above = np.ones(diff_above.size, dtype=np.float64)
        if split:
            w_below[below_split_idx] = 0.5
            w_above[above_split_idx] = 0.5

        # sort(indices, diffHold) — paired sort by diffHold ascending
        sort_idx = np.argsort(diff_hold_arr, kind="stable")
        indices_sorted = kept[sort_idx]
        diff_hold_sorted = diff_hold_arr[sort_idx]

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0
        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below = w_below[idx]
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above = w_above[idx]

        if nonzero_above and nonzero_below:
            sigma = min(_bulk_sigma_lower(w_below, diff_below, counter),
                        _bulk_sigma_lower(w_above, diff_above, counter)) * n_correct_fn(counter)
        elif nonzero_above:
            sigma = _bulk_sigma_lower(w_above, diff_above, counter) * n_correct_fn(counter)
        elif nonzero_below:
            sigma = _bulk_sigma_lower(w_below, diff_below, counter) * n_correct_fn(counter)

        with np.errstate(divide="ignore", invalid="ignore"):
            # sigma collapses to 0 on no-spread data (perfectly-linear /
            # near-constant); the C++ divides anyway and the resulting inf/NaN
            # is absorbed by _bulk_reject (erfcCustom(inf)=0, *NaN < 0.5 = False
            # -> nothing rejected). Suppress NumPy's cosmetic divide/invalid
            # RuntimeWarning while preserving that exact IEEE result.
            ratios_sorted = diff_hold_sorted / sigma
        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma": sigma}


def bulkSingleSigmaRCR(y: np.ndarray, flags: np.ndarray, mu_tech: MuTech,
                       n_correct_fn,
                       non_parametric_model=None,
                       parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:6949 — single-sigma bulk, unweighted.

    Unlike bulkLowerSigmaRCR, this loop DOES update `trueCount` each
    iteration (line 7003 in C++). Single-sigma uses one diff array, no
    below/above split.
    """
    mu = -1.0
    sigma = -1.0
    stop = False
    while not stop:
        kept, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff = np.abs(trueY - mu)
        diff_hold = diff.copy()
        diff_sorted = np.sort(diff, kind="stable")

        # `sort(indices, diffHold)` — sort indices by diffHold ascending
        sort_idx = np.argsort(diff_hold, kind="stable")
        indices_sorted = kept[sort_idx]
        diff_hold_sorted = diff_hold[sort_idx]

        # handleBulkSigmaTechSelect (single) — same DL/SL/68th cascade
        w_ones = np.ones(true_count, dtype=np.float64)
        sigma_raw = _bulk_sigma_lower(w_ones, diff_sorted, true_count)
        # ... but with getSingleFN inside fitDL — _bulk_sigma_lower uses
        # getLowerFN, so we need a single-sigma variant. Recompute here.
        x = stats.getXVec_w(diff_sorted.size, w_ones)
        x_below = stats.countAmountLessThanOne(x)
        if x_below > 2:
            sigma_raw = max(stats.fitDL_w(true_count, w_ones, x, diff_sorted, getSingleFN_w),
                            stats.fitSL_w(w_ones, x, diff_sorted))
        elif x_below > 1:
            sigma_raw = stats.fitSL_w(w_ones, x, diff_sorted)
        else:
            sigma_raw = stats.get68th_w(w_ones, diff_sorted)
        sigma = sigma_raw * n_correct_fn(true_count)

        with np.errstate(divide="ignore", invalid="ignore"):
            # sigma collapses to 0 on no-spread data (perfectly-linear /
            # near-constant); the C++ divides anyway and the resulting inf/NaN
            # is absorbed by _bulk_reject (erfcCustom(inf)=0, *NaN < 0.5 = False
            # -> nothing rejected). Suppress NumPy's cosmetic divide/invalid
            # RuntimeWarning while preserving that exact IEEE result.
            ratios_sorted = diff_hold_sorted / sigma
        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma": sigma}


def bulkEachSigmaRCR(y: np.ndarray, flags: np.ndarray, mu_tech: MuTech,
                     n_correct_fn,
                     non_parametric_model=None,
                     parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:7445 — each-sigma bulk, unweighted.

    Each-sigma bulk: sigmaBelow and sigmaAbove computed separately, no min.
    The rejection ratio for each point is |y-mu| divided by ITS OWN side's
    sigma. Like single/lower bulks, this updates trueCount each iteration."""
    mu = -1.0
    sigma_below = -1.0
    sigma_above = -1.0
    stop = False
    while not stop:
        kept, trueY, mu = _select_candidates(
            flags, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff_below_list: list[float] = []
        diff_above_list: list[float] = []
        diff_hold: list[float] = []
        below_split_idx = -1
        above_split_idx = -1
        split = False
        for i in range(true_count):
            yi = float(trueY[i])
            hold = stats.getDiff(mu, yi)
            diff_hold.append(hold)
            if stats.isEqual(yi, mu):
                diff_below_list.append(hold)
                diff_above_list.append(hold)
                split = True
                below_split_idx = len(diff_below_list) - 1
                above_split_idx = len(diff_above_list) - 1
            elif yi > mu:
                diff_above_list.append(hold)
            else:
                diff_below_list.append(hold)

        diff_below = np.array(diff_below_list, dtype=np.float64)
        diff_above = np.array(diff_above_list, dtype=np.float64)
        diff_hold_arr = np.array(diff_hold, dtype=np.float64)
        w_below = np.ones(diff_below.size, dtype=np.float64)
        w_above = np.ones(diff_above.size, dtype=np.float64)
        if split:
            w_below[below_split_idx] = 0.5
            w_above[above_split_idx] = 0.5

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0

        n_correction = n_correct_fn(true_count)
        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below = w_below[idx]
            # handleBulkSigmaTechSelect (each shares the LOWER FN model)
            sigma_below = _bulk_sigma_lower_each(w_below, diff_below, true_count) * n_correction
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above = w_above[idx]
            sigma_above = _bulk_sigma_lower_each(w_above, diff_above, true_count) * n_correction

        # Normalize diffHold[i] by the appropriate side's sigma. Use the
        # tiny floor when a side's sigma collapsed (see _SIGMA_FLOOR
        # rationale above iterativeEachSigmaRCR).
        s_below = sigma_below if (nonzero_below and sigma_below > 0) else _SIGMA_FLOOR
        s_above = sigma_above if (nonzero_above and sigma_above > 0) else _SIGMA_FLOOR
        for i in range(diff_hold_arr.size):
            yi = float(y[int(kept[i])])
            if yi < mu:
                diff_hold_arr[i] = diff_hold_arr[i] / s_below
            elif yi > mu:
                diff_hold_arr[i] = diff_hold_arr[i] / s_above
            # equal-to-mu case: diff_hold is 0; matches C++ (which doesn't
            # divide for the equal case — neither yi<mu nor yi>mu fires).

        sort_idx = np.argsort(diff_hold_arr, kind="stable")
        indices_sorted = kept[sort_idx]
        ratios_sorted = diff_hold_arr[sort_idx]

        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma_below": sigma_below, "sigma_above": sigma_above}


def _bulk_sigma_lower_each(w: np.ndarray, diff: np.ndarray, counter: int) -> float:
    """handleBulkSigmaTechSelect for each-sigma context — uses getEachFN_w
    inside fitDL's fallback decision (vs getLowerFN_w for the lower-sigma
    bulk variant)."""
    x = stats.getXVec_w(diff.size, w)
    x_below = stats.countAmountLessThanOne(x)
    if x_below > 2:
        return max(stats.fitDL_w(counter, w, x, diff, getEachFN_w),
                   stats.fitSL_w(w, x, diff))
    if x_below > 1:
        return stats.fitSL_w(w, x, diff)
    return stats.get68th_w(w, diff)


def bulkLowerSigmaRCR_w(w: np.ndarray, y: np.ndarray, flags: np.ndarray,
                        mu_tech: MuTech, n_correct_fn_w,
                        non_parametric_model=None,
                        parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:7039. NOTE: unlike unweighted lower bulk,
    the weighted version updates trueCount each iteration (line 7090)."""
    mu = -1.0
    sigma = -1.0
    stop = False
    while not stop:
        kept, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff_below_list: list[float] = []
        diff_above_list: list[float] = []
        w_below_list: list[float] = []
        w_above_list: list[float] = []
        diff_hold: list[float] = []
        for i in range(true_count):
            yi = float(trueY[i])
            wi = float(trueW[i])
            hold = stats.getDiff(mu, yi)
            diff_hold.append(hold)
            if stats.isEqual(yi, mu):
                diff_below_list.append(hold)
                diff_above_list.append(hold)
                w_below_list.append(0.5 * wi)
                w_above_list.append(0.5 * wi)
            elif yi > mu:
                diff_above_list.append(hold)
                w_above_list.append(wi)
            else:
                diff_below_list.append(hold)
                w_below_list.append(wi)

        diff_below = np.array(diff_below_list, dtype=np.float64)
        diff_above = np.array(diff_above_list, dtype=np.float64)
        w_below_arr = np.array(w_below_list, dtype=np.float64)
        w_above_arr = np.array(w_above_list, dtype=np.float64)
        diff_hold_arr = np.array(diff_hold, dtype=np.float64)

        sort_idx = np.argsort(diff_hold_arr, kind="stable")
        indices_sorted = kept[sort_idx]
        diff_hold_sorted = diff_hold_arr[sort_idx]

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0
        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below_arr = w_below_arr[idx]
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above_arr = w_above_arr[idx]

        if nonzero_above and nonzero_below:
            sigma = min(_bulk_sigma_lower(w_below_arr, diff_below, true_count),
                        _bulk_sigma_lower(w_above_arr, diff_above, true_count)) * n_correct_fn_w(true_count, trueW)
        elif nonzero_above:
            sigma = _bulk_sigma_lower(w_above_arr, diff_above, true_count) * n_correct_fn_w(true_count, trueW)
        elif nonzero_below:
            sigma = _bulk_sigma_lower(w_below_arr, diff_below, true_count) * n_correct_fn_w(true_count, trueW)

        with np.errstate(divide="ignore", invalid="ignore"):
            # sigma collapses to 0 on no-spread data (perfectly-linear /
            # near-constant); the C++ divides anyway and the resulting inf/NaN
            # is absorbed by _bulk_reject (erfcCustom(inf)=0, *NaN < 0.5 = False
            # -> nothing rejected). Suppress NumPy's cosmetic divide/invalid
            # RuntimeWarning while preserving that exact IEEE result.
            ratios_sorted = diff_hold_sorted / sigma
        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma": sigma}


def bulkSingleSigmaRCR_w(w: np.ndarray, y: np.ndarray, flags: np.ndarray,
                         mu_tech: MuTech, n_correct_fn_w,
                         non_parametric_model=None,
                         parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:6855. Weighted single-sigma bulk.
    C++ does sort(trueW, diff) — sorts diff ascending with trueW along."""
    mu = -1.0
    sigma = -1.0
    stop = False
    while not stop:
        kept, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff = np.abs(trueY - mu)
        diff_hold = diff.copy()

        # sort(trueW, diff) -- by diff ascending
        sort_idx_diff = np.argsort(diff, kind="stable")
        diff_sorted = diff[sort_idx_diff]
        trueW_sorted = trueW[sort_idx_diff]

        # sort(indices, diffHold) -- by diff_hold ascending
        sort_idx_hold = np.argsort(diff_hold, kind="stable")
        indices_sorted = kept[sort_idx_hold]
        diff_hold_sorted = diff_hold[sort_idx_hold]

        # handleBulkSigmaTechSelect (single) with weighted helpers
        x = stats.getXVec_w(diff_sorted.size, trueW_sorted)
        x_below = stats.countAmountLessThanOne(x)
        if x_below > 2:
            sigma_raw = max(stats.fitDL_w(true_count, trueW_sorted, x, diff_sorted, getSingleFN_w),
                            stats.fitSL_w(trueW_sorted, x, diff_sorted))
        elif x_below > 1:
            sigma_raw = stats.fitSL_w(trueW_sorted, x, diff_sorted)
        else:
            sigma_raw = stats.get68th_w(trueW_sorted, diff_sorted)
        sigma = sigma_raw * n_correct_fn_w(true_count, trueW)

        with np.errstate(divide="ignore", invalid="ignore"):
            # sigma collapses to 0 on no-spread data (perfectly-linear /
            # near-constant); the C++ divides anyway and the resulting inf/NaN
            # is absorbed by _bulk_reject (erfcCustom(inf)=0, *NaN < 0.5 = False
            # -> nothing rejected). Suppress NumPy's cosmetic divide/invalid
            # RuntimeWarning while preserving that exact IEEE result.
            ratios_sorted = diff_hold_sorted / sigma
        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma": sigma}


def bulkEachSigmaRCR_w(w: np.ndarray, y: np.ndarray, flags: np.ndarray,
                       mu_tech: MuTech, n_correct_fn_w,
                       non_parametric_model=None,
                       parametric_model=None) -> dict:
    """Port of cpp/src/RCR.cpp:7301. Weighted each-sigma bulk."""
    mu = -1.0
    sigma_below = -1.0
    sigma_above = -1.0
    stop = False
    while not stop:
        kept, trueW, trueY, mu = _select_candidates_w(
            flags, w, y, mu_tech, non_parametric_model, parametric_model)
        true_count = trueY.size

        diff_below_list: list[float] = []
        diff_above_list: list[float] = []
        w_below_list: list[float] = []
        w_above_list: list[float] = []
        diff_hold: list[float] = []
        for i in range(true_count):
            yi = float(trueY[i])
            wi = float(trueW[i])
            hold = stats.getDiff(mu, yi)
            diff_hold.append(hold)
            if stats.isEqual(yi, mu):
                diff_below_list.append(hold)
                diff_above_list.append(hold)
                w_below_list.append(0.5 * wi)
                w_above_list.append(0.5 * wi)
            elif yi < mu:
                diff_below_list.append(hold)
                w_below_list.append(wi)
            else:
                diff_above_list.append(hold)
                w_above_list.append(wi)

        diff_below = np.array(diff_below_list, dtype=np.float64)
        diff_above = np.array(diff_above_list, dtype=np.float64)
        w_below_arr = np.array(w_below_list, dtype=np.float64)
        w_above_arr = np.array(w_above_list, dtype=np.float64)
        diff_hold_arr = np.array(diff_hold, dtype=np.float64)

        nonzero_above = diff_above.size > 0
        nonzero_below = diff_below.size > 0

        if nonzero_below:
            idx = np.argsort(diff_below, kind="stable")
            diff_below = diff_below[idx]
            w_below_arr = w_below_arr[idx]
            sigma_below = _bulk_sigma_lower_each(w_below_arr, diff_below, true_count) * n_correct_fn_w(true_count, trueW)
        if nonzero_above:
            idx = np.argsort(diff_above, kind="stable")
            diff_above = diff_above[idx]
            w_above_arr = w_above_arr[idx]
            sigma_above = _bulk_sigma_lower_each(w_above_arr, diff_above, true_count) * n_correct_fn_w(true_count, trueW)

        # Tiny floor when a side's sigma collapsed (see _SIGMA_FLOOR
        # rationale above iterativeEachSigmaRCR).
        s_below = sigma_below if (nonzero_below and sigma_below > 0) else _SIGMA_FLOOR
        s_above = sigma_above if (nonzero_above and sigma_above > 0) else _SIGMA_FLOOR
        for i in range(diff_hold_arr.size):
            yi = float(y[int(kept[i])])
            if yi < mu:
                diff_hold_arr[i] = diff_hold_arr[i] / s_below
            elif yi > mu:
                diff_hold_arr[i] = diff_hold_arr[i] / s_above

        sort_idx = np.argsort(diff_hold_arr, kind="stable")
        indices_sorted = kept[sort_idx]
        ratios_sorted = diff_hold_arr[sort_idx]
        stop = _bulk_reject(flags, indices_sorted, ratios_sorted, y,
                            true_y=trueY, parametric_model=parametric_model)

    return {"mu": mu, "sigma_below": sigma_below, "sigma_above": sigma_above}


def _set_final_vectors_w(w: np.ndarray, y: np.ndarray, result_state: dict,
                         flags: np.ndarray, delta: float) -> dict:
    """Port of cpp/src/RCR.cpp:5079. Weighted setFinalVectors."""
    rejected_mask = ~flags
    rejected_y = y[rejected_mask].copy()
    rejected_w = w[rejected_mask].copy()
    clean_y = y[flags].copy()
    clean_w = w[flags].copy()
    mean_hold = stats.getMean_w(clean_w, clean_y)

    cyb: list[float] = []
    cya: list[float] = []
    cwb: list[float] = []
    cwa: list[float] = []
    for yi, wi in zip(clean_y, clean_w):
        adj = float(yi) - mean_hold
        wf = float(wi)
        if stats.isEqual(adj, 0.0):
            cya.append(adj)
            cyb.append(adj)
            cwb.append(0.5 * wf)
            cwa.append(0.5 * wf)
        elif adj > 0.0:
            cya.append(adj)
            cwa.append(wf)
        else:
            cyb.append(adj)
            cwb.append(wf)

    centered = clean_y - mean_hold
    st_dev_total = stats.getStDev_w(delta, clean_w, centered)
    cyb_a = np.array(cyb, dtype=np.float64)
    cya_a = np.array(cya, dtype=np.float64)
    cwb_a = np.array(cwb, dtype=np.float64)
    cwa_a = np.array(cwa, dtype=np.float64)
    st_dev_below = stats.getStDev_w(delta / 2.0, cwb_a, cyb_a) if cyb_a.size else -1.0
    st_dev_above = stats.getStDev_w(delta / 2.0, cwa_a, cya_a) if cya_a.size else -1.0

    result_state["rejected_y"] = rejected_y
    result_state["rejected_w"] = rejected_w
    result_state["original_y"] = y.copy()
    result_state["original_w"] = w.copy()
    result_state["clean_w"] = clean_w
    result_state["st_dev_total"] = st_dev_total
    result_state["st_dev_below"] = st_dev_below
    result_state["st_dev_above"] = st_dev_above
    return result_state


def _set_final_vectors(y: np.ndarray, result_state: dict, flags: np.ndarray,
                       delta: float) -> dict:
    """Port of cpp/src/RCR.cpp:5132 — setFinalVectors(y), unweighted.

    Populates rejectedY, originalY, stDevAbove/Below/Total based on the
    cleanY distribution about its mean."""
    rejected_y = y[~flags].copy()
    clean_y = y[flags].copy()
    mean_hold = stats.getMean(clean_y)

    clean_y_above: list[float] = []
    clean_y_below: list[float] = []
    w_above: list[float] = []
    w_below: list[float] = []
    for v in clean_y:
        adj = float(v) - mean_hold
        if stats.isEqual(adj, 0.0):
            clean_y_above.append(adj)
            clean_y_below.append(adj)
            w_above.append(0.5)
            w_below.append(0.5)
        elif adj > 0.0:
            clean_y_above.append(adj)
            w_above.append(1.0)
        else:
            clean_y_below.append(adj)
            w_below.append(1.0)

    centered = clean_y - mean_hold
    st_dev_total = stats.getStDev(delta, centered)
    cyb = np.array(clean_y_below, dtype=np.float64)
    cya = np.array(clean_y_above, dtype=np.float64)
    wb = np.array(w_below, dtype=np.float64)
    wa = np.array(w_above, dtype=np.float64)
    st_dev_below = stats.getStDev_w(delta / 2.0, wb, cyb) if cyb.size else -1.0
    st_dev_above = stats.getStDev_w(delta / 2.0, wa, cya) if cya.size else -1.0

    result_state["rejected_y"] = rejected_y
    result_state["original_y"] = y.copy()
    result_state["st_dev_total"] = st_dev_total
    result_state["st_dev_below"] = st_dev_below
    result_state["st_dev_above"] = st_dev_above
    return result_state


def performBulkRejection_LS(
    y: np.ndarray,
    rejection_tech_name: str,
    delta: float = 1.0,
    w: np.ndarray | None = None,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:5047 — performBulkRejection(y), unweighted.

    Four-pass:
      1. BULK with user's tech (uses handleBulkSigmaTechSelect's cascade)
      2. ITERATIVE with user's tech (sigmaTech and muTech as aligned)
      3. ITERATIVE median + 68th
      4. ITERATIVE mean + standard deviation
    Then setFinalVectors populates rejectedY/originalY/stDevTotal.
    """
    # Resolve tech → (pass1 mu_tech, pass1 sigma_tech, sigma_choice, CF fn)
    if rejection_tech_name == "LS_MODE_68":
        pass1_mu, pass1_sigma = MuTech.MODE, SigmaTech.SIXTY_EIGHTH_PERCENTILE
        sigma_choice = SigmaChoice.LOWER
        n_correct_unw, n_correct_wgt = getLower68CF, getLower68CF_w
    elif rejection_tech_name == "LS_MODE_DL":
        pass1_mu, pass1_sigma = MuTech.MODE, SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.LOWER
        n_correct_unw, n_correct_wgt = getLowerDLCF, getLowerDLCF_w
    elif rejection_tech_name == "SS_MEDIAN_DL":
        pass1_mu, pass1_sigma = MuTech.MEDIAN, SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.SINGLE
        n_correct_unw, n_correct_wgt = getSingleDLCF, getSingleDLCF_w
    elif rejection_tech_name == "ES_MODE_DL":
        pass1_mu, pass1_sigma = MuTech.MODE, SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.EACH
        n_correct_unw, n_correct_wgt = getEachDLCF, getEachDLCF_w
    else:
        raise NotImplementedError(f"unknown rejection_tech_name {rejection_tech_name!r}")

    flags = np.ones(y.size, dtype=bool)

    # When PARAMETRIC, the C++ sets delta = parameterSpace.size() = M
    # (number of model params) so getStDev's denominator changes.
    if parametric_model is not None:
        delta = float(parametric_model.M)

    mods = (non_parametric_model, parametric_model)

    if w is None:
        # Pass 1: bulk. An empty candidate set at any pass -> nan result (see
        # performRejection_LS); flags keep whatever was rejected first.
        try:
            if sigma_choice is SigmaChoice.LOWER:
                bulkLowerSigmaRCR(y, flags, pass1_mu, n_correct_unw, *mods)
                iter_loop = iterativeLowerSigmaRCR
            elif sigma_choice is SigmaChoice.SINGLE:
                bulkSingleSigmaRCR(y, flags, pass1_mu, n_correct_unw, *mods)
                iter_loop = iterativeSingleSigmaRCR
            else:  # EACH
                bulkEachSigmaRCR(y, flags, pass1_mu, n_correct_unw, *mods)
                iter_loop = iterativeEachSigmaRCR
            iter_loop(y, flags, pass1_mu, pass1_sigma, delta, n_correct_unw, *mods)
            iter_loop(y, flags, MuTech.MEDIAN, SigmaTech.SIXTY_EIGHTH_PERCENTILE, delta, n_correct_unw, *mods)
            state = iter_loop(y, flags, MuTech.MEAN, SigmaTech.STANDARD_DEVIATION, delta, n_correct_unw, *mods)
        except _RCRDegenerate:
            state = _degenerate_state(sigma_choice)

        kept = np.where(flags)[0]
        out = {"flags": flags, "indices": kept, "clean_y": y[kept].copy(), **state}
        _set_final_vectors(y, out, flags, delta)
        return out

    # Weighted bulk
    try:
        if sigma_choice is SigmaChoice.LOWER:
            bulkLowerSigmaRCR_w(w, y, flags, pass1_mu, n_correct_wgt, *mods)
            iter_loop_w = iterativeLowerSigmaRCR_w
        elif sigma_choice is SigmaChoice.SINGLE:
            bulkSingleSigmaRCR_w(w, y, flags, pass1_mu, n_correct_wgt, *mods)
            iter_loop_w = iterativeSingleSigmaRCR_w
        else:
            bulkEachSigmaRCR_w(w, y, flags, pass1_mu, n_correct_wgt, *mods)
            iter_loop_w = iterativeEachSigmaRCR_w
        iter_loop_w(w, y, flags, pass1_mu, pass1_sigma, delta, n_correct_wgt, *mods)
        iter_loop_w(w, y, flags, MuTech.MEDIAN, SigmaTech.SIXTY_EIGHTH_PERCENTILE, delta, n_correct_wgt, *mods)
        state = iter_loop_w(w, y, flags, MuTech.MEAN, SigmaTech.STANDARD_DEVIATION, delta, n_correct_wgt, *mods)
    except _RCRDegenerate:
        state = _degenerate_state(sigma_choice)

    kept = np.where(flags)[0]
    out = {"flags": flags, "indices": kept, "clean_y": y[kept].copy(), **state}
    _set_final_vectors_w(w, y, out, flags, delta)
    return out


def performRejection_LS(
    y: np.ndarray,
    rejection_tech_name: str,
    delta: float = 1.0,
    w: np.ndarray | None = None,
    non_parametric_model=None,
    parametric_model=None,
) -> dict:
    """Port of cpp/src/RCR.cpp:5017 — performRejection(y), restricted to the
    LS_MODE_68 / LS_MODE_DL rejection techs (sigmaChoice=LOWER, muTech=MODE).

    The C++ does three iterative passes, each mutating flags further:
        Pass 1: (mu = MODE,   sigma = user's sigmaTech)  -- aligned by rejection_tech
        Pass 2: (mu = MEDIAN, sigma = 68th percentile)
        Pass 3: (mu = MEAN,   sigma = standard deviation)
    Then `alignTechniques()` resets state — irrelevant once results are
    captured.
    """
    # Resolve (pass1_mu, pass1_sigma, n_correct, sigma_choice) from rejection tech
    # via the alignTechniques table at cpp/src/RCR.cpp:1539.
    if rejection_tech_name == "LS_MODE_68":
        pass1_mu = MuTech.MODE
        pass1_sigma = SigmaTech.SIXTY_EIGHTH_PERCENTILE
        sigma_choice = SigmaChoice.LOWER
        n_correct_unw, n_correct_wgt = getLower68CF, getLower68CF_w
    elif rejection_tech_name == "LS_MODE_DL":
        pass1_mu = MuTech.MODE
        pass1_sigma = SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.LOWER
        n_correct_unw, n_correct_wgt = getLowerDLCF, getLowerDLCF_w
    elif rejection_tech_name == "SS_MEDIAN_DL":
        pass1_mu = MuTech.MEDIAN
        pass1_sigma = SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.SINGLE
        n_correct_unw, n_correct_wgt = getSingleDLCF, getSingleDLCF_w
    elif rejection_tech_name == "ES_MODE_DL":
        pass1_mu = MuTech.MODE
        pass1_sigma = SigmaTech.DOUBLE_LINE
        sigma_choice = SigmaChoice.EACH
        n_correct_unw, n_correct_wgt = getEachDLCF, getEachDLCF_w
    else:
        raise NotImplementedError(
            f"unsupported rejection technique: {rejection_tech_name}"
        )

    flags = np.ones(y.size, dtype=bool)

    if sigma_choice is SigmaChoice.LOWER:
        loop_unw = iterativeLowerSigmaRCR
        loop_wgt = iterativeLowerSigmaRCR_w
    elif sigma_choice is SigmaChoice.SINGLE:
        loop_unw = iterativeSingleSigmaRCR
        loop_wgt = iterativeSingleSigmaRCR_w
    else:  # EACH
        loop_unw = iterativeEachSigmaRCR
        loop_wgt = iterativeEachSigmaRCR_w

    # All loop families now accept (non_parametric_model, parametric_model)
    # as the trailing two args.
    extra_unw = (non_parametric_model, parametric_model)
    extra_wgt = (non_parametric_model, parametric_model)

    # An empty candidate set at any pass (e.g. a NonParametric muFunc where no
    # point has a valid model) -> nan result, the documented "rejection cannot
    # proceed" outcome. flags keep whatever was rejected before the empty pass.
    try:
        if w is None:
            state = loop_unw(y, flags, pass1_mu, pass1_sigma, delta, n_correct_unw, *extra_unw)
            state = loop_unw(y, flags, MuTech.MEDIAN, SigmaTech.SIXTY_EIGHTH_PERCENTILE, delta, n_correct_unw, *extra_unw)
            state = loop_unw(y, flags, MuTech.MEAN, SigmaTech.STANDARD_DEVIATION, delta, n_correct_unw, *extra_unw)
        else:
            state = loop_wgt(w, y, flags, pass1_mu, pass1_sigma, delta, n_correct_wgt, *extra_wgt)
            state = loop_wgt(w, y, flags, MuTech.MEDIAN, SigmaTech.SIXTY_EIGHTH_PERCENTILE, delta, n_correct_wgt, *extra_wgt)
            state = loop_wgt(w, y, flags, MuTech.MEAN, SigmaTech.STANDARD_DEVIATION, delta, n_correct_wgt, *extra_wgt)
    except _RCRDegenerate:
        state = _degenerate_state(sigma_choice)

    indices_kept = np.where(flags)[0]
    indices_rejected = np.where(~flags)[0]

    out = {
        "flags": flags,
        "indices": indices_kept,
        "clean_y": y[indices_kept].copy(),
        "rejected_y": y[indices_rejected].copy(),
        "original_y": y.copy(),
        **state,
    }
    if w is not None:
        out["clean_w"] = w[indices_kept].copy()
    return out
