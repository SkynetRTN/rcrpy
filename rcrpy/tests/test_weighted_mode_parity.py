"""Regression tests for the weighted half-sample-mode parity bug.

Found by scripts/fuzz_oracle_parity.py (2026-06-04). `stats.halfSampleMode_w`
was vectorizing only the C++'s FORWARD branch of getMode(w, y); the BACKWARD
branch (upper-anchored windows) was dropped. That made the weighted mode
disagree with both the unweighted mode (even at uniform weights) and the C++
oracle, so the MODE pass of weighted LS_MODE_68 / LS_MODE_DL / ES_MODE_DL
picked a wrong mu and flipped rejection-boundary decisions, diverging from the
oracle in ~40% of random weighted cases. SS_MEDIAN_DL was unaffected (it never
calls the weighted mode).

Also covers the sibling crash: the weighted CF/FN models
(getLowerFN_w etc.) raised OverflowError on the `10**(10**z)` double-
exponential where the C++ pow() silently overflows to +inf; now routed
through rejection._pow10pow10.
"""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy
from rcrpy import stats

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-9


# ---------------------------------------------------------------------------
# 1. The localized invariant: weighted mode at uniform weights == unweighted
#    mode, and weighted mode == a faithful direct port of the C++ loop.
# ---------------------------------------------------------------------------
def _ref_getMode_w(w, y):
    """Direct, un-vectorized port of cpp/src/RCR.cpp:531 getMode(count,w,y).
    Both branches, sequential — the ground truth the vectorized version must
    reproduce."""
    def isEqual(a, b, rel=1e-8):
        if abs(a - b) < 2.2250738585072014e-308:
            return True
        d = max(abs(a), abs(b))
        return abs((a - b) / d) < rel if d != 0 else True

    n = len(y)
    lo, hi, loin, hiin, fl, fu = 0, n - 1, -1, -1, -1, -1
    while lo != loin or hi != hiin:
        loin, hiin = lo, hi
        size = hi - lo + 1
        minD = 999999.0
        hws = sum(w[lo:hi + 1]) * 0.5
        sVec = [0.0] * size
        sSum = 0.5 * w[lo]
        sVec[0] = sSum
        for i in range(lo + 1, lo + size):
            sSum += w[i - 1] * 0.5 + w[i] * 0.5
            sVec[i - lo] = sSum
        for i in range(size):
            if sVec[i] < hws or isEqual(sVec[i], hws):
                tot = sVec[i] + hws
                k = i
                while k < size and (sVec[k] < tot or isEqual(sVec[k], tot)):
                    k += 1
                k -= 1
                d = abs(y[k + lo] - y[i + lo])
                if isEqual(d, minD):
                    fl, fu = min(fl, i + lo), max(fu, k + lo)
                elif d < minD:
                    minD, fl, fu = d, i + lo, k + lo
            if sVec[i] > hws or isEqual(sVec[i], hws):
                tot = sVec[i] - hws
                k = i
                while k > -1 and (sVec[k] > tot or isEqual(sVec[k], tot)):
                    k -= 1
                k += 1
                d = abs(y[i + lo] - y[k + lo])
                if isEqual(d, minD):
                    fl, fu = min(fl, k + lo), max(fu, i + lo)
                elif d < minD:
                    minD, fl, fu = d, k + lo, i + lo
        lo, hi = fl, fu
    return stats.getMedian_w(np.asarray(w[lo:hi + 1]), np.asarray(y[lo:hi + 1]))


@pytest.mark.parametrize("seed", range(40))
def test_weighted_mode_reduces_to_unweighted_at_uniform_weights(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 50))
    y = np.sort(rng.normal(0, 3, n))
    assert stats.halfSampleMode_w(np.ones(n), y) == stats.halfSampleMode(y)


@pytest.mark.parametrize("seed", range(60))
def test_weighted_mode_matches_faithful_cpp_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 50))
    y = np.sort(rng.normal(0, 3, n))
    w = rng.uniform(0.1, 8.0, n)
    assert stats.halfSampleMode_w(w, y) == _ref_getMode_w(list(w), list(y))


# ---------------------------------------------------------------------------
# 2. End-to-end weighted parity vs the C++ oracle on seeds that previously
#    diverged (mode-using techniques + a control), uniform and random weights.
# ---------------------------------------------------------------------------
def _port(tech, y, w):
    r = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech))
    r.perform_rejection(y.tolist(), w=w.tolist())
    return int(np.sum(r.result.flags)), float(r.result.mu)


def _oracle(tech, y, w):
    r = rcr_oracle.RCR(getattr(rcr_oracle, tech))
    r.performRejection(w.tolist(), y.tolist())
    return int(sum(r.result.flags)), float(r.result.mu)


@pytest.mark.parametrize("tech", ["LS_MODE_68", "LS_MODE_DL", "SS_MEDIAN_DL"])
@pytest.mark.parametrize("wkind", ["ones", "random"])
@pytest.mark.parametrize("seed", [3, 27, 42, 99, 139, 200, 359])
def test_weighted_rejection_parity(tech, wkind, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 80))
    x = np.linspace(0, 10, n)
    y = rng.uniform(-5, 5) + rng.uniform(-3, 3) * x + rng.normal(0, rng.uniform(0.1, 3), n)
    w = np.ones(n) if wkind == "ones" else rng.uniform(0.1, 10, n)

    pk, pm = _port(tech, y, w)
    ok, om = _oracle(tech, y, w)
    assert pk == ok, f"{tech}/{wkind}/seed{seed}: kept port={pk} oracle={ok}"
    np.testing.assert_allclose(pm, om, rtol=RTOL,
                               err_msg=f"{tech}/{wkind}/seed{seed}: mu")


# ---------------------------------------------------------------------------
# 3. The sibling overflow crash: weights that drove getLowerFN_w's
#    10**(10**z) past the double range must no longer raise (C++ -> +inf).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(30))
def test_weighted_cf_overflow_does_not_raise(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 80))
    x = np.linspace(0, 10, n)
    y = rng.uniform(-5, 5) + rng.uniform(-3, 3) * x + rng.normal(0, rng.uniform(0.1, 3), n)
    w = 1.0 + rng.normal(0, 1e-6, n)   # near-unit weights, tiny coeff-of-variation
    pk, pm = _port("LS_MODE_68", y, w)       # must not raise OverflowError
    ok, om = _oracle("LS_MODE_68", y, w)
    assert pk == ok
    np.testing.assert_allclose(pm, om, rtol=RTOL)
