"""Regression tests for the no-spread (sigma -> 0) degenerate case.

When the scatter-vs-scan data is near-constant or perfectly linear, the
parametric fit leaves residuals with no spread, so RCR's lower-sigma /
single-sigma loops compute sigma == 0. The C++ oracle gets `max / 0` for
free from IEEE float division (+inf / NaN) and its reject() short-circuit
stops with nothing rejected -- the fit reduces to a plain (weighted)
least-squares line. The Python port used to raise ZeroDivisionError on the
scalar divide; rcrpy.rejection._reject_ratio now floors sigma so the
behavior matches the oracle.

These cases are *deterministic* (no rejection happens, so the random
combo-sampling MEDIAN/MODE paths never diverge), hence we hold the port to
the tight Phase-1 parity tolerance against the oracle, not the looser 5%
used by the random-sampling sweep tests.
"""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def linear(x, params):
    return params[0] + params[1] * x


def d_linear_b(x, params):
    return 1.0


def d_linear_m(x, params):
    return x


def _port_fit(x, y, weights=None):
    model = rcrpy.FunctionalForm(
        linear, x, y, [d_linear_b, d_linear_m], guess=[0.0, 0.0],
        weights=weights,
    )
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_parametric_model(model)
    if weights is None:
        r.perform_rejection(y.tolist())
    else:
        r.perform_rejection(y.tolist(), w=weights.tolist())
    return np.asarray(model.result.parameters), int(r.result.flags.sum())


def _oracle_fit(x, y, weights=None):
    model = rcr_oracle.FunctionalForm(
        linear, x.tolist(), y.tolist(), [d_linear_b, d_linear_m], [0.0, 0.0],
    )
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_68)
    r.setParametricModel(model)
    if weights is None:
        r.performRejection(y.tolist())
    else:
        r.performRejection(weights.tolist(), y.tolist())
    return np.asarray(model.result.parameters), int(sum(r.result.flags))


def _wls_line(x, y, w=None):
    """Plain (weighted) least-squares intercept+slope, the answer RCR must
    fall back to when there is nothing to reject."""
    if w is None:
        w = np.ones_like(x)
    W = np.diag(w)
    A = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    return beta  # [intercept, slope]


# x grid shared by the degenerate cases.
X = np.linspace(0.0, 10.0, 40)


@pytest.mark.parametrize("label,y", [
    ("perfectly_linear", 2.0 + 0.5 * X),
    ("constant_nonzero", np.full_like(X, 7.0)),
    ("constant_zero", np.zeros_like(X)),
    ("perfectly_linear_negative_slope", 3.0 - 0.25 * X),
])
def test_no_spread_does_not_raise_and_matches_oracle(label, y):
    """The crash repro: near-constant / perfectly-linear data with sigma=0.
    Must (a) not raise, (b) keep every point, (c) match the C++ oracle, and
    (d) equal the plain least-squares line."""
    port_params, port_kept = _port_fit(X, y)        # must not raise
    oracle_params, oracle_kept = _oracle_fit(X, y)

    assert port_kept == X.size, f"{label}: expected all points kept"
    assert port_kept == oracle_kept
    np.testing.assert_allclose(
        port_params, oracle_params, rtol=RTOL,
        err_msg=f"{label}: port={port_params!r} oracle={oracle_params!r}",
    )
    np.testing.assert_allclose(
        port_params, _wls_line(X, y), rtol=1e-9, atol=1e-12,
        err_msg=f"{label}: not the least-squares line",
    )


def test_weighted_no_spread_matches_oracle():
    """Weighted twin: non-uniform weights, still no spread to reject on.
    Verifies the weighted lower-sigma loop (iterativeLowerSigmaRCR_w) is
    guarded too, and that the fallback is the *weighted* LS line."""
    y = 1.0 - 0.4 * X
    w = np.linspace(0.5, 2.0, X.size)

    port_params, port_kept = _port_fit(X, y, weights=w)   # must not raise
    oracle_params, oracle_kept = _oracle_fit(X, y, weights=w)

    assert port_kept == X.size
    assert port_kept == oracle_kept
    np.testing.assert_allclose(port_params, oracle_params, rtol=RTOL)
    np.testing.assert_allclose(
        port_params, _wls_line(X, y, w), rtol=1e-9, atol=1e-12,
    )


@pytest.mark.parametrize("tech_name", [
    "LS_MODE_68", "LS_MODE_DL", "SS_MEDIAN_DL", "ES_MODE_DL",
])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("val", [0.0, 0.18432086, -42.5, 1e6])
def test_single_point_n1(tech_name, weighted, bulk, val):
    """N=1 (a single data point) has zero residual spread, so the variance
    estimators (getStDev/getStDev_w) and weighted CFs (getCFRatio/getFNRatio)
    all divide by zero. C++ gets NaN/inf and keeps the point; the port used to
    raise ZeroDivisionError. Found by the fuzzer (r_tiny_n, N=1)."""
    y = np.array([val])
    w = np.array([2.5]) if weighted else None

    rp = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech_name))
    ro = rcr_oracle.RCR(getattr(rcr_oracle, tech_name))
    if bulk:
        rp.perform_bulk_rejection(y.tolist(), w=(None if w is None else w.tolist()))  # must not raise
        ro.performBulkRejection(y.tolist()) if w is None else ro.performBulkRejection(w.tolist(), y.tolist())
    else:
        rp.perform_rejection(y.tolist(), w=(None if w is None else w.tolist()))       # must not raise
        ro.performRejection(y.tolist()) if w is None else ro.performRejection(w.tolist(), y.tolist())

    assert int(np.sum(rp.result.flags)) == int(sum(ro.result.flags)) == 1
    np.testing.assert_allclose(rp.result.mu, ro.result.mu, rtol=1e-12)


@pytest.mark.parametrize("tech_name", [
    "LS_MODE_68", "LS_MODE_DL", "SS_MEDIAN_DL", "ES_MODE_DL",
])
def test_all_techniques_survive_no_spread(tech_name):
    """Every rejection technique must survive perfectly-linear data without
    raising -- guards the single-sigma (SS) and DL loops in addition to the
    lower-sigma (LS) ones."""
    y = 2.0 + 0.5 * X
    model = rcrpy.FunctionalForm(
        linear, X, y, [d_linear_b, d_linear_m], guess=[0.0, 0.0],
    )
    r = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech_name))
    r.set_parametric_model(model)
    r.perform_rejection(y.tolist())   # must not raise
    params = np.asarray(model.result.parameters)
    assert np.all(np.isfinite(params))
    np.testing.assert_allclose(params, [2.0, 0.5], rtol=1e-6, atol=1e-9)
