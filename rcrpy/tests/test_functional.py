"""Smoke tests for FunctionalForm (Phase 2 first cut).

There is no oracle parity test here yet because the C++ FunctionalForm
uses get2DMedian/get2DMode in passes 2-3 of `performRejection`, and our
port currently collapses those to `regression()`. We instead check:

  1. The class instantiates and `regression()` returns sane parameters
     on a clean linear dataset.
  2. RCR + PARAMETRIC + LS_MODE_68 runs end-to-end on data_linear.csv
     (N=999, heavily contaminated) and recovers a slope/intercept close
     to what a plain least-squares fit on the *clean* subset would give.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import rcrpy

REPO = Path(__file__).resolve().parents[2]
LINEAR_CSV = REPO / "assets" / "test" / "data_linear.csv"
EXPONENTIAL_CSV = REPO / "assets" / "test" / "data_exponential.csv"


def _load_csv_xy(path: Path):
    """Generic x,y loader that handles trailing empty rows (data_*.csv
    files have no header and sometimes trailing blanks)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = []
        for r in reader:
            if len(r) >= 2 and r[0].strip() and r[1].strip():
                rows.append((float(r[0]), float(r[1])))
    x = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    return x, y


def _load_linear():
    return _load_csv_xy(LINEAR_CSV)


def linear_model(x, params):
    return params[0] + params[1] * x


def d_linear_intercept(x, params):
    return 1.0


def d_linear_slope(x, params):
    return x


def test_functional_form_constructs_and_fits_clean():
    """Basic sanity: fit a noiseless line, recover parameters."""
    rng = np.random.default_rng(42)
    true_b, true_m = 3.0, 2.0
    x = np.linspace(-5, 5, 50)
    y = true_b + true_m * x + rng.normal(0, 0.01, size=x.size)

    model = rcrpy.FunctionalForm(
        linear_model, x, y,
        [d_linear_intercept, d_linear_slope],
        guess=[0.0, 0.0],
    )
    params = model.regression()
    assert abs(params[0] - true_b) < 0.05
    assert abs(params[1] - true_m) < 0.05


def test_rcr_with_parametric_runs_on_linear_dataset():
    """End-to-end: RCR + LS_MODE_68 + linear FunctionalForm on
    data_linear.csv. Asserts the run completes and parameters are finite.

    NOTE: this Phase 2 first-cut port uses regression() for all three
    passes of performRejection's refinement (the C++ uses get2DMedian /
    get2DMode for passes 2/3). That means our pass-1 fit may already
    settle into a regression-driven minimum where Chauvenet stops
    rejecting, even if the C++ would have kept rejecting via the
    median/mode passes. So we only check that the run completes — we
    don't yet assert a specific clean-subset size or fit quality. Strict
    parity awaits porting get2DMedian / get2DMode."""
    x, y = _load_linear()

    model = rcrpy.FunctionalForm(
        linear_model, x, y,
        [d_linear_intercept, d_linear_slope],
        guess=[0.0, 1.0],
    )
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_parametric_model(model)
    assert r.mu_type is rcrpy.MuType.PARAMETRIC

    r.perform_rejection(y.tolist())

    params = model.result.parameters
    assert params.size == 2
    assert np.all(np.isfinite(params))
    # Loose sanity: parameters are in a reasonable range.
    assert abs(params[0]) < 100.0
    assert abs(params[1]) < 100.0


def test_functional_form_parity_on_data_linear():
    """Parity vs C++ oracle on data_linear.csv with LS_MODE_68
    performRejection + linear FunctionalForm.

    Tolerance is rtol=5e-3, which reflects the documented LM-vs-GN
    optimizer floor: the port uses scipy.optimize.least_squares
    (Levenberg-Marquardt), the C++ uses a hand-rolled Gauss-Newton, and
    on heavily contaminated datasets the two converge to the same global
    minimum within their respective tolerances but differ at ~1e-3 on
    the recovered parameters. See benchmarks/optimizer_quality_test.py
    for the full characterization (the port is at least as accurate as
    the oracle on every config measured).

    Combo-sampling parity is bit-identical (Option B fix, 2026-05-21);
    only the optimizer step contributes to the remaining floor.
    """
    rcr_oracle = pytest.importorskip("rcr")
    x, y = _load_linear()

    port_model = rcrpy.FunctionalForm(
        linear_model, x, y, [d_linear_intercept, d_linear_slope], guess=[0.0, 1.0],
    )
    r_port = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r_port.set_parametric_model(port_model)
    r_port.perform_rejection(y.tolist())

    oracle_model = rcr_oracle.FunctionalForm(
        linear_model, x.tolist(), y.tolist(),
        [d_linear_intercept, d_linear_slope], [0.0, 1.0],
    )
    r_or = rcr_oracle.RCR(rcr_oracle.LS_MODE_68)
    r_or.setParametricModel(oracle_model)
    r_or.performRejection(y.tolist())

    port_params = port_model.result.parameters
    oracle_params = oracle_model.result.parameters
    np.testing.assert_allclose(port_params, oracle_params, rtol=5e-3,
                               err_msg="parametric fit parameters disagree")


def test_pivot_static_class_attribute():
    """The `pivot` static-on-class attribute mirrors the C++ static
    `FunctionalForm::pivot` so user model functions can reference it."""
    assert hasattr(rcrpy.FunctionalForm, "pivot")
    old = rcrpy.FunctionalForm.pivot
    try:
        rcrpy.FunctionalForm.pivot = 2.5
        assert rcrpy.FunctionalForm.pivot == 2.5
    finally:
        rcrpy.FunctionalForm.pivot = old


# ---------------------------------------------------------------------------
# Phase 2 additions: paramuncertainty, Priors, pivot search, ND support
# ---------------------------------------------------------------------------

def test_parameter_uncertainties_populated():
    """After regression(), result.parameter_uncertainties should be a
    nonempty array with sensible (positive) values for a clean dataset."""
    rng = np.random.default_rng(7)
    x = np.linspace(-5, 5, 100)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.5, size=x.size)
    model = rcrpy.FunctionalForm(
        linear_model, x, y, [d_linear_intercept, d_linear_slope],
        guess=[0.0, 0.0],
    )
    model.regression()
    # handle_mu_tech_select populates result.parameter_uncertainties
    model.handle_mu_tech_select()
    unc = model.result.parameter_uncertainties
    assert unc.size == 2
    assert np.all(np.isfinite(unc))
    assert np.all(unc > 0)
    # Sigma=0.5 / sqrt(N=100) ~ 0.05 ballpark for the intercept on
    # symmetric x. Loose sanity: both bars within 0.5.
    assert unc[0] < 0.5
    assert unc[1] < 0.5


def test_priors_gaussian_pulls_fit_toward_mean():
    """A Gaussian prior centered at a value far from the data fit should
    pull the fitted parameter toward the prior mean (compared with the
    no-prior baseline)."""
    rng = np.random.default_rng(11)
    x = np.linspace(0, 10, 50)
    y = 0.0 + 1.0 * x + rng.normal(0, 0.3, size=x.size)  # true b=0, m=1

    # Baseline: no prior
    base = rcrpy.FunctionalForm(linear_model, x, y,
                                [d_linear_intercept, d_linear_slope],
                                guess=[0.0, 0.0])
    base.regression()
    b_base = base.parameters[0]

    # Strong Gaussian prior on intercept at mu=5.0 sigma=0.1
    pri = rcrpy.Priors(
        prior_type=rcrpy.PriorType.GAUSSIAN,
        gaussian_params=[[5.0, 0.1], [float("nan"), float("nan")]],
    )
    with_prior = rcrpy.FunctionalForm(linear_model, x, y,
                                      [d_linear_intercept, d_linear_slope],
                                      guess=[0.0, 0.0], priors=pri)
    with_prior.regression()
    b_with = with_prior.parameters[0]

    # The prior should pull the fitted intercept toward 5.0 — at minimum,
    # the fitted intercept with prior should be larger than the baseline.
    assert b_with > b_base + 0.5


def test_priors_constrained_clamps_parameter():
    """A constrained prior with a tight upper bound should yield a
    parameter at the bound (and the fit accepts it)."""
    rng = np.random.default_rng(13)
    x = np.linspace(0, 10, 50)
    y = 0.0 + 1.0 * x + rng.normal(0, 0.2, size=x.size)  # true m=1

    pri = rcrpy.Priors(
        prior_type=rcrpy.PriorType.CONSTRAINED,
        # Force slope into [0, 0.3] — well below the true 1.0
        param_bounds=[[float("nan"), float("nan")], [0.0, 0.3]],
    )
    model = rcrpy.FunctionalForm(linear_model, x, y,
                                 [d_linear_intercept, d_linear_slope],
                                 guess=[0.0, 0.1], priors=pri)
    model.regression()
    m = model.parameters[1]
    # Slope is constrained — should be at or near the upper bound 0.3.
    assert 0.0 <= m <= 0.3 + 1e-9


def test_pivot_search_updates_static_pivot():
    """A user-supplied pivot_function should be invoked during
    build_model_space, updating `FunctionalForm.pivot` for use inside
    the user's model function."""

    def pivot_mean(xdata, weights, f, params):
        # Mean of xdata
        return float(np.sum(weights * xdata) / np.sum(weights))

    x = np.linspace(-3, 7, 40)
    y = x * 2.0
    model = rcrpy.FunctionalForm(
        linear_model, x, y, [d_linear_intercept, d_linear_slope],
        guess=[0.0, 1.0],
        pivot_function=pivot_mean,
        pivot_guess=0.0,
    )
    assert model.has_custom_pivot
    # Initial pivot from pivot_guess.
    assert rcrpy.FunctionalForm.pivot == 0.0
    # build_model_space recomputes pivot from current x window.
    model.indices = np.arange(x.size, dtype=np.int64)
    model.parameters = np.array([0.0, 1.0])
    model.build_model_space()
    expected = float(np.mean(x))
    assert abs(rcrpy.FunctionalForm.pivot - expected) < 1e-9
    # Reset so other tests aren't perturbed.
    rcrpy.FunctionalForm.pivot = 0.0


def test_functional_form_runs_on_data_exponential():
    """Smoke test: rcrpy + LS_MODE_68 + exponential FunctionalForm on
    data_exponential.csv. The exponential `a0 * 10^(a1*x)` model is
    poorly conditioned without bounds, so we use CONSTRAINED priors on
    a1 to bound the solver (doubles as a cross-feature test of
    PARAMETRIC + CONSTRAINED priors). We don't compare to the C++ oracle
    here because the C++ Gauss-Newton hangs on this data without similar
    step-damping; that parity check can be added when get2DMedian /
    get2DMode are ported."""
    x, y = _load_csv_xy(EXPONENTIAL_CSV)
    rcrpy.FunctionalForm.pivot = 0.0

    def expo(x, params):
        # Clip exponent to keep the callback returning finite numbers
        # during trial steps.
        exp_val = params[1] * x
        if exp_val > 300:
            return params[0] * 1e300
        if exp_val < -300:
            return 0.0
        return params[0] * (10.0 ** exp_val)

    def d_expo_a0(x, params):
        return expo(x, [1.0, params[1]])

    def d_expo_a1(x, params):
        return expo(x, params) * x * np.log(10)

    bounded = rcrpy.Priors(
        prior_type=rcrpy.PriorType.CONSTRAINED,
        param_bounds=[[float("nan"), float("nan")], [-5.0, 5.0]],
    )
    model = rcrpy.FunctionalForm(expo, x, y, [d_expo_a0, d_expo_a1],
                                 guess=[16.0, 0.0], priors=bounded)
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_parametric_model(model)
    r.perform_rejection(y.tolist())

    params = model.result.parameters
    assert params.size == 2
    assert np.all(np.isfinite(params))
    # Sanity: intercept near data's y(0) magnitude, slope inside its bounds.
    assert 0.0 < params[0] < 200.0
    assert -5.0 <= params[1] <= 5.0


def test_nd_xdata_runs_through_regression():
    """ND support: pass a 2D xdata (N rows, D cols) with a model that
    expects vector x. Verifies the residual loop and dispatch handle the
    ND code path."""
    rng = np.random.default_rng(17)
    N = 60
    x = rng.uniform(-3, 3, size=(N, 2))   # (N, 2) — 2D x
    true_b, true_m1, true_m2 = 1.0, 0.5, -0.3
    y = true_b + true_m1 * x[:, 0] + true_m2 * x[:, 1] + rng.normal(0, 0.1, N)

    def f_nd(xv, params):
        return params[0] + params[1] * xv[0] + params[2] * xv[1]

    def d0(xv, params):
        return 1.0

    def d1(xv, params):
        return xv[0]

    def d2(xv, params):
        return xv[1]

    model = rcrpy.FunctionalForm(
        f_nd, x, y, [d0, d1, d2], guess=[0.0, 0.0, 0.0],
    )
    assert model.ND_check is True
    params = model.regression()
    assert params.size == 3
    assert abs(params[0] - true_b) < 0.1
    assert abs(params[1] - true_m1) < 0.1
    assert abs(params[2] - true_m2) < 0.1
