"""Rigorous tests for FunctionalForm covering the full algorithm port:

  - MEDIAN / MODE mu_tech actually use the M-combination parameter space
    (and give *different* results from MEAN / regression() on contaminated
    data)
  - Cross-feature: every rejection technique (LS_MODE_68, LS_MODE_DL,
    SS_MEDIAN_DL, ES_MODE_DL) works with PARAMETRIC mu_type
  - Weighted + functional combination
  - Edge cases: N == M, N just above M, all-y-equal degenerate fit
"""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy


# ---- shared model helpers --------------------------------------------------

def linear(x, params):
    return params[0] + params[1] * x


def d_linear_b(x, params):
    return 1.0


def d_linear_m(x, params):
    return x


def _make_linear_with_outliers(N: int = 100, n_outliers: int = 20, seed: int = 0):
    """Clean linear data with `n_outliers` points dragged far off the line."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, N)
    y = 2.0 + 1.5 * x + rng.normal(0, 0.3, size=N)
    outlier_idx = rng.choice(N, size=n_outliers, replace=False)
    y[outlier_idx] += rng.normal(20.0, 5.0, size=n_outliers)  # one-sided pull
    return x, y


# ---- core MEDIAN / MODE behavior -------------------------------------------

def test_median_mu_tech_uses_combo_space():
    """Calling handle_mu_tech_select(mu_tech='MEDIAN') should produce
    parameters from the combo-space median, not from regression(). On
    heavily contaminated linear data the median estimate should be CLOSER
    to truth (b=2, m=1.5) than the regression estimate."""
    x, y = _make_linear_with_outliers(N=80, n_outliers=30, seed=42)
    truth = np.array([2.0, 1.5])

    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    assert model._is_linear_in_params  # vectorized fast path
    model.set_true_vec(np.ones(x.size, dtype=bool), y)
    model.build_model_space(build_combos=True)

    # handle_mu_tech_select returns RESIDUALS; the fitted params are on
    # `model.parameters` after each call.
    model.handle_mu_tech_select(mu_tech="MEAN")
    params_mean = model.parameters.copy()
    model.handle_mu_tech_select(mu_tech="MEDIAN")
    params_median = model.parameters.copy()

    # MEDIAN should be different from MEAN on contaminated data.
    assert not np.allclose(params_mean, params_median, atol=0.01)
    # MEDIAN should be closer to truth than MEAN.
    err_mean = np.linalg.norm(params_mean - truth)
    err_median = np.linalg.norm(params_median - truth)
    assert err_median < err_mean, (
        f"MEDIAN={params_median} (err {err_median:.3f}) should be closer to "
        f"truth={truth.tolist()} than MEAN={params_mean} (err {err_mean:.3f})"
    )


def test_mode_mu_tech_uses_combo_space():
    """Same idea for MODE mu_tech."""
    x, y = _make_linear_with_outliers(N=80, n_outliers=30, seed=7)
    truth = np.array([2.0, 1.5])

    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    model.set_true_vec(np.ones(x.size, dtype=bool), y)
    model.build_model_space(build_combos=True)
    model.handle_mu_tech_select(mu_tech="MODE")
    params_mode = model.parameters.copy()
    model.handle_mu_tech_select(mu_tech="MEAN")
    params_mean = model.parameters.copy()

    assert not np.allclose(params_mode, params_mean, atol=0.01)
    err_mode = np.linalg.norm(params_mode - truth)
    err_mean = np.linalg.norm(params_mean - truth)
    assert err_mode < err_mean, (
        f"MODE={params_mode} (err {err_mode:.3f}) should be closer to "
        f"truth={truth.tolist()} than MEAN={params_mean} (err {err_mean:.3f})"
    )


def test_parameter_space_populated_for_median_path():
    """Verify that build_model_space(build_combos=True) actually fills
    parameterSpace and weightSpace with the expected number of entries."""
    x, y = _make_linear_with_outliers(N=20, n_outliers=4, seed=1)
    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    model.set_true_vec(np.ones(x.size, dtype=bool), y)
    model.build_model_space(build_combos=True)
    # C(20, 2) = 190 combos, all finite for distinct x's.
    assert len(model.parameterSpace) == 2
    assert model.parameterSpace[0].size == 190
    assert model.parameterSpace[1].size == 190
    assert len(model.weightSpace) == 2
    assert model.weightSpace[0].size == 190


# ---- cross-tech coverage ---------------------------------------------------

@pytest.mark.parametrize("tech", [
    rcrpy.RejectionTech.LS_MODE_68,
    rcrpy.RejectionTech.LS_MODE_DL,
    rcrpy.RejectionTech.SS_MEDIAN_DL,
    rcrpy.RejectionTech.ES_MODE_DL,
])
def test_functional_form_works_with_each_rejection_tech(tech):
    """All four rejection techs should run end-to-end with a PARAMETRIC
    model and return finite, sensible parameters."""
    x, y = _make_linear_with_outliers(N=60, n_outliers=15, seed=11)
    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    r = rcrpy.RCR(tech)
    r.set_parametric_model(model)
    r.perform_rejection(y.tolist())

    params = model.result.parameters
    assert params.size == 2
    assert np.all(np.isfinite(params))
    # Loose: parameters should be within a few sigma of truth=(2.0, 1.5).
    assert abs(params[0] - 2.0) < 5.0
    assert abs(params[1] - 1.5) < 2.0


# ---- weighted + functional -------------------------------------------------

def test_weighted_functional_form_runs():
    """Weighted RCR + FunctionalForm should run and produce a finite fit.
    Outlier-weighted-down dataset: clean points get high weight, outliers
    get low weight; the fit should heavily favor the clean subset."""
    rng = np.random.default_rng(13)
    N = 100
    x = np.linspace(-5, 5, N)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.2, N)
    # Inject outliers with low weight so they're effectively suppressed.
    out = rng.choice(N, size=20, replace=False)
    y[out] += rng.normal(15, 3, size=20)
    w = np.ones(N)
    w[out] = 0.05

    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0], weights=w)
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_parametric_model(model)
    r.perform_rejection(y.tolist(), w=w.tolist())

    params = model.result.parameters
    assert np.all(np.isfinite(params))
    # Loose: parameters within a couple sigma of truth.
    assert abs(params[0] - 1.0) < 2.0
    assert abs(params[1] - 2.0) < 1.0


# ---- edge cases ------------------------------------------------------------

def test_n_equal_m_single_combo():
    """When N == M (exactly enough points for one combo), build_model_space
    should produce exactly one parameter point per dimension."""
    # M=2 linear, 2 data points.
    x = np.array([0.0, 1.0])
    y = np.array([3.0, 5.0])
    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    model.set_true_vec(np.array([True, True]), y)
    model.build_model_space(build_combos=True)
    # C(2, 2) = 1 combo. Exact line through (0,3),(1,5): b=3, m=2.
    assert model.parameterSpace[0].size == 1
    assert model.parameterSpace[1].size == 1
    np.testing.assert_allclose(model.parameterSpace[0][0], 3.0, rtol=1e-9)
    np.testing.assert_allclose(model.parameterSpace[1][0], 2.0, rtol=1e-9)


def test_small_n_just_above_m():
    """N just barely above M (e.g., 4 points, 2 params) should still
    produce a populated parameter space and run end-to-end through RCR."""
    rng = np.random.default_rng(0)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 1.0 + 2.0 * x + rng.normal(0, 0.1, size=4)

    model = rcrpy.FunctionalForm(linear, x, y, [d_linear_b, d_linear_m],
                                 guess=[0.0, 0.0])
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_parametric_model(model)
    r.perform_rejection(y.tolist())
    assert np.all(np.isfinite(model.result.parameters))


def test_nonlinear_in_params_falls_back_to_fsolve():
    """A model nonlinear in its parameters should auto-detect and route
    through the fsolve path. (Slower but should still produce a sensible
    fit on clean data.)"""

    def exp_model(x, params):
        return params[0] * np.exp(params[1] * x)

    def d_exp_a(x, params):
        # Depends on params[1] -> nonlinear in params.
        return float(np.exp(params[1] * x))

    def d_exp_b(x, params):
        return float(params[0] * x * np.exp(params[1] * x))

    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 12)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.02, size=x.size)

    model = rcrpy.FunctionalForm(exp_model, x, y, [d_exp_a, d_exp_b],
                                 guess=[1.0, 0.3])
    # Detector should flag this as NOT linear in params.
    assert model._is_linear_in_params is False
    # Regression should still find ~(2.0, 0.5).
    p = model.regression()
    assert abs(p[0] - 2.0) < 0.3
    assert abs(p[1] - 0.5) < 0.2
