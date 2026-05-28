"""Parity sweep: rcr2 FunctionalForm vs the C++ oracle on cases where
rejection actually trips, so the MEDIAN / MODE mu_tech paths (and not
just the MEAN-equivalent regression()) get exercised in both.

These tests are LESS STRICT than the Phase 1 parity (rtol=1e-12). The
MEDIAN / MODE paths involve random sampling of M-combinations, and the
C++ uses `std::mt19937` while we use `numpy.random.default_rng` — the
sampled subsets differ. With ~20,000 combos sampled in both, the
weighted median over combo space converges to the same answer at the
level of robust statistical agreement, which we conservatively test at
rtol=5e-2 (5%).
"""
from __future__ import annotations

import numpy as np
import pytest

import rcr2

rcr_oracle = pytest.importorskip("rcr")

# Tolerance for MEDIAN/MODE path parameter agreement. The random-sampling
# difference between Python's RNG and C++'s std::mt19937 means we can't
# hit rtol=1e-12; 5% is what the robust-statistics convergence buys us.
RTOL_PARAMS = 5e-2


def linear(x, params):
    return params[0] + params[1] * x


def d_linear_b(x, params):
    return 1.0


def d_linear_m(x, params):
    return x


def _make_contam(N: int, frac_out: float, slope: float, intercept: float,
                 sigma_clean: float = 0.3, outlier_pull: float = 20.0,
                 seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, N)
    y = intercept + slope * x + rng.normal(0, sigma_clean, size=N)
    n_out = int(round(N * frac_out))
    if n_out > 0:
        out_idx = rng.choice(N, size=n_out, replace=False)
        # One-sided pull, biased positive so MEDIAN-based robust paths get
        # to demonstrate their immunity.
        y[out_idx] += rng.normal(outlier_pull, outlier_pull / 4.0, size=n_out)
    return x, y


def _port_fit(x, y, tech, weights=None):
    model = rcr2.FunctionalForm(
        linear, x, y, [d_linear_b, d_linear_m], guess=[0.0, 0.0],
        weights=weights,
    )
    r = rcr2.RCR(tech)
    r.set_parametric_model(model)
    if weights is None:
        r.perform_rejection(y.tolist())
    else:
        r.perform_rejection(y.tolist(), w=weights.tolist())
    return model.result.parameters, int(r.result.flags.sum())


def _oracle_fit(x, y, tech, weights=None):
    oc_tech = getattr(rcr_oracle, tech.name)
    model = rcr_oracle.FunctionalForm(
        linear, x.tolist(), y.tolist(),
        [d_linear_b, d_linear_m], [0.0, 0.0],
    )
    r = rcr_oracle.RCR(oc_tech)
    r.setParametricModel(model)
    if weights is None:
        r.performRejection(y.tolist())
    else:
        r.performRejection(weights.tolist(), y.tolist())
    return np.asarray(model.result.parameters), int(sum(r.result.flags))


# ---- the actual parity tests ----------------------------------------------

@pytest.mark.parametrize("frac_out,seed", [
    (0.0,  10),   # no contamination — MEDIAN == MEAN, sharp parity
    (0.10, 11),   # moderate
    (0.25, 12),   # heavy enough to trigger rejection
])
def test_lsmode68_parametric_parity(frac_out, seed):
    """LS_MODE_68 + linear FunctionalForm + performRejection.
    Each pass cycles MODE → MEDIAN → MEAN, so all three combo-space
    code paths run."""
    x, y = _make_contam(N=120, frac_out=frac_out, slope=1.5, intercept=2.0,
                         seed=seed)
    port_params, port_kept = _port_fit(x, y, rcr2.RejectionTech.LS_MODE_68)
    oracle_params, oracle_kept = _oracle_fit(x, y, rcr2.RejectionTech.LS_MODE_68)

    np.testing.assert_allclose(
        port_params, oracle_params, rtol=RTOL_PARAMS,
        err_msg=f"frac_out={frac_out}: port={port_params!r} oracle={oracle_params!r}",
    )
    # kept-count should agree to within ~5 points (rejection boundaries
    # depend on tie-breaks in the random combo sampling).
    assert abs(port_kept - oracle_kept) <= max(3, int(0.05 * x.size)), (
        f"frac_out={frac_out}: port kept {port_kept}, oracle kept {oracle_kept}"
    )


@pytest.mark.parametrize("tech_name", [
    "LS_MODE_DL",
    "SS_MEDIAN_DL",
])
def test_other_techs_parametric_parity(tech_name):
    """Same idea but on the other rejection techniques. ES_MODE_DL is
    intentionally omitted — both implementations show occasional
    sensitivity to the residual-split sigma sentinels for parametric+ES
    on heavily contaminated data, and parity at 5% isn't achievable
    cross-implementation without a controlled stress workload."""
    x, y = _make_contam(N=120, frac_out=0.15, slope=1.5, intercept=2.0,
                         seed=42)
    tech = getattr(rcr2.RejectionTech, tech_name)
    port_params, _ = _port_fit(x, y, tech)
    oracle_params, _ = _oracle_fit(x, y, tech)
    np.testing.assert_allclose(
        port_params, oracle_params, rtol=RTOL_PARAMS,
        err_msg=f"{tech_name}: port={port_params!r} oracle={oracle_params!r}",
    )


def test_weighted_parametric_parity():
    """Weighted RCR + parametric + LS_MODE_68. The weight vector down-
    weights the outliers, so the underlying combo weights also shift."""
    rng = np.random.default_rng(7)
    N = 100
    x = np.linspace(-3, 3, N)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.2, size=N)
    out = rng.choice(N, size=20, replace=False)
    y[out] += rng.normal(15, 3, size=20)
    w = np.ones(N)
    w[out] = 0.05  # outliers near-zero-weighted

    port_params, _ = _port_fit(x, y, rcr2.RejectionTech.LS_MODE_68, weights=w)
    oracle_params, _ = _oracle_fit(x, y, rcr2.RejectionTech.LS_MODE_68, weights=w)
    np.testing.assert_allclose(
        port_params, oracle_params, rtol=RTOL_PARAMS,
        err_msg=f"weighted: port={port_params!r} oracle={oracle_params!r}",
    )
