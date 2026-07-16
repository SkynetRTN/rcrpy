"""Smoke tests for the NonParametric API.

There is no oracle test data for non-parametric RCR in `assets/test/`, so
we verify only that:
  1. The default `NonParametric` base class behaves like muType=VALUE
     (passes through all flagged points) — so the parity on data_smoke
     must match the standard LS_MODE_68 result.
  2. A user subclass can override `mu_func` to filter candidates, and
     the loop respects that filter (rejects + final result differ).
"""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy


def test_default_nonparametric_matches_value_path(data_smoke):
    """No-op NonParametric subclass should give the same result as omitting
    it entirely, because the base-class `mu_func` returns all flagged
    points unchanged."""
    y = data_smoke["y"]

    # Baseline: standard MuType.VALUE path.
    r_baseline = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r_baseline.perform_rejection(y.tolist())

    # With default NonParametric (no override) attached.
    r_np = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r_np.set_non_parametric_model(rcrpy.NonParametric())
    assert r_np.mu_type is rcrpy.MuType.NONPARAMETRIC
    r_np.perform_rejection(y.tolist())

    np.testing.assert_allclose(r_np.result.mu, r_baseline.result.mu, rtol=1e-12)
    np.testing.assert_allclose(r_np.result.sigma, r_baseline.result.sigma, rtol=1e-12)
    assert list(r_np.result.flags) == list(r_baseline.result.flags)


def test_custom_nonparametric_subclass_changes_outcome(data_smoke):
    """A subclass that drops half the candidate points should produce a
    measurably different mu than the baseline — proves the model is
    actually being called by the rejection loop."""
    y = data_smoke["y"]

    class DropEverySecond(rcrpy.NonParametric):
        call_count = 0

        def mu_func(self, flags, y):
            type(self).call_count += 1
            idx_all = np.where(flags)[0]
            keep = idx_all[::2]
            self.indices = keep.astype(np.int64)
            return self.indices, y[self.indices]

    model = DropEverySecond()
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.set_non_parametric_model(model)
    r.perform_rejection(y.tolist())

    # Model was actually called by the loop (each iteration of each pass).
    assert DropEverySecond.call_count > 0, "mu_func was never invoked"

    # Baseline for comparison.
    r_base = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r_base.perform_rejection(y.tolist())
    # Mu should differ because the model fed different candidate points
    # into the mu calculation.
    assert r.result.mu != r_base.result.mu, "model had no effect"


def test_setters_are_independent():
    """`set_mu_type` toggles the dispatch; `set_non_parametric_model`
    sets the model AND switches mu_type to NONPARAMETRIC in one call."""
    r = rcrpy.RCR()
    assert r.mu_type is rcrpy.MuType.VALUE

    r.set_mu_type(rcrpy.MuType.NONPARAMETRIC)
    assert r.mu_type is rcrpy.MuType.NONPARAMETRIC

    r.set_mu_type(rcrpy.MuType.VALUE)
    assert r.non_parametric_model is None

    r.set_non_parametric_model(rcrpy.NonParametric())
    assert r.mu_type is rcrpy.MuType.NONPARAMETRIC
    assert r.non_parametric_model is not None
