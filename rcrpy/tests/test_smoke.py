"""Smoke test: the package imports and exposes the expected public surface."""
from __future__ import annotations

import pytest

import rcrpy


def test_public_api_present():
    assert hasattr(rcrpy, "RCR")
    assert hasattr(rcrpy, "RCRResults")
    assert hasattr(rcrpy, "RejectionTech")

    tech = rcrpy.RejectionTech
    assert tech.SS_MEDIAN_DL.value == "SS_MEDIAN_DL"
    assert tech.LS_MODE_68.value == "LS_MODE_68"
    assert tech.LS_MODE_DL.value == "LS_MODE_DL"
    assert tech.ES_MODE_DL.value == "ES_MODE_DL"


def test_rcr_can_be_constructed():
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    assert r.rejection_tech is rcrpy.RejectionTech.LS_MODE_68
    assert isinstance(r.result, rcrpy.RCRResults)


def test_perform_rejection_runs_for_all_techs():
    """All four rejection techniques are now wired up."""
    data = [0.1, 0.2, 0.0, 0.3, -0.1, -0.2, -0.3, 11.0]
    for tech in (rcrpy.RejectionTech.LS_MODE_68, rcrpy.RejectionTech.LS_MODE_DL,
                 rcrpy.RejectionTech.SS_MEDIAN_DL, rcrpy.RejectionTech.ES_MODE_DL):
        r = rcrpy.RCR(tech)
        r.perform_rejection(data)
        assert r.result.flags.size == len(data)


def test_perform_bulk_rejection_runs_for_all_techs():
    """All four rejection techniques are wired up for bulk too."""
    data = [0.1, 0.2, 0.0, 0.3, -0.1, -0.2, -0.3, 11.0]
    for tech in (rcrpy.RejectionTech.LS_MODE_68, rcrpy.RejectionTech.LS_MODE_DL,
                 rcrpy.RejectionTech.SS_MEDIAN_DL, rcrpy.RejectionTech.ES_MODE_DL):
        r = rcrpy.RCR(tech)
        r.perform_bulk_rejection(data)
        assert r.result.flags.size == len(data)
