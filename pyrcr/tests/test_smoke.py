"""Smoke test: the package imports and exposes the expected public surface."""
from __future__ import annotations

import pytest

import pyrcr


def test_public_api_present():
    assert hasattr(pyrcr, "RCR")
    assert hasattr(pyrcr, "RCRResults")
    assert hasattr(pyrcr, "RejectionTech")

    tech = pyrcr.RejectionTech
    assert tech.SS_MEDIAN_DL.value == "SS_MEDIAN_DL"
    assert tech.LS_MODE_68.value == "LS_MODE_68"
    assert tech.LS_MODE_DL.value == "LS_MODE_DL"
    assert tech.ES_MODE_DL.value == "ES_MODE_DL"


def test_rcr_can_be_constructed():
    r = pyrcr.RCR(pyrcr.RejectionTech.LS_MODE_68)
    assert r.rejection_tech is pyrcr.RejectionTech.LS_MODE_68
    assert isinstance(r.result, pyrcr.RCRResults)


def test_perform_rejection_runs_for_all_techs():
    """All four rejection techniques are now wired up."""
    data = [0.1, 0.2, 0.0, 0.3, -0.1, -0.2, -0.3, 11.0]
    for tech in (pyrcr.RejectionTech.LS_MODE_68, pyrcr.RejectionTech.LS_MODE_DL,
                 pyrcr.RejectionTech.SS_MEDIAN_DL, pyrcr.RejectionTech.ES_MODE_DL):
        r = pyrcr.RCR(tech)
        r.perform_rejection(data)
        assert r.result.flags.size == len(data)


def test_perform_bulk_rejection_runs_for_all_techs():
    """All four rejection techniques are wired up for bulk too."""
    data = [0.1, 0.2, 0.0, 0.3, -0.1, -0.2, -0.3, 11.0]
    for tech in (pyrcr.RejectionTech.LS_MODE_68, pyrcr.RejectionTech.LS_MODE_DL,
                 pyrcr.RejectionTech.SS_MEDIAN_DL, pyrcr.RejectionTech.ES_MODE_DL):
        r = pyrcr.RCR(tech)
        r.perform_bulk_rejection(data)
        assert r.result.flags.size == len(data)
