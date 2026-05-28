"""Parity: SS_MEDIAN_DL (single-sigma + median + double-line) vs legacy rcr.

This is the second rejection-tech "family" in Phase 1 — uses the single-sigma
iterative loop and fitDL with the getSingleFN model."""
from __future__ import annotations

import numpy as np
import pytest

import pyrcr

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _oracle_unw(y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.SS_MEDIAN_DL)
    r.performRejection(y_list)
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.stDev,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
    }


def _oracle_w(w_list: list[float], y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.SS_MEDIAN_DL)
    r.performRejection(w_list, y_list)
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.stDev,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
    }


def _port_unw(y: np.ndarray) -> dict:
    r = pyrcr.RCR(pyrcr.RejectionTech.SS_MEDIAN_DL)
    r.perform_rejection(y.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.st_dev,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
    }


def _port_w(w: np.ndarray, y: np.ndarray) -> dict:
    r = pyrcr.RCR(pyrcr.RejectionTech.SS_MEDIAN_DL)
    r.perform_rejection(y.tolist(), w=w.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.st_dev,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
    }


def _assert_parity(port: dict, oracle: dict, *, label: str) -> None:
    for key in ("mu", "sigma", "stDev"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL,
            err_msg=f"[{label}] {key}: port={port[key]!r} oracle={oracle[key]!r}",
        )
    assert port["flags"] == oracle["flags"], f"[{label}] flags disagree"
    assert port["indices"] == oracle["indices"], f"[{label}] indices disagree"
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL)


def test_ss_median_dl_parity_smoke(data_smoke):
    y = data_smoke["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="ss-smoke")


def test_ss_median_dl_parity_singlevalue(data_singlevalue):
    y = data_singlevalue["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="ss-singlevalue")


def test_ss_median_dl_parity_weighted(data_weighted_singlevalue):
    w = data_weighted_singlevalue["w"]
    y = data_weighted_singlevalue["y"]
    _assert_parity(_port_w(w, y), _oracle_w(w.tolist(), y.tolist()), label="ss-weighted")
