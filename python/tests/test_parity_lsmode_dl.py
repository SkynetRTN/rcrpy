"""Parity: LS_MODE_DL (lower-sigma + mode + double-line) vs legacy rcr."""
from __future__ import annotations

import numpy as np
import pytest

import rcr2

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _oracle_unw(y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_DL)
    r.performRejection(y_list)
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.stDev,
        "stDevBelow": r.result.stDevBelow,
        "stDevAbove": r.result.stDevAbove,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
    }


def _oracle_w(w_list: list[float], y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_DL)
    r.performRejection(w_list, y_list)
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.stDev,
        "stDevBelow": r.result.stDevBelow,
        "stDevAbove": r.result.stDevAbove,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
    }


def _port_unw(y: np.ndarray) -> dict:
    r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_DL)
    r.perform_rejection(y.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.st_dev,
        "stDevBelow": r.result.st_dev_below,
        "stDevAbove": r.result.st_dev_above,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
    }


def _port_w(w: np.ndarray, y: np.ndarray) -> dict:
    r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_DL)
    r.perform_rejection(y.tolist(), w=w.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.st_dev,
        "stDevBelow": r.result.st_dev_below,
        "stDevAbove": r.result.st_dev_above,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
    }


def _assert_parity(port: dict, oracle: dict, *, label: str) -> None:
    for key in ("mu", "sigma", "stDev", "stDevBelow", "stDevAbove"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL,
            err_msg=f"[{label}] {key}: port={port[key]!r} oracle={oracle[key]!r}",
        )
    assert port["flags"] == oracle["flags"], f"[{label}] flags disagree"
    assert port["indices"] == oracle["indices"], f"[{label}] indices disagree"
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL)


def test_lsmode_dl_parity_smoke(data_smoke):
    y = data_smoke["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="dl-smoke")


def test_lsmode_dl_parity_singlevalue(data_singlevalue):
    y = data_singlevalue["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="dl-singlevalue")


def test_lsmode_dl_parity_weighted(data_weighted_singlevalue):
    w = data_weighted_singlevalue["w"]
    y = data_weighted_singlevalue["y"]
    _assert_parity(_port_w(w, y), _oracle_w(w.tolist(), y.tolist()), label="dl-weighted")
