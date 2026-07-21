"""Parity: weighted LS_MODE_68 (rcrpy vs legacy rcr) at rtol=1e-12 on
data_weighted_singlevalue.csv (N=200)."""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _run_oracle_w(w_list: list[float], y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_68)
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


def _run_port_w(w_arr: np.ndarray, y_arr: np.ndarray) -> dict:
    r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
    r.perform_rejection(y_arr.tolist(), w=w_arr.tolist())
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


def test_lsmode68_weighted_parity(data_weighted_singlevalue):
    w = data_weighted_singlevalue["w"]
    y = data_weighted_singlevalue["y"]
    oracle = _run_oracle_w(w.tolist(), y.tolist())
    port = _run_port_w(w, y)

    for key in ("mu", "sigma", "stDev", "stDevBelow", "stDevAbove"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL,
            err_msg=f"{key}: port={port[key]!r} oracle={oracle[key]!r}",
        )
    assert port["flags"] == oracle["flags"], "flags disagree"
    assert port["indices"] == oracle["indices"]
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL)
