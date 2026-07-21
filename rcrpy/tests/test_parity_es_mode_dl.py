"""Parity: ES_MODE_DL (each-sigma + mode + double-line) vs legacy rcr.

This is the final rejection-tech "family". Unlike SS/LS, the each-sigma loop
keeps sigmaBelow / sigmaAbove distinct (no min) and uses them independently
in the rejection criterion. The result fields `sigma` and `stDev` are NOT
written by this loop in the C++, so we don't check them.
"""
from __future__ import annotations

import numpy as np
import pytest

import rcrpy

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _oracle_unw(y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.ES_MODE_DL)
    r.performRejection(y_list)
    return _collect(r)


def _oracle_w(w_list: list[float], y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.ES_MODE_DL)
    r.performRejection(w_list, y_list)
    return _collect(r)


def _collect(r) -> dict:
    return {
        "mu": r.result.mu,
        "stDevBelow": r.result.stDevBelow,
        "stDevAbove": r.result.stDevAbove,
        "sigmaBelow": r.result.sigmaBelow,
        "sigmaAbove": r.result.sigmaAbove,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
    }


def _port_unw(y: np.ndarray) -> dict:
    r = rcrpy.RCR(rcrpy.RejectionTech.ES_MODE_DL)
    r.perform_rejection(y.tolist())
    return _collect_port(r)


def _port_w(w: np.ndarray, y: np.ndarray) -> dict:
    r = rcrpy.RCR(rcrpy.RejectionTech.ES_MODE_DL)
    r.perform_rejection(y.tolist(), w=w.tolist())
    return _collect_port(r)


def _collect_port(r) -> dict:
    return {
        "mu": r.result.mu,
        "stDevBelow": r.result.st_dev_below,
        "stDevAbove": r.result.st_dev_above,
        "sigmaBelow": r.result.sigma_below,
        "sigmaAbove": r.result.sigma_above,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
    }


def _assert_parity(port: dict, oracle: dict, *, label: str) -> None:
    for key in ("mu", "stDevBelow", "stDevAbove", "sigmaBelow", "sigmaAbove"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL,
            err_msg=f"[{label}] {key}: port={port[key]!r} oracle={oracle[key]!r}",
        )
    assert port["flags"] == oracle["flags"], f"[{label}] flags disagree"
    assert port["indices"] == oracle["indices"], f"[{label}] indices disagree"
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL)


def test_es_mode_dl_parity_smoke(data_smoke):
    y = data_smoke["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="es-smoke")


def test_es_mode_dl_parity_singlevalue(data_singlevalue):
    y = data_singlevalue["y"]
    _assert_parity(_port_unw(y), _oracle_unw(y.tolist()), label="es-singlevalue")


def test_es_mode_dl_parity_weighted(data_weighted_singlevalue):
    w = data_weighted_singlevalue["w"]
    y = data_weighted_singlevalue["y"]
    _assert_parity(_port_w(w, y), _oracle_w(w.tolist(), y.tolist()), label="es-weighted")
