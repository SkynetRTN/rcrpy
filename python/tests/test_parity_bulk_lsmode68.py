"""Parity: performBulkRejection for LS_MODE_68 vs legacy rcr.

Matches the smoke path in cpp/tests/test.py. This is the only invocation
that populates rejectedY / originalY / stDevTotal (via setFinalVectors).
"""
from __future__ import annotations

import numpy as np
import pytest

import rcr2

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _oracle(y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_68)
    r.performBulkRejection(y_list)
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDevBelow": r.result.stDevBelow,
        "stDevAbove": r.result.stDevAbove,
        "stDevTotal": r.result.stDevTotal,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.cleanY),
        "rejectedY": list(r.result.rejectedY),
        "originalY": list(r.result.originalY),
    }


def _port(y: np.ndarray) -> dict:
    r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
    r.perform_bulk_rejection(y.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDevBelow": r.result.st_dev_below,
        "stDevAbove": r.result.st_dev_above,
        "stDevTotal": r.result.st_dev_total,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
        "rejectedY": list(r.result.rejected_y),
        "originalY": list(r.result.original_y),
    }


def _assert_parity(port: dict, oracle: dict, *, label: str) -> None:
    for key in ("mu", "sigma", "stDevBelow", "stDevAbove", "stDevTotal"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL,
            err_msg=f"[{label}] {key}: port={port[key]!r} oracle={oracle[key]!r}",
        )
    assert port["flags"] == oracle["flags"], f"[{label}] flags disagree"
    assert port["indices"] == oracle["indices"]
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL)
    np.testing.assert_allclose(port["rejectedY"], oracle["rejectedY"], rtol=RTOL)
    np.testing.assert_allclose(port["originalY"], oracle["originalY"], rtol=RTOL)


def test_bulk_lsmode68_smoke(data_smoke):
    y = data_smoke["y"]
    _assert_parity(_port(y), _oracle(y.tolist()), label="bulk-smoke")


def test_bulk_lsmode68_singlevalue(data_singlevalue):
    y = data_singlevalue["y"]
    _assert_parity(_port(y), _oracle(y.tolist()), label="bulk-singlevalue")
