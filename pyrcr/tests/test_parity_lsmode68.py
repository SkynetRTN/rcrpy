"""Parity test: pyrcr port vs. legacy C++ `rcr` module, LS_MODE_68 path.

Tolerance: rtol=1e-12 (per agents/python_vs_rust_plan.md decisions).

Phase 1 scope: just LS_MODE_68 via performRejection (iterative, unweighted)
on data_smoke.csv. Larger CSVs and the bulk/weighted/DL variants are next
session's targets.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyrcr

# Skip cleanly if the oracle isn't installed in this environment.
rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12
ATOL = 0.0  # rtol-only check; tweak if a boundary case requires it.


def _run_oracle(y_list: list[float]) -> dict:
    r = rcr_oracle.RCR(rcr_oracle.LS_MODE_68)
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
        "rejectedY": list(r.result.rejectedY),
    }


def _run_port(y_arr: np.ndarray) -> dict:
    r = pyrcr.RCR(pyrcr.RejectionTech.LS_MODE_68)
    r.perform_rejection(y_arr.tolist())
    return {
        "mu": r.result.mu,
        "sigma": r.result.sigma,
        "stDev": r.result.st_dev,
        "stDevBelow": r.result.st_dev_below,
        "stDevAbove": r.result.st_dev_above,
        "flags": list(r.result.flags),
        "indices": list(r.result.indices),
        "cleanY": list(r.result.clean_y),
        "rejectedY": list(r.result.rejected_y),
    }


def _assert_parity(port: dict, oracle: dict, *, label: str) -> None:
    # Scalars at rtol=1e-12.
    for key in ("mu", "sigma", "stDev", "stDevBelow", "stDevAbove"):
        np.testing.assert_allclose(
            port[key], oracle[key], rtol=RTOL, atol=ATOL,
            err_msg=f"[{label}] {key}: port={port[key]!r} oracle={oracle[key]!r}",
        )

    # Flags + indices bit-identical.
    assert port["flags"] == oracle["flags"], f"[{label}] flags disagree"
    assert port["indices"] == oracle["indices"], f"[{label}] indices disagree"

    # cleanY element-wise. rejectedY/originalY intentionally not checked —
    # C++ performRejection (non-bulk) leaves them empty; only
    # performBulkRejection populates them via setFinalVectors.
    np.testing.assert_allclose(port["cleanY"], oracle["cleanY"], rtol=RTOL, atol=ATOL)


def test_lsmode68_parity_on_smoke(data_smoke):
    y = data_smoke["y"]
    _assert_parity(_run_port(y), _run_oracle(y.tolist()), label="smoke")


def test_lsmode68_parity_on_singlevalue(data_singlevalue):
    """Heavier test: N=1000 with ~85% one-sided contamination. Stresses the
    half-sample-mode iteration depth and exposes tie-breaks at rejection
    boundaries."""
    y = data_singlevalue["y"]
    _assert_parity(_run_port(y), _run_oracle(y.tolist()), label="singlevalue")
