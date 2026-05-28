"""Parity: performBulkRejection for all 4 rejection techniques."""
from __future__ import annotations

import numpy as np
import pytest

import rcr2

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


def _oracle(tech, y_list):
    r = rcr_oracle.RCR(tech)
    r.performBulkRejection(y_list)
    return r


def _port(tech, y):
    r = rcr2.RCR(tech)
    r.perform_bulk_rejection(y.tolist())
    return r


def _assert_common(port_r, oracle_r, label):
    """Fields populated by setFinalVectors regardless of tech."""
    np.testing.assert_allclose(port_r.result.mu, oracle_r.result.mu, rtol=RTOL,
                               err_msg=f"[{label}] mu")
    np.testing.assert_allclose(port_r.result.st_dev_total, oracle_r.result.stDevTotal,
                               rtol=RTOL, err_msg=f"[{label}] stDevTotal")
    np.testing.assert_allclose(port_r.result.st_dev_below, oracle_r.result.stDevBelow,
                               rtol=RTOL, err_msg=f"[{label}] stDevBelow")
    np.testing.assert_allclose(port_r.result.st_dev_above, oracle_r.result.stDevAbove,
                               rtol=RTOL, err_msg=f"[{label}] stDevAbove")
    assert list(port_r.result.flags) == list(oracle_r.result.flags), f"[{label}] flags"
    assert list(port_r.result.indices) == list(oracle_r.result.indices), f"[{label}] indices"
    np.testing.assert_allclose(list(port_r.result.clean_y), list(oracle_r.result.cleanY),
                               rtol=RTOL, err_msg=f"[{label}] cleanY")
    np.testing.assert_allclose(list(port_r.result.rejected_y), list(oracle_r.result.rejectedY),
                               rtol=RTOL, err_msg=f"[{label}] rejectedY")
    np.testing.assert_allclose(list(port_r.result.original_y), list(oracle_r.result.originalY),
                               rtol=RTOL, err_msg=f"[{label}] originalY")


@pytest.mark.parametrize("tech_name,port_tech,oracle_tech,sigma_kind", [
    ("LS_MODE_68",   rcr2.RejectionTech.LS_MODE_68,   None, "lower"),
    ("LS_MODE_DL",   rcr2.RejectionTech.LS_MODE_DL,   None, "lower"),
    ("SS_MEDIAN_DL", rcr2.RejectionTech.SS_MEDIAN_DL, None, "single"),
    ("ES_MODE_DL",   rcr2.RejectionTech.ES_MODE_DL,   None, "each"),
])
def test_bulk_smoke(data_smoke, tech_name, port_tech, oracle_tech, sigma_kind):
    oracle_tech = getattr(rcr_oracle, tech_name)
    y = data_smoke["y"]
    p = _port(port_tech, y)
    o = _oracle(oracle_tech, y.tolist())
    _assert_common(p, o, f"bulk-smoke-{tech_name}")
    if sigma_kind != "each":
        np.testing.assert_allclose(p.result.sigma, o.result.sigma, rtol=RTOL,
                                   err_msg=f"sigma[{tech_name}]")
    else:
        np.testing.assert_allclose(p.result.sigma_below, o.result.sigmaBelow, rtol=RTOL)
        np.testing.assert_allclose(p.result.sigma_above, o.result.sigmaAbove, rtol=RTOL)


@pytest.mark.parametrize("tech_name,port_tech,sigma_kind", [
    ("LS_MODE_68",   rcr2.RejectionTech.LS_MODE_68,   "lower"),
    ("LS_MODE_DL",   rcr2.RejectionTech.LS_MODE_DL,   "lower"),
    ("SS_MEDIAN_DL", rcr2.RejectionTech.SS_MEDIAN_DL, "single"),
    ("ES_MODE_DL",   rcr2.RejectionTech.ES_MODE_DL,   "each"),
])
def test_bulk_singlevalue(data_singlevalue, tech_name, port_tech, sigma_kind):
    oracle_tech = getattr(rcr_oracle, tech_name)
    y = data_singlevalue["y"]
    p = _port(port_tech, y)
    o = _oracle(oracle_tech, y.tolist())
    _assert_common(p, o, f"bulk-single-{tech_name}")
    if sigma_kind != "each":
        np.testing.assert_allclose(p.result.sigma, o.result.sigma, rtol=RTOL,
                                   err_msg=f"sigma[{tech_name}]")
    else:
        np.testing.assert_allclose(p.result.sigma_below, o.result.sigmaBelow, rtol=RTOL)
        np.testing.assert_allclose(p.result.sigma_above, o.result.sigmaAbove, rtol=RTOL)
