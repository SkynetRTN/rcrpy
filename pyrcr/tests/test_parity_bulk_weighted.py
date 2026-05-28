"""Parity: weighted performBulkRejection for all 4 rejection techniques."""
from __future__ import annotations

import numpy as np
import pytest

import pyrcr

rcr_oracle = pytest.importorskip("rcr")

RTOL = 1e-12


@pytest.mark.parametrize("tech_name,port_tech,sigma_kind", [
    ("LS_MODE_68",   pyrcr.RejectionTech.LS_MODE_68,   "lower"),
    ("LS_MODE_DL",   pyrcr.RejectionTech.LS_MODE_DL,   "lower"),
    ("SS_MEDIAN_DL", pyrcr.RejectionTech.SS_MEDIAN_DL, "single"),
    ("ES_MODE_DL",   pyrcr.RejectionTech.ES_MODE_DL,   "each"),
])
def test_bulk_weighted(data_weighted_singlevalue, tech_name, port_tech, sigma_kind):
    oracle_tech = getattr(rcr_oracle, tech_name)
    w = data_weighted_singlevalue["w"]
    y = data_weighted_singlevalue["y"]

    o = rcr_oracle.RCR(oracle_tech)
    o.performBulkRejection(w.tolist(), y.tolist())

    p = pyrcr.RCR(port_tech)
    p.perform_bulk_rejection(y.tolist(), w=w.tolist())

    np.testing.assert_allclose(p.result.mu, o.result.mu, rtol=RTOL,
                               err_msg=f"[{tech_name}] mu")
    np.testing.assert_allclose(p.result.st_dev_total, o.result.stDevTotal, rtol=RTOL)
    np.testing.assert_allclose(p.result.st_dev_below, o.result.stDevBelow, rtol=RTOL)
    np.testing.assert_allclose(p.result.st_dev_above, o.result.stDevAbove, rtol=RTOL)
    assert list(p.result.flags) == list(o.result.flags), f"[{tech_name}] flags"
    assert list(p.result.indices) == list(o.result.indices), f"[{tech_name}] indices"
    np.testing.assert_allclose(list(p.result.clean_y), list(o.result.cleanY), rtol=RTOL)
    np.testing.assert_allclose(list(p.result.clean_w), list(o.result.cleanW), rtol=RTOL)
    np.testing.assert_allclose(list(p.result.rejected_y), list(o.result.rejectedY), rtol=RTOL)
    np.testing.assert_allclose(list(p.result.rejected_w), list(o.result.rejectedW), rtol=RTOL)
    np.testing.assert_allclose(list(p.result.original_y), list(o.result.originalY), rtol=RTOL)
    np.testing.assert_allclose(list(p.result.original_w), list(o.result.originalW), rtol=RTOL)
    if sigma_kind != "each":
        np.testing.assert_allclose(p.result.sigma, o.result.sigma, rtol=RTOL,
                                   err_msg=f"[{tech_name}] sigma")
    else:
        np.testing.assert_allclose(p.result.sigma_below, o.result.sigmaBelow, rtol=RTOL)
        np.testing.assert_allclose(p.result.sigma_above, o.result.sigmaAbove, rtol=RTOL)
