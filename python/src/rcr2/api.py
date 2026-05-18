"""Public API for rcr2.

Mirrors the surface defined in cpp/src/RCR.h:
    enum RejectionTechs { SS_MEDIAN_DL, LS_MODE_68, LS_MODE_DL, ES_MODE_DL }
    struct RCRResults  { mu, sigma, stDev, ..., flags, indices, ... }
    class  RCR         { performRejection, performBulkRejection, ... }

Single-value (non-parametric, non-functional) rejection only in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np


class RejectionTech(Enum):
    SS_MEDIAN_DL = "SS_MEDIAN_DL"
    LS_MODE_68 = "LS_MODE_68"
    LS_MODE_DL = "LS_MODE_DL"
    ES_MODE_DL = "ES_MODE_DL"


@dataclass
class RCRResults:
    mu: float = float("nan")
    sigma: float = float("nan")
    sigma_below: float = float("nan")
    sigma_above: float = float("nan")
    st_dev: float = float("nan")
    st_dev_below: float = float("nan")
    st_dev_above: float = float("nan")
    st_dev_total: float = float("nan")
    flags: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    clean_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rejected_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    clean_w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rejected_w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    original_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    original_w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))


class RCR:
    """Single-value Robust Chauvenet Rejection.

    Phase 1: API skeleton. The real iterative + bulk loops are ported in
    rcr2.rejection and dispatched here.
    """

    def __init__(self, rejection_tech: RejectionTech = RejectionTech.SS_MEDIAN_DL):
        self.rejection_tech = rejection_tech
        self.result = RCRResults()

    def set_rejection_tech(self, tech: RejectionTech) -> None:
        self.rejection_tech = tech

    def perform_rejection(
        self,
        y: Sequence[float],
        w: Sequence[float] | None = None,
    ) -> None:
        from rcr2 import rejection
        y_arr = np.asarray(y, dtype=np.float64)
        w_arr = None if w is None else np.asarray(w, dtype=np.float64)
        out = rejection.performRejection_LS(y_arr, self.rejection_tech.value, w=w_arr)
        r = self.result
        r.mu = out["mu"]
        r.flags = out["flags"]
        r.indices = out["indices"]
        r.clean_y = out["clean_y"]
        # Each-sigma populates sigma_below/above; the other techs populate
        # sigma/st_dev. setFinalVectors-only fields (rejectedY, originalY,
        # clean_w/rejected_w/original_w) remain at default-empty per
        # performRejection's contract.
        if "sigma" in out:
            r.sigma = out["sigma"]
            r.st_dev = out["st_dev"]
        if "sigma_below" in out:
            r.sigma_below = out["sigma_below"]
            r.sigma_above = out["sigma_above"]
        if "st_dev_above" in out:
            r.st_dev_above = out["st_dev_above"]
        if "st_dev_below" in out:
            r.st_dev_below = out["st_dev_below"]
        if "clean_w" in out:
            r.clean_w = out["clean_w"]
        # NOTE: rejected_y / original_y / clean_w / rejected_w / original_w are
        # NOT set here — the C++ performRejection (non-bulk) leaves them at
        # their default-empty state; only performBulkRejection populates them
        # via setFinalVectors. Match that contract for parity.

    def perform_bulk_rejection(
        self,
        y: Sequence[float],
        w: Sequence[float] | None = None,
    ) -> None:
        from rcr2 import rejection
        y_arr = np.asarray(y, dtype=np.float64)
        w_arr = None if w is None else np.asarray(w, dtype=np.float64)
        out = rejection.performBulkRejection_LS(y_arr, self.rejection_tech.value, w=w_arr)
        r = self.result
        r.mu = out["mu"]
        r.flags = out["flags"]
        r.indices = out["indices"]
        r.clean_y = out["clean_y"]
        r.rejected_y = out["rejected_y"]
        r.original_y = out["original_y"]
        r.st_dev_total = out["st_dev_total"]
        r.st_dev_above = out["st_dev_above"]
        r.st_dev_below = out["st_dev_below"]
        # Single/Lower set sigma + st_dev; Each sets sigma_below/sigma_above.
        if "sigma" in out:
            r.sigma = out["sigma"]
            if "st_dev" in out:
                r.st_dev = out["st_dev"]
        if "sigma_below" in out:
            r.sigma_below = out["sigma_below"]
            r.sigma_above = out["sigma_above"]
        if "clean_w" in out:
            r.clean_w = out["clean_w"]
        if "rejected_w" in out:
            r.rejected_w = out["rejected_w"]
        if "original_w" in out:
            r.original_w = out["original_w"]
