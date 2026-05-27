"""rcr2: Robust Chauvenet Rejection, Python reimplementation."""

from rcr2.api import (
    RCR,
    RCRResults,
    RejectionTech,
    MuType,
)
from rcr2.nonparametric import NonParametric
from rcr2.functional import FunctionalForm, FunctionalFormResults, Priors, PriorType

__all__ = [
    "RCR",
    "RCRResults",
    "RejectionTech",
    "MuType",
    "NonParametric",
    "FunctionalForm",
    "FunctionalFormResults",
    "Priors",
    "PriorType",
]

__version__ = "0.1.0"
