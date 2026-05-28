"""pyrcr: Robust Chauvenet Rejection, Python reimplementation."""

from pyrcr.api import (
    RCR,
    RCRResults,
    RejectionTech,
    MuType,
)
from pyrcr.nonparametric import NonParametric
from pyrcr.functional import FunctionalForm, FunctionalFormResults, Priors, PriorType

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
