"""rcrpy: Robust Chauvenet Rejection, Python reimplementation."""

from rcrpy.api import (
    RCR,
    RCRResults,
    RejectionTech,
    MuType,
)
from rcrpy.nonparametric import NonParametric
from rcrpy.functional import FunctionalForm, FunctionalFormResults, Priors, PriorType

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
