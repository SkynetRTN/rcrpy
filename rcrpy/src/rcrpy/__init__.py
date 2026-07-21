"""rcrpy: Robust Chauvenet Rejection, Python reimplementation."""

from rcrpy.api import (
    RCR,
    RCRResults,
    RejectionTech,
    MuType,
)
from rcrpy.nonparametric import NonParametric
from rcrpy.functional import FunctionalForm, FunctionalFormResults, LinearModel, Priors, PriorType


def enable_fast() -> None:
    """Opt in to the Numba-JIT acceleration of the hot DL-sigma kernels.

    Requires the optional ``numba`` dependency. The accelerated kernels keep
    the exact scalar arithmetic of the pure-Python reference but, because libm
    pow/log differ by <=1 ULP across CPython/NumPy/Numba, are not strictly
    bit-identical (~1e-16 relative, far below the rtol=1e-12 parity bar and
    validated to change no rejection outcomes). Off by default; call this once
    after import, or set env RCRPY_FAST=1 before importing rcrpy.
    """
    from rcrpy import _fast
    _fast.install()


__all__ = [
    "RCR",
    "RCRResults",
    "RejectionTech",
    "MuType",
    "NonParametric",
    "FunctionalForm",
    "FunctionalFormResults",
    "LinearModel",
    "Priors",
    "PriorType",
    "enable_fast",
]

__version__ = "0.1.1"

import os as _os
if _os.environ.get("RCRPY_FAST"):
    try:
        enable_fast()
    except Exception:  # numba missing / JIT failure -> stay on pure Python
        pass
