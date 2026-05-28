"""Regression test: the unity tables in pyrcr.tables must hash to the exact
byte-values they had when extracted from cpp/src/RCR.cpp.

The tables are float-literal data copied (via pyrcr/scripts/extract_unity_tables.py)
straight from the C++ source. They have no semantic meaning we can test
independently of the rejection loops, so we lock their bytes with SHA-256
to guard against accidental edits.

If a digest mismatches after a deliberate change (e.g. a corrected literal
upstream), re-run the extractor and update the digests below.
"""
from __future__ import annotations

import hashlib

import pytest

from pyrcr import tables

EXPECTED_SHA256 = {
    "ESUnity":      "a3a42e49e58a77b4b0417b4acf5088b60ac19c498e94daf7666129d31c834f24",
    "SSUnity":      "16306a80d403bdfa93578084a17f2719de403bafc9fd8429a2af8b8d5b049b01",
    "LSUnity":      "84aa816381c9efd0980fdc2f0f97429134414f1aee864fdde80efb75e3272c71",
    "ESDLUnityCF":  "54acc05f8edcdbc279769c1fa96806ea44e47773d62d563f4d9f41e047801883",
    "LSDLUnityCF":  "bad95d828d7f70abace6a4d2031860d9dec1439aac39f65f5ac0e00d4c1f38f4",
    "LS68UnityCF":  "f0fe5840abb212f9c1ef62a078547c3f4d0a7bbac59708bd7794e93e4c4a03b7",
    "SSDLUnityCF":  "8d949f4659cab4f40dde3f7977fa317795d042a17fc1ffbf83f7cc936fbdcda8",
    "SSConstants":  "b9cd59f202006e6d6de2ef50bbbb3a9c5037424e6437f11ee31a4011f821abe8",
}

EXPECTED_SHAPES = {
    "ESUnity":     (1001,),
    "SSUnity":     (1001,),
    "LSUnity":     (1001,),
    "ESDLUnityCF": (101,),
    "LSDLUnityCF": (101,),
    "LS68UnityCF": (101,),
    "SSDLUnityCF": (101,),
    "SSConstants": (2, 8),
}


@pytest.mark.parametrize("name,expected_hash", list(EXPECTED_SHA256.items()))
def test_unity_table_sha256(name: str, expected_hash: str) -> None:
    arr = getattr(tables, name)
    actual = hashlib.sha256(arr.tobytes()).hexdigest()
    assert actual == expected_hash, (
        f"{name} sha256 changed: got {actual}, expected {expected_hash}. "
        f"If the change is deliberate, re-run extract_unity_tables.py and update the digest."
    )


@pytest.mark.parametrize("name,expected_shape", list(EXPECTED_SHAPES.items()))
def test_unity_table_shape(name: str, expected_shape: tuple[int, ...]) -> None:
    arr = getattr(tables, name)
    assert arr.shape == expected_shape
    assert arr.dtype.name == "float64"


def test_ssconstants_known_values() -> None:
    """Spot-check the only 2D table with a few known values from cpp/src/RCR.cpp."""
    assert tables.SSConstants[0, 4] == 0.202399
    assert tables.SSConstants[1, 4] == 0.464231
    assert tables.SSConstants[0, 5] == -0.29158
    assert tables.SSConstants[0, 0] == 0.0
    assert tables.SSConstants[1, 0] == 0.0


def test_unity_table_first_nonzero_indices() -> None:
    """Cross-check a couple of values from the C++ snippet at lines ~1599-1603.
    ESUnity[5] = 8.1666; SSUnity[4] = 2.15681; SSUnity[5] = 8.81255.
    """
    assert tables.ESUnity[5] == 8.1666
    assert tables.SSUnity[4] == 2.15681
    assert tables.SSUnity[5] == 8.81255
    assert tables.ESUnity[0] == 0.0
    assert tables.LSUnity[0] == 0.0
