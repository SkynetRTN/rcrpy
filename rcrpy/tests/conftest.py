"""Shared pytest fixtures for rcrpy tests."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "assets" / "test").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root (no assets/test/ found).")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture(scope="session")
def assets_dir(repo_root: Path) -> Path:
    return repo_root / "assets" / "test"


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols: dict[str, list[float]] = {name: [] for name in header}
        for row in reader:
            for name, val in zip(header, row):
                cols[name].append(float(val))
    return {name: np.array(vals, dtype=np.float64) for name, vals in cols.items()}


@pytest.fixture(scope="session")
def data_smoke(assets_dir: Path) -> dict[str, np.ndarray]:
    return _load_csv(assets_dir / "data_smoke.csv")


@pytest.fixture(scope="session")
def data_singlevalue(assets_dir: Path) -> dict[str, np.ndarray]:
    return _load_csv(assets_dir / "data_singlevalue.csv")


@pytest.fixture(scope="session")
def data_weighted_singlevalue(assets_dir: Path) -> dict[str, np.ndarray]:
    return _load_csv(assets_dir / "data_weighted_singlevalue.csv")
