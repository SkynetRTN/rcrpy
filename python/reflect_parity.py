"""Diagnostic: measure how tightly rcr2 matches the C++ oracle on each parity
case. We assert rtol=1e-12, but how much slack is there really?

For each (tech, dataset, weighted?) tuple, report:
  - max relative diff across all scalar result fields
  - max absolute diff across all clean_y elements
  - whether flags / indices are bit-identical

Run from the repo root:
    python python/reflect_parity.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np

import rcr  # oracle
import rcr2

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets" / "test"


def _load(name: str) -> dict[str, np.ndarray]:
    with open(ASSETS / name, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    cols: dict[str, list[float]] = {h: [] for h in header}
    for row in rows[1:]:
        for h, v in zip(header, row):
            cols[h].append(float(v))
    return {h: np.array(v, dtype=np.float64) for h, v in cols.items()}


def _rel(port: float, oracle: float) -> float:
    if oracle == 0.0:
        return abs(port - oracle)
    return abs(port - oracle) / abs(oracle)


def _scalar_max_rel(port_obj, oracle_obj, fields: list[tuple[str, str]]) -> tuple[float, str]:
    worst = 0.0
    worst_field = ""
    for port_key, oracle_key in fields:
        p = getattr(port_obj, port_key)
        o = getattr(oracle_obj, oracle_key)
        r = _rel(float(p), float(o))
        if r > worst:
            worst = r
            worst_field = f"{port_key}={p!r} vs {o!r}"
    return worst, worst_field


def _vector_max_rel(port_vec, oracle_vec) -> float:
    p = np.asarray(port_vec, dtype=np.float64)
    o = np.asarray(oracle_vec, dtype=np.float64)
    if p.size == 0 and o.size == 0:
        return 0.0
    denom = np.where(np.abs(o) > 0, np.abs(o), 1.0)
    return float(np.max(np.abs(p - o) / denom))


ITERATIVE_FIELDS = [
    ("mu", "mu"), ("sigma", "sigma"), ("st_dev", "stDev"),
    ("st_dev_below", "stDevBelow"), ("st_dev_above", "stDevAbove"),
]
ITERATIVE_EACH_FIELDS = [
    ("mu", "mu"),
    ("sigma_below", "sigmaBelow"), ("sigma_above", "sigmaAbove"),
    ("st_dev_below", "stDevBelow"), ("st_dev_above", "stDevAbove"),
]
SS_FIELDS = [("mu", "mu"), ("sigma", "sigma"), ("st_dev", "stDev")]
BULK_FIELDS = [
    ("mu", "mu"), ("sigma", "sigma"),
    ("st_dev_below", "stDevBelow"), ("st_dev_above", "stDevAbove"),
    ("st_dev_total", "stDevTotal"),
]


def run_case(label: str, tech_enum_port, tech_enum_oracle, dataset: str,
             weighted: bool, fields, oracle_method: str, port_method: str) -> dict:
    d = _load(dataset)
    y = d["y"]
    w = d.get("w")

    port = rcr2.RCR(tech_enum_port)
    oracle = rcr.RCR(tech_enum_oracle)

    if weighted:
        if port_method == "perform_rejection":
            port.perform_rejection(y.tolist(), w=w.tolist())
        else:
            port.perform_bulk_rejection(y.tolist(), w=w.tolist())
        getattr(oracle, oracle_method)(w.tolist(), y.tolist())
    else:
        if port_method == "perform_rejection":
            port.perform_rejection(y.tolist())
        else:
            port.perform_bulk_rejection(y.tolist())
        getattr(oracle, oracle_method)(y.tolist())

    scalar_rel, worst_field = _scalar_max_rel(port.result, oracle.result, fields)
    clean_rel = _vector_max_rel(port.result.clean_y, oracle.result.cleanY)
    flags_ok = list(port.result.flags) == list(oracle.result.flags)
    indices_ok = list(port.result.indices) == list(oracle.result.indices)
    return {
        "label": label,
        "n": y.size,
        "scalar_max_rel": scalar_rel,
        "worst_field": worst_field,
        "clean_y_max_rel": clean_rel,
        "flags_bit_identical": flags_ok,
        "indices_bit_identical": indices_ok,
    }


CASES = [
    # (label, port-tech, oracle-tech, dataset, weighted?, fields, oracle-method, port-method)
    ("LS_MODE_68    smoke ",       rcr2.RejectionTech.LS_MODE_68,   rcr.LS_MODE_68,   "data_smoke.csv",                False, ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("LS_MODE_68    single",       rcr2.RejectionTech.LS_MODE_68,   rcr.LS_MODE_68,   "data_singlevalue.csv",          False, ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("LS_MODE_68    weighted",     rcr2.RejectionTech.LS_MODE_68,   rcr.LS_MODE_68,   "data_weighted_singlevalue.csv", True,  ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("LS_MODE_DL    smoke ",       rcr2.RejectionTech.LS_MODE_DL,   rcr.LS_MODE_DL,   "data_smoke.csv",                False, ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("LS_MODE_DL    single",       rcr2.RejectionTech.LS_MODE_DL,   rcr.LS_MODE_DL,   "data_singlevalue.csv",          False, ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("LS_MODE_DL    weighted",     rcr2.RejectionTech.LS_MODE_DL,   rcr.LS_MODE_DL,   "data_weighted_singlevalue.csv", True,  ITERATIVE_FIELDS,      "performRejection",     "perform_rejection"),
    ("SS_MEDIAN_DL  smoke ",       rcr2.RejectionTech.SS_MEDIAN_DL, rcr.SS_MEDIAN_DL, "data_smoke.csv",                False, SS_FIELDS,             "performRejection",     "perform_rejection"),
    ("SS_MEDIAN_DL  single",       rcr2.RejectionTech.SS_MEDIAN_DL, rcr.SS_MEDIAN_DL, "data_singlevalue.csv",          False, SS_FIELDS,             "performRejection",     "perform_rejection"),
    ("SS_MEDIAN_DL  weighted",     rcr2.RejectionTech.SS_MEDIAN_DL, rcr.SS_MEDIAN_DL, "data_weighted_singlevalue.csv", True,  SS_FIELDS,             "performRejection",     "perform_rejection"),
    ("ES_MODE_DL    smoke ",       rcr2.RejectionTech.ES_MODE_DL,   rcr.ES_MODE_DL,   "data_smoke.csv",                False, ITERATIVE_EACH_FIELDS, "performRejection",     "perform_rejection"),
    ("ES_MODE_DL    single",       rcr2.RejectionTech.ES_MODE_DL,   rcr.ES_MODE_DL,   "data_singlevalue.csv",          False, ITERATIVE_EACH_FIELDS, "performRejection",     "perform_rejection"),
    ("ES_MODE_DL    weighted",     rcr2.RejectionTech.ES_MODE_DL,   rcr.ES_MODE_DL,   "data_weighted_singlevalue.csv", True,  ITERATIVE_EACH_FIELDS, "performRejection",     "perform_rejection"),
    ("LS_MODE_68    BULK smoke",   rcr2.RejectionTech.LS_MODE_68,   rcr.LS_MODE_68,   "data_smoke.csv",                False, BULK_FIELDS,           "performBulkRejection", "perform_bulk_rejection"),
    ("LS_MODE_68    BULK single",  rcr2.RejectionTech.LS_MODE_68,   rcr.LS_MODE_68,   "data_singlevalue.csv",          False, BULK_FIELDS,           "performBulkRejection", "perform_bulk_rejection"),
    ("LS_MODE_DL    BULK smoke",   rcr2.RejectionTech.LS_MODE_DL,   rcr.LS_MODE_DL,   "data_smoke.csv",                False, BULK_FIELDS,           "performBulkRejection", "perform_bulk_rejection"),
    ("LS_MODE_DL    BULK single",  rcr2.RejectionTech.LS_MODE_DL,   rcr.LS_MODE_DL,   "data_singlevalue.csv",          False, BULK_FIELDS,           "performBulkRejection", "perform_bulk_rejection"),
    ("SS_MEDIAN_DL  BULK smoke",   rcr2.RejectionTech.SS_MEDIAN_DL, rcr.SS_MEDIAN_DL, "data_smoke.csv",                False, SS_FIELDS,             "performBulkRejection", "perform_bulk_rejection"),
    ("SS_MEDIAN_DL  BULK single",  rcr2.RejectionTech.SS_MEDIAN_DL, rcr.SS_MEDIAN_DL, "data_singlevalue.csv",          False, SS_FIELDS,             "performBulkRejection", "perform_bulk_rejection"),
    ("ES_MODE_DL    BULK smoke",   rcr2.RejectionTech.ES_MODE_DL,   rcr.ES_MODE_DL,   "data_smoke.csv",                False, ITERATIVE_EACH_FIELDS, "performBulkRejection", "perform_bulk_rejection"),
    ("ES_MODE_DL    BULK single",  rcr2.RejectionTech.ES_MODE_DL,   rcr.ES_MODE_DL,   "data_singlevalue.csv",          False, ITERATIVE_EACH_FIELDS, "performBulkRejection", "perform_bulk_rejection"),
]


def main() -> None:
    print(f"{'case':<28s} {'N':>4s}  {'scalar max rel':>16s}  {'cleanY max rel':>16s}  flags  indices")
    print("-" * 92)
    overall_scalar = 0.0
    overall_vector = 0.0
    for c in CASES:
        r = run_case(*c)
        sr = r["scalar_max_rel"]
        vr = r["clean_y_max_rel"]
        overall_scalar = max(overall_scalar, sr)
        overall_vector = max(overall_vector, vr)
        print(f"{r['label']:<28s} {r['n']:>4d}  {sr:>16.3e}  {vr:>16.3e}   "
              f"{'OK' if r['flags_bit_identical'] else 'FAIL':<5s}  "
              f"{'OK' if r['indices_bit_identical'] else 'FAIL'}")
    print("-" * 92)
    print(f"{'WORST OVERALL':<28s} {'':<4s}  {overall_scalar:>16.3e}  {overall_vector:>16.3e}")
    print()
    print(f"Assertion threshold: rtol=1e-12 = {1e-12:.3e}")
    print(f"Worst observed:      {max(overall_scalar, overall_vector):.3e}")
    headroom = math.log10(1e-12 / max(overall_scalar, overall_vector, 1e-300))
    print(f"Headroom (orders of magnitude below rtol=1e-12): {headroom:.1f}")


if __name__ == "__main__":
    main()
