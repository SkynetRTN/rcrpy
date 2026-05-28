"""Comprehensive Python-port vs C++-oracle diagnostics:
  1. Per-tech wall-clock comparison (timing)
  2. Per-tech worst-case relative scalar disagreement (precision)
  3. Bit-identity of decisions (flags / cleanY)

Run from repository root:
    python pyrcr/diagnostics.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

import rcr  # oracle
import pyrcr

REPO = Path(__file__).resolve().parents[2]  # pyrcr/benchmarks/x.py -> repo root
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


def _rel(a: float, b: float) -> float:
    if b == 0.0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def _max_rel(port_r, oracle_r, fields: list[tuple[str, str]]) -> float:
    worst = 0.0
    for p_key, o_key in fields:
        p = getattr(port_r.result, p_key)
        o = getattr(oracle_r.result, o_key)
        r = _rel(float(p), float(o))
        if r > worst:
            worst = r
    return worst


ITER_FIELDS = [("mu", "mu"), ("sigma", "sigma"), ("st_dev", "stDev"),
               ("st_dev_below", "stDevBelow"), ("st_dev_above", "stDevAbove")]
EACH_FIELDS = [("mu", "mu"), ("sigma_below", "sigmaBelow"), ("sigma_above", "sigmaAbove"),
               ("st_dev_below", "stDevBelow"), ("st_dev_above", "stDevAbove")]
SS_FIELDS = [("mu", "mu"), ("sigma", "sigma"), ("st_dev", "stDev")]
BULK_LOWER_FIELDS = [("mu", "mu"), ("sigma", "sigma"),
                     ("st_dev_total", "stDevTotal"),
                     ("st_dev_below", "stDevBelow"),
                     ("st_dev_above", "stDevAbove")]
BULK_EACH_FIELDS = [("mu", "mu"),
                    ("sigma_below", "sigmaBelow"), ("sigma_above", "sigmaAbove"),
                    ("st_dev_total", "stDevTotal"),
                    ("st_dev_below", "stDevBelow"),
                    ("st_dev_above", "stDevAbove")]


def _time(callable_, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        callable_()
        times.append(time.perf_counter() - t)
    return min(times)  # best-of-3 reduces OS jitter


def run_case(label: str, tech_name: str, dataset: str, weighted: bool, bulk: bool, fields):
    d = _load(dataset)
    y = d["y"]
    w_arr = d.get("w")
    py_tech = getattr(pyrcr.RejectionTech, tech_name)
    oc_tech = getattr(rcr, tech_name)

    def run_oracle():
        o = rcr.RCR(oc_tech)
        if weighted:
            method = o.performBulkRejection if bulk else o.performRejection
            method(w_arr.tolist(), y.tolist())
        else:
            method = o.performBulkRejection if bulk else o.performRejection
            method(y.tolist())
        return o

    def run_port():
        p = pyrcr.RCR(py_tech)
        method = p.perform_bulk_rejection if bulk else p.perform_rejection
        if weighted:
            method(y.tolist(), w=w_arr.tolist())
        else:
            method(y.tolist())
        return p

    oracle = run_oracle()
    port = run_port()

    t_oracle = _time(run_oracle)
    t_port = _time(run_port)

    max_rel = _max_rel(port, oracle, fields)
    flags_ok = list(port.result.flags) == list(oracle.result.flags)
    return {
        "label": label,
        "n": y.size,
        "t_oracle_ms": t_oracle * 1000,
        "t_port_ms": t_port * 1000,
        "slowdown": t_port / t_oracle if t_oracle > 0 else float("inf"),
        "max_rel": max_rel,
        "flags_bit_identical": flags_ok,
    }


CASES = [
    # iterative
    ("LS_MODE_68   iter smoke",   "LS_MODE_68",   "data_smoke.csv",                False, False, ITER_FIELDS),
    ("LS_MODE_68   iter single",  "LS_MODE_68",   "data_singlevalue.csv",          False, False, ITER_FIELDS),
    ("LS_MODE_68   iter weight",  "LS_MODE_68",   "data_weighted_singlevalue.csv", True,  False, ITER_FIELDS),
    ("LS_MODE_DL   iter smoke",   "LS_MODE_DL",   "data_smoke.csv",                False, False, ITER_FIELDS),
    ("LS_MODE_DL   iter single",  "LS_MODE_DL",   "data_singlevalue.csv",          False, False, ITER_FIELDS),
    ("LS_MODE_DL   iter weight",  "LS_MODE_DL",   "data_weighted_singlevalue.csv", True,  False, ITER_FIELDS),
    ("SS_MEDIAN_DL iter smoke",   "SS_MEDIAN_DL", "data_smoke.csv",                False, False, SS_FIELDS),
    ("SS_MEDIAN_DL iter single",  "SS_MEDIAN_DL", "data_singlevalue.csv",          False, False, SS_FIELDS),
    ("SS_MEDIAN_DL iter weight",  "SS_MEDIAN_DL", "data_weighted_singlevalue.csv", True,  False, SS_FIELDS),
    ("ES_MODE_DL   iter smoke",   "ES_MODE_DL",   "data_smoke.csv",                False, False, EACH_FIELDS),
    ("ES_MODE_DL   iter single",  "ES_MODE_DL",   "data_singlevalue.csv",          False, False, EACH_FIELDS),
    ("ES_MODE_DL   iter weight",  "ES_MODE_DL",   "data_weighted_singlevalue.csv", True,  False, EACH_FIELDS),
    # bulk
    ("LS_MODE_68   BULK smoke",   "LS_MODE_68",   "data_smoke.csv",                False, True,  BULK_LOWER_FIELDS),
    ("LS_MODE_68   BULK single",  "LS_MODE_68",   "data_singlevalue.csv",          False, True,  BULK_LOWER_FIELDS),
    ("LS_MODE_68   BULK weight",  "LS_MODE_68",   "data_weighted_singlevalue.csv", True,  True,  BULK_LOWER_FIELDS),
    ("LS_MODE_DL   BULK smoke",   "LS_MODE_DL",   "data_smoke.csv",                False, True,  BULK_LOWER_FIELDS),
    ("LS_MODE_DL   BULK single",  "LS_MODE_DL",   "data_singlevalue.csv",          False, True,  BULK_LOWER_FIELDS),
    ("LS_MODE_DL   BULK weight",  "LS_MODE_DL",   "data_weighted_singlevalue.csv", True,  True,  BULK_LOWER_FIELDS),
    ("SS_MEDIAN_DL BULK smoke",   "SS_MEDIAN_DL", "data_smoke.csv",                False, True,  BULK_LOWER_FIELDS),
    ("SS_MEDIAN_DL BULK single",  "SS_MEDIAN_DL", "data_singlevalue.csv",          False, True,  BULK_LOWER_FIELDS),
    ("SS_MEDIAN_DL BULK weight",  "SS_MEDIAN_DL", "data_weighted_singlevalue.csv", True,  True,  BULK_LOWER_FIELDS),
    ("ES_MODE_DL   BULK smoke",   "ES_MODE_DL",   "data_smoke.csv",                False, True,  BULK_EACH_FIELDS),
    ("ES_MODE_DL   BULK single",  "ES_MODE_DL",   "data_singlevalue.csv",          False, True,  BULK_EACH_FIELDS),
    ("ES_MODE_DL   BULK weight",  "ES_MODE_DL",   "data_weighted_singlevalue.csv", True,  True,  BULK_EACH_FIELDS),
]


def main() -> None:
    print(f"{'case':<28s} {'N':>5s}  {'oracle (ms)':>12s}  {'port (ms)':>11s}  "
          f"{'slowdown':>10s}  {'max rel':>11s}  flags")
    print("-" * 99)
    worst_rel = 0.0
    total_oracle = 0.0
    total_port = 0.0
    for c in CASES:
        r = run_case(*c)
        worst_rel = max(worst_rel, r["max_rel"])
        total_oracle += r["t_oracle_ms"]
        total_port += r["t_port_ms"]
        flag = "OK" if r["flags_bit_identical"] else "FAIL"
        print(f"{r['label']:<28s} {r['n']:>5d}  {r['t_oracle_ms']:>12.2f}  "
              f"{r['t_port_ms']:>11.2f}  {r['slowdown']:>9.1f}x  "
              f"{r['max_rel']:>11.3e}  {flag}")
    print("-" * 99)
    print(f"{'TOTALS':<28s} {'':<5s}  {total_oracle:>12.2f}  {total_port:>11.2f}  "
          f"{total_port/total_oracle:>9.1f}x  {worst_rel:>11.3e}")
    print()
    print(f"Worst-case scalar disagreement: {worst_rel:.3e}")
    print(f"Assertion threshold:            1.000e-12")
    if worst_rel > 0:
        import math
        head = math.log10(1e-12 / worst_rel)
        print(f"Headroom below threshold:       {head:.1f} orders of magnitude")


if __name__ == "__main__":
    main()
