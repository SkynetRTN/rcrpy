"""Quick port-vs-oracle wall-clock comparison on data_singlevalue.csv (N=1000).

Not a parity test — `tests/test_parity_lsmode68.py` covers correctness.
This is a sanity check on how far the port is from the C++ oracle in
absolute terms. Run from the repository root:

    python python/bench_port_vs_oracle.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

import rcr  # legacy C++ oracle
import rcr2

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "assets" / "test" / "data_singlevalue.csv"

with open(CSV, newline="") as f:
    reader = csv.reader(f)
    next(reader)
    y = np.array([float(r[0]) for r in reader], dtype=np.float64)

REPEATS = 3

t0 = time.perf_counter()
for _ in range(REPEATS):
    r = rcr.RCR(rcr.LS_MODE_68)
    r.performRejection(y.tolist())
oracle_t = (time.perf_counter() - t0) / REPEATS

t0 = time.perf_counter()
for _ in range(REPEATS):
    p = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
    p.perform_rejection(y.tolist())
port_t = (time.perf_counter() - t0) / REPEATS

print(f"Dataset:       data_singlevalue.csv (N={y.size})")
print(f"Oracle (cpp):  {oracle_t*1000:8.1f} ms")
print(f"Port (rcr2):   {port_t*1000:8.1f} ms")
print(f"Slowdown:      {port_t/oracle_t:.1f}x")
print(f"Result match:  mu port={p.result.mu:.6f} oracle={r.result.mu:.6f}")
