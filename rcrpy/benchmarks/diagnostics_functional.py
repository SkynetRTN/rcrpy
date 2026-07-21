"""Wall-clock + precision comparison for FunctionalForm fits — rcrpy vs
C++ oracle.

For each case, both implementations run RCR + LS_MODE_68 with the same
parametric model, partials, and initial guess; we time best-of-3 and
report the parameter agreement.

Run from the repo root:
    python rcrpy/diagnostics_functional.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

import rcr   # oracle
import rcrpy

REPO = Path(__file__).resolve().parents[2]  # rcrpy/benchmarks/x.py -> repo root
ASSETS = REPO / "assets" / "test"


def _load(name: str):
    with open(ASSETS / name, newline="") as f:
        reader = csv.reader(f)
        rows = []
        for r in reader:
            if len(r) >= 2 and r[0].strip() and r[1].strip():
                rows.append((float(r[0]), float(r[1])))
    x = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=np.float64)
    return x, y


def linear(x, params):
    return params[0] + params[1] * x


def d_lin0(x, params):
    return 1.0


def d_lin1(x, params):
    return x


def time_best_of_3(fn) -> float:
    times = []
    for _ in range(3):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def case_linear_fit_no_rcr(N: int, label: str):
    rng = np.random.default_rng(7 + N)
    x = np.linspace(-5, 5, N)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.5, size=x.size)

    def run_port():
        m = rcrpy.FunctionalForm(linear, x, y, [d_lin0, d_lin1], guess=[0.0, 0.0])
        m.regression()
        return m.parameters

    def run_oracle():
        m = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                [d_lin0, d_lin1], [0.0, 0.0])
        # The C++ oracle's regression() isn't directly exposed via pybind —
        # use an RCR + LS_MODE_68 run with a single iteration to get a fit.
        r = rcr.RCR(rcr.LS_MODE_68)
        r.setParametricModel(m)
        r.performRejection(y.tolist())
        return m.result.parameters

    t_port = time_best_of_3(run_port)
    t_oracle = time_best_of_3(run_oracle)
    p = run_port()
    o = run_oracle()
    diff = float(max(abs(p[i] - o[i]) for i in range(2)))
    return label, N, t_oracle, t_port, diff


def case_linear_rcr(label: str, x, y):
    def run_port():
        m = rcrpy.FunctionalForm(linear, x, y, [d_lin0, d_lin1], guess=[0.0, 1.0])
        r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
        r.set_parametric_model(m)
        r.perform_rejection(y.tolist())
        return m.result.parameters

    def run_oracle():
        m = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                [d_lin0, d_lin1], [0.0, 1.0])
        r = rcr.RCR(rcr.LS_MODE_68)
        r.setParametricModel(m)
        r.performRejection(y.tolist())
        return m.result.parameters

    t_port = time_best_of_3(run_port)
    t_oracle = time_best_of_3(run_oracle)
    p = run_port()
    o = run_oracle()
    diff = float(max(abs(p[i] - o[i]) for i in range(2)))
    return label, x.size, t_oracle, t_port, diff


def case_nd_fit(N: int, label: str):
    """ND fit benchmarked port-only (oracle's NDpartialsvector route is a
    different pybind surface; not worth wiring just for the bench)."""
    rng = np.random.default_rng(17 + N)
    x = rng.uniform(-3, 3, size=(N, 2))
    y = 1.0 + 0.5 * x[:, 0] + -0.3 * x[:, 1] + rng.normal(0, 0.1, N)

    def f_nd(xv, params):
        return params[0] + params[1] * xv[0] + params[2] * xv[1]

    def d0(xv, params):
        return 1.0

    def d1(xv, params):
        return xv[0]

    def d2(xv, params):
        return xv[1]

    def run_port():
        m = rcrpy.FunctionalForm(f_nd, x, y, [d0, d1, d2], guess=[0.0, 0.0, 0.0])
        m.regression()
        return m.parameters

    t_port = time_best_of_3(run_port)
    return label, N, None, t_port, None


def main() -> None:
    print(f"{'case':<40s} {'N':>5s}  {'oracle (ms)':>12s}  {'port (ms)':>11s}  "
          f"{'slowdown':>10s}  {'max |dp|':>10s}")
    print("-" * 100)

    results = []

    # Clean linear fits (no rejection)
    for N in (50, 200, 1000):
        results.append(case_linear_fit_no_rcr(N, f"linear clean fit"))

    # RCR + functional form on real data
    x, y = _load("data_linear.csv")
    results.append(case_linear_rcr("RCR linear (data_linear.csv)", x, y))

    # ND port-only
    results.append(case_nd_fit(200, "ND linear (3 params, port only)"))

    for label, N, t_or, t_port, diff in results:
        if t_or is None:
            slowdown_s = "  n/a"
            t_or_s = "    n/a"
            diff_s = "   n/a"
        else:
            slowdown_s = f"{t_port/t_or:>9.1f}x"
            t_or_s = f"{t_or*1000:>12.2f}"
            diff_s = f"{diff:>10.3e}"
        print(f"{label:<40s} {N:>5d}  {t_or_s}  {t_port*1000:>11.2f}  "
              f"{slowdown_s}  {diff_s}")


if __name__ == "__main__":
    main()
