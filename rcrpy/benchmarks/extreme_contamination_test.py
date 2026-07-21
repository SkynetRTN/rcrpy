"""Contamination stress test — full_rcr.cpp vs full_rcr.py.

Maples et al. 2018 advertises RCR as robust up to ~85% contamination on
linear functional-form fits. This script puts both implementations
through a configurable contamination level to verify the Python port
preserves the published behavior.

Setup per trial:
  - N=400 synthetic points on y = 2.0 + 1.5*x  with sigma=0.3 inlier noise
  - CONTAM_FRAC of points replaced with heavy-tailed outliers in y
    (mean shift ~15, sigma 5)
  - Both implementations: RCR + LS_MODE_68 on the same data
  - Truth: b=2.0, m=1.5

Run with defaults:
    python rcrpy/benchmarks/extreme_contamination_test.py

Override contamination and trial count:
    python rcrpy/benchmarks/extreme_contamination_test.py \\
        --contam 0.50 --trials 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make full_rcr.py importable from python/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import full_rcr   # the standalone single-file Python build
import rcr        # legacy C++ oracle (proxy for full_rcr.cpp)


TRUTH_B = 2.0
TRUTH_M = 1.5
N = 400
CONTAM_FRAC = 0.85   # overridable via --contam
N_TRIALS = 10        # overridable via --trials


def linear(xv, params):
    return params[0] + params[1] * xv


def d_b(xv, params):
    return 1.0


def d_m(xv, params):
    return xv


def make_data(seed: int, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, N)
    y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, size=N)
    n_out = int(round(CONTAM_FRAC * N))
    idx = rng.choice(N, size=n_out, replace=False)
    if symmetric:
        # Symmetric heavy-tailed contamination — what Maples et al.
        # claim 85% robustness against. Both signs, equal probability.
        signs = rng.choice([-1.0, 1.0], size=n_out)
        y[idx] += signs * rng.normal(15.0, 5.0, size=n_out)
    else:
        # One-sided contamination — much harder; outliers have coherent pull.
        y[idx] += rng.normal(15.0, 5.0, size=n_out)
    return x, y


def run_port(x, y):
    t = time.perf_counter()
    m = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m], guess=[0.0, 1.0])
    r = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
    r.set_parametric_model(m)
    r.perform_rejection(y.tolist())
    dt = time.perf_counter() - t
    b, mm = m.result.parameters
    kept = int(r.result.flags.sum())
    return b, mm, kept, dt


def run_oracle(x, y):
    t = time.perf_counter()
    m = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                            [d_b, d_m], [0.0, 1.0])
    r = rcr.RCR(rcr.LS_MODE_68)
    r.setParametricModel(m)
    r.performRejection(y.tolist())
    dt = time.perf_counter() - t
    b, mm = m.result.parameters
    kept = int(sum(r.result.flags))
    return b, mm, kept, dt


def run_block(label: str, symmetric: bool) -> None:
    print(f"=== {label} ===")
    print(f"Stress test: {int(CONTAM_FRAC*100)}% contamination on linear fit "
          f"(N={N}, {N_TRIALS} trials, "
          f"{'symmetric' if symmetric else 'one-sided'} outliers)")
    print(f"Truth: b={TRUTH_B}, m={TRUTH_M}")
    print(f"Outliers per trial: {int(round(CONTAM_FRAC * N))} of {N} "
          f"(inliers: {N - int(round(CONTAM_FRAC * N))})")
    print()
    print(f"{'seed':>4s}  {'cpp b':>9s} {'cpp m':>9s} {'cpp kept':>9s} "
          f"{'cpp ms':>8s}  {'py b':>9s} {'py m':>9s} {'py kept':>8s} "
          f"{'py ms':>8s}  {'|db|':>8s} {'|dm|':>8s}")
    print("-" * 122)

    rows = []
    for seed in range(N_TRIALS):
        x, y = make_data(seed, symmetric=symmetric)
        ob, om, ok, ot = run_oracle(x, y)
        pb, pm, pk, pt = run_port(x, y)
        db = abs(pb - ob)
        dm = abs(pm - om)
        rows.append((seed, ob, om, ok, ot, pb, pm, pk, pt, db, dm))
        print(f"{seed:>4d}  {ob:>9.4f} {om:>9.4f} {ok:>9d} {ot*1000:>8.1f}  "
              f"{pb:>9.4f} {pm:>9.4f} {pk:>8d} {pt*1000:>8.1f}  "
              f"{db:>8.2e} {dm:>8.2e}")

    print()
    cpp_bs = np.array([r[1] for r in rows])
    cpp_ms = np.array([r[2] for r in rows])
    py_bs = np.array([r[5] for r in rows])
    py_ms = np.array([r[6] for r in rows])
    cpp_times = np.array([r[4] for r in rows])
    py_times = np.array([r[8] for r in rows])

    print("--- Accuracy (recovery of truth, lower is better) ---")
    print(f"  C++  median |b - truth| = {np.median(np.abs(cpp_bs - TRUTH_B)):.4f}, "
          f"max = {np.max(np.abs(cpp_bs - TRUTH_B)):.4f}")
    print(f"  C++  median |m - truth| = {np.median(np.abs(cpp_ms - TRUTH_M)):.4f}, "
          f"max = {np.max(np.abs(cpp_ms - TRUTH_M)):.4f}")
    print(f"  Py   median |b - truth| = {np.median(np.abs(py_bs - TRUTH_B)):.4f}, "
          f"max = {np.max(np.abs(py_bs - TRUTH_B)):.4f}")
    print(f"  Py   median |m - truth| = {np.median(np.abs(py_ms - TRUTH_M)):.4f}, "
          f"max = {np.max(np.abs(py_ms - TRUTH_M)):.4f}")

    print()
    print("--- Parity (Python vs C++ implementation agreement) ---")
    print(f"  median |b_py - b_cpp| = {np.median([r[9]  for r in rows]):.4f}, "
          f"max = {np.max([r[9]  for r in rows]):.4f}")
    print(f"  median |m_py - m_cpp| = {np.median([r[10] for r in rows]):.4f}, "
          f"max = {np.max([r[10] for r in rows]):.4f}")

    print()
    print("--- Performance ---")
    print(f"  median C++ time = {np.median(cpp_times)*1000:.1f} ms")
    print(f"  median Py  time = {np.median(py_times)*1000:.1f} ms")
    print(f"  speedup (cpp/py) = {np.median(cpp_times)/np.median(py_times):.1f}x")
    print()

    # "Who was closer to truth?" tally — answers whether one impl is
    # systematically closer than the other, or whether differences are noise.
    py_closer_b = sum(abs(r[5] - TRUTH_B) < abs(r[1] - TRUTH_B) for r in rows)
    cpp_closer_b = sum(abs(r[1] - TRUTH_B) < abs(r[5] - TRUTH_B) for r in rows)
    tie_b = N_TRIALS - py_closer_b - cpp_closer_b
    py_closer_m = sum(abs(r[6] - TRUTH_M) < abs(r[2] - TRUTH_M) for r in rows)
    cpp_closer_m = sum(abs(r[2] - TRUTH_M) < abs(r[6] - TRUTH_M) for r in rows)
    tie_m = N_TRIALS - py_closer_m - cpp_closer_m

    print("--- Who was closer to truth? (out of {} trials) ---".format(N_TRIALS))
    print(f"  Intercept b:  Py closer in {py_closer_b}, C++ closer in "
          f"{cpp_closer_b}, tie/identical in {tie_b}")
    print(f"  Slope m:      Py closer in {py_closer_m}, C++ closer in "
          f"{cpp_closer_m}, tie/identical in {tie_m}")
    print()
    print("  (If parity is RNG-driven and unbiased, expect ~5/5 splits.)")
    print()


def main() -> None:
    global CONTAM_FRAC, N_TRIALS
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--contam", type=float, default=CONTAM_FRAC,
                        help=f"contamination fraction in [0, 1] "
                             f"(default {CONTAM_FRAC})")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"trials per mode (default {N_TRIALS})")
    args = parser.parse_args()
    CONTAM_FRAC = args.contam
    N_TRIALS = args.trials
    pct = int(round(CONTAM_FRAC * 100))
    run_block(f"SYMMETRIC outliers ({pct}% contamination)", symmetric=True)
    run_block(f"ONE-SIDED outliers ({pct}% contamination)", symmetric=False)


if __name__ == "__main__":
    main()
