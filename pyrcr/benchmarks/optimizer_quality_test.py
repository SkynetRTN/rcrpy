"""Optimizer-quality test: scipy.least_squares (port) vs hand-rolled
Gauss-Newton (C++ oracle) — which recovers truth more accurately?

For each config we generate many noisy realizations with KNOWN true
parameters, fit both implementations, and compute |estimate - truth|.
Reports per-implementation accuracy distribution + a "who won" tally.

Three configs:
  1. Well-conditioned linear data (x spans wide range, low contam) —
     both optimizers should be near-perfect; tests that scipy isn't
     LOSING anything in the easy case.
  2. Ill-conditioned linear data (x clustered in narrow range, near-
     singular Jacobian) — this is the classic case where pure GN
     struggles and LM's damping is supposed to help.
  3. High contamination (60%) — stresses the rejection process more
     than the optimizer; serves as a control.

Truth: b=2.0, m=1.5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import full_rcr
import rcr

TRUTH_B = 2.0
TRUTH_M = 1.5


def linear(xv, p): return p[0] + p[1] * xv
def d_b(xv, p): return 1.0
def d_m(xv, p): return xv


def well_conditioned(seed, N=200, contam_frac=0.20):
    """Clean linear data; x spans wide range. Jacobian well-conditioned."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, N)
    y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, N)
    n_out = int(round(contam_frac * N))
    if n_out > 0:
        idx = rng.choice(N, n_out, replace=False)
        signs = rng.choice([-1.0, 1.0], n_out)
        y[idx] += signs * rng.normal(15.0, 5.0, n_out)
    return x, y


def ill_conditioned(seed, N=200, contam_frac=0.20):
    """Linear data with x clustered in a narrow range — Jacobian becomes
    nearly singular (the constant column and the small-range x column
    become nearly collinear at this scale). This is where Gauss-Newton
    is supposed to struggle and LM's damping is supposed to save the day.
    """
    rng = np.random.default_rng(seed)
    # x clustered in a narrow band around the mean
    x = rng.normal(0.0, 0.05, N)  # range ~ 0.4 vs intercept ~ 2.0
    y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, N)
    n_out = int(round(contam_frac * N))
    if n_out > 0:
        idx = rng.choice(N, n_out, replace=False)
        signs = rng.choice([-1.0, 1.0], n_out)
        y[idx] += signs * rng.normal(15.0, 5.0, n_out)
    return x, y


def high_contam(seed, N=200, contam_frac=0.60):
    """Standard well-conditioned data but 60% contamination — stresses
    rejection more than optimizer."""
    return well_conditioned(seed, N, contam_frac)


def fit_oracle(x, y):
    m = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                            [d_b, d_m], [0.0, 1.0])
    r = rcr.RCR(rcr.LS_MODE_68)
    r.setParametricModel(m)
    r.performRejection(y.tolist())
    return m.result.parameters


def fit_port(x, y):
    m = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m], guess=[0.0, 1.0])
    r = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
    r.set_parametric_model(m)
    r.perform_rejection(y.tolist())
    return m.result.parameters


def run_block(label: str, gen_func, n_trials: int = 50):
    print(f"=== {label} ===")
    err_b_oracle, err_m_oracle = [], []
    err_b_port, err_m_port = [], []
    t_oracle = t_port = 0.0
    failures_oracle = failures_port = 0

    for seed in range(n_trials):
        x, y = gen_func(seed)
        try:
            t0 = time.perf_counter()
            ob, om = fit_oracle(x, y)
            t_oracle += time.perf_counter() - t0
            if not (np.isfinite(ob) and np.isfinite(om)):
                failures_oracle += 1
                continue
        except Exception:
            failures_oracle += 1
            continue
        try:
            t0 = time.perf_counter()
            pb, pm = fit_port(x, y)
            t_port += time.perf_counter() - t0
            if not (np.isfinite(pb) and np.isfinite(pm)):
                failures_port += 1
                continue
        except Exception:
            failures_port += 1
            continue
        err_b_oracle.append(abs(ob - TRUTH_B))
        err_m_oracle.append(abs(om - TRUTH_M))
        err_b_port.append(abs(pb - TRUTH_B))
        err_m_port.append(abs(pm - TRUTH_M))

    eb_o = np.array(err_b_oracle)
    em_o = np.array(err_m_oracle)
    eb_p = np.array(err_b_port)
    em_p = np.array(err_m_port)

    # Who got closer to truth, per trial?
    port_better_b = int(np.sum(eb_p < eb_o))
    oracle_better_b = int(np.sum(eb_o < eb_p))
    tie_b = len(eb_p) - port_better_b - oracle_better_b
    port_better_m = int(np.sum(em_p < em_o))
    oracle_better_m = int(np.sum(em_o < em_p))
    tie_m = len(em_p) - port_better_m - oracle_better_m

    print(f"  N successful trials:  oracle {len(eb_o)}/{n_trials}  "
          f"port {len(eb_p)}/{n_trials}  "
          f"(failures: oracle {failures_oracle}, port {failures_port})")
    print(f"  Recovery of b (truth={TRUTH_B}):")
    print(f"    Oracle median |db|={np.median(eb_o):.4f}  mean={np.mean(eb_o):.4f}  max={np.max(eb_o):.4f}")
    print(f"    Port   median |db|={np.median(eb_p):.4f}  mean={np.mean(eb_p):.4f}  max={np.max(eb_p):.4f}")
    print(f"  Recovery of m (truth={TRUTH_M}):")
    print(f"    Oracle median |dm|={np.median(em_o):.4f}  mean={np.mean(em_o):.4f}  max={np.max(em_o):.4f}")
    print(f"    Port   median |dm|={np.median(em_p):.4f}  mean={np.mean(em_p):.4f}  max={np.max(em_p):.4f}")
    print(f"  Closer-to-truth tally:")
    print(f"    b:  Port {port_better_b}  Oracle {oracle_better_b}  Tie {tie_b}")
    print(f"    m:  Port {port_better_m}  Oracle {oracle_better_m}  Tie {tie_m}")
    print(f"  Wall-clock:  oracle {t_oracle:.1f}s  port {t_port:.1f}s")
    print()


def main():
    print("Optimizer-quality test: scipy.least_squares vs hand-rolled Gauss-Newton")
    print("Question: does scipy actually recover truth more accurately?")
    print()
    run_block("Config 1: well-conditioned linear, 20% contam, N=200, 50 trials",
              well_conditioned, n_trials=50)
    run_block("Config 2: ILL-CONDITIONED (clustered x), 20% contam, N=200, 50 trials",
              ill_conditioned, n_trials=50)
    run_block("Config 3: 60% contamination, N=200, 50 trials (stress on rejection, not optimizer)",
              high_contam, n_trials=50)


if __name__ == "__main__":
    main()
