"""Focused parity test in the *combo-sampling regime* — exercises the
weighted-roulette code path that was the source of the historic RNG
floor between port and oracle.

The functional-form sampler only triggers when C(n_kept, M) > 40000.
At M=2 that means n_kept > ~283. So we need N large enough that even
after some rejection, n_kept stays in the sampling regime.

Configs (all symmetric outliers, LS_MODE_68, linear model):
  - N=400, contam=0%, 10 trials  -> n_kept=400 entire run, sampling every iter
  - N=400, contam=5%, 10 trials  -> n_kept ~ 380, sampling triggered
  - N=600, contam=10%, 10 trials -> n_kept ~ 540, sampling triggered

Compares the post-fix port (MT19937(0xC0FFEE) + C++-matching sampler)
against the deterministic C++ oracle (same fix on the C++ side).

Run:
    python rcrpy/benchmarks/sampling_regime_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import full_rcr
import rcr


def linear(xv, p): return p[0] + p[1] * xv
def d_b(xv, p): return 1.0
def d_m(xv, p): return xv


def make_data(seed: int, N: int, contam_frac: float):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, N)
    y = 2.0 + 1.5 * x + rng.normal(0.0, 0.3, N)
    n_out = int(round(contam_frac * N))
    if n_out > 0:
        idx = rng.choice(N, size=n_out, replace=False)
        signs = rng.choice([-1.0, 1.0], n_out)
        y[idx] += signs * rng.normal(15.0, 5.0, n_out)
    return x, y


def run_config(label: str, N: int, contam: float, n_trials: int):
    diffs_b, diffs_m = [], []
    t_oracle = t_port = 0.0
    for seed in range(n_trials):
        x, y = make_data(seed, N, contam)
        # Oracle
        t0 = time.perf_counter()
        m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                  [d_b, d_m], [0.0, 1.0])
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.setParametricModel(m_o)
        r_o.performRejection(y.tolist())
        t_oracle += time.perf_counter() - t0
        ob, om = m_o.result.parameters
        # Port
        t0 = time.perf_counter()
        m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                       guess=[0.0, 1.0])
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.set_parametric_model(m_p)
        r_p.perform_rejection(y.tolist())
        t_port += time.perf_counter() - t0
        pb, pm = m_p.result.parameters
        diffs_b.append(abs(pb - ob))
        diffs_m.append(abs(pm - om))
    db_arr = np.array(diffs_b)
    dm_arr = np.array(diffs_m)
    bit_id = int(np.sum((db_arr < 1e-14) & (dm_arr < 1e-14)))
    near = int(np.sum((db_arr < 1e-9) & (dm_arr < 1e-9)))
    print(f"  {label:<30s}  bit-id={bit_id:>2d}/{n_trials}  "
          f"<1e-9={near:>2d}/{n_trials}  "
          f"med|db|={np.median(db_arr):.2e}  max|db|={np.max(db_arr):.2e}  "
          f"med|dm|={np.median(dm_arr):.2e}  max|dm|={np.max(dm_arr):.2e}  "
          f"cpp {t_oracle:.1f}s  port {t_port:.1f}s")


def main():
    print("=" * 100)
    print("Combo-sampling-regime parity test")
    print("Pre-fix expectation: ~1e-3 max divergence (RNG floor)")
    print("Post-fix expectation: bit-identical at sampling, residual from LM-vs-GN optimizer only")
    print("=" * 100)
    print()
    print(f"  {'config':<30s}  {'parity':<6}                  divergence stats                              perf")
    print("  " + "-" * 96)
    run_config("N=400, contam=0%", 400, 0.00, 10)
    run_config("N=400, contam=5%", 400, 0.05, 10)
    run_config("N=600, contam=10%", 600, 0.10, 10)


if __name__ == "__main__":
    main()
