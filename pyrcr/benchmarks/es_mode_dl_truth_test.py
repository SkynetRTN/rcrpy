"""Truth-recovery comparison for the one xfail-documented divergence:
ES_MODE_DL + parametric FunctionalForm.

The port and oracle are known to disagree by ~8-20% on this combination
(see test_es_mode_dl_parametric_parity xfail rationale and the
[[rcr2-parity-by-code-path]] memory). But disagreement between port and
oracle says nothing about which one is *correct*. This test directly
measures: when port and oracle give different answers, which gets
closer to the known truth?

Setup per trial:
  - Noisy linear data: y = 2.0 + 1.5*x + noise, with 20% symmetric outliers
  - N=200 points, 50 trials with different RNG seeds
  - Both implementations run RCR + ES_MODE_DL + linear FunctionalForm
  - Report |estimate - truth| for each; tally who was closer

The optimizer_quality_test.py covered LS_MODE_68; this completes the
picture by quantifying ES_MODE_DL specifically.
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
N = 200
CONTAM_FRAC = 0.20
N_TRIALS = 50


def linear(xv, p): return p[0] + p[1] * xv
def d_b(xv, p): return 1.0
def d_m(xv, p): return xv


def make_data(seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, N)
    y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, N)
    n_out = int(round(CONTAM_FRAC * N))
    idx = rng.choice(N, n_out, replace=False)
    signs = rng.choice([-1.0, 1.0], n_out)
    y[idx] += signs * rng.normal(15.0, 5.0, n_out)
    return x, y


def fit_oracle(x, y):
    m = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                            [d_b, d_m], [0.0, 1.0])
    r = rcr.RCR(rcr.ES_MODE_DL)
    r.setParametricModel(m)
    r.performRejection(y.tolist())
    return m.result.parameters


def fit_port(x, y):
    m = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m], guess=[0.0, 1.0])
    r = full_rcr.RCR(full_rcr.RejectionTech.ES_MODE_DL)
    r.set_parametric_model(m)
    r.perform_rejection(y.tolist())
    return m.result.parameters


def main():
    print(f"ES_MODE_DL + parametric truth-recovery comparison")
    print(f"N={N}, contam={int(CONTAM_FRAC*100)}%, {N_TRIALS} trials")
    print(f"Truth: b={TRUTH_B}, m={TRUTH_M}")
    print()

    err_b_o, err_m_o = [], []
    err_b_p, err_m_p = [], []
    disagreements_b, disagreements_m = [], []
    port_failed = oracle_failed = 0
    t_oracle = t_port = 0.0

    for seed in range(N_TRIALS):
        x, y = make_data(seed)
        # Oracle
        try:
            t0 = time.perf_counter()
            ob, om = fit_oracle(x, y)
            t_oracle += time.perf_counter() - t0
            if not (np.isfinite(ob) and np.isfinite(om)):
                oracle_failed += 1
                continue
        except Exception:
            oracle_failed += 1
            continue
        # Port
        try:
            t0 = time.perf_counter()
            pb, pm = fit_port(x, y)
            t_port += time.perf_counter() - t0
            if not (np.isfinite(pb) and np.isfinite(pm)):
                port_failed += 1
                continue
        except Exception:
            port_failed += 1
            continue
        err_b_o.append(abs(ob - TRUTH_B))
        err_m_o.append(abs(om - TRUTH_M))
        err_b_p.append(abs(pb - TRUTH_B))
        err_m_p.append(abs(pm - TRUTH_M))
        disagreements_b.append(abs(pb - ob))
        disagreements_m.append(abs(pm - om))

    n = len(err_b_o)
    print(f"Successful trials: {n}/{N_TRIALS}  "
          f"(failures: oracle {oracle_failed}, port {port_failed})")
    print()

    if n == 0:
        print("No successful comparisons — cannot draw conclusions.")
        return

    eb_o = np.array(err_b_o)
    em_o = np.array(err_m_o)
    eb_p = np.array(err_b_p)
    em_p = np.array(err_m_p)
    db_disagree = np.array(disagreements_b)
    dm_disagree = np.array(disagreements_m)

    print("--- Port-vs-oracle disagreement (confirms ~8-20% xfail rationale) ---")
    print(f"  |b_port - b_oracle|:  median {np.median(db_disagree):.4f}  "
          f"mean {np.mean(db_disagree):.4f}  max {np.max(db_disagree):.4f}")
    print(f"  |m_port - m_oracle|:  median {np.median(dm_disagree):.4f}  "
          f"mean {np.mean(dm_disagree):.4f}  max {np.max(dm_disagree):.4f}")
    print()

    print("--- Truth recovery (lower = closer to truth, the actual question) ---")
    print(f"  Oracle |b - truth|:  median {np.median(eb_o):.4f}  "
          f"mean {np.mean(eb_o):.4f}  max {np.max(eb_o):.4f}")
    print(f"  Port   |b - truth|:  median {np.median(eb_p):.4f}  "
          f"mean {np.mean(eb_p):.4f}  max {np.max(eb_p):.4f}")
    print(f"  Oracle |m - truth|:  median {np.median(em_o):.4f}  "
          f"mean {np.mean(em_o):.4f}  max {np.max(em_o):.4f}")
    print(f"  Port   |m - truth|:  median {np.median(em_p):.4f}  "
          f"mean {np.mean(em_p):.4f}  max {np.max(em_p):.4f}")
    print()

    port_better_b = int(np.sum(eb_p < eb_o))
    oracle_better_b = int(np.sum(eb_o < eb_p))
    tie_b = n - port_better_b - oracle_better_b
    port_better_m = int(np.sum(em_p < em_o))
    oracle_better_m = int(np.sum(em_o < em_p))
    tie_m = n - port_better_m - oracle_better_m

    print("--- Closer-to-truth tally ---")
    print(f"  b:  Port {port_better_b}  Oracle {oracle_better_b}  Tie {tie_b}")
    print(f"  m:  Port {port_better_m}  Oracle {oracle_better_m}  Tie {tie_m}")
    print()
    print(f"  Wall-clock:  oracle {t_oracle:.1f}s  port {t_port:.1f}s")


if __name__ == "__main__":
    main()
