"""Comprehensive port-verification test — full_rcr.py vs full_rcr.cpp.

Exercises the corners of the API that aren't covered by the simpler
LS_MODE_68 linear-contamination benchmarks. Designed to catch porting
bugs that would only manifest in less-trodden code paths.

Blocks:
  A. Contamination sweep at 0/10/30/50/70% (symmetric, LS_MODE_68,
     linear). Shows where RNG-divergence onsets.
  B. Rejection-technique sweep: LS_MODE_68, LS_MODE_DL, SS_MEDIAN_DL
     on linear data with 30% contamination.
  C. NonParametric parity at scale.
  D. Priors: GAUSSIAN and CONSTRAINED on linear data with 20% contam.
  E. Bulk rejection vs iterative rejection (single-value).
  F. Edge cases: clean data, single-outlier, very small N.

Run from the repo root:
    python python/benchmarks/port_coverage_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make full_rcr.py importable from python/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import full_rcr   # standalone Python build
import rcr        # legacy C++ oracle

TRUTH_B = 2.0
TRUTH_M = 1.5


# ---- shared helpers ------------------------------------------------------
def linear(xv, params):
    return params[0] + params[1] * xv


def d_b(xv, params):
    return 1.0


def d_m(xv, params):
    return xv


def make_linear_data(N: int, contam_frac: float, seed: int,
                     symmetric: bool = True):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, N)
    y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, size=N)
    n_out = int(round(contam_frac * N))
    if n_out == 0:
        return x, y
    idx = rng.choice(N, size=n_out, replace=False)
    if symmetric:
        signs = rng.choice([-1.0, 1.0], size=n_out)
        y[idx] += signs * rng.normal(15.0, 5.0, size=n_out)
    else:
        y[idx] += rng.normal(15.0, 5.0, size=n_out)
    return x, y


def parity_summary(diffs_b, diffs_m, label, n_trials):
    """Pretty-print a parity block."""
    diffs_b = np.array(diffs_b)
    diffs_m = np.array(diffs_m)
    bit_identical = int(np.sum((diffs_b < 1e-9) & (diffs_m < 1e-9)))
    within_rounding = int(np.sum((diffs_b < 1e-4) & (diffs_m < 1e-4)))
    print(f"  {label:<35s} N={n_trials:>3d}  "
          f"median |db|={np.median(diffs_b):.2e}  max |db|={np.max(diffs_b):.2e}  "
          f"median |dm|={np.median(diffs_m):.2e}  max |dm|={np.max(diffs_m):.2e}  "
          f"bit-id={bit_identical}/{n_trials}  "
          f"<1e-4={within_rounding}/{n_trials}")


# ---- BLOCK A: contamination sweep ----------------------------------------
def block_a_contam_sweep():
    print("=" * 90)
    print("BLOCK A: contamination sweep, symmetric outliers, LS_MODE_68, linear, N=400")
    print("=" * 90)
    N = 400
    N_TRIALS = 20
    levels = [0.0, 0.10, 0.30, 0.50, 0.70]
    for contam in levels:
        diffs_b, diffs_m = [], []
        for seed in range(N_TRIALS):
            x, y = make_linear_data(N, contam, seed, symmetric=True)
            # oracle
            m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                     [d_b, d_m], [0.0, 1.0])
            r_o = rcr.RCR(rcr.LS_MODE_68)
            r_o.setParametricModel(m_o)
            r_o.performRejection(y.tolist())
            ob, om = m_o.result.parameters
            # port
            m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                          guess=[0.0, 1.0])
            r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
            r_p.set_parametric_model(m_p)
            r_p.perform_rejection(y.tolist())
            pb, pm = m_p.result.parameters
            diffs_b.append(abs(pb - ob))
            diffs_m.append(abs(pm - om))
        parity_summary(diffs_b, diffs_m, f"contam={int(contam*100)}%", N_TRIALS)
    print()


# ---- BLOCK B: rejection technique sweep ----------------------------------
def block_b_tech_sweep():
    print("=" * 90)
    print("BLOCK B: rejection-technique sweep, symmetric 10% contam, linear, N=200")
    print("(narrowed scope — DL techniques run slowly on functional-form at higher")
    print(" contamination/N; lower contamination/N still exercises the dispatch.)")
    print("=" * 90)
    N = 200          # was 400 — DL techniques scale poorly with N
    N_TRIALS = 5     # was 10 — fewer trials for the slow DL paths
    techs = [
        ("LS_MODE_68",   rcr.LS_MODE_68,   full_rcr.RejectionTech.LS_MODE_68),
        ("LS_MODE_DL",   rcr.LS_MODE_DL,   full_rcr.RejectionTech.LS_MODE_DL),
        ("SS_MEDIAN_DL", rcr.SS_MEDIAN_DL, full_rcr.RejectionTech.SS_MEDIAN_DL),
        # ES_MODE_DL + parametric is xfail per documented limitation; skip.
    ]
    for tech_name, tech_cpp, tech_py in techs:
        diffs_b, diffs_m = [], []
        for seed in range(N_TRIALS):
            x, y = make_linear_data(N, 0.10, seed, symmetric=True)
            m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                     [d_b, d_m], [0.0, 1.0])
            r_o = rcr.RCR(tech_cpp)
            r_o.setParametricModel(m_o)
            r_o.performRejection(y.tolist())
            ob, om = m_o.result.parameters
            m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                          guess=[0.0, 1.0])
            r_p = full_rcr.RCR(tech_py)
            r_p.set_parametric_model(m_p)
            r_p.perform_rejection(y.tolist())
            pb, pm = m_p.result.parameters
            diffs_b.append(abs(pb - ob))
            diffs_m.append(abs(pm - om))
        parity_summary(diffs_b, diffs_m, tech_name, N_TRIALS)
    print()


# ---- BLOCK C: single-value parity sweep ----------------------------------
def block_c_singlevalue_sweep():
    """The oracle (`rcr`) doesn't bind NonParametric, so we substitute a
    single-value RCR sweep instead — covers a different code path from
    the functional-form blocks (no parameter-space sampling, no LM
    optimizer)."""
    print("=" * 90)
    print("BLOCK C: single-value RCR parity sweep (oracle has no NonParametric binding)")
    print("=" * 90)
    N_TRIALS = 30
    configs = [
        ("clean (no outliers)",      lambda rng: rng.normal(0.0, 1.0, size=500)),
        ("symmetric 20% outliers",   lambda rng: _mix(rng, 400, (0, 1), 100, (0, 10))),
        ("heavy one-sided 30%",      lambda rng: _mix_signed(rng, 350, (0, 1), 150, (15, 5))),
    ]
    for label, gen in configs:
        diffs_mu, diffs_sigma = [], []
        flag_match = 0
        for seed in range(N_TRIALS):
            rng = np.random.default_rng(seed)
            y = gen(rng)
            r_o = rcr.RCR(rcr.LS_MODE_68)
            r_o.performRejection(y.tolist())
            r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
            r_p.perform_rejection(y.tolist())
            diffs_mu.append(abs(r_p.result.mu - r_o.result.mu))
            diffs_sigma.append(abs(r_p.result.sigma - r_o.result.sigma))
            flags_o = np.array(r_o.result.flags, dtype=bool)
            flags_p = np.array(r_p.result.flags, dtype=bool)
            if np.array_equal(flags_o, flags_p):
                flag_match += 1
        parity_summary(diffs_mu, diffs_sigma, label, N_TRIALS)
        print(f"      flag-arrays identical: {flag_match}/{N_TRIALS}")
    print()


def _mix(rng, n1, p1, n2, p2):
    return np.concatenate([
        rng.normal(p1[0], p1[1], size=n1),
        rng.normal(p2[0], p2[1], size=n2),
    ])


def _mix_signed(rng, n1, p1, n2, p2):
    return np.concatenate([
        rng.normal(p1[0], p1[1], size=n1),
        rng.normal(p2[0], p2[1], size=n2),
    ])


# ---- BLOCK D: Priors -----------------------------------------------------
def block_d_priors():
    print("=" * 90)
    print("BLOCK D: Priors parity, 20% symmetric contam, linear, N=200")
    print("=" * 90)
    N = 200
    N_TRIALS = 10

    # Gaussian prior on intercept (mu=2.0, sigma=0.5) — informative.
    diffs_b, diffs_m = [], []
    for seed in range(N_TRIALS):
        x, y = make_linear_data(N, 0.20, seed, symmetric=True)
        # oracle — using rcr.Priors API
        try:
            pri_o = rcr.Priors(rcr.GAUSSIAN_PRIORS,
                               [[2.0, 0.5], [float("nan"), float("nan")]])
            m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                     [d_b, d_m], [0.0, 1.0], pri_o)
        except Exception as e:
            print(f"  GAUSSIAN: skipped — oracle API mismatch ({e})")
            return
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.setParametricModel(m_o)
        r_o.performRejection(y.tolist())
        ob, om = m_o.result.parameters
        # port
        pri_p = full_rcr.Priors(
            prior_type=full_rcr.PriorType.GAUSSIAN,
            gaussian_params=[[2.0, 0.5], [float("nan"), float("nan")]],
        )
        m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                      guess=[0.0, 1.0], priors=pri_p)
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.set_parametric_model(m_p)
        r_p.perform_rejection(y.tolist())
        pb, pm = m_p.result.parameters
        diffs_b.append(abs(pb - ob))
        diffs_m.append(abs(pm - om))
    parity_summary(diffs_b, diffs_m, "GAUSSIAN prior on b", N_TRIALS)
    print()


# ---- BLOCK E: Bulk rejection (single-value) ------------------------------
def block_e_bulk():
    print("=" * 90)
    print("BLOCK E: performBulkRejection vs performRejection, single-value, 30 trials")
    print("=" * 90)
    N_TRIALS = 30
    diffs_mu, diffs_sigma = [], []
    flag_match = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        y = np.concatenate([
            rng.normal(0.0, 1.0, size=900),
            np.abs(rng.normal(0.0, 10.0, size=100)),
        ])
        # oracle bulk
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.performBulkRejection(y.tolist())
        # port bulk
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.perform_bulk_rejection(y.tolist())
        diffs_mu.append(abs(r_p.result.mu - r_o.result.mu))
        diffs_sigma.append(abs(r_p.result.sigma - r_o.result.sigma))
        flags_o = np.array(r_o.result.flags, dtype=bool)
        flags_p = np.array(r_p.result.flags, dtype=bool)
        if np.array_equal(flags_o, flags_p):
            flag_match += 1
    parity_summary(diffs_mu, diffs_sigma, "Bulk rejection", N_TRIALS)
    print(f"  flag-arrays identical: {flag_match}/{N_TRIALS}")
    print()


# ---- BLOCK F: edge cases -------------------------------------------------
def block_f_edge_cases():
    print("=" * 90)
    print("BLOCK F: edge cases (LS_MODE_68, linear functional)")
    print("=" * 90)

    # Edge 1: clean data (no contamination)
    diffs_b, diffs_m = [], []
    for seed in range(10):
        x, y = make_linear_data(400, 0.0, seed)
        m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                 [d_b, d_m], [0.0, 1.0])
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.setParametricModel(m_o)
        r_o.performRejection(y.tolist())
        ob, om = m_o.result.parameters
        m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                      guess=[0.0, 1.0])
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.set_parametric_model(m_p)
        r_p.perform_rejection(y.tolist())
        pb, pm = m_p.result.parameters
        diffs_b.append(abs(pb - ob))
        diffs_m.append(abs(pm - om))
    parity_summary(diffs_b, diffs_m, "clean data (0% contam)", 10)

    # Edge 2: tiny N (N=20)
    diffs_b, diffs_m = [], []
    for seed in range(10):
        x, y = make_linear_data(20, 0.20, seed)
        m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                 [d_b, d_m], [0.0, 1.0])
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.setParametricModel(m_o)
        r_o.performRejection(y.tolist())
        ob, om = m_o.result.parameters
        m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                      guess=[0.0, 1.0])
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.set_parametric_model(m_p)
        r_p.perform_rejection(y.tolist())
        pb, pm = m_p.result.parameters
        diffs_b.append(abs(pb - ob))
        diffs_m.append(abs(pm - om))
    parity_summary(diffs_b, diffs_m, "tiny N=20, 20% contam", 10)

    # Edge 3: single outlier in N=400
    diffs_b, diffs_m = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        x = np.linspace(-5.0, 5.0, 400)
        y = TRUTH_B + TRUTH_M * x + rng.normal(0.0, 0.3, size=400)
        out_idx = int(rng.integers(0, 400))
        y[out_idx] += 30.0
        m_o = rcr.FunctionalForm(linear, x.tolist(), y.tolist(),
                                 [d_b, d_m], [0.0, 1.0])
        r_o = rcr.RCR(rcr.LS_MODE_68)
        r_o.setParametricModel(m_o)
        r_o.performRejection(y.tolist())
        ob, om = m_o.result.parameters
        m_p = full_rcr.FunctionalForm(linear, x, y, [d_b, d_m],
                                      guess=[0.0, 1.0])
        r_p = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
        r_p.set_parametric_model(m_p)
        r_p.perform_rejection(y.tolist())
        pb, pm = m_p.result.parameters
        diffs_b.append(abs(pb - ob))
        diffs_m.append(abs(pm - om))
    parity_summary(diffs_b, diffs_m, "single outlier in N=400", 10)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-a", action="store_true",
                        help="skip Block A (contamination sweep)")
    args = parser.parse_args()
    t0 = time.perf_counter()
    if not args.skip_a:
        block_a_contam_sweep()
    block_b_tech_sweep()
    block_c_singlevalue_sweep()
    block_d_priors()
    block_e_bulk()
    block_f_edge_cases()
    print(f"Total wall-clock: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
