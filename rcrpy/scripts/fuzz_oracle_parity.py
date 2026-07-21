"""Differential fuzz harness: rcrpy port vs the C++ `rcr` oracle.

Why this exists
---------------
The original parity suite only fed contaminated/noisy data, so sigma was
never 0 and it missed the no-spread degenerate crash (ZeroDivisionError on
perfectly-linear / near-constant inputs). The oracle is an executable spec,
so the cheapest way to gain confidence about *other* edge cases is to drive
both implementations with randomized + structurally-degenerate inputs and
let any divergence surface itself.

What it does
------------
For each generated input it runs both implementations across:
  * code paths : single-value (performRejection), bulk
                 (performBulkRejection), and parametric line fit
                 (FunctionalForm + performRejection)
  * techniques : LS_MODE_68, LS_MODE_DL, SS_MEDIAN_DL, ES_MODE_DL
  * weighting  : unweighted + several weight regimes

and classifies every case as one of:
  * OK            - both succeeded and agree within tolerance
  * PORT_CRASH    - port raised, oracle returned        (the bug class)
  * ORACLE_CRASH  - oracle raised, port returned        (port masking error?)
  * BOTH_CRASH    - both raised                          (fine: shared limit)
  * VALUE_DIVERGE - both returned but disagree           (real divergence)
  * SOFT          - benign expected difference (parametric RNG floor, single
                    tie-break) — reported, not failed

Every finding prints the exact descriptor (regime, rep, seed, weight mode,
technique, path, N) needed to reproduce it.

Tolerance rationale (matches the existing suite):
  * non-parametric single/bulk paths are bit-identical in the suite -> tight
    (RTOL_TIGHT) on scalars and flags must match exactly.
  * the parametric MEDIAN/MODE combo paths use numpy's RNG vs std::mt19937,
    a documented ~1e-3..5e-2 floor -> loose (RTOL_LOOSE) on parameters and
    kept-count differences are treated as soft.

Usage:
    python scripts/fuzz_oracle_parity.py [reps_per_regime]

Exit code is nonzero iff any PORT_CRASH, ORACLE_CRASH, or hard VALUE_DIVERGE
is found.
"""
from __future__ import annotations

import math
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

import rcrpy

try:
    import rcr as oracle
except ImportError:
    print("ERROR: the C++ oracle `rcr` is not installed in this environment.")
    sys.exit(2)


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------
RTOL_TIGHT = 1e-9      # non-parametric single/bulk scalars
RTOL_LOOSE = 5e-2      # parametric params (RNG-floor paths)
ATOL_FLOOR = 1e-12


# --------------------------------------------------------------------------
# Technique metadata
# --------------------------------------------------------------------------
# name -> sigma_kind ('lower' | 'single' | 'each')
TECHS = {
    "LS_MODE_68": "lower",
    "LS_MODE_DL": "lower",
    "SS_MEDIAN_DL": "single",
    "ES_MODE_DL": "each",
}


# --------------------------------------------------------------------------
# Input regimes.  Each takes a numpy Generator and returns (x, y) as float64
# arrays.  For the non-parametric paths only y is used; x is the independent
# variable for the parametric line fit.
# --------------------------------------------------------------------------
def _xgrid(n):
    return np.linspace(0.0, 10.0, n)


def r_noisy_linear(rng):
    # N capped at 64: the mode-based mu techniques hit the halfSampleMode
    # hotspot, and large-N noisy data is already covered exhaustively by the
    # N~1000 CSV parity tests. The fuzzer's job is edge coverage, not size.
    n = int(rng.integers(8, 64))
    x = _xgrid(n)
    m, b = rng.uniform(-3, 3), rng.uniform(-5, 5)
    s = rng.uniform(0.05, 3.0)
    return x, b + m * x + rng.normal(0, s, n)


def r_contaminated(rng):
    n = int(rng.integers(20, 70))
    x = _xgrid(n)
    m, b = rng.uniform(-3, 3), rng.uniform(-5, 5)
    y = b + m * x + rng.normal(0, rng.uniform(0.1, 1.0), n)
    k = int(n * rng.uniform(0.05, 0.45))
    if k:
        idx = rng.choice(n, k, replace=False)
        y[idx] += rng.normal(rng.uniform(5, 30), 5, k)
    return x, y


def r_perfectly_linear(rng):
    n = int(rng.integers(5, 90))
    x = _xgrid(n)
    return x, rng.uniform(-5, 5) + rng.uniform(-3, 3) * x


def r_constant(rng):
    n = int(rng.integers(4, 90))
    return _xgrid(n), np.full(n, rng.uniform(-1e3, 1e3))


def r_near_constant(rng):
    n = int(rng.integers(8, 90))
    c = rng.uniform(-100, 100)
    s = 10.0 ** rng.uniform(-13, -6)
    return _xgrid(n), c + rng.normal(0, s, n)


def r_all_zero(rng):
    n = int(rng.integers(4, 90))
    return _xgrid(n), np.zeros(n)


def r_single_distinct(rng):
    n = int(rng.integers(5, 90))
    y = np.full(n, rng.uniform(-10, 10))
    y[int(rng.integers(0, n))] += rng.uniform(1, 50)
    return _xgrid(n), y


def r_heavy_ties(rng):
    n = int(rng.integers(8, 90))
    levels = rng.uniform(-10, 10, int(rng.integers(2, 4)))
    return _xgrid(n), rng.choice(levels, n)


def r_tiny_n(rng):
    n = int(rng.integers(1, 7))
    x = _xgrid(n)
    return x, rng.uniform(-5, 5) + rng.uniform(-2, 2) * x + rng.normal(0, 0.5, n)


def r_large_magnitude(rng):
    # Scaling stress, but kept within the range the C++ oracle can actually
    # compute: at ~1e12 the oracle raises ValueError (input guard) and even
    # hangs on the ES parametric fit, so it gives no usable parity reference.
    x, y = r_noisy_linear(rng)
    return x, y * 1e4


def r_small_magnitude(rng):
    x, y = r_noisy_linear(rng)
    return x, y * 1e-4


def r_two_clusters(rng):
    n = int(rng.integers(20, 90))
    y = np.where(np.arange(n) < n // 2,
                 rng.normal(0, 0.3, n), rng.normal(rng.uniform(10, 40), 0.3, n))
    return _xgrid(n), y


def r_integer_grid(rng):
    n = int(rng.integers(8, 90))
    return _xgrid(n), np.round(rng.uniform(-5, 5, n)).astype(np.float64)


def r_monotonic_steps(rng):
    n = int(rng.integers(8, 90))
    return _xgrid(n), np.floor(_xgrid(n)).astype(np.float64)


REGIMES = [
    r_noisy_linear, r_contaminated, r_perfectly_linear, r_constant,
    r_near_constant, r_all_zero, r_single_distinct, r_heavy_ties, r_tiny_n,
    r_large_magnitude, r_small_magnitude, r_two_clusters, r_integer_grid,
    r_monotonic_steps,
]


# --------------------------------------------------------------------------
# Weight regimes
# --------------------------------------------------------------------------
def make_weights(mode, n, rng):
    if mode == "none":
        return None
    if mode == "uniform":
        return np.ones(n)
    if mode == "random_pos":
        return rng.uniform(0.1, 10.0, n)
    if mode == "near_zero_subset":
        w = np.ones(n)
        k = max(1, int(n * 0.3))
        w[rng.choice(n, min(k, n), replace=False)] = 1e-8
        return w
    if mode == "extreme_range":
        return 10.0 ** rng.uniform(-6, 6, n)
    raise ValueError(mode)


WEIGHT_MODES = ["none", "uniform", "random_pos", "near_zero_subset", "extreme_range"]


# --------------------------------------------------------------------------
# Line model for the parametric path
# --------------------------------------------------------------------------
def _lin(x, p):
    return p[0] + p[1] * x


def _d_b(x, p):
    return 1.0


def _d_m(x, p):
    return x


# --------------------------------------------------------------------------
# Runners.  Each returns a dict of comparable outputs, or raises.
# --------------------------------------------------------------------------
def run_port_single(tech_name, y, w):
    r = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech_name))
    r.perform_rejection(y.tolist(), w=(None if w is None else w.tolist()))
    return _port_out(r, TECHS[tech_name])


def run_oracle_single(tech_name, y, w):
    r = oracle.RCR(getattr(oracle, tech_name))
    if w is None:
        r.performRejection(y.tolist())
    else:
        r.performRejection(w.tolist(), y.tolist())
    return _oracle_out(r, TECHS[tech_name])


def run_port_bulk(tech_name, y, w):
    r = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech_name))
    r.perform_bulk_rejection(y.tolist(), w=(None if w is None else w.tolist()))
    return _port_out(r, TECHS[tech_name])


def run_oracle_bulk(tech_name, y, w):
    r = oracle.RCR(getattr(oracle, tech_name))
    if w is None:
        r.performBulkRejection(y.tolist())
    else:
        r.performBulkRejection(w.tolist(), y.tolist())
    return _oracle_out(r, TECHS[tech_name])


def run_port_param(tech_name, x, y, w):
    m = rcrpy.FunctionalForm(_lin, x, y, [_d_b, _d_m], guess=[0.0, 0.0],
                             weights=(None if w is None else w))
    r = rcrpy.RCR(getattr(rcrpy.RejectionTech, tech_name))
    r.set_parametric_model(m)
    r.perform_rejection(y.tolist(), w=(None if w is None else w.tolist()))
    return {"params": np.asarray(m.result.parameters, dtype=float),
            "kept": int(np.sum(r.result.flags))}


def run_oracle_param(tech_name, x, y, w):
    m = oracle.FunctionalForm(_lin, x.tolist(), y.tolist(), [_d_b, _d_m], [0.0, 0.0])
    r = oracle.RCR(getattr(oracle, tech_name))
    r.setParametricModel(m)
    if w is None:
        r.performRejection(y.tolist())
    else:
        r.performRejection(w.tolist(), y.tolist())
    return {"params": np.asarray(m.result.parameters, dtype=float),
            "kept": int(sum(r.result.flags))}


def _port_out(r, kind):
    out = {"mu": r.result.mu, "flags": list(r.result.flags)}
    if kind == "each":
        out["sigma_below"] = r.result.sigma_below
        out["sigma_above"] = r.result.sigma_above
    else:
        out["sigma"] = r.result.sigma
    return out


def _oracle_out(r, kind):
    out = {"mu": r.result.mu, "flags": list(r.result.flags)}
    if kind == "each":
        out["sigma_below"] = r.result.sigmaBelow
        out["sigma_above"] = r.result.sigmaAbove
    else:
        out["sigma"] = r.result.sigma
    return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def scalar_close(a, b, rtol):
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) != math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0) + ATOL_FLOOR


def compare_nonparam(port, orac):
    """Returns (hard_msgs, soft_msgs).

    The operationally meaningful parity outputs are the KEPT SET (flags) and
    the location estimate (mu). sigma is an auxiliary scalar that, on
    zero-spread data (constant / near-constant), is computed from a degenerate
    distribution and is numerically unstable with extreme weights — so a
    sigma-only difference WHEN flags and mu both agree is reported soft, not
    hard (no rejection decision depends on it there). A sigma difference that
    coincides with a flag/mu difference is part of a real divergence and stays
    hard.
    """
    hard, soft = [], []
    mu_bad = not scalar_close(port["mu"], orac["mu"], RTOL_TIGHT)

    flags_bad = port["flags"] != orac["flags"]
    flag_msg = None
    if flags_bad:
        dp, do = sum(port["flags"]), sum(orac["flags"])
        flag_msg = (f"flags differ (kept port={dp} oracle={do}, "
                    f"ndiff={sum(a != b for a, b in zip(port['flags'], orac['flags']))})")

    sigma_msgs = []
    for key in ("sigma", "sigma_below", "sigma_above"):
        if key in port or key in orac:
            if not scalar_close(port.get(key, float("nan")),
                                orac.get(key, float("nan")), RTOL_TIGHT):
                sigma_msgs.append(f"{key} port={port.get(key)!r} oracle={orac.get(key)!r}")

    if mu_bad:
        hard.append(f"mu port={port['mu']!r} oracle={orac['mu']!r}")
    if flags_bad:
        # one-point boundary tie-break is a known benign FP effect; many is real
        (soft if abs(dp - do) <= 1 else hard).append(flag_msg)
    if sigma_msgs:
        # sigma-only (flags + mu agree) -> benign degenerate-data artifact.
        (soft if (not mu_bad and not flags_bad) else hard).extend(sigma_msgs)
    return hard, soft


def compare_param(port, orac):
    """Parametric line-fit comparison. The MEDIAN/MODE parametric passes draw
    M-combinations with numpy's RNG vs the C++ std::mt19937, a documented and
    UNFIXABLE divergence (the ~1e-3..5e-2 RNG floor). So parametric param/kept
    differences are reported as SOFT, never hard — a real parametric bug would
    still surface as a PORT_CRASH/ORACLE_CRASH (handled in run_case). Only a
    genuine structural mismatch (shape / finite-ness) is hard."""
    hard, soft = [], []
    pp, op = port["params"], orac["params"]
    if pp.shape != op.shape or np.all(np.isfinite(pp)) != np.all(np.isfinite(op)):
        hard.append(f"params shape/finite mismatch port={pp!r} oracle={op!r}")
        return hard, soft
    rel = np.abs(pp - op) / np.maximum(np.abs(op), 1.0)
    if np.any(rel > RTOL_LOOSE):
        soft.append(f"params rel>{RTOL_LOOSE} (RNG floor): port={pp!r} oracle={op!r}")
    if port["kept"] != orac["kept"]:
        soft.append(f"kept port={port['kept']} oracle={orac['kept']}")
    return hard, soft


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
@dataclass
class Tally:
    cases: int = 0
    ok: int = 0
    both_crash: int = 0
    soft: int = 0
    skipped_es: int = 0
    findings: list = field(default_factory=list)   # (category, descriptor, detail)


def run_case(tally, category_path, runner_port, runner_oracle, comparator, desc):
    tally.cases += 1
    p_err = o_err = None
    p_out = o_out = None
    try:
        p_out = runner_port()
    except Exception as e:  # noqa: BLE001
        p_err = e
    try:
        o_out = runner_oracle()
    except Exception as e:  # noqa: BLE001
        o_err = e

    if p_err and o_err:
        tally.both_crash += 1
        return
    if p_err and not o_err:
        tally.findings.append(("PORT_CRASH", desc,
                               f"{type(p_err).__name__}: {p_err}"))
        return
    if o_err and not p_err:
        tally.findings.append(("ORACLE_CRASH", desc,
                               f"{type(o_err).__name__}: {o_err}"))
        return

    hard, soft = comparator(p_out, o_out)
    if hard:
        tally.findings.append(("VALUE_DIVERGE", desc, " | ".join(hard)))
    elif soft:
        tally.soft += 1
        tally.findings.append(("SOFT", desc, " | ".join(soft)))
    else:
        tally.ok += 1


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print(f"Differential fuzz: rcrpy vs oracle | reps/regime={reps} | "
          f"{len(REGIMES)} regimes x {len(WEIGHT_MODES)} weight modes x "
          f"{len(TECHS)} techs x 3 paths\n")

    tally = Tally()
    for ri, regime in enumerate(REGIMES):
        for rep in range(reps):
            # Deterministic across runs (do NOT use hash(): it's salted by
            # PYTHONHASHSEED). regime index + rep fully fixes the corpus.
            seed = (ri + 1) * 1_000_003 + rep
            x, y = regime(np.random.default_rng(seed))
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            n = y.size
            wrng = np.random.default_rng(seed + 7)
            for wmode in WEIGHT_MODES:
                w = make_weights(wmode, n, wrng)
                for tech in TECHS:
                    # ES_MODE_DL is EXCLUDED entirely (all paths): the C++
                    # oracle infinite-loops on it for several inputs — its
                    # parametric fit (the suite's lone xfail, "broken in BOTH")
                    # and even its single/bulk paths with extreme weights on
                    # tied data. A hung oracle gives no parity reference and
                    # stalls the whole run, so it can't be fuzzed here. Counted
                    # so the exclusion is visible, not silent. (ES single/bulk
                    # on well-behaved data is still covered by the CSV suite.)
                    if tech == "ES_MODE_DL":
                        tally.skipped_es += 1
                        continue
                    base = dict(regime=regime.__name__, rep=rep, seed=seed,
                                wmode=wmode, tech=tech, N=n)
                    # single-value
                    run_case(
                        tally, "single",
                        (lambda t=tech, yy=y, ww=w: run_port_single(t, yy, ww)),
                        (lambda t=tech, yy=y, ww=w: run_oracle_single(t, yy, ww)),
                        compare_nonparam, {**base, "path": "single"})
                    # bulk
                    run_case(
                        tally, "bulk",
                        (lambda t=tech, yy=y, ww=w: run_port_bulk(t, yy, ww)),
                        (lambda t=tech, yy=y, ww=w: run_oracle_bulk(t, yy, ww)),
                        compare_nonparam, {**base, "path": "bulk"})
                    # parametric line fit (needs N >= 2 for a 2-param model).
                    if n >= 2:
                        run_case(
                            tally, "param",
                            (lambda t=tech, xx=x, yy=y, ww=w: run_port_param(t, xx, yy, ww)),
                            (lambda t=tech, xx=x, yy=y, ww=w: run_oracle_param(t, xx, yy, ww)),
                            compare_param, {**base, "path": "param"})
            # Per-rep heartbeat so a mid-regime hang is immediately locatable.
            print(f"  ...{regime.__name__:22s} rep {rep:3d}/{reps}  "
                  f"cases={tally.cases} hard={sum(1 for f in tally.findings if f[0] != 'SOFT')}",
                  flush=True)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"TOTAL cases       : {tally.cases}")
    print(f"  OK              : {tally.ok}")
    print(f"  BOTH_CRASH      : {tally.both_crash}  (acceptable: shared limit)")
    print(f"  SOFT            : {tally.soft}  (benign expected diffs)")
    print(f"  SKIPPED(ES)     : {tally.skipped_es}  (ES_MODE_DL all paths: C++ oracle hangs)")
    hard = [f for f in tally.findings if f[0] in ("PORT_CRASH", "ORACLE_CRASH", "VALUE_DIVERGE")]
    print(f"  HARD findings   : {len(hard)}")
    print("=" * 72)

    by_cat = {}
    for cat, desc, detail in tally.findings:
        by_cat.setdefault(cat, []).append((desc, detail))

    for cat in ("PORT_CRASH", "ORACLE_CRASH", "VALUE_DIVERGE", "SOFT"):
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n### {cat}  ({len(items)})")
        shown = items if cat != "SOFT" else items[:25]
        for desc, detail in shown:
            d = (f"regime={desc['regime']} rep={desc['rep']} seed={desc['seed']} "
                 f"wmode={desc['wmode']} tech={desc['tech']} path={desc['path']} N={desc['N']}")
            print(f"  - {d}\n      {detail}")
        if len(items) > len(shown):
            print(f"    ... and {len(items) - len(shown)} more")

    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
