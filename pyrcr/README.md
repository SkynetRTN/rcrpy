# pyrcr

Python reimplementation of Robust Chauvenet Rejection (RCR) — statistical
outlier rejection that stays accurate on samples that are >85% contaminated.

Based on the algorithm in
[Maples et al. 2018, ApJS 238.1](https://arxiv.org/abs/1807.05276).
This package is a numpy / scipy port of the original C++/pybind11
implementation at [`../cpp/`](../cpp/), preserving algorithmic parity
while running pure Python — no compiler toolchain required.

## Install

```bash
pip install pyrcr
```

Runtime deps: `numpy>=1.26`, `scipy>=1.11`. Python ≥3.11.

## Quick start

### Single-value RCR

Reject outliers from a 1-D dataset and recover the underlying central
value and width.

```python
import pyrcr
import numpy as np

# Heavily contaminated data: mu=0, sigma=1, plus one-sided outliers
rng = np.random.default_rng(42)
y = np.concatenate([
    rng.normal(0, 1, size=150),                # clean
    np.abs(rng.normal(0, 10, size=850)),       # 85% contamination
])

r = pyrcr.RCR(pyrcr.RejectionTech.LS_MODE_68)
r.perform_rejection(y.tolist())

print(f"Recovered mu     = {r.result.mu:.3f}    (truth: 0)")
print(f"Recovered sigma  = {r.result.sigma:.3f} (truth: 1)")
print(f"Points kept      = {r.result.flags.sum()} / {len(y)}")
```

### Functional-form RCR (linear model fit with outlier rejection)

Fit a parametric model to (x, y) data while rejecting outliers.

```python
import pyrcr
import numpy as np

# Linear data with outliers
rng = np.random.default_rng(0)
x = np.linspace(-5, 5, 100)
y = 2.0 + 1.5 * x + rng.normal(0, 0.3, size=x.size)
# Inject 20 outliers
out = rng.choice(x.size, size=20, replace=False)
y[out] += rng.normal(15, 5, size=20)

def linear(x, params):
    return params[0] + params[1] * x

def d_linear_b(x, params):
    return 1.0

def d_linear_m(x, params):
    return x

model = pyrcr.FunctionalForm(
    linear, x, y, [d_linear_b, d_linear_m], guess=[0.0, 0.0],
)
r = pyrcr.RCR(pyrcr.RejectionTech.LS_MODE_68)
r.set_parametric_model(model)
r.perform_rejection(y.tolist())

b, m = model.result.parameters
print(f"Recovered b = {b:.3f} (truth: 2.0)")
print(f"Recovered m = {m:.3f} (truth: 1.5)")
print(f"Kept {int(r.result.flags.sum())} / {len(y)} points")
```

## Rejection techniques

| `RejectionTech.*` | Use case |
|---|---|
| `LS_MODE_68` | Symmetric uncontaminated distribution, one-sided contaminants |
| `LS_MODE_DL` | Mixed one-sided + two-sided contaminants |
| `SS_MEDIAN_DL` | Symmetric uncontaminated distribution, two-sided contaminants |
| `ES_MODE_DL` | Mildly asymmetric uncontaminated distribution and/or very small N |

> ⚠️ **Known algorithm bug**: `ES_MODE_DL` combined with `FunctionalForm`
> (parametric models) exhibits a runaway-rejection cascade that drops
> ~75% of inliers even on perfectly clean data. **This is an algorithm bug
> inherited from the C++ source** — the C++ oracle exhibits the same
> behavior; it is not a port deficiency. Empirical characterization is in
> [`benchmarks/es_mode_dl_truth_test.py`](benchmarks/es_mode_dl_truth_test.py).
> All other combinations of rejection technique × model type work
> correctly. **For functional-form fits, use one of `LS_MODE_68`,
> `LS_MODE_DL`, or `SS_MEDIAN_DL`.** `ES_MODE_DL` remains correct for
> single-value (non-parametric) rejection.

See [agents/python_vs_rust_plan.md](../agents/python_vs_rust_plan.md) and
[Maples et al. 2018](https://arxiv.org/abs/1807.05276) for the science.

## More invocations

| Want… | Call |
|---|---|
| Bulk rejection (faster on large N) | `r.perform_bulk_rejection(y)` |
| Weighted RCR | `r.perform_rejection(y, w=weights)` |
| User-defined "candidate" mu | subclass `pyrcr.NonParametric`, `r.set_non_parametric_model(model)` |
| Add Gaussian / bounded priors | `pyrcr.Priors(prior_type=..., gaussian_params=...)` → pass via `priors=` to `FunctionalForm` |
| Pivot search for power-law / exponential fits | pass `pivot_function=...` and `pivot_guess=...` to `FunctionalForm` |
| ND independent variable | pass `xdata` as `(N, D)` array; user functions take vector `x` |

## Status

| Slice | Status |
|---|---|
| Phase 1 single-value RCR (4 techs × iter/bulk × weighted/unweighted) | ✅ 100%; bit-identical to C++ oracle at rtol=1e-12 (worst case 1.7e-15 = 8 ULPs) |
| Phase 2 functional form (1D + ND, priors, pivot search, paramuncertainty) | ✅ 99%; parity with C++ at rtol=5% on 6 sweep scenarios (RNG-limited) |
| Packaging / CI / docs | 🚧 0.1.0 release; collecting real-user feedback before 1.0 |

85 tests passing (+ 1 documented xfail). Reproduce parity with
[`benchmarks/reflect_parity.py`](benchmarks/reflect_parity.py); benchmark
with [`benchmarks/diagnostics.py`](benchmarks/diagnostics.py) and
[`benchmarks/diagnostics_functional.py`](benchmarks/diagnostics_functional.py).

## Layout

```
python/
├── pyproject.toml
├── README.md             (you are here)
├── LICENSE               (mirrors ../cpp/LICENSE — academic / non-commercial)
├── src/pyrcr/
│   ├── __init__.py       public API re-exports
│   ├── api.py            RCR class, RCRResults, RejectionTech, MuType
│   ├── stats.py          median, halfSampleMode, robust-sigma helpers
│   ├── rejection.py      iterative + bulk loops (lower/single/each sigma)
│   ├── tables.py         unity tables ported verbatim from cpp/src/RCR.cpp
│   ├── nonparametric.py  NonParametric base class
│   └── functional.py     FunctionalForm + Priors for parametric fits
├── tests/                pytest suite (85 + 1 xfailed)
├── docs/                 markdown tutorials + PUBLISHING checklist
├── benchmarks/           parity + perf scripts (not part of the wheel)
└── scripts/              one-off codegen + post-install smoke check
```

## Citing

If you use `pyrcr` in academic work, please cite the original paper:

```bibtex
@article{maples2018robust,
    title  = {Robust Chauvenet Outlier Rejection},
    author = {Maples, M.P. and Reichart, D.E. and Konz, N.C. and Berger, T.A.
              and Trotter, A.S. and Martin, J.R. and Dutton, D.A.
              and Paggen, M.L. and Joyner, R.E. and Salemi, C.P.},
    journal = {The Astrophysical Journal Supplement Series},
    volume  = {238}, number = {1}, pages = {2}, year = {2018},
}
```

## License

See [LICENSE](LICENSE) — mirrors the upstream C++ license. Free for
academic and non-commercial use; contact UNC's Office of
Commercialization and Economic Development for commercial licensing.
