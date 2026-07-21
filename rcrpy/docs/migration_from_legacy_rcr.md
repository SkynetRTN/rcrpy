# Migrating from legacy `rcr` to `rcrpy`

The legacy package on PyPI (`rcr`) is a pybind11 wrapper around the C++
implementation at [`../../cpp/`](../../cpp/). `rcrpy` is a pure-Python
reimplementation: same algorithm, same answers (Phase 1 bit-identical at
rtol=1e-12; Phase 2 within rtol=5%), but no C++ toolchain needed and
significantly faster on functional-form fits.

The APIs are intentionally close. This page is a direct translation
table.

## Imports

```python
# Before
import rcr

# After
import rcrpy
```

## Constants and enums

| Legacy `rcr` | `rcrpy` |
|---|---|
| `rcr.LS_MODE_68` | `rcrpy.RejectionTech.LS_MODE_68` |
| `rcr.LS_MODE_DL` | `rcrpy.RejectionTech.LS_MODE_DL` |
| `rcr.SS_MEDIAN_DL` | `rcrpy.RejectionTech.SS_MEDIAN_DL` |
| `rcr.ES_MODE_DL` | `rcrpy.RejectionTech.ES_MODE_DL` |
| `rcr.CUSTOM_PRIORS` | `rcrpy.PriorType.CUSTOM` |
| `rcr.GAUSSIAN_PRIORS` | `rcrpy.PriorType.GAUSSIAN` |
| `rcr.CONSTRAINED_PRIORS` | `rcrpy.PriorType.CONSTRAINED` |
| `rcr.MIXED_PRIORS` | `rcrpy.PriorType.MIXED` |

## Methods (camelCase → snake_case)

| Legacy `rcr` | `rcrpy` |
|---|---|
| `r = rcr.RCR(rcr.LS_MODE_68)` | `r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)` |
| `r.setRejectionTech(...)` | `r.set_rejection_tech(...)` |
| `r.performRejection(y)` | `r.perform_rejection(y)` |
| `r.performBulkRejection(y)` | `r.perform_bulk_rejection(y)` |
| `r.performRejection(w, y)` | `r.perform_rejection(y, w=w)` ← note kwarg |
| `r.setParametricModel(model)` | `r.set_parametric_model(model)` |
| `r.setNonParametricModel(model)` | `r.set_non_parametric_model(model)` |

## Result fields (camelCase → snake_case)

| Legacy `r.result.*` | `rcrpy r.result.*` |
|---|---|
| `mu`, `sigma` | `mu`, `sigma` |
| `stDev`, `stDevBelow`, `stDevAbove`, `stDevTotal` | `st_dev`, `st_dev_below`, `st_dev_above`, `st_dev_total` |
| `sigmaBelow`, `sigmaAbove` | `sigma_below`, `sigma_above` |
| `flags` (list of bool) | `flags` (numpy bool array) |
| `indices` (list of int) | `indices` (numpy int64 array) |
| `cleanY`, `rejectedY` | `clean_y`, `rejected_y` |
| `cleanW`, `rejectedW` | `clean_w`, `rejected_w` |
| `originalY`, `originalW` | `original_y`, `original_w` |

## FunctionalForm

Constructor signature is the same; result fields renamed.

```python
# Before
model = rcr.FunctionalForm(f, xdata, ydata, partials, guess,
                            weights=w, error_y=ey)
print(model.result.parameters)
print(model.result.parameter_uncertainties)
print(model.result.pivot)
print(rcr.FunctionalForm.pivot)

# After  (identical, modulo `rcr → rcrpy` and field naming)
model = rcrpy.FunctionalForm(f, xdata, ydata, partials, guess,
                             weights=w, error_y=ey)
print(model.result.parameters)
print(model.result.parameter_uncertainties)
print(model.result.pivot)
print(rcrpy.FunctionalForm.pivot)
```

The `pivot` static-class attribute works the same way — your model
function can reference `rcrpy.FunctionalForm.pivot` inline.

## Priors

Same constructor signatures and meanings; argument names converted to
snake_case.

```python
# Before
mypriors = rcr.Priors(rcr.MIXED_PRIORS, gaussianParams, paramBounds)

# After
mypriors = rcrpy.Priors(prior_type=rcrpy.PriorType.MIXED,
                        gaussian_params=gaussian_params,
                        param_bounds=param_bounds)
```

## NonParametric

Subclass the same way; override `mu_func` (and optionally `mu_func_w`)
in Python instead of `muFunc` in C++.

```python
class MyModel(rcrpy.NonParametric):
    def mu_func(self, flags, y):
        # Decide which points contribute to the mu estimate this iteration.
        idx = your_filter(flags, y)
        self.indices = idx.astype(np.int64)
        return idx, y[idx]
```

## What `rcrpy` does NOT include from legacy `rcr`

| Feature | Status |
|---|---|
| Direct pybind11 binding to C++ | Not applicable — `rcrpy` is pure Python |
| Sphinx-rendered docstrings exactly matching legacy | `rcrpy` has docstrings; format may differ slightly |
| `extraParameterSpace` for "runaway" combos | Not implemented (our solvers don't produce the legacy's M+1 signal) |
| 100% bit-identical MEDIAN/MODE for parametric | Different RNG between Python and C++'s `std::mt19937` means parity is rtol=5% on the combo-sampled paths |

For Phase 1 (single-value RCR), parity is bit-identical at rtol=1e-12.

## Performance

| Workload | Legacy (C++) | `rcrpy` |
|---|---|---|
| Single-value RCR, large N | Baseline | ~10× slower |
| Functional-form fit, no rejection | Baseline | **80×–6000× FASTER** |
| Functional-form fit + LS_MODE_68 | Baseline | ~10–400× faster |

The functional-form speedup comes from `scipy.optimize.least_squares`
making far fewer Python↔C++ callback round-trips than the legacy's
hand-rolled Gauss-Newton solver. See [`../benchmarks/diagnostics_functional.py`](../benchmarks/diagnostics_functional.py).
