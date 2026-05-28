# Functional-form RCR

When you have `(x, y)` data and want to fit a model `y = f(x, params)`
while rejecting outliers, use `FunctionalForm` + `set_parametric_model`.

## Minimal example: linear fit

```python
import numpy as np
import pyrcr

# Linear data with one-sided outliers
rng = np.random.default_rng(0)
x = np.linspace(-5, 5, 100)
y = 2.0 + 1.5 * x + rng.normal(0, 0.3, size=x.size)
out = rng.choice(x.size, size=20, replace=False)
y[out] += rng.normal(15, 5, size=20)  # outliers

# 1. Define the model and its partial derivatives w.r.t. each parameter.
def linear(x, params):
    return params[0] + params[1] * x

def d_linear_b(x, params):
    return 1.0          # df/d(params[0])

def d_linear_m(x, params):
    return x            # df/d(params[1])

# 2. Build the FunctionalForm with an initial-guess parameter vector.
model = pyrcr.FunctionalForm(
    linear, x, y, [d_linear_b, d_linear_m], guess=[0.0, 0.0],
)

# 3. Run RCR with the model attached.
r = pyrcr.RCR(pyrcr.RejectionTech.LS_MODE_68)
r.set_parametric_model(model)
r.perform_rejection(y.tolist())

# 4. Inspect results.
b, m = model.result.parameters
print(f"intercept = {b:.3f} (truth 2.0)")
print(f"slope     = {m:.3f} (truth 1.5)")
print(f"kept {int(r.result.flags.sum())} / {len(y)} points")
```

## Other model shapes

The model function and its partial derivatives can be any Python callables.
Examples (mirrors of [`../../cpp/src/functional_calc_test.py`](../../cpp/src/functional_calc_test.py)):

### Quadratic with pivot

```python
def quad(x, params):
    a0, a1, a2 = params
    p = pyrcr.FunctionalForm.pivot
    return a0 + a1 * (x - p) + a2 * (x - p)**2

def d_quad_a0(x, params): return 1.0
def d_quad_a1(x, params): return x - pyrcr.FunctionalForm.pivot
def d_quad_a2(x, params): return (x - pyrcr.FunctionalForm.pivot)**2

model = pyrcr.FunctionalForm(quad, x, y,
                             [d_quad_a0, d_quad_a1, d_quad_a2],
                             guess=[0.0, 0.0, 0.0])
```

### Power-law (with optional pivot search — see [advanced_functional.md](advanced_functional.md))

```python
def powerlaw(x, params):
    a0, a1 = params
    return a0 * (x / 10**pyrcr.FunctionalForm.pivot) ** a1

def d_pl_a0(x, params):
    return (x / 10**pyrcr.FunctionalForm.pivot) ** params[1]

def d_pl_a1(x, params):
    return powerlaw(x, params) * np.log(x / 10**pyrcr.FunctionalForm.pivot)
```

## With error bars

```python
sigma_y = np.full(y.size, 0.3)  # per-point measurement uncertainties

model = pyrcr.FunctionalForm(
    linear, x, y, [d_linear_b, d_linear_m],
    guess=[0.0, 0.0],
    error_y=sigma_y,
)
```

When `error_y` is provided, `model.result.parameter_uncertainties` is
populated from the post-fit Jacobian covariance.

## With weights

```python
weights = np.ones(x.size)
weights[outlier_indices] = 0.1   # downweight known-suspect points

model = pyrcr.FunctionalForm(
    linear, x, y, [d_linear_b, d_linear_m],
    guess=[0.0, 0.0],
    weights=weights,
)
r.perform_rejection(y.tolist(), w=weights.tolist())
```

## Multi-dimensional independent variable (ND)

```python
# x is shape (N, 2) — two independent variables per data point
x = np.column_stack([x1, x2])

def f_nd(xv, params):
    return params[0] + params[1] * xv[0] + params[2] * xv[1]

def d_nd_p0(xv, params): return 1.0
def d_nd_p1(xv, params): return xv[0]
def d_nd_p2(xv, params): return xv[1]

model = pyrcr.FunctionalForm(f_nd, x, y, [d_nd_p0, d_nd_p1, d_nd_p2],
                             guess=[0.0, 0.0, 0.0])
```

The ND detection is automatic — if `x.ndim == 2`, the ND code path
activates and the model function receives a vector `xv` instead of a
scalar.

## Performance

`pyrcr`'s functional-form path is **significantly FASTER than the C++**
on this kind of workload — typically 80×–6000× depending on N — because
scipy's nonlinear-least-squares solver makes far fewer Python↔C++
callback round-trips than the C++'s hand-rolled Gauss-Newton solver.
Reproduce with [`../benchmarks/diagnostics_functional.py`](../benchmarks/diagnostics_functional.py).
