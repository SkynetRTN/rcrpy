# Advanced functional-form features

## Priors

Constrain the fit with bounds or Gaussian priors.

### Gaussian priors

Pull a parameter toward a known value with a known uncertainty:

```python
import rcrpy

# Gaussian prior on the intercept at mu=2.0, sigma=0.1.
# NaN values disable the prior on that parameter.
priors = rcrpy.Priors(
    prior_type=rcrpy.PriorType.GAUSSIAN,
    gaussian_params=[[2.0, 0.1],            # intercept: N(2.0, 0.1)
                     [float("nan"), float("nan")]],   # slope: no prior
)

model = rcrpy.FunctionalForm(linear, x, y, [d_b, d_m], guess=[0.0, 0.0],
                             priors=priors)
```

The Gaussian prior is implemented as an extra penalty residual
`(params[k] - mu_k) / sigma_k` appended to the least-squares residual
vector.

### Constrained priors (bounds)

Force a parameter into a fixed interval:

```python
priors = rcrpy.Priors(
    prior_type=rcrpy.PriorType.CONSTRAINED,
    param_bounds=[[0.0, float("nan")],     # intercept: lower-bounded by 0
                  [-5.0, 5.0]],             # slope: [-5, 5]
)
```

`NaN` means unbounded on that side. Constrained priors are passed to
`scipy.optimize.least_squares` as its `bounds=` argument (so the solver
respects them throughout the optimization, not just as a soft penalty).

### Mixed priors

Combine Gaussian + bounded:

```python
priors = rcrpy.Priors(
    prior_type=rcrpy.PriorType.MIXED,
    gaussian_params=[[2.0, 0.1], [float("nan"), float("nan")]],
    param_bounds=[[float("nan"), float("nan")], [0.0, 5.0]],
)
```

### Custom priors

Pass a user function `p(params) -> ndarray`. The returned values are
appended verbatim to the residual vector, so this lets you implement
any log-posterior penalty shape:

```python
def my_prior(params):
    # Sharp penalty if slope is negative
    return np.array([max(0.0, -params[1]) * 100.0])

priors = rcrpy.Priors(prior_type=rcrpy.PriorType.CUSTOM, p=my_prior)
```

## Pivot points

For power-law and exponential models, the value of `params[0]` is often
correlated with the slope/decay parameter. Choosing a "pivot point" near
the data centroid de-correlates them and gives a better-conditioned fit.

The pivot is exposed as a static attribute on `FunctionalForm` so your
model function can reference it inline:

```python
def powerlaw(x, params):
    a0, a1 = params
    return a0 * (x / 10**rcrpy.FunctionalForm.pivot) ** a1
```

### Static pivot

If you know the right pivot:

```python
rcrpy.FunctionalForm.pivot = 0.5   # set once before constructing the model
```

### Pivot search

Have RCR recompute the pivot each iteration from a user-supplied
function:

```python
def get_pivot_powerlaw(xdata, weights, f, params):
    # Pivot definition for power-law fits: weighted log-mean of x.
    top = np.sum(weights * np.log10(xdata) * np.power(10., -2.*f(xdata, params)))
    bot = np.sum(weights * np.power(10., -2.*f(xdata, params)))
    return top / bot

model = rcrpy.FunctionalForm(
    powerlaw, x, y, partials, guess=[1.0, 0.5],
    pivot_function=get_pivot_powerlaw,
    pivot_guess=0.0,
)
```

Each iteration `model.build_model_space()` calls your pivot function on
the current flagged subset; the result is stored in
`rcrpy.FunctionalForm.pivot` and in `model.result.pivot`.

## Parameter uncertainties

After `regression()` (called internally by `r.perform_rejection`),
`model.result.parameter_uncertainties` holds the post-fit standard
errors:

```python
unc = model.result.parameter_uncertainties
print(f"b = {model.result.parameters[0]} ± {unc[0]}")
print(f"m = {model.result.parameters[1]} ± {unc[1]}")
```

These are computed via the standard nonlinear-least-squares covariance
estimator (`sqrt(diag(inv(J^T J)))`) using the Jacobian that
`scipy.optimize.least_squares` returns. Note: this differs in formula
from the C++'s `paramuncertainty`, which uses an unusual M-sized
reduction that looks like an indexing bug; we substitute the textbook
formula.

## MEDIAN / MODE mu_tech for parametric

When `r.perform_rejection(...)` runs with a `set_parametric_model(model)`
attached, the 3-pass refinement uses each `mu_tech` in turn (MODE,
MEDIAN, MEAN). The MEDIAN/MODE paths sample up to **20,000
M-combinations** of (x, y) data points; for each, the model is fit
exactly through those M points; then the per-parameter weighted median
(or half-sample mode) over the combo space is taken.

This is the **robust** part of robust parameter estimation — it's what
makes the fit immune to single bad points pulling the global solution
sideways.

| Path | When | How |
|---|---|---|
| Vectorized fast path | Linear-in-params (auto-detected at construction) | One batched `np.linalg.solve` call across all combos |
| Per-combo fallback | Nonlinear-in-params or ND | `scipy.optimize.fsolve` per combo |

The vectorized path handles 20,000 combos in milliseconds. The fallback
caps at 500 combos to keep latency reasonable on nonlinear models.

Reproduce parity against the C++ oracle (at `rtol=5%`, limited by RNG
differences) with the test sweep at
[`../tests/test_functional_parity_sweep.py`](../tests/test_functional_parity_sweep.py).
