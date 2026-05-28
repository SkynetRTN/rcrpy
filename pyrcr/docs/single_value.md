# Single-value RCR

When you have a 1-D dataset and want to recover the underlying central
value (mu) and width (sigma) in the presence of outliers, single-value
RCR is what you want.

## The four rejection techniques

```python
import rcr2

# Pick one based on your data:
rcr2.RejectionTech.LS_MODE_68    # Symmetric uncontaminated, one-sided contaminants
rcr2.RejectionTech.LS_MODE_DL    # Mixed one-sided + two-sided contaminants
rcr2.RejectionTech.SS_MEDIAN_DL  # Symmetric uncontaminated, two-sided contaminants
rcr2.RejectionTech.ES_MODE_DL    # Mildly asymmetric / very small N
```

See section 3 of [Maples et al. 2018](https://arxiv.org/abs/1807.05276)
for a decision tree.

## Unweighted, iterative rejection

```python
import numpy as np
import rcr2

# Some data: clean Gaussian + heavy one-sided contamination
rng = np.random.default_rng(42)
y = np.concatenate([
    rng.normal(0, 1, size=150),
    np.abs(rng.normal(0, 10, size=850)),
])

r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
r.perform_rejection(y.tolist())

print(f"mu       = {r.result.mu:.3f}")
print(f"sigma    = {r.result.sigma:.3f}")
print(f"stDev    = {r.result.st_dev:.3f}")
print(f"kept     = {int(r.result.flags.sum())} / {len(y)}")
```

## Bulk rejection (faster on large N)

The bulk variant rejects many points in each pass instead of one at a
time — substantially faster, with the same final result for well-behaved
data:

```python
r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
r.perform_bulk_rejection(y.tolist())
# Same r.result fields, plus rejectedY / originalY / stDevTotal which
# the bulk path populates via the C++'s setFinalVectors equivalent.
```

## Weighted rejection

Pass per-point weights:

```python
weights = np.ones(y.size)
weights[some_indices] = 0.5   # downweight some points

r = rcr2.RCR(rcr2.RejectionTech.LS_MODE_68)
r.perform_rejection(y.tolist(), w=weights.tolist())
```

## Results

All four invocations populate `r.result`, which is an `RCRResults`
dataclass with:

| Field | What |
|---|---|
| `mu` | Recovered central value (mean / median / mode) |
| `sigma` | Recovered robust 68.3-percentile deviation |
| `st_dev` | Recovered standard deviation |
| `st_dev_below`, `st_dev_above` | Asymmetric width (when applicable) |
| `sigma_below`, `sigma_above` | Asymmetric robust width (ES_MODE_DL) |
| `st_dev_total` | Combined width (populated by `perform_bulk_rejection`) |
| `flags` | bool array, `True` for kept points |
| `indices` | int indices of kept points |
| `clean_y`, `clean_w` | Kept y / w |
| `rejected_y`, `rejected_w`, `original_y`, `original_w` | populated by bulk path |

## Performance

`rcr2` runs single-value RCR at roughly **10× the C++ implementation's
runtime** on large datasets (N=1000). Reproduce with
[`../benchmarks/diagnostics.py`](../benchmarks/diagnostics.py).
