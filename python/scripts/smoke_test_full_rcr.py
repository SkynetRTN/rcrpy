"""End-to-end smoke test of the standalone `full_rcr.py`.

Imports it directly (no rcrpy package needed), runs both single-value
and functional-form quick-starts, and verifies the results are sensible.
Designed to be run in a sterile venv with only numpy + scipy.
"""
import sys
from pathlib import Path

# Make full_rcr.py importable: it lives one level up from this script
# (python/full_rcr.py, while this is python/scripts/smoke_test_full_rcr.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import full_rcr

print(f"full_rcr version: {full_rcr.__version__}")

# ---- single-value --------------------------------------------------------
rng = np.random.default_rng(42)
y = np.concatenate([
    rng.normal(0, 1, size=150),
    np.abs(rng.normal(0, 10, size=850)),
])
r = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
r.perform_rejection(y.tolist())
print(f"single-value: mu={r.result.mu:.3f} sigma={r.result.sigma:.3f} "
      f"kept={int(r.result.flags.sum())}/{len(y)}")
assert abs(r.result.mu) < 5.0, f"mu way off: {r.result.mu}"
assert 0.1 < r.result.sigma < 10.0, f"sigma way off: {r.result.sigma}"

# ---- functional form -----------------------------------------------------
rng = np.random.default_rng(0)
x = np.linspace(-5, 5, 100)
y2 = 2.0 + 1.5 * x + rng.normal(0, 0.3, size=x.size)
out = rng.choice(x.size, size=20, replace=False)
y2[out] += rng.normal(15, 5, size=20)


def linear(xv, params):
    return params[0] + params[1] * xv


def d_b(xv, params):
    return 1.0


def d_m(xv, params):
    return xv


model = full_rcr.FunctionalForm(linear, x, y2, [d_b, d_m], guess=[0.0, 0.0])
r = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
r.set_parametric_model(model)
r.perform_rejection(y2.tolist())
b, m = model.result.parameters
print(f"functional:   b={b:.3f} (truth 2.0)  m={m:.3f} (truth 1.5)  "
      f"kept={int(r.result.flags.sum())}/{len(y2)}")
assert abs(b - 2.0) < 1.5, f"intercept off: {b}"
assert abs(m - 1.5) < 0.5, f"slope off: {m}"

# ---- Priors --------------------------------------------------------------
pri = full_rcr.Priors(
    prior_type=full_rcr.PriorType.GAUSSIAN,
    gaussian_params=[[5.0, 0.1], [float("nan"), float("nan")]],
)
model2 = full_rcr.FunctionalForm(linear, x, y2, [d_b, d_m], guess=[0.0, 0.0],
                                  priors=pri)
model2.regression()
print(f"with Gaussian prior on b: b={model2.parameters[0]:.3f}, "
      f"m={model2.parameters[1]:.3f}")

# ---- NonParametric -------------------------------------------------------
npmodel = full_rcr.NonParametric()
r = full_rcr.RCR(full_rcr.RejectionTech.LS_MODE_68)
r.set_non_parametric_model(npmodel)
r.perform_rejection(np.array([0.1, 0.2, 0.0, 0.3, -0.1, -0.2, -0.3, 11.0]).tolist())
print(f"non-parametric: mu={r.result.mu:.4f}  rejected_index={np.where(~r.result.flags)[0].tolist()}")

print("\nOK: full_rcr.py works standalone with numpy + scipy only.")
