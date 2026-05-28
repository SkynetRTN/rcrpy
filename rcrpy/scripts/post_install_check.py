"""End-to-end smoke test of the installed rcrpy wheel: runs both the
single-value and functional form quick-start examples from the README
and verifies the results are sensible. Run inside the .venv-clean to
prove the public wheel is usable with no source tree access.
"""
import numpy as np

import rcrpy

print(f"rcrpy version: {rcrpy.__version__}")

# --- single-value ----------------------------------------------------------
rng = np.random.default_rng(42)
y = np.concatenate([
    rng.normal(0, 1, size=150),
    np.abs(rng.normal(0, 10, size=850)),
])
r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
r.perform_rejection(y.tolist())
print(f"single-value: mu={r.result.mu:.3f} sigma={r.result.sigma:.3f} "
      f"kept={int(r.result.flags.sum())}/{len(y)}")
# Heavily contaminated 1-sided data: mu won't recover to exactly 0, but
# should be much lower than the pre-RCR mean. Loose sanity.
assert abs(r.result.mu) < 5.0, f"single-value mu way off: {r.result.mu}"
assert 0.1 < r.result.sigma < 10.0, f"single-value sigma way off: {r.result.sigma}"

# --- functional form -------------------------------------------------------
rng = np.random.default_rng(0)
x = np.linspace(-5, 5, 100)
y = 2.0 + 1.5 * x + rng.normal(0, 0.3, size=x.size)
out = rng.choice(x.size, size=20, replace=False)
y[out] += rng.normal(15, 5, size=20)

def linear(xv, params):
    return params[0] + params[1] * xv

def d_lin_b(xv, params):
    return 1.0

def d_lin_m(xv, params):
    return xv

model = rcrpy.FunctionalForm(linear, x, y, [d_lin_b, d_lin_m], guess=[0.0, 0.0])
r = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
r.set_parametric_model(model)
r.perform_rejection(y.tolist())
b, m = model.result.parameters
print(f"functional:   b={b:.3f} (truth 2.0)  m={m:.3f} (truth 1.5)  "
      f"kept={int(r.result.flags.sum())}/{len(y)}")
assert abs(b - 2.0) < 1.5, f"intercept off: {b}"
assert abs(m - 1.5) < 0.5, f"slope off: {m}"

print("\nOK: rcrpy wheel installs and runs cleanly.")
