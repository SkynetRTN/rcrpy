# rcrpy documentation

Pragmatic markdown docs for `rcrpy`. For the deeper science see
[Maples et al. 2018](https://arxiv.org/abs/1807.05276); for the original
C++/pybind11 implementation see the Sphinx site under
[`../../cpp/docs/`](../../cpp/docs/).

## Guides

| Topic | File |
|---|---|
| Single-value RCR | [single_value.md](single_value.md) |
| Functional-form RCR (model fitting + outlier rejection) | [functional_form.md](functional_form.md) |
| Priors, pivots, error bars (advanced functional form) | [advanced_functional.md](advanced_functional.md) |
| Migrating from the legacy `rcr` (C++/pybind11) package | [migration_from_legacy_rcr.md](migration_from_legacy_rcr.md) |
| Publishing a new release (maintainer) | [PUBLISHING.md](PUBLISHING.md) |

## Reference

The public API is exported from `rcrpy`:

| Symbol | What |
|---|---|
| `RCR` | Main driver class |
| `RCRResults` | Result bundle (`mu`, `sigma`, `flags`, …) |
| `RejectionTech` | Enum: `LS_MODE_68`, `LS_MODE_DL`, `SS_MEDIAN_DL`, `ES_MODE_DL` |
| `MuType` | Enum: `VALUE`, `PARAMETRIC`, `NONPARAMETRIC` |
| `FunctionalForm` | Parametric model class for model-fitting RCR |
| `FunctionalFormResults` | Result bundle for parametric fits |
| `Priors` / `PriorType` | Parameter priors (Gaussian, Constrained, Mixed, Custom) |
| `NonParametric` | Base class for user-defined mu computation |

Inspect docstrings on each class for full method descriptions.
