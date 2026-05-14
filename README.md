# Robust Chauvenet Rejection (RCR)

> Advanced, but easy-to-use, statistical outlier rejection — robust enough
> for datasets that are >85% contaminated.

This repository hosts the **multi-language reimplementation effort** for RCR,
originally published in [Maples et al. 2018](https://arxiv.org/abs/1807.05276)
and shipped as a C++ library with Python bindings.

The work here is a fresh restructuring around three tracked implementations:

| Track | Status | Purpose |
|---|---|---|
| [`cpp/`](cpp/) | **Reference**. Frozen upstream code. | Source of truth for algorithmic behavior. |
| `python/` | **Planned.** | Pure-Python reimplementation — readability, hackability, parity tests. |
| `rust/` | **Planned.** | High-performance reimplementation — safety, portability, optional Python bindings via PyO3. |

The Python and Rust ports are intentionally separate trees so each can have
its own idiomatic layout (Cargo workspace, `pyproject.toml`, etc.) without
contorting around the legacy C++ tree.

## What is RCR?

The naive form of outlier rejection is **sigma clipping**: drop points more
than *k* standard deviations from the mean. Chauvenet (1863) gave a
principled prescription for *k* as a function of sample size. Iterating this
rule is "traditional Chauvenet rejection".

The catch: the mean and standard deviation are themselves polluted by the
very outliers they are being used to reject. That makes traditional
Chauvenet useless on heavily contaminated samples.

**Robust Chauvenet Rejection (RCR)** swaps in robust replacements — the
median or half-sample mode for the center, and calibrated robust
replacements for the spread — and applies the iterative rejection on top.
It has been simulated extensively and works on samples with contamination
fractions in excess of 90%.

For the full method, see the [paper](cpp/RCR_paper.pdf) (included) and the
[short preprint](https://arxiv.org/abs/2301.07838).

## Repository layout

```
robust-chauvenet-rejection/
├── README.md            <- you are here
├── AGENT.md             <- ground rules for AI agents working in this repo
├── assets/
│   └── test/            <- canonical test datasets, shared across implementations
│       ├── README.md
│       ├── data_smoke.csv
│       ├── data_singlevalue.csv
│       ├── data_weighted_singlevalue.csv
│       ├── data_linear.csv
│       └── data_exponential.csv
├── agents/              <- exploratory plans and notes for the reimplementation
│   ├── README.md
│   ├── python_vs_rust_plan.md
│   └── test_data_generation.md
├── cpp/                 <- frozen reference implementation (pybind11 module)
│   ├── src/             <- C++ source + pybind11 glue
│   ├── tests/           <- existing pybind smoke tests
│   ├── testdata/        <- original CSVs (also mirrored to assets/test/)
│   ├── docs/            <- Sphinx site sources
│   └── ...
├── python/              <- (planned) pure-Python implementation
└── rust/                <- (planned) Rust implementation
```

## Quickstart (current state)

The only currently runnable implementation is the upstream C++/pybind11
module under [`cpp/`](cpp/).

```bash
python3 -m pip install rcr        # installs from PyPI, not this checkout
python3 cpp/tests/test.py         # smoke-test the installed module
```

To build from this checkout instead:

```bash
cd cpp
python3 -m pip install pybind11
python3 -m pip install -e .
python3 tests/maintest.py
```

The Python and Rust ports do not exist yet — see
[`agents/python_vs_rust_plan.md`](agents/python_vs_rust_plan.md) for the
plan to bring them up.

## Testing data

All implementations are validated against the same CSVs under
[`assets/test/`](assets/test/). New implementations should add a thin loader
that reads from that directory rather than re-generating the data, so that
cross-implementation results stay comparable. See
[`assets/test/README.md`](assets/test/README.md) for per-file expectations.

## Citation

If you use RCR in academic work, cite Maples et al. 2018:

```bibtex
@article{maples2018robust,
    title  = {Robust Chauvenet Outlier Rejection},
    author = {Maples, M.P. and Reichart, D.E. and Konz, N.C. and Berger, T.A.
              and Trotter, A.S. and Martin, J.R. and Dutton, D.A.
              and Paggen, M.L. and Joyner, R.E. and Salemi, C.P.},
    journal = {The Astrophysical Journal Supplement Series},
    volume  = {238}, number = {1}, pages = {2}, year = {2018},
    publisher = {IOP Publishing}
}
```

## License

See [`cpp/LICENSE`](cpp/LICENSE). RCR is free for academic and non-commercial
use; contact the authors for commercial licensing.

## Acknowledgements

Original C++/Python author: Nick C. Konz. Former author: Michael Maples.
Originally developed at the Department of Physics and Astronomy, University
of North Carolina at Chapel Hill.
