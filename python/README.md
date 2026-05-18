# rcr2

Python reimplementation of Robust Chauvenet Rejection (RCR 2.0).

Status: Phase 1 — single-value RCR is parity-green at `rtol=1e-12` against
the legacy C++ `rcr` module across all four rejection techniques.

### Parity coverage

**Iterative path (`performRejection`)** — all 4 rejection techs × 3 CSVs:

| Tech | smoke (N=8) | singlevalue (N=1000) | weighted singlevalue (N=200) |
|---|---|---|---|
| `LS_MODE_68` | ✅ | ✅ | ✅ |
| `LS_MODE_DL` (uses fitDL/mFinder) | ✅ | ✅ | ✅ |
| `SS_MEDIAN_DL` (single-sigma loop) | ✅ | ✅ | ✅ |
| `ES_MODE_DL` (each-sigma loop) | ✅ | ✅ | ✅ |

**Bulk path (`performBulkRejection`)** — all 4 rejection techs, weighted + unweighted:

| Tech | smoke (N=8) | singlevalue (N=1000) | weighted (N=200) |
|---|---|---|---|
| `LS_MODE_68` | ✅ | ✅ | ✅ |
| `LS_MODE_DL` | ✅ | ✅ | ✅ |
| `SS_MEDIAN_DL` | ✅ | ✅ | ✅ |
| `ES_MODE_DL` | ✅ | ✅ | ✅ |

### Measured agreement (post-halfSampleMode vectorization)

Asserted at `rtol=1e-12`; actual worst-case scalar disagreement across all
24 diagnostic cases is **1.99×10⁻¹⁵** (≈9 ULPs of double precision; 2.7
orders of magnitude below the threshold). Every `flags`/`indices` list
is **bit-identical** to the oracle on every test case.

### Performance (after halfSampleMode vectorization)

Reproduce with [`diagnostics.py`](diagnostics.py). Highlights on N=1000:

| Tech (N=1000) | Oracle | rcr2 | Slowdown |
|---|---|---|---|
| LS_MODE_68 iter | 128 ms | 435 ms | **3.4×** |
| LS_MODE_DL iter | 309 ms | 8722 ms | 28× (fitDL/mFinder is the new hotspot) |
| SS_MEDIAN_DL iter | 2.6 ms | 124 ms | 47× |
| ES_MODE_DL iter | 32 ms | 1304 ms | 40× |
| LS_MODE_68 BULK | 1.6 ms | 53 ms | 34× |

Vectorizing `halfSampleMode` dropped LS_MODE_68 N=1000 from ~12,500 ms to
435 ms (**29× speedup**). Next perf wins, in priority order:

1. Vectorize `mFinder` / `fitDL` (the inner DOUBLE_LINE optimization loop) — the
   new hotspot for LS_MODE_DL and ES_MODE_DL.
2. Vectorize `halfSampleMode_w` (weighted) — bottleneck for weighted iterative.
3. Replace remaining `for i in range(true_count)` loops in the rejection
   loop bodies with vectorized split/diff/dispatch.

### Phase 1: COMPLETE ✅

All four rejection techniques × both invocation paths × both weight modes
= **24 parity scenarios**, all green at `rtol=1e-12`.

### Phase 2 — not yet started

- Functional form (parametric model fitting) — `cpp/src/FunctionalForm.cpp`
- Non-parametric — `cpp/src/NonParametric.cpp`
- Pivot point machinery / priors
- `halfSampleMode` vectorization (~80× perf hotspot)

See [`../agents/python_vs_rust_plan.md`](../agents/python_vs_rust_plan.md)
for the full plan.

## Development install

From the repository root:

```bash
python -m pip install -e python/[dev]
```

The `[dev]` extra pulls in `pytest` and the legacy `rcr` C++ module that
parity tests compare against.

## Layout

```
python/
├── pyproject.toml
├── src/rcr2/
│   ├── __init__.py    public API re-exports
│   ├── api.py         RCR class
│   ├── stats.py       median, half-sample mode, robust-sigma helpers
│   ├── rejection.py   iterative + bulk rejection loops
│   └── tables.py      unity tables, ported verbatim from cpp/src/RCR.cpp
└── tests/
    ├── conftest.py    locates assets/test/ via repo root
    ├── test_smoke.py  data_smoke.csv known-answer test
    └── test_parity.py parity vs. the installed rcr C++ module
```

## Parity target

Tests use `rtol=1e-12` against the legacy C++ `rcr` module on the CSVs in
[`../assets/test/`](../assets/test/). Current results on `data_singlevalue.csv`:

| Path | Wall-clock (mean of 3) |
|---|---|
| Oracle (C++, `rcr`) | ~158 ms |
| Port (`rcr2`, pure-numpy) | ~12,500 ms |

The ~80× gap is entirely in `halfSampleMode` (nested Python loops). Vectorize
that before any other perf work. See [`bench_port_vs_oracle.py`](bench_port_vs_oracle.py)
to reproduce.
