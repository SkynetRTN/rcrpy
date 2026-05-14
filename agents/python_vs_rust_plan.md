# Python vs. Rust Re-implementation Plan

> **Status:** Exploratory. This plan compares two candidate re-implementations
> of [RCR](../README.md) and proposes how to evaluate them. It is not a
> commitment to ship both — it is the artifact that should let us decide.

## 1. Why re-implement at all?

The existing [`cpp/`](../cpp/) tree is a working pybind11 module with a
decade of accumulated logic, but it has real friction:

- **Build surface.** Manylinux + macOS + Windows + Visual C++ Build Tools is
  the standard pybind11 install nightmare. Users routinely hit it.
- **Hackability.** `RCR.cpp` is a single 267 kB translation unit; iterating on
  the algorithm in C++ is slow and error-prone.
- **Maintenance.** Original authors have moved on; the code is effectively
  read-only.
- **No native fallback.** There is no implementation that works without a
  C++ toolchain — annoying for read-only environments and CI.

Two candidate ports address different subsets of those problems:

| Track | Solves | Cost |
|---|---|---|
| **Pure Python (numpy/scipy)** | Hackability, easy install (wheels are trivial), readable reference. | Slower than C++ (probably 5–50× on the hot loop, depending on vectorization). |
| **Rust (+ PyO3 bindings)** | Native fallback, performance, modern toolchain, easier to cross-compile, optional WASM. | Larger up-front cost; smaller pool of contributors familiar with the algorithm *and* Rust. |

They are not mutually exclusive. The realistic outcome is **both** — Python
for the readable reference and pedagogy, Rust for production users — but the
ordering matters.

## 2. Goals and non-goals

### Goals

1. **Algorithmic parity** with `cpp/` on the validation CSVs in
   [`assets/test/`](../assets/test/), within a documented numerical
   tolerance.
2. **Self-contained installs.** No C++ toolchain required.
3. A **single Python API** that can be backed by either the Python or the
   Rust implementation, selected at install time.
4. Validation against the **shared CSVs**, not per-implementation fixtures.

### Non-goals (for v1)

- Algorithmic changes or "improvements" to RCR. Parity first, papers later.
- A new web calculator (the existing one under `cpp/webcalculator/` is fine).
- A C ABI or non-Python language bindings beyond what PyO3 gives us.
- WASM, GPU, or distributed RCR.

## 3. Decision criteria

We will judge each track on:

| Criterion | How we measure |
|---|---|
| **Parity** | Recovered `mu`/`sigma` (and functional-fit parameters) on `assets/test/` within tolerance vs. `cpp/`. |
| **Performance** | Wall-clock on `data_singlevalue.csv` (N=1000) and `data_linear.csv` (N=999), normalized to `cpp/`. |
| **Readability** | LOC + cyclomatic complexity, plus subjective "could a new contributor land a fix in a week?" |
| **Install ergonomics** | Time and surface area: `pip install` on a clean macOS, Linux, Windows VM. |
| **Maintenance load** | Number of distinct toolchains needed to ship a release (Python = 1; Rust = 2). |

Numbers go into this file as they are gathered. Don't wait for all five
before sharing — partial measurements are useful.

## 4. Anchor: the reference implementation

The C++ tree under [`cpp/`](../cpp/) is the algorithmic ground truth. Key
files:

- [`cpp/src/RCR.h`](../cpp/src/RCR.h) — public API, enums, struct shapes.
  This is the single best file to read first.
- [`cpp/src/RCR.cpp`](../cpp/src/RCR.cpp) — main iterative + bulk rejection
  loops, the unity tables, and the rejection rules. 267 kB; large but
  approachable in chunks.
- [`cpp/src/FunctionalForm.{h,cpp}`](../cpp/src/) — model-fitting variant.
- [`cpp/src/NonParametric.{h,cpp}`](../cpp/src/) — non-parametric variant.
- [`cpp/src/MiscFunctions.{h,cpp}`](../cpp/src/) — statistics helpers
  (median, half-sample mode, etc.).
- [`cpp/src/RCR_python.cpp`](../cpp/src/RCR_python.cpp) — pybind11 surface;
  defines the *Python API the ports must match*.

The public surface to match (from `RCR.h`):

```cpp
enum MuTechs        { MEAN, MEDIAN, MODE };
enum SigmaTechs     { STANDARD_DEVIATION, SIXTY_EIGHTH_PERCENTILE,
                      SINGLE_LINE, DOUBLE_LINE };
enum SigmaChoices   { SINGLE, LOWER, EACH };
enum RejectionTechs { SS_MEDIAN_DL, LS_MODE_68, LS_MODE_DL, ES_MODE_DL };
enum MuTypes        { VALUE, PARAMETRIC, NONPARAMETRIC };

struct RCRResults { mu, stDev, stDev{Below,Above,Total},
                    sigma, sigma{Below,Above},
                    flags, indices,
                    cleanW, cleanY, rejectedW, rejectedY,
                    originalW, originalY };

class RCR {
    RCR(RejectionTechs);
    void performRejection(vector<double> y);
    void performBulkRejection(vector<double> y);
    void performRejection(vector<double> w, vector<double> y);
    void performBulkRejection(vector<double> w, vector<double> y);
    void setParametricModel(FunctionalForm&);
    void setNonParametricModel(NonParametric&);
    void setMuType(MuTypes);
    void setInitialModel(vector<double>);
};
```

Both ports should expose this shape (modulo naming conventions and idiomatic
result types). Existing Python users should be able to swap implementations
by changing the import.

## 5. Track A — Pure Python (numpy + scipy)

### 5.1 Scaffolding

```
python/
├── pyproject.toml
├── src/rcr/
│   ├── __init__.py            # re-exports RCR, enums, FunctionalForm, ...
│   ├── api.py                 # the public RCR class
│   ├── stats.py               # median, half-sample mode, robust-sigma helpers
│   ├── rejection.py           # iterative + bulk loops
│   ├── functional.py          # model-fitting variant
│   ├── nonparametric.py
│   └── tables.py              # unity tables (ported as static arrays)
└── tests/
    ├── conftest.py            # finds assets/test/ via repo root
    ├── test_singlevalue.py
    ├── test_weighted.py
    ├── test_functional.py
    └── test_parity_vs_cpp.py  # opt-in; skipped if cpp module unavailable
```

### 5.2 Approach

1. **Port the helpers first.** `MiscFunctions.cpp` is mostly self-contained
   statistics — median, half-sample mode, CF/FN tables. Port these into
   `stats.py` and write unit tests against scalar inputs from `cpp/`.
2. **Port the unity tables verbatim.** Don't try to regenerate them; copy
   the numeric literals from `cpp/src/RCR.cpp` into `tables.py`. Add a test
   that hashes the table to detect accidental edits.
3. **Port the iterative rejection loop.** Single-sigma first
   (`iterativeSingleSigmaRCR`), then lower-sigma, then each-sigma. Write
   a parity test per loop using `data_singlevalue.csv` and the matching
   call into the installed `rcr` C++ module as the oracle.
4. **Port bulk rejection.** Same order; bulk is faster but more subtle.
5. **Functional + non-parametric.** Defer until single-value parity is
   green.

### 5.3 Risks

- **Performance.** Pure-Python `for` loops over 1000+ points with per-step
  median/mode recomputation will be slow. Vectorize aggressively with
  numpy; if a loop is still hot, lean on numba or scipy primitives **before**
  reaching for Cython. The whole point of this track is portability, so
  don't reintroduce a compiler dependency.
- **Float ordering.** The C++ uses `std::sort` plus index-based passes;
  numpy's `argsort` is stable on different keys. Watch for tie-breaks at
  rejection boundaries — they can flip a single flag and cascade.
- **Unity-table indexing.** The tables are indexed by sample count after
  rejection; off-by-one here is the most likely silent parity break.

### 5.4 Expected effort

- Stats + unity tables: **0.5–1 week.**
- Single-value RCR (all three sigma choices): **1–2 weeks.**
- Functional-form variant (Levenberg–Marquardt-ish loop): **1–2 weeks.**
- Non-parametric variant: **0.5 week.**
- Polish + parity hardening: **0.5 week.**

Total: **roughly 4–6 weeks** of focused work for one engineer.

## 6. Track B — Rust (+ optional PyO3)

### 6.1 Scaffolding

A Cargo workspace, so the core lib and the Python binding can ship and test
independently:

```
rust/
├── Cargo.toml                  # workspace
├── crates/
│   ├── rcr-core/
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs          # public API
│   │   │   ├── stats.rs
│   │   │   ├── rejection.rs
│   │   │   ├── functional.rs
│   │   │   ├── nonparametric.rs
│   │   │   └── tables.rs       # ported unity tables as `&[f64]` statics
│   │   └── tests/              # cargo tests; read from ../../../assets/test/
│   ├── rcr-py/
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml      # maturin
│   │   └── src/lib.rs          # #[pyclass] wrappers
│   └── rcr-cli/                # optional: thin CLI for the CSVs
└── README.md
```

### 6.2 Approach

1. **Define the core types in Rust idiom** but mirror C++ semantics:
   - `enum RejectionTech { SsMedianDl, LsMode68, LsModeDl, EsModeDl }`
   - `struct RcrResult { mu: f64, sigma: f64, flags: Vec<bool>, ... }`
   - Return `Result<RcrResult, RcrError>` instead of exceptions.
2. **Port the unity tables** as `pub(crate) static UNITY_xx: &[f64] = &[...];`
   with the same hash-based regression test as the Python port.
3. **Port the loops.** Start with `iterativeSingleSigmaRCR`. Use `f64`
   slices and avoid allocations inside the rejection inner loop —
   pre-allocate `flags` and `working` buffers once.
4. **Bindings (PyO3 + maturin) come last.** Build them only after the core
   crate passes parity against `cpp/`.

### 6.3 Dependency stance

Keep the core crate **lean**:

| Use case | Crate | Why |
|---|---|---|
| Numeric arrays | `Vec<f64>` / `ndarray` | Prefer plain `Vec` for the core; reach for `ndarray` only if a multi-D fit needs it. |
| Statistics | hand-rolled | Avoid `statrs` etc. — we want to match `cpp/`'s exact rules, not "a" reasonable median. |
| Errors | `thiserror` | Boring, conventional, no `anyhow` in libraries. |
| Tests | built-in + `approx` | `approx::assert_relative_eq!` for parity checks. |
| Python binding | `pyo3` + `maturin` | The standard combo. Keep behind a `python` feature flag so `rcr-core` is pure Rust. |

### 6.4 Risks

- **Algorithm ports that look right but aren't.** Rust's type system catches
  type errors, not algorithmic ones. The parity tests are the safety net,
  so write them *before* you port, not after.
- **API ergonomics.** The C++ class is stateful (`setParametricModel`,
  `performRejection` mutating `result`); a direct Rust port of that is
  unidiomatic. Decide early whether to mirror the stateful object or offer
  a builder + free function. Recommend: **stateful struct mirroring C++**
  for the first cut, since it keeps the parity diff small.
- **Build-time on CI.** PyO3 + maturin + manylinux wheels are
  well-trodden but slow. Budget a day for cibuildwheel setup.

### 6.5 Expected effort

- Core stats + unity tables: **1 week.**
- Single-value RCR: **1.5–2 weeks** (longer than Python because of
  ownership wrangling on the first port).
- Functional + non-parametric: **2 weeks.**
- PyO3 bindings + maturin packaging: **0.5–1 week.**
- Wheels on CI for macOS / Linux / Windows: **0.5 week.**

Total: **roughly 5–7 weeks** of focused work for one engineer who has
shipped a Rust library before. Add ~50% if not.

## 7. Comparison at a glance

| Aspect | Pure Python | Rust (+ PyO3) |
|---|---|---|
| Time to first parity-green test | Short (days). | Medium (1–2 weeks). |
| Performance vs. `cpp/` | 5–50× slower, depending on vectorization. | Within 1–2× of `cpp/`; sometimes faster on bulk. |
| Install surface | `pip install rcr-py` — pure-Python wheel, works everywhere. | `pip install rcr-rs` — needs prebuilt wheels per platform, but no compiler on the user side. |
| Contributor pool | Large (any Python developer). | Smaller (Rust + numerics overlap is rarer). |
| Cross-compile to WASM | Painful. | Reasonable (long-term option). |
| Risk of silent algorithmic drift | Higher (dynamic typing, floats-as-doubles-as-Decimals confusion). | Lower (typed, no implicit promotion). |
| Useful as a teaching/reference impl | Yes — the codebase reads like the paper. | Less so — ownership noise. |
| Useful as a production engine | Borderline for huge datasets. | Yes. |

## 8. Recommended phased roadmap

**Phase 0 — Shared groundwork (this PR).**
- `assets/test/` populated. ✔
- `README.md` + `AGENT.md` written. ✔
- This plan committed. ✔

**Phase 1 — Pure-Python port, single-value only.** ~2 weeks.
- Scaffold `python/` with `pyproject.toml`.
- Port `MiscFunctions` helpers + unity tables.
- Port `iterativeSingleSigmaRCR` and `bulkSingleSigmaRCR` for the three
  rejection techs.
- Parity tests against `cpp/` using `data_singlevalue.csv`,
  `data_weighted_singlevalue.csv`, and `data_smoke.csv`.
- Decision point: continue Python-only, or kick off Rust in parallel.

**Phase 2 — Pure-Python functional + non-parametric.** ~2 weeks.
- Port `FunctionalForm` and `NonParametric`.
- Parity tests against `data_linear.csv` and `data_exponential.csv`.
- Ship as `rcr-py` on TestPyPI.

**Phase 3 — Rust core crate.** ~3 weeks.
- Scaffold workspace under `rust/`.
- Port helpers + single-value RCR with parity tests reading
  `assets/test/`.
- Port functional + non-parametric.

**Phase 4 — PyO3 bindings + ABI parity.** ~1 week.
- `rcr-rs` Python package via maturin, exposing the same public surface
  as `rcr-py`.
- Cross-implementation parity test: same CSV, two backends, results must
  match within tolerance.

**Phase 5 — Decision.**
- Compare against the criteria in §3.
- Pick a default backend; keep the other as a documented alternative.
- Retire (or freeze) the `cpp/` tree in favor of one of the ports.

The decision in Phase 5 might be "ship both indefinitely" — that's fine, as
long as we've validated it's worth the maintenance.

## 9. Open questions

These need a human decision before the next phase starts. Ping the
requester rather than guessing.

1. **Numerical tolerance for parity.** Bit-exact, `rtol=1e-12`, or
   `rtol=1e-6`? RCR's iterative loops mean tiny float differences can flip
   a flag at the rejection boundary.
2. **Package names.** `rcr` is taken on PyPI by the existing C++ package.
   Use `rcr-py` and `rcr-rs`? Reuse `rcr` and version-bump? Coordinate with
   Nick Konz?
3. **Python version floor.** The current `setup.py` is permissive. Modern
   numpy/scipy strongly prefer Python ≥3.10. Pin to 3.10+?
4. **Rust MSRV.** Pin to a recent stable (e.g. 1.85) or stay on the
   `--edition 2024` defaults?
5. **License continuity.** The C++ code carries a custom academic license
   (see `cpp/LICENSE`). Ports inherit it — confirm with the original
   authors before the first public release.

## 10. Where to look next

- For the algorithm itself: `cpp/RCR_paper.pdf` (full version) and
  [arXiv:2301.07838](https://arxiv.org/abs/2301.07838) (concise preprint).
- For the existing Python API: `cpp/src/RCR_python.cpp` and the tutorials
  under `cpp/docs/source/tutorials/`.
- For shared test data: [`assets/test/README.md`](../assets/test/README.md).
- For working norms in this repo: [`../AGENT.md`](../AGENT.md).
