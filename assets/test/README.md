# Test Data Sets

Reference datasets for validating Robust Chauvenet Rejection (RCR)
implementations. The same files back the Python and Rust reference tests so
results can be compared across implementations.

All files are UTF-8 CSV with a one-line header. No missing values; no NaNs.

## Files

| File | Rows | Columns | Use case |
|---|---|---|---|
| `data_smoke.csv` | 8 | `y` | Tiny known-answer smoke test (one obvious outlier: `11`). |
| `data_singlevalue.csv` | 1000 | `y` | Heavily contaminated 1-D set (~85% one-sided contaminants). Recovers `mu=0`, `sigma=1`. |
| `data_weighted_singlevalue.csv` | 200 | `w,y` | Weighted 1-D set; ~30% high-mean contaminants given low weights. Recovers `mu=5`, `sigma=2`. |
| `data_linear.csv` | 999 | `x,y` | Linear functional-form fit with outliers. Originally from `cpp/testdata/`. |
| `data_exponential.csv` | 199 | `x,y` | Exponential functional-form fit with outliers. Originally from `cpp/testdata/`. |

## Generation

The two functional-form CSVs (`data_linear.csv`, `data_exponential.csv`) were
carried over verbatim from the upstream C++ codebase's `testdata/` directory
(provenance: Maples et al. 2018).

The three single-value CSVs were generated with `random.seed(42)` using the
Python standard library, so they are reproducible without numpy. The exact
generation logic is recorded in [`agents/test_data_generation.md`](../../agents/test_data_generation.md)
should the datasets ever need to be regenerated.

## Expected results (for cross-implementation validation)

These targets describe the underlying *generating* distribution, not exact
empirical values for the sampled CSV. Implementations should land close to
these but need not match each other bit-for-bit unless they share an
algorithm.

- `data_smoke.csv` — RCR with `LS_MODE_68` (bulk) must reject `11.0` and keep
  the remaining 7 points.
- `data_singlevalue.csv` — recovered `mu ~ 0`, `sigma ~ 1` after rejection,
  with ~150 points retained.
- `data_weighted_singlevalue.csv` — recovered `mu ~ 5`, `sigma ~ 2`.
- `data_linear.csv` — slope and intercept consistent with the clean
  sub-population; see the upstream `singlevalue`/`functional` tutorial.
- `data_exponential.csv` — same.
