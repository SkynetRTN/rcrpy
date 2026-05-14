# Test data generation

Reproducibility log for the datasets under
[`assets/test/`](../assets/test/). All implementations should validate
against these files unchanged; this document only matters if a dataset
needs to be regenerated or extended.

## Provenance summary

| File | Origin | Reproducible? |
|---|---|---|
| `data_linear.csv` | Copied verbatim from `cpp/testdata/data_linear.csv`. Source: Maples et al. 2018 supplementary material. | No (legacy artifact). Treat as fixed. |
| `data_exponential.csv` | Copied verbatim from `cpp/testdata/data_exponential.csv`. Same source. | No (legacy artifact). Treat as fixed. |
| `data_smoke.csv` | Hand-authored from `cpp/tests/maintest.py` line 20 (`y = [0.1, 0.2, 0, 0.3, -0.1, -0.2, -0.3, 11]`). | Yes (trivially). |
| `data_singlevalue.csv` | Generated below with `random.seed(42)`. | Yes — Python stdlib only, no numpy. |
| `data_weighted_singlevalue.csv` | Generated below with the **same seeded RNG instance** as `data_singlevalue.csv`. | Yes — but you must regenerate **both** files in one run to reproduce, because they consume from the same RNG stream. |

## Regeneration recipe (Python stdlib only)

Run from the repository root. This reproduces `data_singlevalue.csv` and
`data_weighted_singlevalue.csv` exactly. It does **not** rewrite the legacy
CSVs.

```python
import random
random.seed(42)

# 1) data_singlevalue.csv — N=1000, 85% one-sided half-normal contamination.
N = 1000
n_contam = int(round(N * 0.85))
n_good = N - n_contam
good = [random.gauss(0.0, 1.0) for _ in range(n_good)]
contam = [abs(random.gauss(0.0, 10.0)) for _ in range(n_contam)]
data = good + contam
random.shuffle(data)
with open('assets/test/data_singlevalue.csv', 'w') as f:
    f.write('y\n')
    for y in data:
        f.write(f'{y:.6e}\n')

# 2) data_weighted_singlevalue.csv — N=200, 30% high-mean contamination.
N2 = 200
n_c2 = int(round(N2 * 0.30))
n_g2 = N2 - n_c2
good2 = [random.gauss(5.0, 2.0) for _ in range(n_g2)]
contam2 = [random.gauss(30.0, 5.0) for _ in range(n_c2)]
weights = ([random.uniform(0.5, 1.5) for _ in range(n_g2)]
           + [random.uniform(0.1, 0.6) for _ in range(n_c2)])
paired = list(zip(weights, good2 + contam2))
random.shuffle(paired)
with open('assets/test/data_weighted_singlevalue.csv', 'w') as f:
    f.write('w,y\n')
    for w, y in paired:
        f.write(f'{w:.6e},{y:.6e}\n')
```

### Why stdlib instead of numpy

The two synthetic datasets were generated with the Python standard
library's `random` module so the recipe is portable and seedable without an
optional numpy dependency. If you regenerate with numpy you will get
different bytes even with the same seed.

### Format conventions

- UTF-8, Unix line endings, one-line header.
- Scientific notation with 6 digits of precision (`%.6e`).
- No trailing newline-only rows.

## When to add a new dataset

Add a new CSV under `assets/test/` when:

1. A behavior is hard to provoke with the existing data (e.g., bimodal
   contamination, very small N, ties at the median, NaN handling).
2. You have a **specific** target output to assert against.

Don't add data just because "more is better" — every new file becomes a
parity-test burden across implementations.

When you do add one, in the same change:

- Update [`assets/test/README.md`](../assets/test/README.md) (per-file row
  in the table plus expected-result blurb).
- Add an entry to the provenance table at the top of this file and, if the
  dataset is synthetic, drop the generation snippet in this file.
- Commit the seed and any non-default RNG choices alongside the snippet.
