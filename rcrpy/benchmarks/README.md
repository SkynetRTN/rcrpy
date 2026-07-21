# `rcrpy` benchmarks

Standalone scripts for measuring `rcrpy` correctness and performance.
None of these are part of the installed package — they're maintainer
tools that live alongside the source.

| Script | What it does |
|---|---|
| [`reflect_parity.py`](reflect_parity.py) | Sweep all 20 Phase 1 parity scenarios (single-value RCR × every tech × iterative+bulk × weighted+unweighted). Reports max relative scalar disagreement vs the C++ oracle. Worst case: 1.7×10⁻¹⁵. |
| [`diagnostics.py`](diagnostics.py) | Per-scenario wall-clock + precision sweep for single-value RCR (port vs. C++ oracle). |
| [`diagnostics_functional.py`](diagnostics_functional.py) | Same for functional-form RCR. Demonstrates the port is FASTER than the oracle on this workload (80×–6000× depending on N). |
| [`bench_port_vs_oracle.py`](bench_port_vs_oracle.py) | One-shot bench on `data_singlevalue.csv` for quick perf checks. |

## Running

Each script is self-contained. From the repo root:

```bash
python rcrpy/benchmarks/reflect_parity.py
python rcrpy/benchmarks/diagnostics.py
python rcrpy/benchmarks/diagnostics_functional.py
```

All four require the legacy `rcr` C++ module (the oracle) plus `rcrpy`
itself installed in the active environment — i.e., `pip install -e
./python[dev]` covers everything.
