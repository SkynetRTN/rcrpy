# `agents/` — Planning notes for the RCR reimplementation

This directory holds **exploratory plans, design notes, and reproducibility
logs** for the Python and Rust re-implementations of RCR. It is intended
for both human contributors and AI coding agents.

These documents are *living* — update them in place rather than appending
new files when the scope is the same. Use one file per topic, not one file
per discussion.

## Contents

| File | Purpose |
|---|---|
| [`python_vs_rust_plan.md`](python_vs_rust_plan.md) | The main exploratory plan comparing Python and Rust reimplementations: goals, criteria, scaffolding, parity strategy, and a phased roadmap. |
| [`test_data_generation.md`](test_data_generation.md) | Provenance and regeneration recipe for the CSVs under `assets/test/`. |

## How to use this directory

- Before starting work on a port, **read `python_vs_rust_plan.md`** and
  agree (with the requester) on which phase you are tackling.
- When you make a non-obvious design decision (e.g., picking ndarray over
  Vec, choosing thiserror over anyhow), record it as a short bullet in the
  relevant plan, *with a why*.
- If you create new test datasets, log the generation recipe in
  `test_data_generation.md` so anyone can reproduce them.

This directory is not a substitute for code comments or PR descriptions —
keep it focused on *cross-cutting* concerns that span more than one PR or
that future agents would have no way to recover from the diff alone.
