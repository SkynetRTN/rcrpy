# AGENT.md

Ground rules and orientation for AI agents (Claude Code, Cursor, etc.)
working inside this repository. Humans are welcome to read it too — it is
essentially a brief contributor's guide.

## TL;DR

- This repo is being **restructured for a Python and a Rust port** of the
  existing C++ RCR library. The C++ tree under [`cpp/`](cpp/) is the
  **frozen reference** — do not refactor it.
- All implementations validate against the **shared CSVs** in
  [`assets/test/`](assets/test/). Do not duplicate or regenerate that data;
  read from it.
- Exploratory plans and design notes live in [`agents/`](agents/). When you
  produce a multi-file plan, drop it there rather than scattering it.

## What this project is

Robust Chauvenet Rejection (RCR) — statistical outlier rejection that stays
accurate on heavily contaminated samples. See [`README.md`](README.md) for
the science background, and [`cpp/RCR_paper.pdf`](cpp/RCR_paper.pdf) /
[arXiv:1807.05276](https://arxiv.org/abs/1807.05276) for the algorithm.

The active work is **re-implementing RCR in Python and Rust**, keeping the
C++ implementation as the algorithmic ground truth.

## Directory map

| Path | What it is | What you can do |
|---|---|---|
| [`cpp/`](cpp/) | Original C++ + pybind11 implementation. | Read for behavior; **do not modify** unless explicitly asked. |
| [`assets/test/`](assets/test/) | Shared validation CSVs. | Read. Add new datasets only when a new test case demands one — and document it in the asset README. |
| [`agents/`](agents/) | Exploratory plans, design notes, dataset-generation logs. | Add new `.md` files here for planning artifacts. Update existing ones in place when their scope is the same. |
| `python/` (planned) | Pure-Python reimplementation. | To be created per [`agents/python_vs_rust_plan.md`](agents/python_vs_rust_plan.md). |
| `rust/` (planned) | Rust reimplementation. | Same. |

## House rules

### Reference parity over local cleverness

The C++ code is the authoritative algorithm. When porting:

1. Find the matching routine in [`cpp/src/`](cpp/src/) (`RCR.cpp`,
   `FunctionalForm.cpp`, `MiscFunctions.cpp`, `NonParametric.cpp`).
2. Translate semantics first; idiomatic refactors come after parity tests pass.
3. If the C++ does something subtle (precomputed unity tables, magic
   constants, branch ordering), preserve it and leave a one-line comment
   pointing back to the file/line in `cpp/`.

If you find what looks like a bug in `cpp/`, **flag it**; do not silently
"fix" it in the port. Behavior changes belong in a follow-up after parity
is established.

### Test data is shared, not duplicated

Implementations read from [`assets/test/`](assets/test/). Do not copy CSVs
into a per-implementation `tests/` directory — use relative paths to the
shared assets. If a new dataset is needed, generate it deterministically
(seed your RNG), add it to the asset directory, and update both
[`assets/test/README.md`](assets/test/README.md) and
[`agents/test_data_generation.md`](agents/test_data_generation.md).

### Don't bridge implementations prematurely

It is tempting to bind the existing C++ from Rust, or to call the Rust core
from the Python port. Resist that until the standalone Python and Rust
ports each pass parity against `cpp/`. Premature bridging hides bugs.

### Don't churn the C++ tree

`cpp/` is preserved for provenance and as an executable spec. Do not
reformat, refactor, or "tidy" it. The only acceptable edits are: fixing a
build break on a modern toolchain, or annotating a confirmed bug with a
comment that points to the port's workaround.

### Scope of new work

Stay inside the explicitly requested scope. The plan in
[`agents/python_vs_rust_plan.md`](agents/python_vs_rust_plan.md) is
**exploratory** — it should be read and updated, not blindly executed
end-to-end. If a task says "scaffold the Python module," do that and stop;
do not also scaffold the Rust crate unless asked.

## Workflow

1. **Read the plan.** Start with [`agents/python_vs_rust_plan.md`](agents/python_vs_rust_plan.md).
2. **Anchor on a test.** Pick a dataset under [`assets/test/`](assets/test/)
   and the expected behavior from its README. Drive the port from that
   test, not from staring at C++.
3. **Port one routine at a time.** Match C++ behavior, then refactor.
4. **Update notes, not memory.** When you discover an algorithmic subtlety
   worth recording, append it to the relevant doc under `agents/`. Keep
   commit messages and PR descriptions short — explanations belong in the
   plan files.

## Build and run hints

- **C++ / Python binding (current):** `cd cpp && python3 -m pip install -e .`
  then `python3 tests/maintest.py`. Requires pybind11.
- **Python port:** TBD. Will live under `python/` with a `pyproject.toml`.
- **Rust port:** TBD. Will live under `rust/` as a Cargo workspace. PyO3
  bindings are a stretch goal, not a v1 requirement.

## When in doubt

Ask the human. The restructuring touches three implementation tracks at
once, and small early decisions (vector vs. ndarray, error type, where
unity tables live) propagate widely. A two-line clarification beats a
500-line refactor.
