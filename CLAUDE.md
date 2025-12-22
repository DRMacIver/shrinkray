# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
just install

# Run all tests
just test

# Run a single test file
just test tests/test_sat.py

# Run a specific test
just test tests/test_sat.py::test_name -v

# Lint and type-check
just lint

# Run shrinkray CLI
uv run shrinkray <interestingness_test> <file_to_reduce>
```

Note: Some tests require `minisat` to be installed (`apt-get install minisat` on Ubuntu).

## Architecture Overview

Shrink Ray is a multiformat test-case reducer built on Trio for async/parallelism. The key architectural concepts:

### Core Abstractions (src/shrinkray/)

**ReductionProblem[T]** (`problem.py`): The central interface representing a reduction task.
- `current_test_case: T` - The current state being reduced
- `is_interesting(test_case: T) -> bool` - Tests if a candidate triggers the bug
- `sort_key(test_case: T)` - Shortlex ordering: (length, lexicographic) for reproducibility
- Problems can be "viewed" through Formats to reduce structured data

**Format[S, T]** (`passes/definitions.py`): Bridges bytes ↔ structured data.
- `parse(input: S) -> T` and `dumps(output: T) -> S`
- Enables format-agnostic passes to work on bytes, JSON, Python AST, etc.
- `compose(format, pass)` wraps a pass to work through a format layer

**ReductionPass** (`passes/definitions.py`): A callable `(ReductionProblem[T]) -> Awaitable[None]` that makes reduction attempts until no progress.

**Reducer** (`reducer.py`): Orchestrates passes. `ShrinkRay` is the main reducer for bytes, organizing passes into tiers (initial_cuts → great_passes → ok_passes → last_ditch_passes).

### Parallelism Model

**WorkContext** (`work.py`): Manages Trio-based parallelism with methods like `map()`, `filter()`, `find_first_value()`. Uses lazy prefetching with backpressure.

**PatchApplier** (`passes/patching.py`): Handles parallel patch application with binary-search merging to find maximal compatible patch sets.

### Format-Specific Support (src/shrinkray/passes/)

- `bytes.py` - Generic byte-level cuts (lines, blocks, tokens)
- `genericlanguages.py` - Language-agnostic text passes (brackets, whitespace)
- `python.py` - libcst-based AST reductions
- `json.py` - JSON-specific passes via SetPatches
- `sat.py` - DIMACS CNF format with unit propagation
- `clangdelta.py` - C/C++ support via creduce's clang_delta tool

### Data Flow

```
CLI → ShrinkRayState → ReductionProblem → Reducer → Passes → Patches → is_interesting → Update state
```

Passes work by generating patches (typically `Cuts` for deletions), applying them in parallel, and updating the problem state when reductions succeed.
