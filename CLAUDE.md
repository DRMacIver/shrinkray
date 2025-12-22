# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install dependencies
just install

# Run tests (skips slow tests by default)
just test

# Run all tests including slow ones
just test -m ""

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

## Detailed Documentation

See the `notes/` directory for detailed architecture documentation:
- `notes/core-abstractions.md` - ReductionProblem, Format, passes, pumps
- `notes/patching-system.md` - How parallel patch application works
- `notes/reduction-passes.md` - Catalog of all reduction passes with examples
- `notes/parallelism-model.md` - WorkContext and speculative execution

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

**ReductionPump** (`passes/definitions.py`): Like a pass but can temporarily INCREASE size (e.g., inlining functions). Returns a new test case that the reducer tries to reduce below the original.

**Reducer** (`reducer.py`): Orchestrates passes. `ShrinkRay` is the main reducer for bytes, organizing passes into tiers:
1. **initial_cuts**: Fast high-value passes (comments, hollow, large blocks) with timeout-based cancellation
2. **great_passes**: Core loop (line deletion, token deletion, lift_braces) - loops until no progress
3. **ok_passes**: Run when great_passes stop making progress
4. **last_ditch_passes**: Expensive or low-yield passes (byte lowering, substitutions)

### Parallelism Model

**WorkContext** (`work.py`): Manages Trio-based parallelism with methods like `map()`, `filter()`, `find_first_value()`. Uses lazy prefetching with backpressure.

**PatchApplier** (`passes/patching.py`): Handles parallel patch application with binary-search merging to find maximal compatible patch sets. Key insight: if patches A and B work independently, trying both together often works.

### Format-Specific Support (src/shrinkray/passes/)

- `bytes.py` - Generic byte-level cuts (lines, blocks, tokens, whitespace)
- `sequences.py` - Operations on sequences (block_deletion, delete_duplicates)
- `genericlanguages.py` - Language-agnostic text passes (comments, brackets, identifiers)
- `python.py` - libcst-based AST reductions (lift blocks, strip annotations)
- `json.py` - JSON-specific passes (delete keys recursively)
- `sat.py` - DIMACS CNF format with unit propagation
- `clangdelta.py` - C/C++ support via creduce's clang_delta tool (pumps)

### Data Flow

```
CLI → ShrinkRayState → ReductionProblem → Reducer → Passes → Patches → is_interesting → Update state
```

Passes work by generating patches (typically `Cuts` for deletions), applying them in parallel, and updating the problem state when reductions succeed.

### TUI Architecture (src/shrinkray/tui.py, src/shrinkray/subprocess/)

The interactive TUI uses **textual** (not urwid) and runs in a subprocess architecture:

```
Main Process (asyncio/textual)     Subprocess (trio)
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  ShrinkRayApp (textual)     │    │  ReducerWorker              │
│  - StatsDisplay widget      │◄───│  - Runs actual reduction    │
│  - ContentPreview widget    │    │  - Emits ProgressUpdate     │
│  - SubprocessClient         │───►│  - Handles start/cancel     │
└─────────────────────────────┘    └─────────────────────────────┘
         stdin/stdout JSON protocol
```

**Why subprocess?** Textual requires asyncio, but the reducer uses trio. They're incompatible in the same process.

**Protocol** (`subprocess/protocol.py`):
- `Request`: Commands sent to worker (start, cancel, status)
- `Response`: Command acknowledgments with results
- `ProgressUpdate`: Periodic stats (size, calls, reductions, parallelism, content preview)

**Key files**:
- `tui.py` - Textual app with StatsDisplay and ContentPreview widgets
- `subprocess/worker.py` - Entry point for reducer subprocess (`shrinkray-worker`)
- `subprocess/client.py` - SubprocessClient manages communication
- `subprocess/protocol.py` - Message dataclasses and JSON serialization

### Key Design Decisions

1. **Shortlex ordering**: Ensures reproducibility - same minimal result regardless of reduction path
2. **Cache clearing on reduction**: When a smaller test case is found, old cached results are no longer useful (derived from old test case)
3. **View caching**: `problem.view(format)` caches parsed views to avoid redundant parsing
4. **Speculative parallelism**: Multiple candidates tested concurrently; first success wins, others are "wasted" but harmless

## Testing Best Practices

### Async Tests
- **Use pytest-trio** - This project uses pytest-trio, so async test functions work directly. Do NOT create sync wrapper functions that call `trio.run(async_test)`. Simply define `async def test_something():` and pytest-trio will run it.

### Test Parametrization
- **Parametrize similar tests** - If multiple tests differ only in a single value (e.g., testing with values 0, 2, 4, 64, 100), use `@pytest.mark.parametrize` instead of duplicating the test function.
- **Use meaningful IDs** - Use `pytest.param(value, id="name")` with comments explaining why each value was chosen:
  ```python
  @pytest.mark.parametrize("target", [
      pytest.param(0, id="zero"),       # Edge case: boundary condition
      pytest.param(4, id="boundary"),   # Where linear scan ends
      pytest.param(64, id="power_of_two"),  # Tests binary search
  ])
  ```
- **Parametrize by parallelism** - Tests that involve WorkContext should typically be parametrized by parallelism `[1, 2]` to catch bugs that only manifest in parallel execution.

### Test Organization
- Group related tests with section comments (e.g., `# === View tests ===`)
- Keep tests fast (< 5 seconds each, ideally much less)
- Test edge cases explicitly with meaningful test names

### TUI Testing
- **Snapshot tests** (`tests/test_tui_snapshots.py`) use `pytest-textual-snapshot` for visual regression testing
- Run `just test tests/test_tui_snapshots.py --snapshot-update` to update snapshots after intentional UI changes
- View snapshot gallery: `uv run pytest tests/test_tui_snapshots.py --snapshot-report` then open `snapshot_report.html`
- **Unit tests** (`tests/test_tui.py`) use `FakeSubprocessClient` to test TUI logic without spawning subprocesses
- Textual's `refresh(layout=True)` is needed when updating reactive properties that affect widget height
