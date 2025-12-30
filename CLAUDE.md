# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Shrink Ray is the best test-case reducer in the world. It's a personal hobby/research project, but it's important that it be high-quality, robust, and user-friendly software so it can be widely used. C-Reduce is the only comparable tool, and Shrink Ray has surpassed it.

Because this is a hobby project worked on haphazardly, there's accumulated technical debt and the codebase quality is uneven. Don't assume existing code patterns are correct just because they exist - much of it reflects partial refactors or rushed work.

## Claude's Role and Quality Expectations

The goal is to achieve software quality beyond what a single person working part-time can accomplish. This means taking on maintenance work (fixing type errors, improving test coverage, refactoring) and doing it *well*.

### Be Meticulous, Not Sloppy

- **Fix problems properly** - Don't just suppress errors or use workarounds. If there's a type error, understand why and fix the underlying issue. If a test is hard to write, think about what would make the code more testable.
- **Don't give up on hard things** - When something seems difficult (like covering certain code paths), get creative. Refactor for testability, use Hypothesis for property-based testing, run subprocesses and gather coverage, write generators to produce edge-case inputs. Persistence matters more than speed.
- **Self-review before presenting work** - Before committing or presenting code, review it critically: "Is this sloppy? Did I take shortcuts? If I were reviewing someone else's PR with this code, what would I flag?" This catches many issues before they waste the maintainer's time.

### Avoid Suppressions

Never use these as shortcuts - they're code smells that indicate a problem to fix:

- **`# type: ignore`** - Fix the type error properly. Use proper type annotations, add type guards, or refactor to make types work. If you're adding `type: ignore`, ask yourself why the code is confusing the type checker and fix that.
- **`# pragma: no cover`** - Write a test that covers the code. If coverage is hard to achieve, refactor for testability (extract methods, use dependency injection, etc.).
- **`# pragma: no branch`** - Same as above - find a way to exercise the branch.
- **`# noqa`** - Fix the lint error. If the linter is wrong, it's usually a sign the code could be clearer.

When tempted to add a suppression, instead:
1. Understand why the tool is complaining
2. Fix the underlying issue (refactor, add types, write tests)
3. Only if the tool is genuinely wrong AND there's no cleaner solution, consider suppression - but this should be rare

**Allowed exceptions** (where suppression is acceptable):
- `# noqa: B027` for intentionally empty methods on abstract classes that serve as optional override hooks (no better alternative exists)

If you encounter a case where suppression seems genuinely necessary and principled, ask the maintainer about adding it to this list.

### Commits

- Make small, logically self-contained commits
- Each commit should ideally be lint-clean and pass tests
- Commits are a good checkpoint for self-review
- Just commit when ready - don't ask for permission
- Use the `/checkpoint` skill to ensure consistent quality at each commit
- **Always use `git add` with specific file paths** - Never use `git add -A`, `git add .`, or `git add <directory>`. Always list the specific files you intend to stage. This prevents accidentally committing unrelated files (test scripts, debug files, etc.).

### No Backward Compatibility

Shrink Ray is a standalone application, not a library. Do not add backward compatibility shims, re-exports, or compatibility layers when refactoring. When moving code between modules, update all imports directly rather than re-exporting from the old location.

### CLAUDE.md as Source of Truth

This file is the source of truth for project conventions. However:
- Update it based on feedback from the maintainer
- Update it based on your own judgment about what would improve the project
- Ask about specific style decisions you notice and record them here

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
- `sort_key(test_case: T)` - Ordering for reproducibility (natural ordering for text, shortlex for binary)
- Problems can be "viewed" through Formats to reduce structured data

**Format[S, T]** (`problem.py`): Bridges bytes ↔ structured data.
- `parse(input: S) -> T` and `dumps(output: T) -> S`
- Enables format-agnostic passes to work on bytes, JSON, Python AST, etc.
- `compose(format, pass)` in `passes/definitions.py` wraps a pass to work through a format layer

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

1. **Natural ordering**: Ensures reproducibility - same minimal result regardless of reduction path. Uses a multi-tier heuristic for text (length, average squared line length, line count, then character ordering) and shortlex for binary data.
2. **Cache clearing on reduction**: When a smaller test case is found, old cached results are no longer useful (derived from old test case)
3. **View caching**: `problem.view(format)` caches parsed views to avoid redundant parsing
4. **Speculative parallelism**: Multiple candidates tested concurrently; first success wins, others are "wasted" but harmless

## Development Process: Test-Driven Development

This codebase is complex. The TUI runs in asyncio, the reducer runs in trio, they communicate via subprocess with a JSON protocol, and there are multiple layers of abstraction. **This complexity makes bugs easy to introduce and hard to find.** The tooling (tests, coverage, lints) exists to catch these bugs early. Use it properly.

### The Core Principle

**A feature is NOT complete until it has 100% test coverage.** This is not a suggestion—it is the definition of "done." Attempts to work around this (suppressions, claiming something is "too hard to test") are actively harmful because they leave bugs in the codebase that will cause problems later.

The following keywords are used per RFC 2119:
- **MUST** / **MUST NOT** - Absolute requirements
- **SHOULD** / **SHOULD NOT** - Strong recommendations with exceptions only when fully understood
- **MAY** - Truly optional

### Why TDD Matters Here

This codebase has repeatedly demonstrated that:
1. Code written without tests first is usually broken
2. "It's too hard to test" usually means the code needs refactoring
3. Coverage gaps always correspond to bugs discovered later
4. The tests catch bugs that seem obvious in hindsight but weren't caught during implementation

TDD is not bureaucracy—it's the most efficient path to working code in a codebase this complex.

### The Development Workflow

#### Phase 1: Understand the Testing Surface

Before writing any code, you MUST identify which layers need testing:

**Layer 1: Pure Logic (no I/O)**
- Data transformations, protocol serialization, algorithms
- Test with: Direct unit tests, Hypothesis property tests
- Example: `PassStatsData` serialization, history file parsing

**Layer 2: Worker/Reducer (trio async)**
- The `ReducerWorker` class, reduction passes, problem state
- Test with: `pytest-trio`, `@pytest.mark.trio`, mock I/O streams
- Example: `test_subprocess_worker.py` tests worker commands in isolation

**Layer 3: TUI Components (asyncio)**
- Individual widgets like `StatsDisplay`, `ContentPreview`, modals
- Test with: `asyncio.run()` wrapper, `FakeReductionClient`
- Example: Testing that a modal displays correct data without spawning a real worker

**Layer 4: Integration (subprocess communication)**
- Full TUI ↔ Worker communication
- Test with: Real subprocess spawning (mark as `@pytest.mark.slow`)
- Example: `test_tui_history_modal_during_reduction`

You MUST test each layer independently before testing them together. If a bug exists in the worker layer, you SHOULD be able to reproduce it with a worker-only test, not by running the full TUI.

#### Phase 2: Write Tests First

For each feature, you MUST write tests before implementation:

1. **Start with the contract** - What should this feature do? Write a test that asserts the expected behavior.

2. **Write failing tests** - The test MUST fail before you write the implementation. If it passes, either the test is wrong or the feature already exists.

3. **Test error cases first** - Error handling is where most bugs hide. Write tests for:
   - Missing required parameters
   - Invalid input
   - State not initialized
   - Network/I/O failures

4. **Test the happy path** - Only after error cases are covered.

Example for adding a new worker command:

```python
# Step 1: Test missing parameter
@pytest.mark.trio
async def test_handle_new_command_missing_param():
    worker = ReducerWorker()
    response = await worker._handle_new_command("id", {})
    assert response.error == "param is required"

# Step 2: Test invalid state
@pytest.mark.trio
async def test_handle_new_command_no_state():
    worker = ReducerWorker()
    response = await worker._handle_new_command("id", {"param": "value"})
    assert response.error == "State not available"

# Step 3: Test success case
@pytest.mark.trio
async def test_handle_new_command_success():
    worker = ReducerWorker()
    worker.state = MagicMock()
    # ... setup ...
    response = await worker._handle_new_command("id", {"param": "value"})
    assert response.result == {"status": "success"}
```

#### Phase 3: Implement Minimal Code

Write the minimum code to make tests pass:

1. You MUST NOT write code that isn't exercised by a test
2. You SHOULD run tests after each small change
3. If you find yourself writing complex logic, stop and write more tests first

#### Phase 4: Check Coverage

After implementation, you MUST run coverage:

```bash
uv run coverage run -m pytest tests/ -m "not slow" --ignore=tests/test_tui_snapshots.py -q
uv run coverage report --fail-under=100
```

**If coverage is not 100%, the feature is not complete.**

When coverage shows missing lines:
1. You MUST NOT add `# pragma: no cover`
2. You MUST write a test that exercises the missing code
3. If the code is genuinely unreachable, you MUST delete it
4. If testing is difficult, you SHOULD refactor the code to be more testable

Common refactoring patterns for testability:
- Extract pure functions from methods with side effects
- Use dependency injection for external services
- Split large methods into smaller, testable pieces
- Add parameters to control behavior in tests (e.g., `_for_testing` flags with proper tests)

#### Phase 5: Run Lints

After coverage passes, you MUST run lints:

```bash
just lint
```

Lint errors indicate code quality issues. You MUST NOT suppress them with `# noqa` except for the explicitly allowed cases in the "Avoid Suppressions" section.

#### Phase 6: Self-Review

Before committing, ask yourself:
- Did I write tests first, or did I retrofit them?
- Is every code path tested?
- Would I be embarrassed if someone reviewed this code?
- Did I take any shortcuts?

### Testing Different Architectural Layers

#### Testing the Worker Without the TUI

Most worker features can be tested without the TUI:

```python
@pytest.mark.trio
async def test_worker_feature():
    # Create worker with fake I/O
    input_stream = BidirectionalInputStream()
    output = MemoryOutputStream()
    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    # Send commands directly
    input_stream.send_command(Request(id="1", command="start", params={...}))

    # Run worker in nursery
    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.run)
        # ... interact with worker ...
        nursery.cancel_scope.cancel()
```

This is MUCH faster than spawning a real subprocess and avoids TUI complexity.

#### Testing the TUI Without a Real Worker

Use `FakeReductionClient` to test TUI behavior:

```python
def test_tui_feature():
    async def run():
        updates = [ProgressUpdate(status="Running", size=100, ...)]
        client = FakeReductionClient(updates=updates)

        app = ShrinkRayApp(file_path="test.txt", test=["true"], client=client)
        async with app.run_test() as pilot:
            await pilot.press("x")  # Test keyboard interaction
            # ... assert widget state ...

    asyncio.run(run())
```

This tests TUI logic without subprocess overhead.

#### When to Use Integration Tests

Integration tests (real TUI + real worker) SHOULD be used sparingly:
- Verifying the protocol works end-to-end
- Testing timing-sensitive behavior
- Catching bugs that only appear with real async scheduling

Integration tests MUST be marked `@pytest.mark.slow`.

### What "Too Hard to Test" Actually Means

When you think something is too hard to test, it usually means one of:

1. **The code is poorly structured** - Refactor it. Extract testable pieces.
2. **You don't understand the testing tools** - Read the existing tests for examples.
3. **The test would be slow** - That's fine, mark it `@pytest.mark.slow`.
4. **You need to mock something** - Use `unittest.mock.patch` or dependency injection.

"Too hard to test" is almost never a valid reason to skip tests. If you genuinely believe something cannot be tested, explain the specific technical barrier and propose a refactoring that would make it testable.

### Async Tests
- **Use pytest-trio for most async tests** - This project uses pytest-trio, so async test functions work directly. Simply define `async def test_something():` and pytest-trio will run it.
- **Exception: SubprocessClient tests use asyncio** - The `SubprocessClient` class uses asyncio (for textual TUI compatibility), so tests for it must use `asyncio.run()` wrappers:
  ```python
  def test_subprocess_client_something():
      async def run():
          client = SubprocessClient()
          # ... test logic ...
      asyncio.run(run())
  ```
  This isolates asyncio tests from the trio event loop.

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
- **Use module-level functions, not classes** - Write tests as `def test_something():` at module level, not inside `class TestSomething:`. Group related tests using section comments instead of classes.
- Group related tests with section comments (e.g., `# === View tests ===`)
- Keep tests fast (< 500ms each, ideally much less)
- Test edge cases explicitly with meaningful test names
- **All imports at top of file** - Never put imports inside test functions. Put all imports (including `from unittest.mock import patch`) at the top of the test file. To check for violations: `git grep '    import' tests/`

### Slow Tests
- **Mark genuinely slow tests with `@pytest.mark.slow`** - Tests that inherently require waiting (timeouts, full reductions, integration tests) should be marked as slow
- **By default, `just test` skips slow tests** - This keeps the feedback loop fast during development
- **Run `just test -m ""` to run all tests including slow ones** - Use this before committing or in CI
- **Maintain 100% coverage from non-slow tests** - If a slow test provides unique coverage, either:
  1. Add a fast test that covers the same code path differently
  2. Accept the test as non-slow if the coverage is essential
  3. Refactor the code to be more testable without slow operations
- **Use `--durations=10` to monitor test performance** - The justfile includes this by default to show slowest tests

### TUI Testing
- **Snapshot tests** (`tests/test_tui_snapshots.py`) use a vendored copy of `pytest-textual-snapshot` in `tests/pytest_textual_snapshot.py`
  - The vendored copy fixes syrupy 5.0 compatibility (upstream uses `_file_extension` but syrupy 5.0 changed to `file_extension`)
  - Snapshots are saved as `.svg` files for easy viewing
- Run `just test tests/test_tui_snapshots.py --snapshot-update` to update snapshots after intentional UI changes
- View snapshot gallery: `uv run pytest tests/test_tui_snapshots.py --snapshot-report` then open `snapshot_report.html`
- **Unit tests** (`tests/test_tui.py`) use `FakeSubprocessClient` to test TUI logic without spawning subprocesses
- Textual's `refresh(layout=True)` is needed when updating reactive properties that affect widget height
