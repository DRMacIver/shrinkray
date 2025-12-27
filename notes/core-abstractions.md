# Core Abstractions

This document describes the foundational abstractions in Shrink Ray.

## ReductionProblem[T]

The central abstraction is `ReductionProblem[T]` from `problem.py`. This represents a test-case reduction task over values of type T.

### Key Properties

- **current_test_case**: The best known-interesting test case (may temporarily differ during backtracking from pumped test cases)
- **is_interesting(test_case) -> bool**: Async predicate that tests if a candidate preserves the bug
- **sort_key(test_case)**: Returns a comparable key for ordering, ensuring reproducible minimal results
- Note: The **size** of a test case is used as a heuristic for how good a reduction pass is: passes that reduce size are much more valuable than non-size-reducing passes.
- **size(test_case) -> int**: Returns the size of a test case (for metrics)

### Ordering Strategies

Shrink Ray uses different ordering strategies depending on the type of test case:

#### Natural Ordering (for text)

For text-based test cases (bytes that decode as valid Unicode in any encoding), Shrink Ray uses a **natural ordering** that produces human-readable minimal results. This ordering uses a chain of heuristics:

1. **Total length** - shorter strings are always preferred
2. **Average squared line length** - penalizes very long lines, preferring balanced code
   - Formula: `sum(len(line)²) / count(lines)²`
   - This makes "well-formatted" code with balanced lines preferred over single-line messes
3. **Number of lines** - fewer lines is better (after accounting for balance)
4. **List of line lengths** - lexicographically compare line length sequences
5. **Natural character order** - `whitespace < digits < lowercase < uppercase`
   - Unknown characters (punctuation, unicode) sort after letters by Unicode code point

This ordering is implemented via `LazyChainedSortKey`, which evaluates comparison functions lazily until one returns different values for the two inputs.

#### Shortlex Ordering (fallback for non-text)

If bytes cannot be decoded as valid Unicode, Shrink Ray falls back to **shortlex ordering**: `(length, lexicographic_value)`. This means:
1. Shorter test cases are always preferred
2. Among equal-length test cases, lexicographically smaller is preferred

Note: Bytes that decode as valid text are always preferred over bytes that don't, regardless of other ordering criteria.

#### Dict Ordering (for directories)

For dict-based test cases (used in directory reduction), ordering compares:
1. Total size of all values
2. Number of keys
3. Values for each key (in reverse order of their initial file sizes), using the appropriate ordering for each value

### Normalisation

Consistent ordering is crucial for **normalisation** - the idea that every interestingness test should ideally have a single canonical minimal result. See "One Test to Rule Them All" by Alex Groce and Josie Holmes for background on this concept.

### Callbacks and Statistics

The problem tracks statistics and supports callbacks:
- `on_reduce(callback)`: Register a callback to be called whenever a smaller interesting test case is found
- `stats`: Property returning `ReductionStats` with calls, reductions, timing info

### Views

`problem.view(format)` creates a new problem that parses input through a Format and serializes output back. This allows format-specific passes to work on structured data while the underlying problem operates on bytes.

## Format[S, T]

A `Format` bridges between two types, typically bytes to structured data.

```python
class Format[S, T]:
    def parse(self, input: S) -> T: ...
    def dumps(self, output: T) -> S: ...
    def is_valid(self, input: S) -> bool: ...
```

Examples:
- `Split(delimiter)`: bytes -> list of bytes (lines, semicolon-separated)
- `Tokenize()`: bytes -> list of tokens (respecting quotes/braces)
- `JSON`: bytes -> Python dict/list
- `DimacsCNF`: bytes -> list of clauses

The `compose(format, pass)` function wraps a pass to work through a format layer:
```python
# This pass deletes duplicate elements from sequences
compose(Split(b"\n"), delete_duplicates)  # Deletes duplicate lines
```

## ReductionPass[T]

A reduction pass is a callable: `(ReductionProblem[T]) -> Awaitable[None]`

Passes make reduction attempts by calling `problem.is_interesting()` with smaller candidates. When a reduction succeeds, the problem's `current_test_case` is automatically updated.

### Pass Categories in ShrinkRay

ShrinkRay organizes byte passes into tiers:

1. **initial_cuts**: Run first, fast high-value passes (comment removal, hollow, block deletion)
2. **great_passes**: Core passes run in a loop until no progress
3. **ok_passes**: Run when great passes stop making progress
4. **last_ditch_passes**: Expensive or low-value passes run at the end

## ReductionPump[T]

A pump is like a pass but can **temporarily increase** test case size:
`(ReductionProblem[T]) -> Awaitable[T]`

Example: `clang_delta` can inline a function (making code larger) which then allows other passes to delete more code.

The pump returns a (possibly larger) test case. The `Reducer` class has a `backtrack()` context manager that temporarily switches to reducing from this larger test case:

```python
# In Reducer.pump():
pumped = await pump(self.target)  # Get larger test case
with self.backtrack(pumped):      # Temporarily switch target
    await self.run_great_passes() # Try to reduce it
    # If result is smaller than original, it's kept
```

The `Reducer.backtrack()` context manager internally calls `problem.backtrack()` which creates a new `BasicReductionProblem` starting from the pumped test case.

## WorkContext

Manages parallelism using Trio. Key methods:
- `map(fn, items)`: Parallel map with lazy evaluation
- `filter(predicate, items)`: Parallel filter
- `find_first_value(items, predicate)`: Find first item satisfying predicate
- `find_large_integer(predicate)`: Find an n such that `predicate(n)` is true and `predicate(n+1)` is false. Note: this only finds the boundary where the predicate transitions from true to false; if the predicate is not monotonic, this may not be the largest n for which it holds.
