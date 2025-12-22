# Core Abstractions

This document describes the foundational abstractions in Shrink Ray.

## ReductionProblem[T]

The central abstraction is `ReductionProblem[T]` from `problem.py`. This represents a test-case reduction task over values of type T.

### Key Properties

- **current_test_case**: The best-known interesting test case
- **is_interesting(test_case) -> bool**: Async predicate that tests if a candidate preserves the bug
- **sort_key(test_case)**: Returns a comparable key for ordering - defaults to shortlex ordering
- **size(test_case) -> int**: Returns the size of a test case (for metrics)

### Shortlex Ordering

Shrink Ray uses shortlex ordering: `(length, lexicographic_value)`. This means:
1. Shorter test cases are always preferred
2. Among equal-length test cases, lexicographically smaller is preferred

This is crucial for **reproducibility** - regardless of which reduction path is taken, the final result should be the same minimal test case.

### Reduction Events

The problem tracks statistics and fires events:
- `on_reduce`: Called whenever a smaller interesting test case is found
- `stats`: Tracks calls, cache hits, reductions, timing

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

The pump returns a (possibly larger) test case, and the reducer runs passes on it using `backtrack()` to try to reduce it below the original size.

## WorkContext

Manages parallelism using Trio. Key methods:
- `map(fn, items)`: Parallel map with lazy evaluation
- `filter(predicate, items)`: Parallel filter
- `find_first_value(items, predicate)`: Find first item satisfying predicate
- `find_large_integer(predicate)`: Binary search for largest n where predicate(n) holds
