"""Type definitions and utilities for reduction passes.

This module defines the core type aliases and abstractions for reduction:

- ReductionPass[T]: A function that attempts to reduce a test case
- ReductionPump[T]: A function that may temporarily increase test case size
- Format[S, T]: A bidirectional transformation between types
- compose(): Combines a Format with a pass to work on a different type

These abstractions enable format-agnostic reduction: the same pass
(e.g., "delete duplicate elements") can work on bytes, lines, tokens,
JSON arrays, or any other sequence-like type.

Note: Format, ParseError, and DumpError are defined in shrinkray.problem
and re-exported here for backward compatibility.
"""

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

from shrinkray.problem import (
    DumpError,
    Format,
    ParseError,
    ReductionProblem,
)


# Re-export for backward compatibility
__all__ = [
    "DumpError",
    "Format",
    "ParseError",
    "ReductionPass",
    "ReductionPump",
    "compose",
]


S = TypeVar("S")
T = TypeVar("T")


# A reduction pass takes a problem and attempts to reduce it.
# The pass modifies the problem by calling is_interesting() with smaller candidates.
# When a reduction succeeds, problem.current_test_case is automatically updated.
ReductionPass = Callable[[ReductionProblem[T]], Awaitable[None]]

# A reduction pump can temporarily INCREASE test case size.
# Example: inlining a function makes code larger, but may enable further reductions.
# The reducer runs passes on the pumped result using backtrack() to try to
# reduce it below the original size.
ReductionPump = Callable[[ReductionProblem[T]], Awaitable[T]]


def compose(format: Format[S, T], reduction_pass: ReductionPass[T]) -> ReductionPass[S]:
    """Wrap a reduction pass to work through a Format transformation.

    This is the key combinator for format-agnostic reduction. It takes
    a pass that works on type T and returns a pass that works on type S,
    by parsing S->T before the pass and dumping T->S after.

    Example:
        # delete_duplicates works on sequences
        # Split(b"\\n") parses bytes into lines
        # The composed pass deletes duplicate lines from bytes
        line_dedup = compose(Split(b"\\n"), delete_duplicates)

    If parsing fails, the composed pass returns immediately (no-op).
    """

    @wraps(reduction_pass)
    async def wrapped_pass(problem: ReductionProblem[S]) -> None:
        view = problem.view(format)

        try:
            view.current_test_case
        except ParseError:
            return

        await reduction_pass(view)

    wrapped_pass.__name__ = f"{format.name}/{reduction_pass.__name__}"

    return wrapped_pass
