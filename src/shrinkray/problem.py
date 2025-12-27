"""Core abstractions for test-case reduction.

This module defines the fundamental interfaces for reduction problems:

- ReductionProblem[T]: The central abstraction representing a reduction task
- BasicReductionProblem[T]: A concrete implementation with caching and callbacks
- View[S, T]: A problem wrapper that parses through a Format

The key insight is that all reduction is about finding the smallest test case
that satisfies an "interestingness" predicate. The problem abstraction hides
the details of caching, parallelism, and state management.
"""

import hashlib
import string
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sized
from datetime import timedelta
from functools import total_ordering
from typing import (
    Any,
    Protocol,
    TypeVar,
    cast,
)

import attrs
import trio
from attrs import define
from humanize import naturalsize, precisedelta

from shrinkray.formatting import try_decode
from shrinkray.work import WorkContext


S = TypeVar("S")
T = TypeVar("T")


class PassStatsProtocol(Protocol):
    """Protocol for pass statistics tracking.

    This allows problem.py to track stats without importing from reducer.py,
    avoiding circular dependencies.
    """

    test_evaluations: int
    successful_reductions: int
    bytes_deleted: int


def shortlex[SizedT: Sized](value: SizedT) -> tuple[int, SizedT]:
    """Return a comparison key for shortlex ordering.

    Shortlex ordering compares first by length, then lexicographically.
    This ensures shorter test cases are always preferred, and among
    equal-length test cases, lexicographically smaller ones win.

    This ordering is crucial for reproducibility: regardless of which
    reduction path is taken, the final result should be the same minimal
    test case.

    Example:
        >>> shortlex(b"aa") < shortlex(b"aaa")  # shorter wins
        True
        >>> shortlex(b"ab") < shortlex(b"ba")   # same length, lex order
        True
    """
    return (len(value), value)


@total_ordering
class LazyChainedSortKey:
    """A comparison key that lazily evaluates a chain of comparison functions.

    This class provides an ordering that compares values by applying a sequence
    of functions in order. The first function that produces different values
    for two inputs determines the ordering. If all functions return equal
    values, the inputs are considered equal.

    This is used to implement the natural ordering for strings, which compares
    by length, then average squared line length, then number of lines, etc.

    The "lazy" aspect is that comparison functions are only evaluated until
    one returns different values, avoiding unnecessary computation.
    """

    def __init__(self, functions: list[Callable[[T], Any]], value: T):
        self.functions = functions
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, LazyChainedSortKey):
            return NotImplemented
        assert len(self.functions) == len(other.functions)
        return self.value == other.value

    def __lt__(self, other):
        if self == other:
            return False
        if not isinstance(other, LazyChainedSortKey):
            return NotImplemented
        for f in self.functions:
            self_key = f(self.value)
            other_key = f(other.value)
            if self_key < other_key:
                return True
            elif self_key > other_key:
                return False
        # All comparison functions returned equal values for different inputs.
        # This shouldn't happen with the current functions (natural_string_lex
        # compares character-by-character) but if it does, neither is less.
        return False


# Natural character ordering: whitespace < digits < lowercase < uppercase.
# Characters not in this string are sorted by ord() after all known characters.
NATURAL_CHARACTER_ORDER = (
    string.whitespace + string.digits + string.ascii_lowercase + string.ascii_uppercase
)
NATURAL_CHARACTER_ORDER_INDEX = {s: i for i, s in enumerate(NATURAL_CHARACTER_ORDER)}


def character_index(c: str) -> int:
    """Return the sorting index for a character in natural ordering.

    Characters in NATURAL_CHARACTER_ORDER get their position in that string.
    Unknown characters (punctuation, unicode, etc.) sort after all known
    characters, ordered by their Unicode code point.
    """
    return NATURAL_CHARACTER_ORDER_INDEX.get(c, len(NATURAL_CHARACTER_ORDER) + ord(c))


def natural_string_lex(s: str) -> list[int]:
    """Convert a string to a list of character indices for lexicographic comparison.

    This transforms the string so that comparing the resulting lists gives
    the natural character ordering (whitespace < digits < lowercase < uppercase).
    """
    return list(map(character_index, s))


# The chain of comparison functions used for natural string ordering.
# Each function is tried in sequence; the first that differs determines order.
#
# 1. Total length - shorter strings are always preferred
# 2. Average squared line length - penalizes very long lines, preferring balanced code
#    Formula: sum(len(line)²) / count(lines)²
# 3. Number of lines - fewer lines is better (after accounting for balance)
# 4. List of line lengths - lexicographically compare line length sequences
# 5. Natural character order - whitespace < digits < lowercase < uppercase
NATURAL_ORDERING_FUNCTIONS: list[Callable[[str], Any]] = [
    len,
    lambda s: sum(len(line) ** 2 for line in s.split("\n")) / len(s.split("\n")) ** 2,
    lambda s: len(s.splitlines()),
    lambda s: list(map(len, s.splitlines())),
    natural_string_lex,
]


def natural_key(s: str) -> LazyChainedSortKey:
    """Return a comparison key for natural string ordering.

    Natural ordering uses a chain of heuristics to determine which string
    is "smaller" (more reduced). This is designed to produce human-readable
    minimal test cases with balanced line lengths and natural character choices.

    See NATURAL_ORDERING_FUNCTIONS for the complete ordering criteria.
    """
    return LazyChainedSortKey(functions=NATURAL_ORDERING_FUNCTIONS, value=s)


def sort_key_for_initial(initial: Any) -> Callable[[Any], Any]:
    """Create a sort key function appropriate for the given initial value.

    This examines the initial test case and returns a comparison function
    that will be used to order all test cases during reduction.

    For bytes:
        - If decodable as text, uses natural ordering on the decoded string
        - Falls back to shortlex for binary data that can't be decoded

    For dicts:
        - Orders by total size of values, then number of keys
        - Then compares values for each key in order of largest-first

    For other types:
        - Falls back to natural ordering on repr()

    The returned function can be used as a sort key or comparison key.
    """
    if isinstance(initial, bytes):
        encoding, _ = try_decode(initial)
        if encoding is None:
            return shortlex
        else:

            def natural_for_encoding(b: bytes) -> Any:
                try:
                    s = b.decode(encoding)
                    return (0, natural_key(s))
                except UnicodeDecodeError:
                    return (1, shortlex(b))

            return natural_for_encoding
    elif isinstance(initial, dict):
        keys = sorted(initial, key=lambda k: shortlex(initial[k]), reverse=True)
        natural_keys = {k: sort_key_for_initial(v) for k, v in initial.items()}

        def dict_total_size(s):
            return sum(len(v) for v in s.values())

        def key_sort_key(k):
            def f(x):
                try:
                    v = x[k]
                except KeyError:
                    return (0,)
                else:
                    return (1, natural_keys[k](v))

            return f

        functions = [
            dict_total_size,
            len,
        ] + [key_sort_key(k) for k in keys]

        def dict_sort_key(v):
            return LazyChainedSortKey(
                functions=functions,
                value=v,
            )

        return dict_sort_key
    else:
        # We don't use this branch in the main app, but this
        # function is also used in tests.
        def fallback_sort_key(s):
            return natural_key(repr(s))

        return fallback_sort_key


def default_sort_key(value: Any) -> Any:
    """Return a comparison key for a value using type-appropriate ordering.

    This is a simpler alternative to sort_key_for_initial that doesn't
    examine the initial value to determine the best ordering.

    - bytes: shortlex ordering (length, then lexicographic)
    - str: natural ordering (length, line balance, character order)
    - other: shortlex on repr()

    Note: This really should return some sort of Comparable type, but Python
    doesn't have a built-in protocol for that.
    """
    if isinstance(value, bytes):
        return shortlex(value)
    elif isinstance(value, str):
        return natural_key(value)
    else:
        return shortlex(repr(value))


def default_display(value: Any) -> str:
    r = repr(value)
    if len(r) < 50:
        return f"{r} (size {len(value)})"
    return f"value of size {len(value)}"


class ParseError(Exception):
    """Raised when a Format cannot parse its input."""

    pass


class DumpError(Exception):
    """Raised when a Format cannot serialize its output.

    This occurs because not all internal representations map to valid
    output in the target format. For example, a reduction might create
    an invalid AST structure that cannot be converted back to source code.
    """

    pass


class Format[S, T](ABC):
    """A bidirectional transformation between two types.

    Formats enable format-agnostic passes by abstracting the
    parse/serialize cycle. For example:

    - Split(b"\\n"): bytes <-> list[bytes] (lines)
    - Tokenize(): bytes <-> list[bytes] (tokens)
    - JSON: bytes <-> Any (Python objects)
    - DimacsCNF: bytes <-> list[list[int]] (SAT clauses)

    A Format must satisfy the round-trip property:
        dumps(parse(x)) should be equivalent to x
        (possibly with normalization)

    Example usage:
        # Delete duplicate lines
        compose(Split(b"\\n"), delete_duplicates)

        # Reduce integer literals in source code
        compose(IntegerFormat(), reduce_integer)
    """

    @property
    def name(self) -> str:
        """Human-readable name for this format, used in pass names."""
        return repr(self)

    @abstractmethod
    def parse(self, input: S) -> T:
        """Parse input into the target type. Raises ParseError on failure."""
        ...

    def is_valid(self, input: S) -> bool:
        """Check if input can be parsed by this format."""
        try:
            self.parse(input)
            return True
        except ParseError:
            return False

    @abstractmethod
    def dumps(self, input: T) -> S:
        """Serialize the target type back to the source type."""
        ...


def default_size(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


@define
class ReductionStats:
    reductions: int = 0
    failed_reductions: int = 0

    calls: int = 0
    interesting_calls: int = 0
    wasted_interesting_calls: int = 0

    time_of_last_reduction: float = 0.0
    start_time: float = attrs.Factory(time.time)

    initial_test_case_size: int = 0
    current_test_case_size: int = 0

    def time_since_last_reduction(self) -> float:
        return time.time() - self.time_of_last_reduction

    def display_stats(self) -> str:
        runtime = time.time() - self.start_time
        if self.reductions > 0:
            reduction_percentage = (
                1.0 - self.current_test_case_size / self.initial_test_case_size
            ) * 100
            reduction_rate = (
                self.initial_test_case_size - self.current_test_case_size
            ) / runtime
            reduction_msg = (
                f"Current test case size: {naturalsize(self.current_test_case_size)} "
                f"({reduction_percentage:.2f}% reduction, {naturalsize(reduction_rate)} / second)"
            )
        else:
            reduction_msg = (
                f"Current test case size: {self.current_test_case_size} bytes"
            )

        return "\n".join(
            [
                reduction_msg,
                f"Total runtime: {precisedelta(timedelta(seconds=runtime))}",
                (
                    (
                        f"Calls to interestingness test: {self.calls} ({self.calls / runtime:.2f} calls / second, "
                        f"{self.interesting_calls / self.calls * 100.0:.2f}% interesting, "
                        f"{self.wasted_interesting_calls / self.calls * 100:.2f}% wasted)"
                    )
                    if self.calls > 0
                    else "Not yet called interestingness test"
                ),
                (
                    f"Time since last reduction: {self.time_since_last_reduction():.2f}s ({self.reductions / runtime:.2f} reductions / second)"
                    if self.reductions
                    else "No reductions yet"
                ),
            ]
        )


@define(slots=False)
class ReductionProblem[T](ABC):
    """Abstract base class representing a test-case reduction task.

    A ReductionProblem encapsulates everything needed to reduce a test case:
    - The current best-known interesting test case
    - A predicate to test if candidates are "interesting" (trigger the bug)
    - An ordering to determine which test cases are "smaller"

    Reduction passes work by generating candidate test cases and calling
    is_interesting() on them. When a smaller interesting test case is found,
    current_test_case is automatically updated.

    The problem maintains a cache of interestingness results and tracks
    statistics about the reduction process.

    Subclasses must implement:
    - current_test_case: Property returning the current best test case
    - is_interesting(test_case): Async method testing if a candidate works
    - sort_key(test_case): Returns a comparable key for ordering
    - size(test_case): Returns the size of a test case
    - display(value): Returns a human-readable representation
    """

    work: WorkContext
    # Track current pass stats for real-time updates (set by reducer)
    current_pass_stats: PassStatsProtocol | None = None

    def __attrs_post_init__(self) -> None:
        # Cache of View objects for each Format, to avoid re-parsing
        self.__view_cache: dict[Any, ReductionProblem[Any]] = {}

    def view(self, format: Format[T, S] | type[Format[T, S]]) -> "ReductionProblem[S]":
        """Create a view of this problem through a Format.

        A View wraps this problem, parsing the current test case through
        the format's parse() method and serializing candidates back through
        dumps(). This allows format-specific passes to work on structured
        data while the underlying problem operates on bytes.

        Example:
            # Work on lines instead of raw bytes
            line_problem = byte_problem.view(Split(b"\\n"))

            # Work on JSON structure
            json_problem = byte_problem.view(JSON)

        Views are cached: calling view() with the same format returns the
        same View object, avoiding redundant parsing.
        """
        try:
            return cast(ReductionProblem[S], self.__view_cache[format])
        except KeyError:
            pass

        concrete_format: Format[T, S] = format() if isinstance(format, type) else format

        result: View[T, S] = View(
            problem=self,
            work=self.work,
            dump=concrete_format.dumps,
            parse=concrete_format.parse,
        )

        return cast(ReductionProblem[S], self.__view_cache.setdefault(format, result))

    async def setup(self) -> None:  # noqa: B027
        """Initialize the problem before reduction begins.

        Subclasses may override this to perform validation or initialization.
        The default implementation does nothing.
        """

    @property
    @abstractmethod
    def current_test_case(self) -> T: ...

    @property
    @abstractmethod
    def stats(self) -> ReductionStats: ...

    @abstractmethod
    async def is_interesting(self, test_case: T) -> bool:
        pass

    async def is_reduction(self, test_case: T) -> bool:
        """Check if test_case would be a valid reduction from current state.

        A valid reduction is an interesting test case that is smaller than
        the current one (according to sort_key). This is a convenience method
        that short-circuits if the candidate is larger.
        """
        if test_case == self.current_test_case:
            return True
        if self.sort_key(test_case) > self.sort_key(self.current_test_case):
            return False
        return await self.is_interesting(test_case)

    @abstractmethod
    def sort_key(self, test_case: T) -> Any: ...

    @abstractmethod
    def size(self, test_case: T) -> int:
        return len(test_case)  # type: ignore

    @property
    def current_size(self) -> int:
        return self.size(self.current_test_case)

    @abstractmethod
    def display(self, value: T) -> str: ...

    def backtrack(self, new_test_case: T) -> "ReductionProblem[T]":
        """Create a new problem starting from a different test case.

        This is used by reduction pumps to try larger test cases temporarily.
        The new problem shares the same is_interesting predicate but starts
        from new_test_case instead of current_test_case.

        If reduction succeeds and the result is smaller than the original
        current_test_case, it can be adopted into the main problem.

        Example:
            # Pump inlines a function, making code larger
            pumped = await pump(problem)  # Returns larger test case
            backtracked = problem.backtrack(pumped)
            # Try to reduce the larger test case
            await run_passes(backtracked)
            # If result is smaller than original, keep it
        """
        return BasicReductionProblem(
            initial=new_test_case,
            is_interesting=self.is_interesting,
            work=self.work,
            sort_key=self.sort_key,
            size=self.size,
            display=self.display,
        )


class InvalidInitialExample(ValueError):
    pass


def default_cache_key(value: Any) -> str:
    if not isinstance(value, bytes):
        if not isinstance(value, str):
            value = repr(value)
        value = value.encode("utf-8")

    hex = hashlib.sha1(value).hexdigest()[:8]
    return f"{len(value)}:{hex}"


class BasicReductionProblem(ReductionProblem[T]):
    """Concrete implementation of ReductionProblem for in-memory reduction.

    This is the main implementation used by Shrink Ray. It provides:
    - Caching of interestingness results (by content hash)
    - Statistics tracking (calls, cache hits, timing)
    - Callbacks for reduction events
    - Automatic cache clearing when a reduction succeeds

    The cache clearing is a practical choice: when we find a smaller test case,
    cached results for candidates derived from the old test case are no longer
    useful (we're now reducing from a different starting point). Clearing the
    cache saves memory and avoids serving stale cache entries that won't help.
    """

    def __init__(
        self,
        initial: T,
        is_interesting: Callable[[T], Awaitable[bool]],
        work: WorkContext,
        sort_key: Callable[[T], Any] = default_sort_key,
        size: Callable[[T], int] = default_size,
        display: Callable[[T], str] = default_display,
        stats: ReductionStats | None = None,
        cache_key: Callable[[Any], str] = default_cache_key,
    ):
        super().__init__(work=work)
        self.__current = initial
        self.__sort_key = sort_key
        self.__size = size
        self.__display = display
        if stats is None:
            self._stats = ReductionStats()
            self._stats.initial_test_case_size = self.size(initial)
            self._stats.current_test_case_size = self.size(initial)
        else:
            self._stats = stats

        self.__is_interesting_cache: dict[str, bool] = {}
        self.__cache_key = cache_key
        self.__is_interesting = is_interesting
        self.__on_reduce_callbacks: list[Callable[[T], Awaitable[None]]] = []
        self.__current = initial
        self.__has_set_up = False

    async def setup(self) -> None:
        if self.__has_set_up:
            return
        self.__has_set_up = True
        if not await self.__is_interesting(self.current_test_case):
            raise InvalidInitialExample(
                f"Initial example ({self.display(self.current_test_case)}) does not satisfy interestingness test."
            )

    def display(self, value: T) -> str:
        return self.__display(value)

    @property
    def stats(self) -> ReductionStats:
        return self._stats

    def sort_key(self, test_case: T) -> Any:
        return self.__sort_key(test_case)

    def size(self, test_case: T) -> int:
        return self.__size(test_case)

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        """Every time `is_interesting` is called with a successful reduction,
        call `fn` with the new value. Note that these are called outside the lock."""
        self.__on_reduce_callbacks.append(callback)

    async def is_interesting(self, test_case: T) -> bool:
        """Returns true if this test_case is interesting."""
        await trio.lowlevel.checkpoint()
        if test_case == self.current_test_case:
            return True
        cache_key = self.__cache_key(test_case)
        try:
            return self.__is_interesting_cache[cache_key]
        except KeyError:
            pass
        result = await self.__is_interesting(test_case)
        self.__is_interesting_cache[cache_key] = result
        self.stats.failed_reductions += 1
        self.stats.calls += 1

        # Update current pass stats if a pass is running
        if self.current_pass_stats is not None:
            self.current_pass_stats.test_evaluations += 1

        if result:
            self.stats.interesting_calls += 1
            if self.sort_key(test_case) < self.sort_key(self.current_test_case):
                self.__is_interesting_cache.clear()
                self.stats.failed_reductions -= 1
                self.stats.reductions += 1
                self.stats.time_of_last_reduction = time.time()

                # Update current pass stats for reductions
                if self.current_pass_stats is not None:
                    self.current_pass_stats.successful_reductions += 1
                    size_diff = self.size(self.current_test_case) - self.size(test_case)
                    self.current_pass_stats.bytes_deleted += size_diff

                self.stats.current_test_case_size = self.size(test_case)
                self.__current = test_case
                for f in self.__on_reduce_callbacks:
                    await f(test_case)
            else:
                self.stats.wasted_interesting_calls += 1
        return result

    @property
    def current_test_case(self) -> T:
        return self.__current


class View[S, T](ReductionProblem[T]):
    """A view of a ReductionProblem through a parse/dump transformation.

    View wraps an underlying problem, presenting it as a different type.
    For example, a problem over bytes can be viewed as a problem over
    lists of lines, or JSON structures, or AST nodes.

    The View:
    - Parses the underlying problem's test case on access
    - Dumps candidates back to the underlying type for testing
    - Caches the parsed representation for efficiency
    - Delegates interestingness testing to the underlying problem

    The caching is subtle: when the underlying problem's test case changes,
    the View re-parses it. But it only updates its cached value if the new
    parsed value is "smaller" (according to sort_key), to maintain
    monotonicity of reduction.
    """

    def __init__(
        self,
        problem: ReductionProblem[S],
        parse: Callable[[S], T],
        dump: Callable[[T], S],
        work: WorkContext | None = None,
        sort_key: Callable[[T], Any] | None = None,
    ):
        super().__init__(work=work or problem.work)
        self.__problem = problem
        self.__parse = parse
        self.__dump = dump
        self.__sort_key = sort_key

        current = problem.current_test_case
        self.__prev = current
        self.__current = parse(current)

    def display(self, value: T) -> str:
        return default_display(value)

    @property
    def stats(self) -> ReductionStats:
        return self.__problem.stats

    @property
    def current_test_case(self) -> T:
        current = self.__problem.current_test_case
        if current != self.__prev:
            self.__prev = current
            new_value = self.__parse(current)
            if self.__sort_key is None or self.__sort_key(new_value) < self.__sort_key(
                self.__current
            ):
                self.__current = new_value
        return self.__current

    async def is_interesting(self, test_case: T) -> bool:
        try:
            return await self.__problem.is_interesting(self.__dump(test_case))
        except DumpError:
            return False

    def sort_key(self, test_case: T) -> Any:
        if self.__sort_key is not None:
            return self.__sort_key(test_case)
        return self.__problem.sort_key(self.__dump(test_case))

    def size(self, test_case: T) -> int:
        return self.__problem.size(self.__dump(test_case))
