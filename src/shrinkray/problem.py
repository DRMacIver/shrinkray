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
from functools import lru_cache, total_ordering
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
from shrinkray.nondeterminism import (
    ANCHOR_SEED_RUNS,
    DETECTION_REPLAYS,
    GATE_RUNS,
    VERIFY_INTERVAL,
    Evidence,
    NondeterminismPolicy,
    Verdict,
    confirmation_bar,
    gauntlet,
)
from shrinkray.reformat import basic_format, canonical_distance
from shrinkray.work import WorkContext


S = TypeVar("S")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class BacktrackHistory(Protocol[T_co]):
    """The test cases a reduction has adopted, oldest first: the original
    input at index 0, then each adopted reduction. Entries may be read
    lazily (from disk), so only indexing and length are required."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> T_co: ...


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


# The cache only needs a handful of live entries: each comparison touches
# two strings (the candidate and the current test case, which is re-keyed
# on every comparison), plus a few more for candidates concurrently in
# flight under parallelism. Each entry retains both the raw string and its
# canonicalised form (roughly twice the test case size), so a large cache
# would pin many copies of a multi-megabyte test case in memory for no
# extra hits.
@lru_cache(maxsize=8)
def reflow_sort_key(s: str) -> Any:
    """Canonicalisation-based ordering key for a text test case.

    Orders primarily by the natural key of the *reflowed* (whitespace-
    canonicalised) form, so ordering is immune to layout differences (a cramped
    one-liner and its readable form compare equal on this term). Ties are broken
    toward the raw form that is CLOSEST to its canonical -- i.e. the cleanest,
    least-reformatted valid form -- then by line-count closeness to the (readable)
    canonical, then by the raw natural key as a deterministic total-order tail.

    This makes reduction prefer readable, well-structured test cases while still
    minimising content, and is immune to the raw-whitespace pathologies (cramped
    output, blank lines, over/under-splitting) that a length-first key rewards.
    Memoised: the reflow is O(n) and this key is recomputed for the current test
    case on every candidate comparison.
    """
    canonical = basic_format(s)
    return (
        natural_key(canonical),
        canonical_distance(s, canonical),
        abs(len(s.splitlines()) - len(canonical.splitlines())),
        natural_key(s),
    )


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
                    return (0, reflow_sort_key(s))
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
        return reflow_sort_key(value)
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

    # Calls made by the patch applier's merge probes (testing combinations
    # of patches that each passed on their own), and how many probes.
    merge_probes: int = 0
    merge_probe_calls: int = 0
    # Under a nondeterministic test: confirmation sweeps run, and the
    # calls made while one was running.
    confirmation_sweeps: int = 0
    confirmation_sweep_calls: int = 0

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
    # Called after every real evaluation of the interestingness test (not
    # cache hits), letting the reducer observe a running pass's progress
    # (set by the reducer for the duration of a pass run).
    pass_call_monitor: Callable[[], None] | None = None

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
        try:
            if self.sort_key(test_case) > self.sort_key(self.current_test_case):
                return False
        except DumpError:
            # Views compute sort keys by dumping the candidate. A candidate
            # that can't be dumped can't be tested, so it's not a reduction.
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

    @property
    def nondeterministic(self) -> bool:
        """Whether the interestingness test has been observed to be
        nondeterministic. Reducers consult this because a pass that made
        no progress on a nondeterministic test may still make progress on
        the same test case when re-run."""
        return False

    async def attempt_unstick(self) -> bool:
        """Called by reducers when a full round of reduction made no progress.

        Returns True if something changed (e.g. the test timeout was raised,
        invalidating cached timeout-failures) such that another round of
        reduction might now make progress. The default implementation has
        nothing to change.
        """
        await trio.lowlevel.checkpoint()
        return False

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
            nondeterministic_source=lambda: self.nondeterministic,
        )


class InvalidInitialExample(ValueError):
    pass


@define
class InterestingnessResult:
    """The result of running the interestingness test on a candidate.

    Interestingness predicates may return one of these instead of a plain
    bool when the result should only be cached conditionally: `cache_valid`
    is consulted on each cache hit and the result is re-tested once it
    returns False. This is used for candidates that timed out, which might
    succeed if retried under a larger timeout.
    """

    interesting: bool
    cache_valid: Callable[[], bool] | None = None
    # Whether the run timed out. A timeout is a failed run for reduction
    # purposes but says nothing about whether the test is deterministic,
    # so nondeterminism detection ignores it.
    timed_out: bool = False


@define
class Ledger:
    """What is known about one candidate.

    Under a deterministic test one run decides and `verdict` is latched
    immediately. Under a nondeterministic test `evidence` accumulates
    across every run of the candidate (including retries by later
    passes), `min_hits` pins the number of interesting runs an accept
    needs, fixed the first time the candidate is proposed so its
    stopping rule never changes mid-test, and `verdict` latches only
    once the gauntlet reaches a decision.
    """

    evidence: Evidence = attrs.Factory(Evidence)
    verdict: bool | None = None
    cache_valid: Callable[[], bool] | None = None
    min_hits: int | None = None
    # A snapshot of the evidence at the moment the gauntlet accepted, so
    # the runs after it (which the stopping rule did not select on) can
    # be told apart when the anchor is raised.
    accepted_at: Evidence | None = None

    def unselected_evidence(self) -> Evidence:
        """The runs recorded after the accept decision."""
        assert self.accepted_at is not None
        return self.evidence.since(self.accepted_at)

    def latched(self) -> bool:
        return self.verdict is not None and (
            self.cache_valid is None or self.cache_valid()
        )


def default_cache_key(value: Any) -> str:
    if not isinstance(value, bytes):
        if not isinstance(value, str):
            value = repr(value)
        value = value.encode("utf-8")

    # 16 hex digits = 64 bits. A big reduction can test hundreds of
    # thousands of candidates of the same length; at 32 bits a birthday
    # collision (silently serving the wrong cached result) becomes likely.
    hex = hashlib.sha1(value).hexdigest()[:16]
    return f"{len(value)}:{hex}"


class BasicReductionProblem(ReductionProblem[T]):
    """Concrete implementation of ReductionProblem for in-memory reduction.

    This is the main implementation used by Shrink Ray. It provides:
    - Caching of interestingness results (by content hash)
    - Statistics tracking (calls, cache hits, timing)
    - Callbacks for reduction events
    - Handling of nondeterministic interestingness tests (see
      shrinkray.nondeterminism), when given a policy

    Cached results are kept for the whole reduction (the cache holds only
    small content hashes, so this is cheap). Ordinary reduction rarely
    retries a candidate, but the restart phase deliberately replays earlier
    rounds' attempts, and those replays are answered from the cache instead
    of re-running the interestingness test.
    """

    def __init__(
        self,
        initial: T,
        is_interesting: Callable[[T], Awaitable[bool | InterestingnessResult]],
        work: WorkContext,
        sort_key: Callable[[T], Any] = default_sort_key,
        size: Callable[[T], int] = default_size,
        display: Callable[[T], str] = default_display,
        stats: ReductionStats | None = None,
        cache_key: Callable[[Any], str] = default_cache_key,
        unstick: Callable[[], Awaitable[bool]] | None = None,
        policy: NondeterminismPolicy | None = None,
        history: BacktrackHistory[T] | None = None,
        nondeterministic_source: Callable[[], bool] | None = None,
    ):
        """
        `policy`, when given, turns on nondeterminism handling: the initial
        test case is replayed at setup and the current one periodically and
        at the end to detect a nondeterministic test, and once detected,
        candidates must clear the policy's gauntlet before adoption. Without
        one every run is taken as a verdict.

        `history` is the sequence of test cases this reduction has adopted,
        oldest first (the original input, then each reduction), used to
        backtrack to a reproducing test case when nondeterminism is only
        detected after the reducer has adopted candidates on single runs.
        Without one, only the original input is available to backtrack to.

        `nondeterministic_source` is for problems that delegate their
        evaluations to another problem (the restart phase, pumps), so the
        reducer running them sees the delegate's determinism status.
        """
        super().__init__(work=work)
        self.__initial = initial
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

        self.__ledgers: dict[str, Ledger] = {}
        self.__cache_key = cache_key
        self.__is_interesting = is_interesting
        self.__unstick = unstick
        self.__policy = policy
        self.__history = history
        self.__nondeterministic_source = nondeterministic_source
        self.__on_reduce_callbacks: list[Callable[[T], Awaitable[None]]] = []
        self.__has_set_up = False
        self.__calls_at_last_verify = 0

    async def setup(self) -> None:
        if self.__has_set_up:
            return
        self.__has_set_up = True
        result, timed_out, _ = await self.__execute(self.current_test_case)
        if self.__policy is None:
            if not result:
                raise self.__invalid_initial()
            return
        # The initial test case is the one run we did not select for being
        # interesting, so its first run counts as evidence like any replay.
        # A first run that misses is not yet a verdict either: a flaky
        # initial test case may fail its first run and reproduce on the
        # replays, in which case it is interesting and the test is
        # nondeterministic.
        evidence = Evidence()
        if not timed_out:
            evidence.record(result)
        all_reproduced = await self.__replays_all_reproduce(
            evidence, DETECTION_REPLAYS - 1, stop_on_miss=result
        )
        # An initial test case that has not reproduced yet gets the
        # confirmation bar's gate: it is only invalid once it has missed
        # GATE_RUNS times in a row, so a rarely-reproducing one is not
        # refused on a handful of unlucky runs.
        attempts = DETECTION_REPLAYS
        while evidence.interesting == 0 and attempts < GATE_RUNS:
            attempts += 1
            await self.__replays_all_reproduce(evidence, 1)
        if evidence.interesting == 0:
            raise self.__invalid_initial()
        if not (result and all_reproduced):
            await self.__flip(evidence)

    def __invalid_initial(self) -> InvalidInitialExample:
        return InvalidInitialExample(
            f"Initial example ({self.display(self.current_test_case)}) does not satisfy interestingness test."
        )

    async def __execute(
        self, test_case: T, *, replay: str | None = None
    ) -> tuple[bool, bool, Callable[[], bool] | None]:
        """Run the underlying predicate once and normalize its result to
        (interesting, timed_out, cache_valid), updating the call statistics.
        A replay is any run beyond a candidate's first; `replay` names the
        site spending it (see NondeterminismPolicy.replay_sites)."""
        outcome = await self.__is_interesting(test_case)
        if isinstance(outcome, InterestingnessResult):
            result = (outcome.interesting, outcome.timed_out, outcome.cache_valid)
        else:
            result = (outcome, False, None)
        self.stats.calls += 1
        if result[0]:
            self.stats.interesting_calls += 1
        if self.current_pass_stats is not None:
            self.current_pass_stats.test_evaluations += 1
        if self.__policy is not None:
            if replay is not None:
                self.__policy.record_replay(result[0], replay)
            if self.__policy.confirming:
                self.stats.confirmation_sweep_calls += 1
        return result

    def display(self, value: T) -> str:
        return self.__display(value)

    @property
    def stats(self) -> ReductionStats:
        return self._stats

    @property
    def policy(self) -> NondeterminismPolicy | None:
        return self.__policy

    @property
    def nondeterministic(self) -> bool:
        if self.__policy is not None:
            return self.__policy.active
        if self.__nondeterministic_source is not None:
            return self.__nondeterministic_source()
        return False

    def ledger(self, test_case: T) -> Ledger:
        """What is known about `test_case`, for inspection."""
        return self.__ledgers[self.__cache_key(test_case)]

    @property
    def backtrack_history(self) -> BacktrackHistory[T] | None:
        return self.__history

    def sort_key(self, test_case: T) -> Any:
        return self.__sort_key(test_case)

    def size(self, test_case: T) -> int:
        return self.__size(test_case)

    def on_reduce(self, callback: Callable[[T], Awaitable[None]]) -> None:
        """Every time `is_interesting` is called with a successful reduction,
        call `fn` with the new value. Note that these are called outside the lock."""
        self.__on_reduce_callbacks.append(callback)

    async def attempt_unstick(self) -> bool:
        await trio.lowlevel.checkpoint()
        if self.__policy is not None:
            if not self.__policy.active:
                # The final check for a test that looked deterministic all
                # the way through: replay the result before believing it.
                evidence = Evidence()
                if not await self.__replays_all_reproduce(evidence, DETECTION_REPLAYS):
                    await self.__flip(evidence)
                    return True
            elif not self.__policy.confirming:
                # Fast sweeps reject a candidate on one missed run, so a
                # fixpoint reached that way is not a certificate. Run one
                # more round in which every candidate is driven to a bound
                # verdict; adoption during it drops back to fast sweeps.
                self.__policy.confirming = True
                self.stats.confirmation_sweeps += 1
                return True
            else:
                # A confirmation sweep found nothing. Before believing the
                # result, measure it: an incumbent that does not reproduce
                # at the required rate is recovered from, and the next
                # round starts from the recovered test case.
                self.__policy.confirming = False
                evidence = await self.measure_current(ANCHOR_SEED_RUNS)
                for _ in range(evidence.interesting):
                    self.__policy.record_incumbent_run(True)
                for _ in range(evidence.runs - evidence.interesting):
                    self.__policy.record_incumbent_run(False)
                if self.__policy.incumbent_failing():
                    await self.__recover()
                    return True
        if self.__unstick is None:
            return False
        return await self.__unstick()

    async def measure_current(self, runs: int) -> Evidence:
        """Replay the current test case `runs` times and report how many
        reproduced, for the final report."""
        evidence = Evidence()
        await self.__replays_all_reproduce(
            evidence, runs, stop_on_miss=False, site="report"
        )
        return evidence

    async def __replays_all_reproduce(
        self,
        evidence: Evidence,
        runs: int,
        *,
        stop_on_miss: bool = True,
        site: str = "detection",
    ) -> bool:
        """Replay the current test case `runs` times concurrently, recording
        the outcomes into `evidence` (timeouts excluded: they say nothing
        about determinism). Returns whether every completed run reproduced.
        With `stop_on_miss` the remaining replays are cancelled on the
        first miss."""
        test_case = self.current_test_case
        missed = False

        async with trio.open_nursery() as nursery:

            async def replay() -> None:
                nonlocal missed
                interesting, timed_out, _ = await self.__execute(test_case, replay=site)
                if timed_out:
                    return
                evidence.record(interesting)
                if not interesting:
                    missed = True
                    if stop_on_miss:
                        nursery.cancel_scope.cancel()

            for _ in range(runs):
                nursery.start_soon(replay)
        return not missed

    async def __flip(self, evidence: Evidence) -> None:
        """Switch into nondeterministic handling, `evidence` being the
        replays of the current test case that revealed it."""
        assert self.__policy is not None
        self.__policy.flip()
        self.work.warn(
            "Nondeterministic interestingness test detected: the current test "
            "case did not reproduce on a replay. From now on candidates are "
            "confirmed by repeated runs before being adopted, which costs "
            "more calls but keeps the result reproducing."
        )
        # Nothing recorded before the flip was a verdict, but every run was
        # a sample: keep the evidence and forget the conclusions.
        for ledger in self.__ledgers.values():
            ledger.verdict = None
            ledger.cache_valid = None
        self.__policy.confirming = False
        await self.__confirm_or_backtrack(evidence)

    async def __confirm_or_backtrack(self, evidence: Evidence) -> None:
        """The current test case was adopted on single runs. Confirm that
        it reproduces, and if it does not, back up through the adopted
        history to the newest test case that does."""
        assert self.__policy is not None
        if await self.__confirmation_batch(self.current_test_case, evidence):
            return
        accepted = await self.__scan_history(self.__confirmation_batch)
        if accepted is None:
            self.work.warn(
                "The current test case reproduces rarely (interesting on "
                f"{evidence.interesting} of {evidence.runs} replays) and no "
                "earlier test case did better. Continuing from it, but "
                "reduction may be slow."
            )
            return
        test_case, accepted_evidence = accepted
        self.work.warn(
            "Backtracking to an earlier test case that reproduces (interesting "
            f"on {accepted_evidence.interesting} of {accepted_evidence.runs} "
            "replays)."
        )
        await self.__set_current(test_case)

    async def __confirmation_batch(self, test_case: T, evidence: Evidence) -> bool:
        """Drive `evidence` about `test_case` to a confirmation-bar verdict,
        extending an accept to the anchor seed size and offering the
        extension (the runs the bar did not select on) to the anchor."""
        assert self.__policy is not None
        while True:
            verdict = confirmation_bar(evidence)
            if verdict == Verdict.REJECT:
                return False
            if verdict == Verdict.ACCEPT:
                accepted_at = Evidence(evidence.interesting, evidence.runs)
                while evidence.runs < ANCHOR_SEED_RUNS:
                    interesting, _, _ = await self.__execute(test_case, replay="seed")
                    evidence.record(interesting)
                self.__policy.raise_anchor(evidence.since(accepted_at))
                return True
            interesting, _, _ = await self.__execute(test_case, replay="confirmation")
            evidence.record(interesting)

    async def __scan_history(
        self, judge: Callable[[T, Evidence], Awaitable[bool]]
    ) -> tuple[T, Evidence] | None:
        """Find the newest adopted test case that `judge` accepts.

        Entries are scanned newest first at geometrically growing
        distances, then the boundary between the last rejected and the
        first accepted entry is refined by bisection. Under uncertainty
        this is biased towards older (larger) entries, which is the safe
        direction: an older entry can only reproduce better.
        """
        history: BacktrackHistory[T] = (
            self.__history if self.__history is not None else [self.__initial]
        )
        current = self.current_test_case
        positions = [
            i for i in range(len(history) - 1, -1, -1) if history[i] != current
        ]
        if not positions:
            return None

        async def test(k: int) -> tuple[T, Evidence] | None:
            test_case = history[positions[k]]
            evidence = Evidence()
            if await judge(test_case, evidence):
                return (test_case, evidence)
            return None

        # Every position is tested at most once: the geometric scan stops
        # at the first accept, the oldest entry is only tried when the scan
        # skipped it, and the bisection stays strictly between the last
        # reject and the accept.
        rejected = -1
        accepted: tuple[int, tuple[T, Evidence]] | None = None
        k = 0
        while k < len(positions):
            result = await test(k)
            if result is not None:
                accepted = (k, result)
                break
            rejected = k
            k = 2 * k + 1
        if accepted is None:
            oldest = len(positions) - 1
            result = await test(oldest) if oldest > rejected else None
            if result is None:
                return None
            accepted = (oldest, result)
        while accepted[0] - rejected > 1:
            mid = (accepted[0] + rejected) // 2
            result = await test(mid)
            if result is not None:
                accepted = (mid, result)
            else:
                rejected = mid
        return accepted[1]

    async def __set_current(self, test_case: T) -> None:
        """Adopt `test_case` by backtracking rather than by reduction. It
        still counts as a reduction event and fires the reduction
        callbacks: the file on disk and the history directory must follow
        the current test case, whichever direction it moved."""
        self.stats.reductions += 1
        self.stats.time_of_last_reduction = time.time()
        self.stats.current_test_case_size = self.size(test_case)
        self.__current = test_case
        for f in self.__on_reduce_callbacks:
            await f(test_case)

    async def is_interesting(self, test_case: T) -> bool:
        """Returns true if this test_case is interesting."""
        await trio.lowlevel.checkpoint()
        if test_case == self.current_test_case:
            return True
        cache_key = self.__cache_key(test_case)
        ledger = self.__ledgers.get(cache_key)
        if ledger is not None and ledger.latched():
            assert ledger.verdict is not None
            return ledger.verdict

        if self.__policy is not None:
            await self.__maybe_verify_current()

        if self.__policy is None or not self.__policy.active:
            result, _, cache_valid = await self.__execute(test_case)
            ledger = Ledger(verdict=result, cache_valid=cache_valid)
            ledger.evidence.record(result)
            self.__ledgers[cache_key] = ledger
        else:
            if ledger is None:
                ledger = Ledger()
                self.__ledgers[cache_key] = ledger
            result = await self.__run_gauntlet(test_case, ledger)

        self.stats.failed_reductions += 1
        if result:
            if self.sort_key(test_case) < self.sort_key(self.current_test_case):
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
                if self.__policy is not None and self.__policy.active:
                    # The one validated event that may move the anchor: an
                    # accepted candidate becoming the incumbent, priced on
                    # its own gauntlet evidence. An improvement also ends
                    # any confirmation sweep, since the fixpoint it was
                    # certifying is gone.
                    self.__policy.raise_anchor(ledger.unselected_evidence())
                    self.__policy.adopt(ledger.unselected_evidence())
                    self.__policy.confirming = False
                for f in self.__on_reduce_callbacks:
                    await f(test_case)
            else:
                self.stats.wasted_interesting_calls += 1
        if self.pass_call_monitor is not None:
            self.pass_call_monitor()
        return result

    async def __maybe_verify_current(self) -> None:
        """Periodically replay the current test case. While the test still
        looks deterministic this catches a nondeterministic test that
        reproduces most of the time before the reducer has walked too far
        on single-run verdicts. Under nondeterministic handling it feeds
        the incumbent monitor, which catches an incumbent that does not
        reproduce at the rate the gauntlet required (a false accept, or
        an anchor the incumbent never lived up to)."""
        assert self.__policy is not None
        if self.stats.calls - self.__calls_at_last_verify < VERIFY_INTERVAL:
            return
        self.__calls_at_last_verify = self.stats.calls
        if not self.__policy.active:
            evidence = Evidence()
            if not await self.__replays_all_reproduce(evidence, 1):
                await self.__flip(evidence)
            return
        evidence = Evidence()
        await self.__replays_all_reproduce(
            evidence, 1, stop_on_miss=False, site="monitor"
        )
        if evidence.runs:
            self.__policy.record_incumbent_run(evidence.interesting == 1)
        if self.__policy.incumbent_failing():
            await self.__recover()

    async def __recover(self) -> None:
        """The incumbent does not reproduce at the rate the gauntlet
        required: treat it as a false accept, demand more of later
        candidates, and back up through the adopted history to the newest
        test case that clears the gauntlet."""
        assert self.__policy is not None
        policy = self.__policy
        policy.record_false_accept()
        incumbent = policy.incumbent
        self.work.warn(
            "The current test case reproduces less often than expected "
            f"(interesting on {incumbent.interesting} of {incumbent.runs} "
            f"replays); candidates now need {policy.min_hits} interesting "
            "runs to be adopted."
        )
        accepted = await self.__scan_history(self.__gauntlet_batch)
        if accepted is None:
            self.work.warn("No earlier test case reproduces better; continuing.")
            policy.adopt(Evidence())
            return
        test_case, evidence = accepted
        self.work.warn(
            "Backtracking to an earlier test case that reproduces (interesting "
            f"on {evidence.interesting} of {evidence.runs} replays)."
        )
        await self.__set_current(test_case)
        policy.adopt(Evidence())

    async def __gauntlet_batch(self, test_case: T, evidence: Evidence) -> bool:
        """Drive `evidence` about `test_case` to a gauntlet verdict against
        the current anchor, at the hit minimum in force."""
        assert self.__policy is not None
        while True:
            verdict = gauntlet(evidence, self.__policy.anchor, self.__policy.min_hits)
            if verdict != Verdict.CONTINUE:
                return verdict == Verdict.ACCEPT
            interesting, _, _ = await self.__execute(test_case, replay="confirmation")
            evidence.record(interesting)

    async def __run_gauntlet(self, test_case: T, ledger: Ledger) -> bool:
        """Judge a candidate under nondeterministic handling. Its first
        run recruits it: in a fast sweep a miss rejects it at the cost of
        that one run (the evidence is kept, so a later retry has more
        power), while a hit starts the gauntlet against the incumbent's
        anchor. An accept tops the ledger up to the anchor seed size before
        latching, so the bound that may raise the anchor is not biased by
        the stopping rule."""
        assert self.__policy is not None
        policy = self.__policy
        if ledger.min_hits is None:
            ledger.min_hits = policy.min_hits
        interesting, _, _ = await self.__execute(test_case)
        ledger.evidence.record(interesting)
        if not interesting and not policy.confirming:
            return False
        while True:
            verdict = gauntlet(ledger.evidence, policy.anchor, ledger.min_hits)
            if verdict == Verdict.REJECT:
                ledger.verdict = False
                return False
            if verdict == Verdict.ACCEPT:
                ledger.accepted_at = Evidence(
                    ledger.evidence.interesting, ledger.evidence.runs
                )
                # Top the ledger up towards the seed size, but only while
                # the runs could still raise the anchor: that is their
                # only purpose, and once a miss or the anchor's level has
                # put a raise out of reach they are wasted.
                while (
                    ledger.evidence.runs < ANCHOR_SEED_RUNS
                    and policy.raise_reachable(
                        ledger.unselected_evidence(),
                        ANCHOR_SEED_RUNS - ledger.evidence.runs,
                    )
                ):
                    interesting, _, _ = await self.__execute(test_case, replay="seed")
                    ledger.evidence.record(interesting)
                ledger.verdict = True
                return True
            interesting, _, _ = await self.__execute(test_case, replay="gauntlet")
            ledger.evidence.record(interesting)

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

    async def attempt_unstick(self) -> bool:
        return await self.__problem.attempt_unstick()

    @property
    def nondeterministic(self) -> bool:
        return self.__problem.nondeterministic

    def sort_key(self, test_case: T) -> Any:
        if self.__sort_key is not None:
            return self.__sort_key(test_case)
        return self.__problem.sort_key(self.__dump(test_case))

    def size(self, test_case: T) -> int:
        return self.__problem.size(self.__dump(test_case))
