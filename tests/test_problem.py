"""Unit tests for problem module utilities and classes."""

import random
import time

import pytest
import trio

from shrinkray.nondeterminism import (
    ANCHOR_SEED_RUNS,
    DETECTION_REPLAYS,
    GATE_RUNS,
    GAUNTLET_FLOOR,
    GAUNTLET_MIN_HITS,
    VERIFY_INTERVAL,
    Evidence,
    NondeterminismPolicy,
)
from shrinkray.problem import (
    BasicReductionProblem,
    DumpError,
    Format,
    InterestingnessResult,
    InvalidInitialExample,
    ReductionProblem,
    ReductionStats,
    View,
    default_cache_key,
    default_display,
    default_size,
    default_sort_key,
    shortlex,
)
from shrinkray.work import WorkContext


# =============================================================================
# shortlex function tests
# =============================================================================


def test_shortlex_shorter_wins():
    assert shortlex(b"ab") < shortlex(b"abc")


def test_shortlex_same_length_lexicographic():
    assert shortlex(b"ab") < shortlex(b"ba")


def test_shortlex_equal():
    assert shortlex(b"ab") == shortlex(b"ab")


def test_shortlex_empty():
    assert shortlex(b"") < shortlex(b"a")


def test_shortlex_string():
    assert shortlex("ab") < shortlex("abc")


# =============================================================================
# default_sort_key function tests
# =============================================================================


def test_default_sort_key_bytes_ordering():
    assert default_sort_key(b"a") < default_sort_key(b"ab")


def test_default_sort_key_string_ordering():
    """default_sort_key uses natural_key for strings."""
    # Shorter strings are preferred
    assert default_sort_key("a") < default_sort_key("ab")
    # Same length, character order matters
    assert default_sort_key("a") < default_sort_key("b")


# =============================================================================
# default_display function tests
# =============================================================================


def test_default_display_short():
    result = default_display(b"hi")
    assert "b'hi'" in result
    assert "size 2" in result


def test_default_display_long():
    long_value = b"x" * 100
    result = default_display(long_value)
    assert "value of size 100" in result


def test_default_display_list():
    result = default_display([1, 2, 3])
    assert "size 3" in result


# =============================================================================
# default_size function tests
# =============================================================================


def test_default_size_bytes():
    assert default_size(b"hello") == 5


def test_default_size_list():
    assert default_size([1, 2, 3]) == 3


def test_default_size_no_len():
    # Objects without len() return 0
    assert default_size(42) == 0


# =============================================================================
# default_cache_key function tests
# =============================================================================


def test_default_cache_key_bytes():
    key = default_cache_key(b"hello")
    assert key.startswith("5:")  # length prefix
    # 64 bits of hash after the length prefix, to keep birthday
    # collisions unlikely across the lifetime of a big reduction.
    assert len(key.split(":")[1]) == 16


def test_default_cache_key_string():
    key = default_cache_key("hello")
    assert key.startswith("5:")


def test_default_cache_key_other():
    # Non-string/bytes uses repr
    key = default_cache_key([1, 2, 3])
    assert ":" in key


def test_default_cache_key_same_content_same_key():
    assert default_cache_key(b"test") == default_cache_key(b"test")


def test_default_cache_key_different_content_different_key():
    assert default_cache_key(b"test1") != default_cache_key(b"test2")


# =============================================================================
# ReductionStats tests
# =============================================================================


def test_reduction_stats_defaults():
    stats = ReductionStats()
    assert stats.reductions == 0
    assert stats.failed_reductions == 0
    assert stats.calls == 0
    assert stats.interesting_calls == 0
    assert stats.wasted_interesting_calls == 0


def test_reduction_stats_time_since_last_reduction():
    stats = ReductionStats()
    stats.time_of_last_reduction = stats.start_time
    # Should be close to 0
    assert stats.time_since_last_reduction() >= 0


def test_reduction_stats_display_no_reductions():
    stats = ReductionStats()
    stats.current_test_case_size = 100
    display = stats.display_stats()
    assert "100 bytes" in display
    assert "No reductions yet" in display


def test_reduction_stats_display_with_reductions():
    stats = ReductionStats()
    stats.initial_test_case_size = 1000
    stats.current_test_case_size = 500
    stats.reductions = 5
    stats.calls = 10
    stats.interesting_calls = 5
    stats.wasted_interesting_calls = 1
    stats.start_time = (
        time.time() - 10
    )  # Set start time 10 seconds ago to avoid division by zero
    display = stats.display_stats()
    assert "50.00% reduction" in display


def test_reduction_stats_display_no_calls():
    stats = ReductionStats()
    stats.current_test_case_size = 100
    display = stats.display_stats()
    assert "Not yet called interestingness test" in display


# =============================================================================
# BasicReductionProblem tests
# =============================================================================


async def test_basic_problem_current_test_case():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert problem.current_test_case == b"hello"


async def test_basic_problem_is_interesting_same_value():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    # Same value should return True without calling predicate
    result = await problem.is_interesting(b"hello")
    assert result is True


async def test_basic_problem_is_interesting_reduces():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    result = await problem.is_interesting(b"hi")
    assert result is True
    assert problem.current_test_case == b"hi"  # Should have reduced


async def test_basic_problem_is_interesting_not_reduction():
    async def is_interesting(x):
        return x == b"hello" or x == b"goodbye"

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    # "goodbye" is longer, so not a reduction
    result = await problem.is_interesting(b"goodbye")
    assert result is True
    assert problem.current_test_case == b"hello"  # Should NOT have changed


async def test_basic_problem_is_interesting_not_interesting():
    async def is_interesting(x):
        return x == b"hello"

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    result = await problem.is_interesting(b"hi")
    assert result is False
    assert problem.current_test_case == b"hello"


async def test_basic_problem_caching():
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    # First call
    await problem.is_interesting(b"x")
    count_after_first = call_count[0]

    # Second call with same value should be cached
    await problem.is_interesting(b"x")
    assert call_count[0] == count_after_first  # No additional call


async def test_basic_problem_setup_invalid():
    async def is_interesting(x):
        return False

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    with pytest.raises(InvalidInitialExample):
        await problem.setup()


async def test_basic_problem_setup_valid():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    await problem.setup()  # Should not raise


async def test_basic_problem_is_reduction():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    # Smaller value
    assert await problem.is_reduction(b"hi") is True
    # Same value
    assert await problem.is_reduction(problem.current_test_case) is True


async def test_basic_problem_is_reduction_larger():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hi",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    # Larger value should return False without calling predicate
    assert await problem.is_reduction(b"hello") is False


async def test_basic_problem_on_reduce_callback():
    reductions = []

    async def callback(value):
        reductions.append(value)

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    problem.on_reduce(callback)
    await problem.is_interesting(b"hi")
    assert reductions == [b"hi"]


async def test_basic_problem_stats_updated():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    await problem.is_interesting(b"hi")
    assert problem.stats.calls == 1
    assert problem.stats.reductions == 1
    assert problem.stats.interesting_calls == 1


async def test_basic_problem_backtrack():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    backtracked = problem.backtrack(b"world")
    assert backtracked.current_test_case == b"world"
    # Original unchanged
    assert problem.current_test_case == b"hello"


def test_basic_problem_size():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert problem.size(b"hello") == 5
    assert problem.current_size == 5


def test_basic_problem_display():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hi",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    display = problem.display(b"hi")
    assert "b'hi'" in display


def test_basic_problem_with_provided_stats():
    """Test that BasicReductionProblem uses provided stats."""

    async def is_interesting(x):
        return True

    custom_stats = ReductionStats()
    custom_stats.initial_test_case_size = 100
    custom_stats.current_test_case_size = 50

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        stats=custom_stats,
    )
    # Should use the provided stats, not create new ones
    assert problem.stats.initial_test_case_size == 100
    assert problem.stats.current_test_case_size == 50


# =============================================================================
# View tests
# =============================================================================


def test_view_current_test_case():
    """Test View parses underlying test case."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )
    assert view.current_test_case == "hello"


async def test_view_is_interesting_delegates():
    """Test View delegates is_interesting to underlying problem."""

    async def is_interesting(x):
        return x == b"hello"

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )

    assert await view.is_interesting("hello") is True
    assert await view.is_interesting("world") is False


async def test_view_is_interesting_handles_dump_error():
    """Test View returns False when dump raises DumpError."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    def dump(s):
        if s == "bad":
            raise DumpError("Cannot dump 'bad'")
        return s.encode("utf-8")

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=dump,
    )

    assert await view.is_interesting("hello") is True
    assert await view.is_interesting("bad") is False


async def test_view_is_reduction_handles_dump_error():
    """Regression test: is_reduction computes the candidate's sort key
    before testing interestingness. For a View the sort key dumps the
    candidate, so an undumpable candidate raised DumpError out of
    is_reduction (crashing patch merging) instead of being rejected."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    def dump(s):
        if s == "bad":
            raise DumpError("Cannot dump 'bad'")
        return s.encode("utf-8")

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=dump,
    )

    assert await view.is_reduction("bad") is False


def test_view_stats_delegates():
    """Test View returns underlying problem's stats."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )

    assert view.stats is problem.stats


def test_view_size_delegates():
    """Test View delegates size to underlying problem."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )

    assert view.size("hello") == 5


def test_view_sort_key_with_custom():
    """Test View uses custom sort_key when provided."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=lambda s: (len(s), s),
    )

    assert view.sort_key("hi") == (2, "hi")
    assert view.sort_key("hello") == (5, "hello")


def test_view_sort_key_without_custom():
    """Test View delegates sort_key to underlying problem when not provided."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )

    # Should use problem's sort_key on the dumped value
    assert view.sort_key("hi") == problem.sort_key(b"hi")


def test_view_display():
    """Test View uses default_display for display."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
    )

    display = view.display("hi")
    assert "size 2" in display


async def test_view_caches_parsed_value():
    """Test View caches parsed value and only updates when smaller."""
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    parse_calls = [0]

    def counting_parse(b):
        parse_calls[0] += 1
        return b.decode("utf-8")

    view = View(
        problem=problem,
        parse=counting_parse,
        dump=lambda s: s.encode("utf-8"),
        sort_key=len,
    )

    # Initial parse
    assert parse_calls[0] == 1
    initial = view.current_test_case
    assert initial == "hello"

    # Accessing again without underlying change shouldn't re-parse
    _ = view.current_test_case
    assert parse_calls[0] == 1

    # Reduce underlying problem
    await problem.is_interesting(b"hi")
    assert problem.current_test_case == b"hi"

    # Now accessing should re-parse
    current = view.current_test_case
    assert parse_calls[0] == 2
    assert current == "hi"


async def test_view_only_accepts_smaller_parse_results():
    """Test View only updates cached value if new value is smaller."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"5chars",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=len,  # Size-based sorting
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=len,
    )

    assert view.current_test_case == "5chars"

    # Reduce to smaller
    await problem.is_interesting(b"hi")
    assert view.current_test_case == "hi"

    # If somehow underlying got larger (shouldn't happen in practice),
    # view would keep the smaller cached value
    # This is tested by the sort_key comparison in the property


async def test_view_keeps_cached_value_if_parse_is_larger():
    """Test View keeps cached value when parsed result is larger."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hi",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=len,
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=len,
    )

    assert view.current_test_case == "hi"

    # Force underlying problem to have a larger value by directly manipulating
    # This simulates an edge case where the underlying changes but is larger
    problem._BasicReductionProblem__current = b"longer"  # type: ignore[attr-defined]

    # View should still return the cached smaller value
    current = view.current_test_case
    assert current == "hi"


async def test_basic_problem_setup_called_twice():
    """Test that setup() is idempotent - second call does nothing."""
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    await problem.setup()
    first_call_count = call_count[0]

    # Second call should not call is_interesting again
    await problem.setup()
    assert call_count[0] == first_call_count


# =============================================================================
# ReductionProblem.view() tests
# =============================================================================


def test_reduction_problem_view_method():
    """Test that view() creates a View with correct format."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    class StringFormat(Format[bytes, str]):
        @staticmethod
        def parse(data: bytes) -> str:
            return data.decode("utf-8")

        @staticmethod
        def dumps(value: str) -> bytes:
            return value.encode("utf-8")

    view = problem.view(StringFormat)
    assert view.current_test_case == "hello"


def test_reduction_problem_view_caches():
    """Test that view() returns cached View for same format."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    class StringFormat(Format[bytes, str]):
        @staticmethod
        def parse(data: bytes) -> str:
            return data.decode("utf-8")

        @staticmethod
        def dumps(value: str) -> bytes:
            return value.encode("utf-8")

    view1 = problem.view(StringFormat)
    view2 = problem.view(StringFormat)
    assert view1 is view2


def test_reduction_problem_view_with_instance():
    """Test that view() works with format instance."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    class StringFormat(Format[bytes, str]):
        @staticmethod
        def parse(data: bytes) -> str:
            return data.decode("utf-8")

        @staticmethod
        def dumps(value: str) -> bytes:
            return value.encode("utf-8")

    format_instance = StringFormat()
    view = problem.view(format_instance)
    assert view.current_test_case == "hello"


async def test_reduction_problem_base_setup():
    """Test that base ReductionProblem.setup() does nothing."""

    # Create a minimal concrete implementation
    class MinimalProblem(ReductionProblem[bytes]):
        def __init__(self, work: WorkContext):
            super().__init__(work=work)
            self._current = b"test"

        @property
        def current_test_case(self) -> bytes:
            return self._current

        @property
        def stats(self) -> ReductionStats:
            return ReductionStats()

        async def is_interesting(self, test_case: bytes) -> bool:
            return True

        def sort_key(self, test_case: bytes):
            return len(test_case)

        def size(self, test_case: bytes) -> int:
            return len(test_case)

        def display(self, value: bytes) -> str:
            return str(value)

    problem = MinimalProblem(work=WorkContext(parallelism=1))
    # Base setup should do nothing and not raise
    await problem.setup()


async def test_abstract_method_default_implementations():
    """Test that abstract method default implementations can be called via super().

    The abstract methods is_interesting and size have default implementations
    that can be used by subclasses calling super().
    """

    # Create a subclass that explicitly calls super() for abstract methods
    class SubclassThatCallsSuper(ReductionProblem[bytes]):
        def __init__(self, work: WorkContext):
            super().__init__(work=work)
            self._current = b"test"

        @property
        def current_test_case(self) -> bytes:
            return self._current

        @property
        def stats(self) -> ReductionStats:
            return ReductionStats()

        async def is_interesting(self, test_case: bytes) -> bool:
            # Call the base class implementation (which is just pass)
            await ReductionProblem.is_interesting(self, test_case)
            return True

        def sort_key(self, test_case: bytes):
            return len(test_case)

        def size(self, test_case: bytes) -> int:
            # Call the base class implementation
            return ReductionProblem.size(self, test_case)

        def display(self, value: bytes) -> str:
            return str(value)

    problem = SubclassThatCallsSuper(work=WorkContext(parallelism=1))

    # Test that size() calls base implementation (which returns len())
    assert problem.size(b"hello") == 5

    # Test that is_interesting() calls base implementation (which is pass)
    result = await problem.is_interesting(b"hello")
    assert result is True  # Our implementation returns True after calling super


# =============================================================================
# InterestingnessResult caching tests
# =============================================================================


async def test_outcome_true_is_adopted_as_reduction():
    async def is_interesting(x):
        return InterestingnessResult(interesting=True)

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert await problem.is_interesting(b"hi") is True
    assert problem.current_test_case == b"hi"


async def test_outcome_false_is_cached_while_valid():
    calls = [0]
    valid = [True]

    async def is_interesting(x):
        calls[0] += 1
        return InterestingnessResult(interesting=False, cache_valid=lambda: valid[0])

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert await problem.is_interesting(b"hi") is False
    assert await problem.is_interesting(b"hi") is False
    assert calls[0] == 1  # Cached while valid

    valid[0] = False
    assert await problem.is_interesting(b"hi") is False
    assert calls[0] == 2  # Invalidated: re-tested


async def test_invalidated_outcome_is_replaced_in_cache():
    calls = [0]

    async def is_interesting(x):
        calls[0] += 1
        if calls[0] == 1:
            # First run: conditionally cacheable and immediately invalid.
            return InterestingnessResult(interesting=False, cache_valid=lambda: False)
        # Second run: unconditionally cacheable.
        return InterestingnessResult(interesting=False)

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert await problem.is_interesting(b"hi") is False
    assert await problem.is_interesting(b"hi") is False
    assert calls[0] == 2
    # The replacement entry is unconditional, so no further calls.
    assert await problem.is_interesting(b"hi") is False
    assert calls[0] == 2


async def test_plain_bool_results_are_cached_unconditionally():
    calls = [0]

    async def is_interesting(x):
        calls[0] += 1
        return False

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert await problem.is_interesting(b"hi") is False
    assert await problem.is_interesting(b"hi") is False
    assert calls[0] == 1


# =============================================================================
# attempt_unstick tests
# =============================================================================


async def test_attempt_unstick_default_is_false():
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    assert await problem.attempt_unstick() is False


async def test_attempt_unstick_calls_callback():
    results = [True, False]

    async def unstick():
        return results.pop(0)

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        unstick=unstick,
    )
    assert await problem.attempt_unstick() is True
    assert await problem.attempt_unstick() is False
    assert results == []


async def test_view_attempt_unstick_delegates():
    unstick_calls = [0]

    async def unstick():
        unstick_calls[0] += 1
        return True

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        unstick=unstick,
    )
    view = View(problem=problem, parse=lambda x: x, dump=lambda x: x)
    assert await view.attempt_unstick() is True
    assert unstick_calls[0] == 1


# =============================================================================
# Concurrent is_interesting tests (parallelism > 1)
# =============================================================================


async def test_concurrent_interesting_candidates_settle_on_smallest():
    """Concurrent successful candidates must leave the problem on the
    sort-key-smallest one, with consistent statistics, regardless of the
    order in which their tests complete."""

    async def is_interesting(tc: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return True

    problem = BasicReductionProblem(
        initial=b"aaaaaaaa",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=4),
    )

    candidates = [b"aaaa", b"aa", b"aaaaaa", b"a"]
    results: list[bool] = []

    async with trio.open_nursery() as nursery:
        for candidate in candidates:

            async def check(candidate: bytes = candidate) -> None:
                results.append(await problem.is_interesting(candidate))

            nursery.start_soon(check)

    assert results == [True] * len(candidates)
    assert problem.current_test_case == b"a"

    # Every interesting call either was adopted as a reduction or was
    # counted as wasted, and failed_reductions tracks the rest.
    stats = problem.stats
    assert stats.calls == len(candidates)
    assert stats.interesting_calls == len(candidates)
    assert stats.reductions + stats.wasted_interesting_calls == len(candidates)
    assert stats.failed_reductions == stats.calls - stats.reductions
    assert stats.reductions >= 1
    assert stats.current_test_case_size == 1


async def test_concurrent_reductions_fire_callbacks_in_decreasing_order():
    """on_reduce callbacks fire once per adopted reduction, and adopted
    test cases get strictly smaller over time even under concurrency."""

    async def is_interesting(tc: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return True

    problem = BasicReductionProblem(
        initial=b"aaaaaaaaaa",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=4),
    )

    adopted: list[bytes] = []

    async def callback(tc: bytes) -> None:
        adopted.append(tc)

    problem.on_reduce(callback)

    candidates = [b"aaaaaaaa", b"aaaa", b"aaaaaa", b"aa", b"a", b"aaaaaaaaa"]
    async with trio.open_nursery() as nursery:
        for candidate in candidates:
            nursery.start_soon(problem.is_interesting, candidate)

    assert len(adopted) == problem.stats.reductions
    assert adopted[-1] == b"a"
    for before, after in zip(adopted, adopted[1:], strict=False):
        assert problem.sort_key(after) < problem.sort_key(before)


async def test_concurrent_uninteresting_candidates_are_cached():
    """A candidate tested while no reduction happens is served from cache
    on later calls, including after concurrent duplicate tests."""

    call_count = 0

    async def is_interesting(tc: bytes) -> bool:
        nonlocal call_count
        call_count += 1
        await trio.lowlevel.checkpoint()
        return False

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    async with trio.open_nursery() as nursery:
        nursery.start_soon(problem.is_interesting, b"x")
        nursery.start_soon(problem.is_interesting, b"y")

    calls_after_concurrent_phase = call_count
    # Later duplicate tests are answered from the cache.
    assert await problem.is_interesting(b"x") is False
    assert await problem.is_interesting(b"y") is False
    assert call_count == calls_after_concurrent_phase


# =============================================================================
# Nondeterminism handling
# =============================================================================


class Flaky:
    """A predicate that is interesting with a fixed probability per run,
    driven by a seeded RNG so tests are reproducible."""

    def __init__(self, rate, *, seed=0, condition=None, timeouts=()):
        self.rate = rate
        self.random = random.Random(seed)
        self.condition = condition or (lambda tc: True)
        self.calls = 0
        self.timeouts = set(timeouts)

    async def __call__(self, tc):
        await trio.lowlevel.checkpoint()
        self.calls += 1
        if self.calls in self.timeouts:
            return InterestingnessResult(interesting=False, timed_out=True)
        return self.condition(tc) and self.random.random() < self.rate


def policy(problem) -> NondeterminismPolicy:
    """The problem's nondeterminism policy, which these tests always set."""
    result = problem.policy
    assert result is not None
    return result


def nd_problem(is_interesting, *, initial=b"hello world", history=None, **kwargs):
    return BasicReductionProblem(
        initial=initial,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        policy=NondeterminismPolicy(),
        history=history,
        **kwargs,
    )


async def test_deterministic_setup_pays_the_detection_replays():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return True

    problem = nd_problem(is_interesting)
    await problem.setup()
    assert calls == DETECTION_REPLAYS
    assert not problem.nondeterministic
    assert problem.stats.calls == DETECTION_REPLAYS
    assert policy(problem).replay_calls == DETECTION_REPLAYS - 1


async def test_no_policy_means_no_detection():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return True

    problem = BasicReductionProblem(
        initial=b"hello", is_interesting=is_interesting, work=WorkContext()
    )
    await problem.setup()
    assert calls == 1
    assert not problem.nondeterministic
    assert problem.policy is None


async def test_a_missed_startup_replay_flips_and_confirms_the_incumbent():
    flaky = Flaky(0.5, seed=1)
    problem = nd_problem(flaky)
    await problem.setup()
    assert problem.nondeterministic
    # The confirmation batch extends to the seed size so the anchor is an
    # honest estimate rather than the stopping rule.
    assert policy(problem).anchor > 0.0
    assert flaky.calls >= ANCHOR_SEED_RUNS
    assert problem.current_test_case == b"hello world"


async def test_startup_detection_is_reported_once():
    flaky = Flaky(0.5, seed=1)
    problem = nd_problem(flaky)
    messages = []
    problem.work.report = lambda msg, level: messages.append(msg)
    await problem.setup()
    assert len([m for m in messages if "ondeterministic" in m]) == 1


async def test_timed_out_replays_are_not_detection_evidence():
    # Replays 2 and 3 time out; the fourth reproduces. A timeout says
    # nothing about determinism.
    flaky = Flaky(1.0, timeouts={2, 3})
    problem = nd_problem(flaky)
    await problem.setup()
    assert not problem.nondeterministic


async def test_invalid_initial_is_still_rejected_before_detection():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return False

    problem = nd_problem(is_interesting)
    with pytest.raises(InvalidInitialExample):
        await problem.setup()
    # A test case that never reproduces is only refused once it has
    # missed the confirmation bar's gate.
    assert calls == GATE_RUNS


async def test_initial_that_reproduces_rarely_is_found_within_the_gate():
    outcomes = iter([False] * (GATE_RUNS - 1) + [True] * 200)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    await problem.setup()
    assert problem.nondeterministic


async def test_initial_that_misses_once_but_reproduces_is_nondeterministic():
    # A flaky initial test case may fail its very first run; the detection
    # replays are what decide whether it is interesting at all.
    outcomes = iter([False, True, False, True] + [True] * 100)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    await problem.setup()
    assert problem.nondeterministic
    assert problem.current_test_case == b"hello world"
    assert policy(problem).anchor > 0.0


async def test_initial_that_only_times_out_is_invalid():
    async def is_interesting(tc):
        return InterestingnessResult(interesting=False, timed_out=True)

    problem = nd_problem(is_interesting)
    with pytest.raises(InvalidInitialExample):
        await problem.setup()


async def test_an_incumbent_that_never_reproduces_is_kept_with_a_warning():
    # Interesting exactly once (the initial run), then never again.
    calls = []

    async def is_interesting(tc):
        calls.append(tc)
        return len(calls) == 1

    problem = nd_problem(is_interesting)
    messages = []
    problem.work.report = lambda msg, level: messages.append(msg)
    await problem.setup()
    assert problem.nondeterministic
    assert problem.current_test_case == b"hello world"
    assert policy(problem).anchor < GAUNTLET_FLOOR
    assert any("rarely" in m for m in messages)


async def test_fast_reject_costs_one_run_and_retains_evidence():
    flaky = Flaky(0.5, seed=3)
    problem = nd_problem(flaky)
    await problem.setup()
    policy(problem).flip()
    calls_before = flaky.calls

    # Force the candidate's first run to miss.
    flaky.random = random.Random(0)
    flaky.rate = 0.0
    assert await problem.is_interesting(b"hello") is False
    assert flaky.calls == calls_before + 1
    # Not latched: a retry runs again and accumulates.
    assert await problem.is_interesting(b"hello") is False
    assert flaky.calls == calls_before + 2
    assert problem.ledger(b"hello").evidence.runs == 2
    assert problem.ledger(b"hello").verdict is None


async def test_gauntlet_accept_extends_to_the_seed_and_adopts():
    flaky = Flaky(1.0)
    problem = nd_problem(flaky)
    await problem.setup()
    policy(problem).flip()
    calls_before = flaky.calls
    assert await problem.is_interesting(b"hello") is True
    assert problem.current_test_case == b"hello"
    assert flaky.calls - calls_before == ANCHOR_SEED_RUNS
    ledger = problem.ledger(b"hello")
    assert ledger.verdict is True
    assert ledger.evidence.runs == ANCHOR_SEED_RUNS
    # The anchor is raised from the runs after the accept decision only.
    unselected = ANCHOR_SEED_RUNS - GAUNTLET_MIN_HITS
    assert ledger.accepted_at == Evidence(GAUNTLET_MIN_HITS, GAUNTLET_MIN_HITS)
    assert policy(problem).anchor == pytest.approx(
        Evidence(unselected, unselected).lower_bound()
    )
    # Latched: no further runs.
    assert await problem.is_interesting(b"hello") is True
    assert flaky.calls - calls_before == ANCHOR_SEED_RUNS


async def test_gauntlet_rejects_a_candidate_below_the_anchor():
    # First run interesting, then reliably not.
    outcomes = iter([True] + [False] * 100)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    policy(problem).flip()
    policy(problem).raise_anchor(Evidence(ANCHOR_SEED_RUNS, ANCHOR_SEED_RUNS))
    assert await problem.is_interesting(b"hello") is False
    assert problem.ledger(b"hello").verdict is False
    assert problem.current_test_case == b"hello world"
    # Latched reject: no more runs.
    assert await problem.is_interesting(b"hello") is False
    assert problem.ledger(b"hello").evidence.runs == 2


async def test_accepted_candidates_that_are_not_smaller_are_not_adopted():
    flaky = Flaky(1.0)
    problem = nd_problem(flaky)
    await problem.setup()
    policy(problem).flip()
    anchor = policy(problem).anchor
    assert await problem.is_interesting(b"hello world!!") is True
    assert problem.current_test_case == b"hello world"
    assert policy(problem).anchor == anchor
    assert problem.stats.wasted_interesting_calls == 1


async def test_the_anchor_never_falls():
    flaky = Flaky(1.0)
    problem = nd_problem(flaky)
    await problem.setup()
    policy(problem).flip()
    policy(problem).raise_anchor(Evidence(ANCHOR_SEED_RUNS, ANCHOR_SEED_RUNS))
    high = policy(problem).anchor
    # A candidate at 90% clears nothing at the high water, but even an
    # accepted one at a lower measured rate must not lower the anchor.
    outcomes = iter([True] * 19 + [False] + [True] * 100)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    policy(problem).flip()
    policy(problem).raise_anchor(Evidence(15, 20))
    low = policy(problem).anchor
    assert await problem.is_interesting(b"hello") is True
    assert policy(problem).anchor >= low
    assert policy(problem).anchor < high


async def test_confirming_mode_drives_a_missed_first_run_to_a_verdict():
    outcomes = iter([False] + [True] * 100)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    policy(problem).flip()
    policy(problem).confirming = True
    assert await problem.is_interesting(b"hello") is True
    assert problem.current_test_case == b"hello"
    ledger = problem.ledger(b"hello")
    assert ledger.verdict is True
    assert ledger.evidence.runs >= ANCHOR_SEED_RUNS


async def test_adoption_ends_the_confirmation_sweep():
    problem = nd_problem(Flaky(1.0))
    policy(problem).flip()
    policy(problem).confirming = True
    assert await problem.is_interesting(b"hello") is True
    assert not policy(problem).confirming


async def test_hit_minimum_is_pinned_per_candidate():
    problem = nd_problem(Flaky(1.0))
    policy(problem).flip()
    policy(problem).budget.min_hits = 6
    assert await problem.is_interesting(b"hello") is True
    assert problem.ledger(b"hello").min_hits == 6


async def test_pass_call_monitor_fires_once_per_verdict():
    problem = nd_problem(Flaky(1.0))
    policy(problem).flip()
    fired = []
    problem.pass_call_monitor = lambda: fired.append(1)
    await problem.is_interesting(b"hello")
    assert len(fired) == 1


async def test_periodic_verify_catches_late_nondeterminism():
    # Deterministic for the first VERIFY_INTERVAL calls, then the current
    # test case stops reproducing.
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        if calls <= DETECTION_REPLAYS:
            return True
        return tc != b"hello world"

    problem = nd_problem(is_interesting)
    await problem.setup()
    for i in range(VERIFY_INTERVAL):
        await problem.is_interesting(b"x" * 100 + bytes([i]))
    assert problem.nondeterministic


async def test_periodic_verify_ignores_timeouts():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        if calls > DETECTION_REPLAYS and tc == b"hello world":
            return InterestingnessResult(interesting=False, timed_out=True)
        return tc == b"hello world"

    problem = nd_problem(is_interesting)
    await problem.setup()
    for i in range(2 * VERIFY_INTERVAL):
        await problem.is_interesting(b"x" * 100 + bytes([i % 256, i // 256]))
    assert not problem.nondeterministic


async def test_unstick_replays_the_final_result_and_flips_on_a_miss():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return calls <= DETECTION_REPLAYS or calls % 2 == 0

    problem = nd_problem(is_interesting)
    await problem.setup()
    assert not problem.nondeterministic
    assert await problem.attempt_unstick() is True
    assert problem.nondeterministic


async def test_unstick_of_a_deterministic_result_delegates():
    unstick_calls = []

    async def unstick():
        unstick_calls.append(1)
        return False

    problem = nd_problem(Flaky(1.0), unstick=unstick)
    await problem.setup()
    assert await problem.attempt_unstick() is False
    assert unstick_calls == [1]
    assert not problem.nondeterministic


async def test_unstick_under_nondeterminism_runs_one_confirmation_sweep():
    unstick_calls = []

    async def unstick():
        unstick_calls.append(1)
        return False

    problem = nd_problem(Flaky(1.0), unstick=unstick)
    await problem.setup()
    policy(problem).flip()
    assert await problem.attempt_unstick() is True
    assert policy(problem).confirming
    assert unstick_calls == []
    # A confirmation sweep that changed nothing ends the sweep and falls
    # through to the other unstick hooks.
    assert await problem.attempt_unstick() is False
    assert not policy(problem).confirming
    assert unstick_calls == [1]


async def test_measure_current_reports_fresh_evidence():
    flaky = Flaky(1.0)
    problem = nd_problem(flaky)
    await problem.setup()
    before = flaky.calls
    evidence = await problem.measure_current(7)
    assert (evidence.interesting, evidence.runs) == (7, 7)
    assert flaky.calls == before + 7


async def test_backtrack_reverts_to_the_newest_reproducing_entry():
    # History: original reproduces always, the later entries never do.
    reproducing = {b"hello world", b"hello worl"}
    entries = [b"hello world", b"hello worl", b"hello wor", b"hello wo", b"hello w"]

    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        if calls <= DETECTION_REPLAYS:
            return True
        return tc in reproducing

    problem = nd_problem(is_interesting, history=entries)
    await problem.setup()
    # Walk the incumbent down to the last entry as a deterministic reducer would.
    reproducing.update(entries)
    for entry in entries[1:]:
        assert await problem.is_interesting(entry) is True
    assert problem.current_test_case == b"hello w"
    reproducing.clear()
    reproducing.update({b"hello world", b"hello worl"})
    reverted = []
    problem.on_reduce(lambda tc: _record(reverted, tc))

    assert await problem.attempt_unstick() is True
    assert problem.nondeterministic
    assert problem.current_test_case == b"hello worl"
    assert reverted == [b"hello worl"]
    assert policy(problem).anchor > 0.5


async def _record(into, tc):
    into.append(tc)


async def test_backtrack_with_no_history_reverts_to_the_original():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return calls <= DETECTION_REPLAYS + 1 or tc == b"hello world"

    problem = nd_problem(is_interesting)
    await problem.setup()
    assert await problem.is_interesting(b"hello") is True
    assert await problem.attempt_unstick() is True
    assert problem.current_test_case == b"hello world"


async def test_backtrack_keeps_the_incumbent_when_nothing_reproduces():
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return calls <= DETECTION_REPLAYS + 1

    problem = nd_problem(is_interesting, history=[b"hello world"])
    await problem.setup()
    assert await problem.is_interesting(b"hello") is True
    messages = []
    problem.work.report = lambda msg, level: messages.append(msg)
    assert await problem.attempt_unstick() is True
    assert problem.current_test_case == b"hello"
    assert any("rarely" in m for m in messages)


async def test_backtrack_scans_geometrically_then_refines():
    # Ten entries; only the first four reproduce once the run goes
    # nondeterministic. The scan must find entry three (the newest
    # reproducing one) without testing every entry.
    entries = [bytes([i]) * (20 - i) for i in range(10)]
    reproducing = set(entries)
    tested = []
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        if calls > DETECTION_REPLAYS:
            tested.append(tc)
        return tc in reproducing

    problem = nd_problem(is_interesting, initial=entries[0], history=entries)
    await problem.setup()
    for entry in entries[1:]:
        assert await problem.is_interesting(entry) is True
    assert problem.current_test_case == entries[9]
    del tested[:]
    reproducing.intersection_update(entries[:4])

    assert await problem.attempt_unstick() is True
    assert problem.current_test_case == entries[3]
    distinct = {tc for tc in tested if tc != entries[9]}
    assert entries[3] in distinct
    assert entries[4] in distinct
    assert entries[8] in distinct
    # The geometric scan skips most entries and the bisection only looks
    # between the last reject and the first accept.
    assert entries[0] not in distinct
    assert entries[2] not in distinct


async def test_backtrack_problem_delegates_nondeterminism():
    problem = nd_problem(Flaky(1.0))
    policy(problem).flip()
    inner = problem.backtrack(b"hello world again")
    assert inner.nondeterministic
    assert isinstance(inner, BasicReductionProblem)
    assert inner.policy is None


async def test_view_delegates_nondeterminism():
    problem = nd_problem(Flaky(1.0))
    view = View(problem=problem, parse=lambda x: list(x), dump=bytes)
    assert not view.nondeterministic
    policy(problem).flip()
    assert view.nondeterministic


async def test_stats_count_every_replay_as_a_call():
    flaky = Flaky(1.0)
    problem = nd_problem(flaky)
    await problem.setup()
    policy(problem).flip()
    await problem.is_interesting(b"hello")
    assert problem.stats.calls == flaky.calls
    assert problem.stats.reductions == 1


async def test_measure_current_records_misses():
    outcomes = iter([True] * DETECTION_REPLAYS + [False, True, False])

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    await problem.setup()
    evidence = await problem.measure_current(3)
    assert (evidence.interesting, evidence.runs) == (1, 3)


async def test_backtrack_falls_back_to_the_oldest_entry():
    # Six entries; only the original reproduces. The geometric scan skips
    # the oldest entry, so it must be tried separately.
    entries = [bytes([i]) * (20 - i) for i in range(6)]
    reproducing = set(entries)
    calls = 0

    async def is_interesting(tc):
        nonlocal calls
        calls += 1
        return tc in reproducing

    problem = nd_problem(is_interesting, initial=entries[0], history=entries)
    await problem.setup()
    for entry in entries[1:]:
        assert await problem.is_interesting(entry) is True
    reproducing.intersection_update(entries[:1])
    assert await problem.attempt_unstick() is True
    assert problem.current_test_case == entries[0]


def test_base_problem_is_deterministic_by_default():
    class Minimal(ReductionProblem[bytes]):
        def __init__(self):
            super().__init__(work=WorkContext())

        @property
        def current_test_case(self):
            return b""

        @property
        def stats(self):
            return ReductionStats()

        async def is_interesting(self, test_case):
            return True

        def sort_key(self, test_case):
            return test_case

        def size(self, test_case):
            return len(test_case)

        def display(self, value):
            return repr(value)

    assert Minimal().nondeterministic is False


async def test_replays_are_attributed_to_their_sites():
    # Startup detection misses once, the incumbent confirms (extended to
    # the seed size), a candidate then runs the gauntlet and is topped
    # up, and the report measures the result.
    outcomes = iter([True, False] + [True] * 500)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    await problem.setup()
    sites = policy(problem).replay_sites
    assert sites["detection"] == DETECTION_REPLAYS - 1
    assert sites["confirmation"] + sites["seed"] == ANCHOR_SEED_RUNS - DETECTION_REPLAYS
    before = dict(sites)
    assert await problem.is_interesting(b"hello") is True
    # The candidate's first run recruits it; the gauntlet reruns it until
    # its bound clears the anchor, and the seed top-up brings the ledger
    # to the seed size.
    assert sites["gauntlet"] >= GAUNTLET_MIN_HITS - 1
    assert 1 + sites["gauntlet"] + sites["seed"] - before["seed"] == ANCHOR_SEED_RUNS
    await problem.measure_current(3)
    assert sites["report"] == 3
    assert sum(sites.values()) == policy(problem).replay_calls


async def test_confirmation_sweep_calls_are_counted():
    problem = nd_problem(Flaky(1.0))
    await problem.setup()
    policy(problem).flip()
    assert await problem.attempt_unstick() is True
    assert problem.stats.confirmation_sweeps == 1
    calls_before = problem.stats.calls
    await problem.is_interesting(b"hello")
    assert problem.stats.confirmation_sweep_calls == problem.stats.calls - calls_before


async def test_anchor_is_raised_only_from_unselected_runs():
    # Four straight hits accept the candidate at the floor; the sixteen
    # runs after the decision all miss. The lucky start must not move
    # the anchor.
    outcomes = iter([True] * GAUNTLET_MIN_HITS + [False] * 100)

    async def is_interesting(tc):
        return next(outcomes)

    problem = nd_problem(is_interesting)
    policy(problem).flip()
    assert await problem.is_interesting(b"hello") is True
    assert problem.current_test_case == b"hello"
    assert policy(problem).anchor == 0.0
    assert policy(problem).anchor_attempts == 1
