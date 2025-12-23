"""Unit tests for problem module utilities and classes."""

import pytest

from shrinkray.problem import (
    BasicReductionProblem,
    InvalidInitialExample,
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


def test_default_sort_key_bytes():
    key = default_sort_key(b"hello")
    assert key == (5, b"hello")


def test_default_sort_key_string():
    key = default_sort_key("hello")
    assert key == (5, "hello")


def test_default_sort_key_list():
    # For non-string/bytes, uses repr
    key = default_sort_key([1, 2, 3])
    assert key[0] == len(repr([1, 2, 3]))


def test_default_sort_key_ordering():
    assert default_sort_key(b"a") < default_sort_key(b"ab")


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
    import time

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
    from shrinkray.passes.definitions import DumpError

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
        sort_key=lambda s: len(s),
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
        sort_key=lambda x: len(x),  # Size-based sorting
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=lambda s: len(s),
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
        sort_key=lambda x: len(x),
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=lambda s: len(s),
    )

    assert view.current_test_case == "hi"

    # Force underlying problem to have a larger value by directly manipulating
    # This simulates an edge case where the underlying changes but is larger
    problem._BasicReductionProblem__current = b"longer"

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
    from shrinkray.passes.definitions import Format

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
    from shrinkray.passes.definitions import Format

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
    from shrinkray.passes.definitions import Format

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
    from shrinkray.problem import ReductionProblem

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
    from shrinkray.problem import ReductionProblem

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
