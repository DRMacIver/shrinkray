"""Unit tests for problem module utilities and classes."""

import pytest

from shrinkray.problem import (
    BasicReductionProblem,
    InvalidInitialExample,
    ReductionStats,
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
    stats = ReductionStats()
    stats.initial_test_case_size = 1000
    stats.current_test_case_size = 500
    stats.reductions = 5
    stats.calls = 10
    stats.interesting_calls = 5
    stats.wasted_interesting_calls = 1
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
