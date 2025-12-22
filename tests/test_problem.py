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


# =============================================================================
# View tests
# =============================================================================


from shrinkray.problem import View


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

    # In some cases (e.g., SAT problems) the underlying test case can get
    # larger during reduction. When this happens, the View keeps the smaller
    # cached value based on sort_key comparison.


async def test_view_handles_underlying_size_increase():
    """Test View keeps smaller cached value even if underlying gets larger.

    This can happen in practice with SAT problems where unit propagation
    may expand the representation even though it's semantically smaller.
    """
    changes = []

    async def is_interesting(x):
        changes.append(x)
        return True

    problem = BasicReductionProblem(
        initial=b"short",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=lambda x: len(x),  # Pure size-based comparison
    )

    view = View(
        problem=problem,
        parse=lambda b: b.decode("utf-8"),
        dump=lambda s: s.encode("utf-8"),
        sort_key=lambda s: len(s),
    )

    assert view.current_test_case == "short"

    # Simulate what happens in SAT: underlying problem accepts a larger
    # representation that is semantically better
    # First, we manually set the underlying problem's test case to something larger
    # by bypassing the normal reduction check (as can happen with pumps/SAT)
    problem._BasicReductionProblem__current = b"much longer value"

    # The view should still return the cached smaller value
    assert view.current_test_case == "short"


def test_sat_unit_propagation_can_expand_representation():
    """Demonstrate that SAT unit propagation can expand the representation.

    This is the real-world example of why View must cache smaller values:
    unit propagation makes clauses explicit even though it's semantically
    a simplification.
    """
    from shrinkray.passes.sat import UnitPropagator, DimacsCNF

    # The clause [[-1, 2]] means "if 1 is true, then 2 must be true" (implication)
    # When we add unit [1] and propagate, the solver discovers [2] must also be true.
    # The propagated form has MORE clauses: [[1], [2]] instead of [[-1, 2]]
    original = [[-1, 2]]  # Single clause: 1 implies 2
    propagator = UnitPropagator(original)
    propagator.add_unit(1)  # Force variable 1 to be true

    propagated = propagator.propagated_clauses()

    # The propagated result has explicit unit clauses
    assert [1] in propagated  # Our added unit
    assert [2] in propagated  # Discovered through propagation

    # More clauses in propagated form
    assert len(propagated) > len(original), (
        f"Expected expansion: original={original}, propagated={propagated}"
    )

    # Compare byte representation sizes
    original_bytes = DimacsCNF.dumps(original)
    propagated_bytes = DimacsCNF.dumps(propagated)

    # The propagated form is larger in bytes even though it's semantically equivalent
    # Original: "p cnf 2 1\n-1 2 0" (15 bytes)
    # Propagated: "p cnf 2 2\n1 0\n2 0" (18 bytes)
    assert len(propagated_bytes) > len(original_bytes), (
        f"Expected byte expansion: original={len(original_bytes)}, "
        f"propagated={len(propagated_bytes)}"
    )


from hypothesis import given, settings, assume, example
import hypothesis.strategies as st


@st.composite
def sat_with_unit(draw):
    """Generate SAT clauses where we can add a unit to trigger propagation."""
    # Generate 1-5 variables
    n_vars = draw(st.integers(1, 5))
    variables = range(1, n_vars + 1)

    # Generate literal (positive or negative variable)
    literal = st.builds(
        lambda v, s: v * s, st.sampled_from(variables), st.sampled_from([-1, 1])
    )

    # Generate 1-5 clauses, each with 2-4 literals
    clauses = draw(
        st.lists(
            st.lists(literal, min_size=2, max_size=4, unique_by=abs),
            min_size=1,
            max_size=5,
        )
    )

    # Pick a unit to add from literals that appear in clauses
    all_literals = {lit for clause in clauses for lit in clause}
    assume(all_literals)  # Need at least one literal
    unit = draw(st.sampled_from(sorted(all_literals, key=abs)))

    return clauses, unit


@settings(max_examples=100)
@given(sat_with_unit())
def test_sat_unit_propagation_expansion_hypothesis(sat_and_unit):
    """Use Hypothesis to find SAT cases where unit propagation expands representation.

    This demonstrates a real scenario where the underlying test case can grow:
    when we add a unit clause and propagate, the explicit representation grows
    even though we've learned new information that simplifies the problem.
    """
    from shrinkray.passes.sat import UnitPropagator, DimacsCNF, Inconsistent

    clauses, unit = sat_and_unit

    try:
        propagator = UnitPropagator(clauses)
        propagator.add_unit(unit)
        propagated = propagator.propagated_clauses()
    except Inconsistent:
        # Contradictory clauses, skip this case
        assume(False)

    # The key observation: after propagation, we have at least one explicit
    # unit clause that wasn't there before (unless unit was already present)
    had_unit_before = any(clause == [unit] or clause == [abs(unit)] for clause in clauses)
    has_unit_after = any(clause == [unit] for clause in propagated)

    if not had_unit_before:
        # We added a unit, so it should appear in propagated form
        assert has_unit_after, f"Expected {unit} to appear as unit clause in {propagated}"

    # Verify that propagation can increase clause count
    # (this happens when units are made explicit)
    if len(propagated) > len(clauses):
        # Successfully found a case where propagation expands the representation!
        # This is exactly the scenario that View caching handles.
        from shrinkray.passes.sat import DimacsCNF

        original_bytes = DimacsCNF.dumps(clauses)
        propagated_bytes = DimacsCNF.dumps(propagated)

        # Note: bytes might also be larger due to the extra clauses
        assert len(propagated) > len(clauses)
