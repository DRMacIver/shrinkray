"""Unit tests for reducer module."""

import pytest

from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import BasicReducer, Reducer, RestartPass, UpdateKeys, KeyProblem
from shrinkray.work import WorkContext


# =============================================================================
# Reducer base class tests
# =============================================================================


def test_reducer_backtrack_context():
    """Test Reducer.backtrack temporarily changes target."""
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def dummy_pass(p):
        pass

    reducer = BasicReducer(target=problem, reduction_passes=[dummy_pass])

    assert reducer.target.current_test_case == b"hello"

    with reducer.backtrack(b"world"):
        assert reducer.target.current_test_case == b"world"

    assert reducer.target.current_test_case == b"hello"


def test_reducer_status_default():
    """Test Reducer.status default is empty string."""
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Create a concrete subclass for testing
    class TestReducer(Reducer[bytes]):
        async def run(self):
            pass

    reducer = TestReducer(target=problem)
    assert reducer.status == ""


# =============================================================================
# BasicReducer tests
# =============================================================================


async def test_basic_reducer_runs_passes():
    """Test BasicReducer runs all passes."""
    call_log = []

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def pass1(p):
        call_log.append("pass1")

    async def pass2(p):
        call_log.append("pass2")

    reducer = BasicReducer(target=problem, reduction_passes=[pass1, pass2])
    await reducer.run()

    assert "pass1" in call_log
    assert "pass2" in call_log


async def test_basic_reducer_stops_when_no_progress():
    """Test BasicReducer stops when no reduction is made."""
    call_count = [0]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def counting_pass(p):
        call_count[0] += 1
        # Never makes progress

    reducer = BasicReducer(target=problem, reduction_passes=[counting_pass])
    await reducer.run()

    # Should run exactly once and then stop since no progress was made
    assert call_count[0] == 1


async def test_basic_reducer_loops_on_progress():
    """Test BasicReducer loops when progress is made."""
    pass_call_count = [0]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def reducing_pass(p):
        pass_call_count[0] += 1
        # Make a reduction - each time make it smaller
        current_size = len(p.current_test_case)
        if current_size > 1:
            await p.is_interesting(b"x" * (current_size - 1))

    reducer = BasicReducer(target=problem, reduction_passes=[reducing_pass])
    await reducer.run()

    # Should have reduced from 5 chars down to 1
    assert problem.current_test_case == b"x"
    # Pass should have been called multiple times as progress was made
    assert pass_call_count[0] >= 4  # At least: 5->4->3->2->1, plus final no-progress run


async def test_basic_reducer_status_updates():
    """Test BasicReducer updates status during run."""
    statuses_seen = []

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def named_pass(p):
        statuses_seen.append(reducer.status)

    named_pass.__name__ = "my_special_pass"

    reducer = BasicReducer(target=problem, reduction_passes=[named_pass])
    await reducer.run()

    assert any("my_special_pass" in s for s in statuses_seen)


async def test_basic_reducer_with_pumps():
    """Test BasicReducer runs pumps correctly."""
    pump_called = [False]
    pass_called_under_pump = [False]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def simple_pass(p):
        if pump_called[0]:
            pass_called_under_pump[0] = True

    async def simple_pump(p):
        pump_called[0] = True
        # Return a different (larger) test case to trigger pump reduction
        return b"hello world"

    simple_pump.__name__ = "simple_pump"

    reducer = BasicReducer(
        target=problem,
        reduction_passes=[simple_pass],
        pumps=[simple_pump],
    )
    await reducer.run()

    assert pump_called[0]
    assert pass_called_under_pump[0]


async def test_basic_reducer_pump_backtrack():
    """Test BasicReducer properly backtracks after pump."""
    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    async def simple_pass(p):
        pass

    async def expanding_pump(p):
        # Return an expanded version
        return b"hello world expanded"

    expanding_pump.__name__ = "expanding_pump"

    reducer = BasicReducer(
        target=problem,
        reduction_passes=[simple_pass],
        pumps=[expanding_pump],
    )
    await reducer.run()

    # Original problem should still be at original value (no reduction made)
    # since pump's backtrack ensures we only update if reduction is better
    assert len(problem.current_test_case) <= len(b"hello")


# =============================================================================
# RestartPass exception tests
# =============================================================================


def test_restart_pass_is_exception():
    """Test RestartPass can be raised and caught."""
    with pytest.raises(RestartPass):
        raise RestartPass()


# =============================================================================
# UpdateKeys Patches tests
# =============================================================================


def test_update_keys_empty():
    """Test UpdateKeys.empty returns empty dict."""
    uk = UpdateKeys()
    assert uk.empty == {}


def test_update_keys_combine():
    """Test UpdateKeys.combine merges dicts."""
    uk = UpdateKeys()
    result = uk.combine({"a": b"1"}, {"b": b"2"})
    assert result == {"a": b"1", "b": b"2"}


def test_update_keys_combine_overlap():
    """Test UpdateKeys.combine with overlapping keys takes later."""
    uk = UpdateKeys()
    result = uk.combine({"a": b"1"}, {"a": b"2"})
    assert result == {"a": b"2"}


def test_update_keys_apply():
    """Test UpdateKeys.apply updates target dict."""
    uk = UpdateKeys()
    target = {"a": b"1", "b": b"2"}
    result = uk.apply({"a": b"changed"}, target)
    assert result == {"a": b"changed", "b": b"2"}


def test_update_keys_apply_new_key():
    """Test UpdateKeys.apply can add new keys."""
    uk = UpdateKeys()
    target = {"a": b"1"}
    result = uk.apply({"b": b"2"}, target)
    assert result == {"a": b"1", "b": b"2"}


def test_update_keys_size():
    """Test UpdateKeys.size returns number of keys."""
    uk = UpdateKeys()
    assert uk.size({}) == 0
    assert uk.size({"a": b"1"}) == 1
    assert uk.size({"a": b"1", "b": b"2"}) == 2


# =============================================================================
# KeyProblem tests
# =============================================================================


async def test_key_problem_current_test_case():
    """Test KeyProblem.current_test_case returns value for key."""
    from shrinkray.passes.patching import PatchApplier

    async def is_interesting(x):
        return True

    base = BasicReductionProblem(
        initial={"file1": b"content1", "file2": b"content2"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(patches=UpdateKeys(), problem=base)
    kp = KeyProblem(base_problem=base, applier=applier, key="file1")

    assert kp.current_test_case == b"content1"


def test_key_problem_size():
    """Test KeyProblem.size returns byte length."""
    from shrinkray.passes.patching import PatchApplier

    async def is_interesting(x):
        return True

    base = BasicReductionProblem(
        initial={"file1": b"hello"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(patches=UpdateKeys(), problem=base)
    kp = KeyProblem(base_problem=base, applier=applier, key="file1")

    assert kp.size(b"hello") == 5
    assert kp.size(b"hi") == 2


def test_key_problem_sort_key():
    """Test KeyProblem.sort_key uses shortlex."""
    from shrinkray.passes.patching import PatchApplier
    from shrinkray.problem import shortlex

    async def is_interesting(x):
        return True

    base = BasicReductionProblem(
        initial={"file1": b"hello"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(patches=UpdateKeys(), problem=base)
    kp = KeyProblem(base_problem=base, applier=applier, key="file1")

    assert kp.sort_key(b"hello") == shortlex(b"hello")


def test_key_problem_display():
    """Test KeyProblem.display returns repr."""
    from shrinkray.passes.patching import PatchApplier

    async def is_interesting(x):
        return True

    base = BasicReductionProblem(
        initial={"file1": b"hello"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(patches=UpdateKeys(), problem=base)
    kp = KeyProblem(base_problem=base, applier=applier, key="file1")

    assert kp.display(b"hello") == "b'hello'"
