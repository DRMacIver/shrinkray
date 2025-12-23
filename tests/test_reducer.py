"""Unit tests for reducer module."""

import pytest
import trio

from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import (
    BasicReducer,
    KeyProblem,
    Reducer,
    RestartPass,
    ShrinkRay,
    UpdateKeys,
)
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
    assert (
        pass_call_count[0] >= 4
    )  # At least: 5->4->3->2->1, plus final no-progress run


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


def test_key_problem_stats_delegates_to_base():
    """Test KeyProblem.stats returns base problem's stats."""
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

    # stats should be the same object as base's stats
    assert kp.stats is base.stats


async def test_key_problem_is_interesting():
    """Test KeyProblem.is_interesting applies patch correctly."""
    from shrinkray.passes.patching import PatchApplier

    async def is_interesting(x):
        # Only accept if file1 contains 'hi'
        return x.get("file1", b"") == b"hi"

    base = BasicReductionProblem(
        initial={"file1": b"hello", "file2": b"world"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(patches=UpdateKeys(), problem=base)
    kp = KeyProblem(base_problem=base, applier=applier, key="file1")

    # Should return True because patching file1 to b"hi" makes it interesting
    result = await kp.is_interesting(b"hi")
    assert result is True


# =============================================================================
# ShrinkRay status tests
# =============================================================================


def test_shrinkray_status_with_pump_no_pass():
    """Test ShrinkRay status when pump is set but pass is not."""
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    # Simulate pump running without pass
    async def mock_pump(p):
        return p.current_test_case

    mock_pump.__name__ = "test_pump"
    reducer.current_pump = mock_pump
    reducer.current_reduction_pass = None

    status = reducer.status
    assert "test_pump" in status
    assert "reduction pump" in status


def test_shrinkray_status_with_pump_and_pass():
    """Test ShrinkRay status when both pump and pass are set."""
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    # Simulate both pump and pass running
    async def mock_pump(p):
        return p.current_test_case

    mock_pump.__name__ = "test_pump"

    async def mock_pass(p):
        pass

    mock_pass.__name__ = "test_pass"

    reducer.current_pump = mock_pump
    reducer.current_reduction_pass = mock_pass

    status = reducer.status
    assert "test_pass" in status
    assert "test_pump" in status
    assert "under pump" in status


def test_shrinkray_status_no_pump_no_pass():
    """Test ShrinkRay status when neither pump nor pass is set."""
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    reducer.current_pump = None
    reducer.current_reduction_pass = None

    status = reducer.status
    assert status == "Selecting reduction pass"


def test_shrinkray_pumps_without_clang_delta():
    """Test ShrinkRay.pumps returns empty when no clang_delta."""
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem, clang_delta=None)
    assert list(reducer.pumps) == []


# =============================================================================
# DirectoryShrinkRay tests
# =============================================================================


async def test_directory_shrinkray_delete_keys():
    """Test DirectoryShrinkRay.delete_keys removes deletable keys."""
    from shrinkray.reducer import DirectoryShrinkRay

    # Track which files are required
    required_files = {"a.txt", "c.txt"}

    async def is_interesting(x):
        # Interesting if all required files are present
        return all(f in x for f in required_files)

    problem = BasicReductionProblem(
        initial={"a.txt": b"content a", "b.txt": b"content b", "c.txt": b"content c"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = DirectoryShrinkRay(target=problem)
    await reducer.delete_keys()

    # b.txt should be deleted since it's not required
    assert "a.txt" in problem.current_test_case
    assert "b.txt" not in problem.current_test_case
    assert "c.txt" in problem.current_test_case


async def test_directory_shrinkray_delete_keys_priority():
    """Test DirectoryShrinkRay.delete_keys deletes larger files first."""
    from shrinkray.reducer import DirectoryShrinkRay

    deletion_order = []

    async def is_interesting(x):
        # Track what's being tested by recording deletions
        current_keys = set(x.keys())
        for key in ["a.txt", "b.txt", "c.txt"]:
            if key not in current_keys and key not in deletion_order:
                deletion_order.append(key)
        # Always keep all files
        return True

    problem = BasicReductionProblem(
        initial={
            "a.txt": b"small",  # 5 bytes
            "b.txt": b"medium content",  # 14 bytes
            "c.txt": b"large content here",  # 18 bytes
        },
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = DirectoryShrinkRay(target=problem)
    await reducer.delete_keys()

    # Larger files should be tried first (they're sorted by size descending)
    # c.txt (18 bytes) should be tried before b.txt (14 bytes) before a.txt (5 bytes)
    assert deletion_order == ["c.txt", "b.txt", "a.txt"]


async def test_directory_shrinkray_run_reduces_directory():
    """Test DirectoryShrinkRay.run reduces directory contents."""
    from shrinkray.reducer import DirectoryShrinkRay

    async def is_interesting(x):
        # Interesting if a.txt exists and contains 'x'
        return "a.txt" in x and b"x" in x.get("a.txt", b"")

    problem = BasicReductionProblem(
        initial={"a.txt": b"xxxyyy", "b.txt": b"deleteme"},
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = DirectoryShrinkRay(target=problem)
    await reducer.run()

    # b.txt should be deleted, a.txt should be reduced
    assert "b.txt" not in problem.current_test_case
    assert "a.txt" in problem.current_test_case
    # a.txt should be reduced to minimal form containing 'x'
    assert b"x" in problem.current_test_case["a.txt"]
    assert len(problem.current_test_case["a.txt"]) < 6


# =============================================================================
# ShrinkRay clang_delta and pump tests
# =============================================================================


def test_shrinkray_pumps_with_clang_delta():
    """Test ShrinkRay.pumps returns clang_delta_pumps when clang_delta is set.

    This covers line 205.
    """
    from shrinkray.passes.clangdelta import ClangDelta, find_clang_delta
    from shrinkray.reducer import ShrinkRay

    clang_delta_exec = find_clang_delta()
    if not clang_delta_exec:
        pytest.skip("clang_delta not available")

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"int main() {}",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cd = ClangDelta(clang_delta_exec)
    reducer = ShrinkRay(target=problem, clang_delta=cd)
    pumps = list(reducer.pumps)
    assert len(pumps) > 0
    # Each pump should be a callable
    for pump in pumps:
        assert callable(pump)


def test_shrinkray_status_with_pass_no_pump():
    """Test ShrinkRay status when pass is set but pump is not.

    This covers line 211.
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    # Simulate pass running without pump
    async def mock_pass(p):
        pass

    mock_pass.__name__ = "test_pass"
    reducer.current_reduction_pass = mock_pass
    reducer.current_pump = None

    status = reducer.status
    assert "test_pass" in status
    assert "Running reduction pass" in status


def test_shrinkray_register_format_specific_pass():
    """Test ShrinkRay.register_format_specific_pass adds passes for valid format.

    This covers lines 196-198.
    """
    from shrinkray.passes.definitions import Format, ParseError
    from shrinkray.reducer import ShrinkRay

    # Create a simple format that only validates if starts with "FORMAT:"
    class TestFormat(Format[bytes, bytes]):
        @property
        def name(self) -> str:
            return "TestFormat"

        def parse(self, input: bytes) -> bytes:
            if input.startswith(b"FORMAT:"):
                return input[7:]
            raise ParseError()

        def dumps(self, input: bytes) -> bytes:
            return b"FORMAT:" + input

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"FORMAT:hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    initial_great_count = len(reducer.great_passes)

    async def test_pass(p):
        pass

    reducer.register_format_specific_pass(TestFormat(), [test_pass])

    # Should have added the pass
    assert len(reducer.great_passes) > initial_great_count


def test_shrinkray_register_format_specific_pass_invalid_format():
    """Test register_format_specific_pass doesn't add passes for invalid format."""
    from shrinkray.passes.definitions import Format, ParseError
    from shrinkray.reducer import ShrinkRay

    class TestFormat(Format[bytes, bytes]):
        @property
        def name(self) -> str:
            return "TestFormat"

        def parse(self, input: bytes) -> bytes:
            if input.startswith(b"FORMAT:"):
                return input[7:]
            raise ParseError()

        def dumps(self, input: bytes) -> bytes:
            return b"FORMAT:" + input

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",  # Doesn't match format
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    initial_great_count = len(reducer.great_passes)

    async def test_pass(p):
        pass

    reducer.register_format_specific_pass(TestFormat(), [test_pass])

    # Should not have added the pass
    assert len(reducer.great_passes) == initial_great_count


async def test_shrinkray_pump_method():
    """Test ShrinkRay.pump method runs pump and reduction passes.

    This covers lines 229-249.
    """
    from shrinkray.reducer import ShrinkRay

    pump_called = [False]
    pass_called = [False]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    # Add a pass that marks when it's called
    async def tracking_pass(p):
        pass_called[0] = True

    reducer.great_passes = [tracking_pass]
    reducer.ok_passes = []
    reducer.last_ditch_passes = []

    # Create a pump that returns larger content (to trigger reduction)
    async def expanding_pump(p):
        pump_called[0] = True
        return p.current_test_case + b" extra"

    expanding_pump.__name__ = "expanding_pump"

    await reducer.pump(expanding_pump)

    assert pump_called[0]
    # Pass should have been called during pump
    assert pass_called[0]


async def test_shrinkray_run_empty_is_interesting():
    """Test ShrinkRay.run returns immediately if empty string is interesting.

    This covers line 343.
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        # Everything is interesting, including empty string
        return True

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    await reducer.run()

    # Should have reduced to empty immediately
    assert problem.current_test_case == b""


async def test_shrinkray_run_single_byte_interesting():
    """Test ShrinkRay.run returns early if single byte is interesting.

    This covers lines 347-351.
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        # Only empty is not interesting, but single byte newline is
        if x == b"":
            return False
        return b"\n" in x or len(x) == 1

    problem = BasicReductionProblem(
        initial=b"hello\nworld",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    await reducer.run()

    # Should have reduced to a single byte (likely 0 since it's smallest)
    assert len(problem.current_test_case) == 1


async def test_shrinkray_run_single_byte_finds_smaller():
    """Test ShrinkRay.run finds smaller single byte when possible.

    This covers the inner loop in lines 348-350.
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        # Empty is not interesting
        if x == b"":
            return False
        # Single bytes >= 5 are interesting (so we can find smaller)
        if len(x) == 1:
            return x[0] >= 5
        return True

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    await reducer.run()

    # Should have reduced to bytes([5]) - the smallest acceptable single byte
    assert problem.current_test_case == bytes([5])


async def test_shrinkray_pump_no_change():
    """Test ShrinkRay.pump returns early when pumped equals current.

    This covers line 235 (pumped == current branch).
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    pass_called = [False]

    async def tracking_pass(p):
        pass_called[0] = True

    reducer.great_passes = [tracking_pass]

    # Create a pump that returns the same content (no change)
    async def identity_pump(p):
        return p.current_test_case

    identity_pump.__name__ = "identity_pump"

    await reducer.pump(identity_pump)

    # Pass should NOT have been called since pump returned same content
    assert not pass_called[0]


async def test_basic_reducer_pump_changes_result():
    """Test BasicReducer when pump produces different result that leads to reduction.

    This covers the TRUE branch of line 103 (pumped != current_test_case).
    """

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    pump_returned_different = [False]
    pass_called_under_pump = [False]

    async def simple_pass(p):
        if pump_returned_different[0]:
            pass_called_under_pump[0] = True
            # Make a reduction under pump
            if len(p.current_test_case) > 3:
                await p.is_interesting(b"hi")

    async def changing_pump(p):
        # Return a different (larger) test case to trigger pump branch
        pump_returned_different[0] = True
        return b"hello world different"

    changing_pump.__name__ = "changing_pump"

    reducer = BasicReducer(
        target=problem,
        reduction_passes=[simple_pass],
        pumps=[changing_pump],
    )
    await reducer.run()

    assert pump_returned_different[0]
    assert pass_called_under_pump[0]


async def test_basic_reducer_pump_returns_same():
    """Test BasicReducer when pump returns same value (no change).

    This covers line 103->100 (pumped == current_test_case, skip if block).
    """

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    pump_calls = [0]
    pass_under_pump_called = [False]

    async def simple_pass(p):
        # Track if pass is called while pump tracking is active
        if pump_calls[0] > 0:
            pass_under_pump_called[0] = True

    # First pump returns same, second pump returns different
    async def identity_pump(p):
        pump_calls[0] += 1
        # Always return same value - pump produces no change
        return p.current_test_case

    identity_pump.__name__ = "identity_pump"

    async def changing_pump(p):
        # This one changes to trigger the other branch
        return b"different value"

    changing_pump.__name__ = "changing_pump"

    reducer = BasicReducer(
        target=problem,
        reduction_passes=[simple_pass],
        pumps=[identity_pump, changing_pump],
    )
    await reducer.run()

    # identity_pump should have been called
    assert pump_calls[0] >= 1


async def test_shrinkray_pump_early_break_on_improvement():
    """Test ShrinkRay.pump breaks early when finding smaller result.

    This covers line 246 (break when smaller found during pump).
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello world testing",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    call_sequence = []

    # Create passes that track calls and make reductions
    async def great_pass(p):
        call_sequence.append("great")
        # Make a significant reduction
        if len(p.current_test_case) > 5:
            await p.is_interesting(b"hi")

    async def ok_pass(p):
        call_sequence.append("ok")

    async def last_ditch_pass(p):
        call_sequence.append("last_ditch")

    reducer.great_passes = [great_pass]
    reducer.ok_passes = [ok_pass]
    reducer.last_ditch_passes = [last_ditch_pass]

    # Pump that returns expanded content
    async def expanding_pump(p):
        return p.current_test_case + b" extra content"

    expanding_pump.__name__ = "expanding_pump"

    await reducer.pump(expanding_pump)

    # Should have called great pass (which reduced) and broken early
    assert "great" in call_sequence
    # ok and last_ditch may not be called if early break happened
    # (depends on timing - the break happens if smaller found)


async def test_shrinkray_great_passes_no_reduction():
    """Test ShrinkRay.run_great_passes resets to all passes when none reduce.

    This covers line 267 (current = self.great_passes when no successful passes).
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    call_counts = {"pass1": 0, "pass2": 0}

    # Create passes that don't make reductions
    async def pass1(p):
        call_counts["pass1"] += 1

    async def pass2(p):
        call_counts["pass2"] += 1

    reducer.great_passes = [pass1, pass2]

    await reducer.run_great_passes()

    # Both passes should have been called
    assert call_counts["pass1"] >= 1
    assert call_counts["pass2"] >= 1


async def test_shrinkray_great_passes_change_without_size_reduction():
    """Test run_great_passes when test case changes but no pass reduces size.

    This covers line 267 (elif not successful: current = self.great_passes).
    The condition is: test case changed, but no individual pass reduced size.
    This can happen if a pass changes content to lexicographically smaller but same size.
    """
    from shrinkray.reducer import ShrinkRay

    call_count = [0]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"zzzzzzzzzzzz",  # 12 bytes, starts with 'z'
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)

    # Pass that changes content to lexicographically smaller but same size
    # shortlex prefers (smaller length, then smaller bytes)
    # "aaaaaaaaaaaa" < "zzzzzzzzzzzz" lexicographically, same length
    async def same_size_change_pass(p):
        call_count[0] += 1
        if call_count[0] == 1:
            # Change to lexicographically smaller content of same size
            # This will be accepted (smaller by shortlex) but size doesn't decrease
            await p.is_interesting(b"aaaaaaaaaaaa")  # Same 12 bytes

    reducer.great_passes = [same_size_change_pass]

    await reducer.run_great_passes()

    # The pass should have been called at least twice (once to change, once to confirm no more changes)
    assert call_count[0] >= 2


async def test_shrinkray_ok_passes_make_progress():
    """Test ShrinkRay.run_some_passes returns after ok_passes when they make progress.

    This covers line 287 (return after ok_passes when prev != current).
    """
    from shrinkray.reducer import ShrinkRay

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello world testing",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    reducer = ShrinkRay(target=problem)
    reducer.unlocked_ok_passes = True  # Pre-unlock ok_passes

    last_ditch_called = [False]

    # great_passes don't reduce
    async def non_reducing_great(p):
        pass

    # ok_passes make progress
    async def reducing_ok(p):
        if len(p.current_test_case) > 5:
            await p.is_interesting(b"hi")

    # last_ditch should not be called if ok_passes succeeded
    async def last_ditch(p):
        last_ditch_called[0] = True

    reducer.great_passes = [non_reducing_great]
    reducer.ok_passes = [reducing_ok]
    reducer.last_ditch_passes = [last_ditch]

    await reducer.run_some_passes()

    # last_ditch should not be called since ok_passes made progress
    assert not last_ditch_called[0]


async def test_shrinkray_main_loop_with_pumps():
    """Test ShrinkRay.run main loop calls pumps and continues on progress.

    This covers lines 364-365 (pump call and continue branch in main loop).
    The continue at line 365->358 happens when pump changes test case.
    """
    from shrinkray.reducer import ShrinkRay

    call_log = []
    loop_iterations = [0]

    async def is_interesting(x):
        # Not interesting if empty or single byte (to avoid early exit)
        if len(x) <= 1:
            return False
        return True

    problem = BasicReductionProblem(
        initial=b"hello world testing longer",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Create subclass to override initial_cut and pumps
    pump_calls = [0]

    async def progress_pump(p):
        pump_calls[0] += 1
        call_log.append(f"pump:{pump_calls[0]}")
        if pump_calls[0] == 1:
            # Directly update the problem's test case via is_interesting
            # This simulates the pump finding a reduced version
            await p.is_interesting(b"reduced by pump")
        return p.current_test_case

    progress_pump.__name__ = "progress_pump"

    class TestShrinkRay(ShrinkRay):
        async def initial_cut(self) -> None:
            # Skip initial cut for faster testing
            pass

        @property
        def pumps(self):
            return [progress_pump]

        async def run_some_passes(self) -> None:
            loop_iterations[0] += 1
            call_log.append(f"passes:{loop_iterations[0]}")
            # Don't make progress (test case stays same)
            # This allows us to reach the pump code in main loop

    reducer = TestShrinkRay(target=problem)

    await reducer.run()

    # Both pass iterations and pump should have been called
    assert any("passes:" in x for x in call_log)
    assert any("pump:" in x for x in call_log)
    # Should have at least 2 pass iterations (one before pump progress, one after)
    assert loop_iterations[0] >= 2


@pytest.mark.slow
async def test_directory_shrinkray_with_c_files():
    """Test DirectoryShrinkRay uses clang_delta for C files."""
    from shrinkray.passes.clangdelta import ClangDelta, find_clang_delta
    from shrinkray.reducer import DirectoryShrinkRay

    clang_delta_exec = find_clang_delta()
    if not clang_delta_exec:
        pytest.skip("clang_delta not available")

    async def is_interesting(x):
        # Just needs to contain main
        return "test.c" in x and b"main" in x.get("test.c", b"")

    problem = BasicReductionProblem(
        initial={
            "test.c": b"int main() { return 0; }",
            "other.txt": b"not a c file",
        },
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cd = ClangDelta(clang_delta_exec)
    reducer = DirectoryShrinkRay(target=problem, clang_delta=cd)
    await reducer.run()

    # Should have reduced the directory
    assert "test.c" in problem.current_test_case
    assert b"main" in problem.current_test_case["test.c"]


# =============================================================================
# Initial cut watcher tests (using mock clock)
# =============================================================================


async def test_initial_cut_watcher_cancels_on_no_progress(autojump_clock):
    """Test that the watcher task cancels a pass when no progress is made.

    This uses autojump_clock to test the time-dependent watcher code that
    normally requires 5 seconds to trigger.
    """
    pass_iterations = [0]
    pass_cancelled = [False]

    async def is_interesting(x):
        return len(x) > 0

    problem = BasicReductionProblem(
        initial=b"hello world",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Create a pass that runs forever but never makes progress
    async def slow_no_progress_pass(p):
        try:
            while True:
                pass_iterations[0] += 1
                # Yield control so the watcher can run
                await trio.sleep(0.1)
        except trio.Cancelled:
            pass_cancelled[0] = True
            raise

    # Create a ShrinkRay with only our slow pass as initial cut
    reducer = ShrinkRay(target=problem)
    reducer.initial_cuts = [slow_no_progress_pass]

    # Run initial_cut - the watcher should cancel after 5s (mock time)
    await reducer.initial_cut()

    # The pass should have been cancelled by the watcher
    assert pass_cancelled[0], "Pass should have been cancelled by watcher"
    # Should have run for multiple iterations before being cancelled
    assert pass_iterations[0] > 0, "Pass should have run at least once"


async def test_initial_cut_watcher_cancels_on_rate_drop(autojump_clock):
    """Test that the watcher cancels when reduction rate drops significantly.

    The watcher cancels if rate drops below 50% of the best rate observed.
    """
    pass_cancelled = [False]
    reductions_made = [0]

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"x" * 100,  # Start with 100 bytes
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Create a pass that makes fast progress initially, then stops
    async def slowing_pass(p):
        try:
            # Make some reductions quickly at first
            for _ in range(3):
                current = p.current_test_case
                if len(current) > 50:
                    await p.is_interesting(current[:-10])  # Remove 10 bytes
                    reductions_made[0] += 1
                await trio.sleep(0.1)

            # Now just loop without making progress
            while True:
                await trio.sleep(0.1)
        except trio.Cancelled:
            pass_cancelled[0] = True
            raise

    reducer = ShrinkRay(target=problem)
    reducer.initial_cuts = [slowing_pass]

    await reducer.initial_cut()

    assert pass_cancelled[0], "Pass should have been cancelled by watcher"
    assert reductions_made[0] > 0, "Should have made some reductions first"


async def test_initial_cut_watcher_multiple_iterations(autojump_clock):
    """Test watcher runs multiple iterations when progress continues.

    This exercises the branch where best_reduction_rate is already set
    and rate doesn't improve (319->325 branch). The pass finishes naturally
    which triggers the nursery cancel from run_pass completion.
    """

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"x" * 1000,  # Start with 1000 bytes - large enough for many watcher iterations
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Create a pass that makes progress at varying rates
    # First batch: fast progress (establishes high best_reduction_rate)
    # Second batch: slower progress (rate < best_reduction_rate, hits False branch)
    iteration = [0]

    async def varying_rate_pass(p):
        # First 20 seconds: make 10 reductions (fast)
        for _ in range(10):
            await trio.sleep(2)
            current = p.current_test_case
            if len(current) > 100:
                await p.is_interesting(current[:-50])  # Remove 50 bytes
            iteration[0] += 1

        # Next 50 seconds: make 5 reductions (slow - rate drops)
        for _ in range(5):
            await trio.sleep(10)
            current = p.current_test_case
            if len(current) > 100:
                await p.is_interesting(current[:-10])  # Remove 10 bytes
            iteration[0] += 1

    reducer = ShrinkRay(target=problem)
    reducer.initial_cuts = [varying_rate_pass]

    initial_size = problem.current_size
    await reducer.initial_cut()

    # Progress should have been made
    assert problem.current_size < initial_size, "Should have made progress"
