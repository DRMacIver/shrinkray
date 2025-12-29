"""Tests for state management."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import trio

from shrinkray.cli import InputType
from shrinkray.problem import InvalidInitialExample, shortlex
from shrinkray.state import (
    DYNAMIC_TIMEOUT_MIN,
    OutputCaptureManager,
    ShrinkRayDirectoryState,
    ShrinkRayStateSingleFile,
    TimeoutExceededOnInitial,
    sort_key_for_initial,
)
from shrinkray.work import Volume


# === TimeoutExceededOnInitial tests ===


def test_timeout_exceeded_stores_runtime_and_timeout():
    exc = TimeoutExceededOnInitial(runtime=5.5, timeout=2.0)
    assert exc.runtime == 5.5
    assert exc.timeout == 2.0


def test_timeout_exceeded_message_includes_timeout():
    exc = TimeoutExceededOnInitial(runtime=5.5, timeout=2.0)
    assert "2.0s" in str(exc)
    assert "timeout" in str(exc).lower()


# === ShrinkRayStateSingleFile tests ===


@pytest.fixture
def simple_state(tmp_path):
    """Create a simple state for testing."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello world")

    return ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )


def test_single_file_state_creates_reducer(simple_state):
    reducer = simple_state.reducer
    assert reducer is not None


def test_single_file_state_reducer_is_cached(simple_state):
    reducer1 = simple_state.reducer
    reducer2 = simple_state.reducer
    assert reducer1 is reducer2


def test_single_file_state_problem_property(simple_state):
    problem = simple_state.problem
    assert problem is not None
    assert problem.current_test_case == b"hello world"


async def test_single_file_state_parallel_tasks_tracking(simple_state):
    # Before any calls
    assert simple_state.parallel_tasks_running == 0


async def test_single_file_state_write_test_case(tmp_path, simple_state):
    target = tmp_path / "output.txt"
    await simple_state.write_test_case_to_file(str(target), b"test data")
    assert target.read_bytes() == b"test data"


async def test_single_file_state_format_data_with_none_formatter(simple_state):
    # With formatter="none", format_data should return the input unchanged
    result = await simple_state.format_data(b"test data")
    assert result == b"test data"


async def test_single_file_state_run_formatter_command(simple_state):
    # Test running a simple formatter command
    result = await simple_state.run_formatter_command(["cat"], b"hello")
    assert result.stdout == b"hello"
    assert result.returncode == 0


# === ShrinkRayDirectoryState tests ===


@pytest.fixture
def directory_state(tmp_path):
    """Create a directory state for testing."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "target"
    target.mkdir()
    (target / "a.txt").write_text("file a")
    (target / "b.txt").write_text("file b")

    return ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="target",
        parallelism=1,
        initial={"a.txt": b"file a", "b.txt": b"file b"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )


def test_directory_state_creates_reducer(directory_state):
    reducer = directory_state.reducer
    assert reducer is not None


def test_directory_state_extra_problem_kwargs(directory_state):
    kwargs = directory_state.extra_problem_kwargs
    assert "size" in kwargs


def test_directory_state_size_function(directory_state):
    kwargs = directory_state.extra_problem_kwargs
    size_fn = kwargs["size"]
    test_case = {"a.txt": b"hello", "b.txt": b"world!"}
    assert size_fn(test_case) == 11  # 5 + 6


def test_directory_state_sort_key_from_initial(directory_state):
    """Sort key is derived from initial test case via sort_key_for_initial."""
    sort_key_fn = sort_key_for_initial(directory_state.initial)

    tc1 = {"a.txt": b"hi"}
    tc2 = {"a.txt": b"hello"}
    tc3 = {"a.txt": b"hi", "b.txt": b"x"}

    # Smaller total size should come first
    assert sort_key_fn(tc1) < sort_key_fn(tc2)
    # Fewer total bytes wins even with more files
    assert sort_key_fn(tc1) < sort_key_fn(tc3)


def test_sort_key_for_initial_binary_data():
    """sort_key_for_initial returns shortlex for binary (non-text) data."""
    # Binary data that's not valid text
    binary_data = bytes([0x80, 0x81, 0x82])
    sort_key_fn = sort_key_for_initial(binary_data)

    # Should be shortlex for binary data
    assert sort_key_fn is shortlex


def test_sort_key_for_initial_text_with_decode_error():
    """sort_key_for_initial handles UnicodeDecodeError during comparison."""
    # UTF-8 encoded text
    text_data = b"hello"
    sort_key_fn = sort_key_for_initial(text_data)

    # Valid UTF-8 gets natural key
    valid_result = sort_key_fn(b"hello")
    assert valid_result[0] == 0  # Prefix 0 for successful decode

    # Invalid UTF-8 gets shortlex fallback
    invalid_data = bytes([0x80, 0x81, 0x82])
    invalid_result = sort_key_fn(invalid_data)
    assert invalid_result[0] == 1  # Prefix 1 for failed decode


def test_sort_key_for_initial_dict_missing_key():
    """sort_key_for_initial handles missing keys in dict comparisons."""
    # Initial has keys a.txt and b.txt
    initial = {"a.txt": b"file a", "b.txt": b"file b"}
    sort_key_fn = sort_key_for_initial(initial)

    # To test the missing key branch, we need two dicts with:
    # - Same total size (so dict_total_size is equal)
    # - Same number of keys (so len is equal)
    # - One has a key the other doesn't, forcing key_sort_key to compare
    tc1 = {"a.txt": b"xxxxx", "b.txt": b"yyyyy"}  # 10 bytes, 2 keys
    tc2 = {"a.txt": b"xxxxxxxxxx", "c.txt": b""}  # 10 bytes, 2 keys, but b.txt missing

    result1 = sort_key_fn(tc1)
    result2 = sort_key_fn(tc2)

    # tc2 is missing b.txt (which initial has), so when comparing key_sort_key
    # for b.txt, tc2 returns (0,) for missing, tc1 returns (1, ...) for present.
    # (0,) < (1, ...) so tc2 should come first
    assert result2 < result1


async def test_directory_state_write_creates_directory(tmp_path, directory_state):
    target = tmp_path / "output_dir"
    test_case = {"sub/a.txt": b"content a", "b.txt": b"content b"}

    await directory_state.write_test_case_to_file(str(target), test_case)

    assert target.is_dir()
    assert (target / "sub" / "a.txt").read_bytes() == b"content a"
    assert (target / "b.txt").read_bytes() == b"content b"


async def test_directory_state_format_data_returns_none(directory_state):
    # Directory formatting is not implemented
    result = await directory_state.format_data({"a.txt": b"test"})
    assert result is None


async def test_directory_state_run_formatter_command_raises(directory_state):
    with pytest.raises(NotImplementedError):
        await directory_state.run_formatter_command(["cat"], {"a.txt": b"test"})


# === attempt_format tests ===


async def test_attempt_format_returns_data_when_cannot_format(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # With formatter=none, can_format should be False
    assert state.can_format is False
    result = await state.attempt_format(b"test")
    assert result == b"test"


# === run_for_exit_code tests ===


async def test_run_for_exit_code_returns_script_exit_code(tmp_path):
    """Test that run_for_exit_code returns the script's exit code."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 42")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    exit_code = await state.run_for_exit_code(b"hello")
    assert exit_code == 42


async def test_run_for_exit_code_with_stdin_input_type(tmp_path):
    """Test that stdin input type pipes data correctly."""
    # Script that exits 0 if stdin contains 'magic'
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\ngrep -q magic && exit 0 || exit 1")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("magic word")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.stdin,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"magic word",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Should exit 0 because stdin contains 'magic'
    exit_code = await state.run_for_exit_code(b"magic word")
    assert exit_code == 0

    # Should exit 1 because stdin doesn't contain 'magic'
    exit_code = await state.run_for_exit_code(b"other word")
    assert exit_code == 1


async def test_run_for_exit_code_in_place_mode(tmp_path):
    """Test in_place mode writes to original file location."""
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/bash\ncat "$1" | grep -q hello && exit 0 || exit 1')
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello world")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=True,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Should exit 0 because file contains 'hello'
    exit_code = await state.run_for_exit_code(b"hello there")
    assert exit_code == 0


# === is_interesting tests ===


async def test_is_interesting_returns_true_for_exit_zero(tmp_path):
    """Test that is_interesting returns True when script exits 0."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    result = await state.is_interesting(b"hello")
    assert result is True
    # Also verify parallel tasks tracking worked
    assert state.parallel_tasks_running == 0


async def test_is_interesting_returns_false_for_non_zero_exit(tmp_path):
    """Test that is_interesting returns False when script exits non-zero."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    result = await state.is_interesting(b"hello")
    assert result is False


# === attempt_format additional tests ===


async def test_attempt_format_with_working_formatter(tmp_path):
    """Test attempt_format returns formatted data when formatter works."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        # Use cat as formatter (just returns input)
        formatter="cat",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Formatter should work and return data
    result = await state.attempt_format(b"hello")
    assert result == b"hello"


async def test_attempt_format_disables_on_failure(tmp_path):
    """Test attempt_format disables formatting when formatter fails."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        # Use a formatter that will fail
        formatter="false",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Initially can_format is True
    assert state.can_format is True

    # After format failure, should return original data and disable
    result = await state.attempt_format(b"test data")
    assert result == b"test data"
    assert state.can_format is False


# === parallel task tracking tests ===


async def test_is_interesting_tracks_parallel_tasks(tmp_path):
    """Test that is_interesting properly tracks parallel task count."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nsleep 0.1\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=2,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Run two tasks in parallel
    results = []

    async def check_parallel():
        # Record the parallel count during execution
        results.append(await state.is_interesting(b"test"))

    async with trio.open_nursery() as nursery:
        nursery.start_soon(check_parallel)
        nursery.start_soon(check_parallel)

    # Both should succeed
    assert results == [True, True]
    # After completion, parallel count should be 0
    assert state.parallel_tasks_running == 0


# === first_call tracking tests ===


async def test_first_call_flag_is_cleared(tmp_path):
    """Test that first_call flag is cleared after first run_for_exit_code call."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call flag should be True initially
    assert state.first_call is True

    await state.run_for_exit_code(b"hello")

    # First call flag should be cleared after first call
    assert state.first_call is False
    # Initial exit code should be recorded
    assert state.initial_exit_code == 0


# === print_exit_message tests ===


async def test_print_exit_message_directory(directory_state, capsys):
    """Test directory state print_exit_message."""
    problem = directory_state.problem
    await directory_state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "done" in captured.out.lower()


async def test_print_exit_message_already_reduced(tmp_path, capsys):
    """Test print_exit_message when test case was already minimal."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("x")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"x",
        formatter="none",
        trivial_is_error=False,  # Don't error on trivial
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    problem = state.problem
    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "already maximally reduced" in captured.out.lower()


async def test_print_exit_message_reduced(tmp_path, capsys):
    """Test print_exit_message when size was reduced."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    # Start with a longer file
    target.write_text("hello world this is a test")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world this is a test",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    problem = state.problem
    # Reduce the test case
    await problem.is_interesting(b"hello")
    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "Deleted" in captured.out


# === report_error tests ===


async def test_report_error_timeout_exceeded(tmp_path, capsys):
    """Test report_error with TimeoutExceededOnInitial."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=1.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    exc = TimeoutExceededOnInitial(runtime=5.5, timeout=1.0)
    with pytest.raises(SystemExit):
        await state.report_error(exc)
    captured = capsys.readouterr()
    assert "timeout" in captured.err.lower()
    assert "5.5" in captured.err or "5.50" in captured.err


async def test_run_for_exit_code_no_input_type_arg(tmp_path):
    """Test run_for_exit_code with input_type that doesn't include arg."""
    script = tmp_path / "test.sh"
    # Script that exits 0 always (testing that command is called without arg)
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.stdin,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Should run without the file argument
    exit_code = await state.run_for_exit_code(b"hello")
    assert exit_code == 0


async def test_run_for_exit_code_in_place_not_basename(tmp_path):
    """Test run_for_exit_code in_place mode but not basename input type."""
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/bash\ntest -f "$1" && exit 0 || exit 1')
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=True,  # in_place but using arg not basename
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Should create a temporary file with unique name
    exit_code = await state.run_for_exit_code(b"hello world")
    assert exit_code == 0


# === Additional error path tests ===


async def test_report_error_non_timeout_rerun_fails(tmp_path, capsys):
    """Test report_error when initial test fails with non-zero exit.

    Exercises the debug rerun path where the script produces a non-zero exit code.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1")  # Always fails
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Pass a non-timeout exception to trigger the else branch
    with pytest.raises(SystemExit):
        await state.report_error(ValueError("test error"))
    captured = capsys.readouterr()
    assert "exit" in captured.err.lower() or "debug" in captured.err.lower()


async def test_report_error_cwd_dependent(tmp_path, capsys, monkeypatch):
    """Test report_error when test fails in temp dir but works locally.

    Exercises the cwd dependency detection in report_error.
    """
    call_count = {"value": 0}

    script = tmp_path / "test.sh"
    # Script that always succeeds
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Mock run_for_exit_code to fail (simulating temp dir failure)
    original_run_for_exit_code = state.run_for_exit_code

    async def mock_run_for_exit_code(test_case, debug=False):
        call_count["value"] += 1
        if call_count["value"] == 1:
            # First call in report_error should fail
            return 1
        return await original_run_for_exit_code(test_case, debug)

    monkeypatch.setattr(state, "run_for_exit_code", mock_run_for_exit_code)

    with pytest.raises(SystemExit):
        await state.report_error(ValueError("test error"))
    captured = capsys.readouterr()
    # Should mention running in directory
    assert "directory" in captured.err.lower()


async def test_print_exit_message_trivial_error(tmp_path, capsys):
    """Test print_exit_message when result is trivial and trivial_is_error=True.

    Exercises the trivial result error path in print_exit_message.
    """

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("x")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"x",
        formatter="none",
        trivial_is_error=True,  # This is key
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Mock a problem with a 1-byte result
    problem = MagicMock()
    problem.current_test_case = b"x"  # Single byte - trivial

    with pytest.raises(SystemExit) as exc_info:
        await state.print_exit_message(problem)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "trivial" in captured.out.lower()


async def test_print_exit_message_no_reduction(tmp_path, capsys):
    """Test print_exit_message when changes made but no bytes deleted.

    Exercises the 'no bytes deleted' message path in print_exit_message.
    """

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello world")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world",  # 11 bytes
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Mock a problem where the result is DIFFERENT but SAME LENGTH as initial
    # This triggers the "Some changes were made but no bytes were deleted" branch
    problem = MagicMock()
    problem.current_test_case = (
        b"world hello"  # Different content, same length (11 bytes)
    )
    problem.stats.initial_test_case_size = len(b"hello world")
    problem.stats.start_time = 0

    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "no bytes were deleted" in captured.out.lower()


async def test_run_script_on_file_nonexistent(tmp_path):
    """Test run_script_on_file raises when file doesn't exist.

    Exercises the FileNotFoundError path in run_script_on_file.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Try to run on a non-existent file
    with pytest.raises(ValueError, match="No such file"):
        await state.run_script_on_file(
            working=str(tmp_path / "nonexistent.txt"),
            debug=False,
            cwd=str(tmp_path),
        )


async def test_check_formatter_failure(tmp_path, capsys):
    """Test check_formatter when formatter exits non-zero.

    Exercises the formatter failure path in check_formatter.
    """
    # Create a formatter that always fails
    formatter = tmp_path / "bad_formatter.sh"
    formatter.write_text("#!/bin/bash\necho 'error' >&2\nexit 1")
    formatter.chmod(0o755)

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    with pytest.raises(SystemExit):
        await state.check_formatter()
    captured = capsys.readouterr()
    assert "formatter" in captured.err.lower() or "unexpected" in captured.err.lower()


async def test_check_formatter_makes_uninteresting(tmp_path, capsys):
    """Test check_formatter when formatting makes test case uninteresting.

    Exercises the uninteresting-after-format path in check_formatter.
    """
    # Create a formatter that outputs different content
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/bash\necho 'different'")
    formatter.chmod(0o755)

    # Script that only accepts 'hello'
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/bash\ngrep -q "hello" "$1" && exit 0 || exit 1')
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    with pytest.raises(SystemExit):
        await state.check_formatter()
    captured = capsys.readouterr()
    assert "uninteresting" in captured.err.lower()


async def test_default_formatter_fallback(tmp_path):
    """Test default formatter when no formatter command is determined.

    Exercises the default_reformat_data fallback path in format_data.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Use a file extension that won't match any known formatter
    target = tmp_path / "test.xyz"
    target.write_text("  hello  world  ")  # Has extra spaces

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.xyz",
        parallelism=1,
        initial=b"  hello  world  ",
        formatter="default",  # Use default formatter, not "none"
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # format_data should use default_reformat_data
    result = await state.format_data(b"  hello  world  ")
    # default_reformat_data normalizes whitespace
    assert result is not None
    assert result != b"  hello  world  "  # Should be normalized


async def test_attempt_format_with_formatter(tmp_path):
    """Test attempt_format when can_format is True.

    Exercises the can_format disabled path in attempt_format.
    """
    # Create a formatter that outputs something different
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/bash\necho 'formatted'")
    formatter.chmod(0o755)

    # Script that doesn't accept 'formatted'
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/bash\ngrep -q "hello" "$1" && exit 0 || exit 1')
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Initially can_format should be True (formatter is set)
    assert state.can_format is True

    # attempt_format should try formatting, but 'formatted' won't be interesting
    # so it should set can_format to False and return original
    result = await state.attempt_format(b"hello")
    assert result == b"hello"
    assert state.can_format is False


async def test_is_interesting_tracks_first_call_time(tmp_path):
    """Test that is_interesting sets first_call_time on first call.

    Exercises the first_call_time initialization in is_interesting.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    # Use ShrinkRayDirectoryState to test the base class is_interesting
    # (ShrinkRayStateSingleFile has its own implementation)
    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial={"a.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # first_call_time should be None initially
    assert state.first_call_time is None

    # Call is_interesting
    await state.is_interesting({"a.txt": b"hello"})

    # first_call_time should now be set
    assert state.first_call_time is not None


async def test_print_exit_message_formatting_increase(tmp_path, capsys):
    """Test print_exit_message when formatting increases size.

    Exercises the formatting increase message path in print_exit_message.
    """

    # Create a formatter that adds content (increases size)
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/bash\ncat; echo 'extra content'")
    formatter.chmod(0o755)

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("ab")  # Very short

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"abcdefghij",  # 10 bytes initially
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Mock a problem where the result is smaller than initial but formatting adds bytes
    problem = MagicMock()
    problem.current_test_case = b"ab"  # 2 bytes (reduced from 10)
    problem.stats.initial_test_case_size = 10
    problem.stats.start_time = 0

    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    # Should show the deletion stats
    assert "deleted" in captured.out.lower() or "increase" in captured.out.lower()


async def test_run_for_exit_code_in_place_basename(tmp_path):
    """Test run_for_exit_code in_place mode with basename input type.

    Exercises the in_place with basename input type path in run_for_exit_code.
    """

    # Change to tmp_path so the script can find the file by basename
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        script = tmp_path / "test.sh"
        # Script that checks the file by basename exists in cwd
        script.write_text('#!/bin/bash\ntest -f "test.txt" && exit 0 || exit 1')
        script.chmod(0o755)

        target = tmp_path / "test.txt"
        target.write_text("hello")

        state = ShrinkRayStateSingleFile(
            input_type=InputType.basename,
            in_place=True,  # in_place with basename input type
            test=[str(script)],
            filename=str(target),
            timeout=5.0,
            base="test.txt",
            parallelism=1,
            initial=b"hello",
            formatter="none",
            trivial_is_error=True,
            seed=0,
            volume=Volume.quiet,
            clang_delta_executable=None,
        )

        # Should write to the original filename and run the script
        exit_code = await state.run_for_exit_code(b"hello world")
        assert exit_code == 0
    finally:
        os.chdir(original_cwd)


async def test_check_formatter_none(tmp_path):
    """Test check_formatter returns immediately when formatter is None.

    Exercises the early return path in check_formatter when no formatter is set.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",  # No formatter
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # check_formatter should return immediately without doing anything
    # (no exception, no side effects)
    await state.check_formatter()


async def test_is_interesting_multiple_calls(tmp_path):
    """Test is_interesting when called multiple times (first_call_time already set).

    Exercises the skip path in is_interesting when first_call_time is already set.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial={"a.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call sets first_call_time
    await state.is_interesting({"a.txt": b"hello"})
    first_time = state.first_call_time

    # Second call should skip setting first_call_time
    await state.is_interesting({"a.txt": b"hello"})
    second_time = state.first_call_time

    # first_call_time should remain the same
    assert first_time == second_time


async def test_report_error_flaky_test(tmp_path, capsys):
    """Test report_error when test is flaky (different exit codes).

    Exercises the flaky test detection in report_error when the script
    returns different exit codes on repeated runs.
    """
    # Create a script that returns different exit codes
    counter_file = tmp_path / "counter"
    counter_file.write_text("0")

    script = tmp_path / "test.sh"
    # Script sequence:
    # Call 1 (temp dir): exits 1 (fails)
    # Call 2 (cwd, local_exit_code): exits 0 (succeeds)
    # Call 3 (cwd, other_exit_code): exits 1 (different from 0 = flaky!)
    script.write_text(
        f"""#!/bin/bash
COUNTER=$(cat "{counter_file}")
COUNTER=$((COUNTER + 1))
echo $COUNTER > "{counter_file}"
if [ "$COUNTER" -eq 1 ]; then
    exit 1  # First call fails (temp dir)
elif [ "$COUNTER" -eq 2 ]; then
    exit 0  # Second call succeeds (local_exit_code)
else
    exit 1  # Third call fails (other_exit_code, different from 0 = flaky!)
fi
"""
    )
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Set initial_exit_code to 0 (what it would be if initial test passed)
    # Also set first_call = False to prevent run_for_exit_code from overwriting it
    state.initial_exit_code = 0
    state.first_call = False

    with pytest.raises(SystemExit):
        await state.report_error(ValueError("test error"))
    captured = capsys.readouterr()
    # Should mention flaky
    assert "flaky" in captured.err.lower()


async def test_report_error_nondeterministic(tmp_path, capsys):
    """Test report_error when initial was non-zero but now exits 0.

    Exercises the nondeterministic behavior detection in report_error when
    the test now succeeds but previously failed.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")  # Always succeeds now
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Set initial_exit_code to non-zero (as if initial test returned non-zero)
    # Also set first_call = False to prevent run_for_exit_code from overwriting it
    state.initial_exit_code = 1
    state.first_call = False

    with pytest.raises(SystemExit):
        await state.report_error(ValueError("test error"))
    captured = capsys.readouterr()
    # Should mention nondeterministic
    assert "nondeterministic" in captured.err.lower()


async def test_print_exit_message_reformatted_is_interesting(tmp_path, capsys):
    """Test print_exit_message when reformatted result is interesting.

    Exercises the formatter application path in print_exit_message.
    """
    # Create a formatter that transforms content
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/bash\necho 'formatted'")
    formatter.chmod(0o755)

    # Script accepts anything
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello world this is long")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world this is long",  # 24 bytes
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Get the problem and reduce it
    problem = state.problem
    await problem.is_interesting(b"hello")  # 5 bytes

    # Now print_exit_message should format it, and the formatted version
    # should be interesting and written to file
    await state.print_exit_message(problem)

    # Check the file was updated with formatted content
    content = target.read_bytes()
    assert b"formatted" in content or content == b"hello"


async def test_check_formatter_reformatted_is_interesting(tmp_path):
    """Test check_formatter when reformatted result IS interesting.

    This covers branch 299->exit (formatter passes - condition is False).
    """
    # Create a formatter that transforms content
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/bash\ncat")  # Just passes through
    formatter.chmod(0o755)

    # Script accepts anything
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter=str(formatter),
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # check_formatter should pass without error (formatter works and result is interesting)
    await state.check_formatter()


async def test_timeout_on_first_call(tmp_path):
    """Test that TimeoutExceededOnInitial is raised when first call exceeds timeout.

    Exercises the timeout check on first call in run_for_exit_code.
    """
    # Create a script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nsleep 0.5\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=0.1,  # Very short timeout
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call should raise TimeoutExceededOnInitial (wrapped in ExceptionGroup by trio)
    with pytest.raises(ExceptionGroup) as exc_info:
        await state.run_for_exit_code(b"hello")

    # Find the TimeoutExceededOnInitial in the group
    timeout_exc = None
    for exc in exc_info.value.exceptions:
        if isinstance(exc, ExceptionGroup):
            for inner_exc in exc.exceptions:
                if isinstance(inner_exc, TimeoutExceededOnInitial):
                    timeout_exc = inner_exc
                    break
        elif isinstance(exc, TimeoutExceededOnInitial):
            timeout_exc = exc
            break

    assert timeout_exc is not None
    assert timeout_exc.timeout == 0.1
    assert timeout_exc.runtime >= 0.1


async def test_process_killed_on_timeout(tmp_path):
    """Test that process is killed when it doesn't terminate before wait timeout.

    Exercises the _interrupt_wait_and_kill call when the process exceeds timeout.
    """
    # Create a script that sleeps for 2 seconds
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nsleep 2\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=0.05,  # 50ms timeout, wait timeout = 500ms, but script sleeps 2s
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call should raise TimeoutExceededOnInitial and also kill the process
    with pytest.raises(ExceptionGroup) as exc_info:
        await state.run_for_exit_code(b"hello")

    # Find the TimeoutExceededOnInitial in the group
    timeout_exc = None
    for exc in exc_info.value.exceptions:
        if isinstance(exc, ExceptionGroup):
            for inner_exc in exc.exceptions:
                if isinstance(inner_exc, TimeoutExceededOnInitial):
                    timeout_exc = inner_exc
                    break
        elif isinstance(exc, TimeoutExceededOnInitial):
            timeout_exc = exc
            break

    assert timeout_exc is not None
    # The process should have been killed via the timeout handler


async def test_directory_cleanup_in_place_mode(tmp_path):
    """Test directory cleanup in in_place mode.

    Exercises the shutil.rmtree cleanup path for directories in in_place mode.
    """
    # Create a script that creates a directory instead of file
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test_dir"
    target.mkdir()
    (target / "a.txt").write_text("content")

    # Use directory state instead of file state
    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=True,  # in_place mode
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test_dir",
        parallelism=1,
        initial={"a.txt": b"content"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # This should exercise the directory cleanup code
    exit_code = await state.run_for_exit_code({"a.txt": b"modified"})
    assert exit_code == 0


# === Debug mode tests ===


async def test_run_for_exit_code_debug_mode_timeout_on_first_call(tmp_path):
    """Test timeout handling in debug mode on first call.

    Exercises the timeout check in debug mode.
    """
    # Create a script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nsleep 0.5\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=0.1,  # Very short timeout
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call in debug mode should raise TimeoutExceededOnInitial
    with pytest.raises(TimeoutExceededOnInitial) as exc_info:
        await state.run_for_exit_code(b"hello", debug=True)

    assert exc_info.value.timeout == 0.1
    assert exc_info.value.runtime >= 0.1
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_exit_code_debug_mode_dynamic_timeout(tmp_path):
    """Test dynamic timeout computation in debug mode on first call.

    Exercises the dynamic timeout computation path in debug mode.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=None,  # Dynamic timeout
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call in debug mode with None timeout should compute dynamic timeout
    assert state.timeout is None
    exit_code = await state.run_for_exit_code(b"hello", debug=True)
    assert exit_code == 0
    # After first call, timeout should be computed
    assert state.timeout is not None
    assert state.timeout > 0
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_exit_code_dynamic_timeout_non_debug(tmp_path):
    """Test dynamic timeout computation in non-debug mode on first call.

    Exercises the dynamic timeout computation path in non-debug mode,
    which uses run_script_on_file instead of the debug path.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=None,  # Dynamic timeout
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First call in non-debug mode with None timeout should compute dynamic timeout
    assert state.timeout is None
    exit_code = await state.run_for_exit_code(b"hello", debug=False)
    assert exit_code == 0
    # After first call, timeout should be computed
    assert state.timeout is not None
    assert state.timeout > 0
    # Verify minimum timeout is respected
    assert state.timeout >= DYNAMIC_TIMEOUT_MIN
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_exit_code_debug_mode_captures_stdout(tmp_path):
    """Test that debug mode captures stdout output.

    Exercises the stdout capture in debug mode.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho 'hello from stdout'\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Run in debug mode
    exit_code = await state.run_for_exit_code(b"hello", debug=True)
    assert exit_code == 0

    # Check that stdout was captured
    assert "hello from stdout" in state._last_debug_output


async def test_run_for_exit_code_debug_mode_captures_stderr(tmp_path):
    """Test that debug mode captures stderr output.

    Exercises the stderr capture in debug mode.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho 'error from stderr' >&2\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # Run in debug mode
    exit_code = await state.run_for_exit_code(b"hello", debug=True)
    assert exit_code == 0

    # Check that stderr was captured
    assert "error from stderr" in state._last_debug_output


async def test_build_error_message_includes_debug_output(tmp_path):
    """Test that build_error_message includes debug output.

    Exercises the debug output inclusion in build_error_message.
    """

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho 'diagnostic output' >&2\nexit 1")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
    )

    # First, ensure first_call is set so we can trigger the right code path
    state.first_call = False
    state.initial_exit_code = 0

    # Create an InvalidInitialExample exception
    exc = InvalidInitialExample("Test error")

    # Build the error message (this calls run_for_exit_code with debug=True internally)
    error_message = await state.build_error_message(exc)

    # The error message should include the captured debug output
    assert "diagnostic output" in error_message


async def test_build_error_message_includes_cwd_debug_output(tmp_path):
    """Test build_error_message includes debug output from cwd run."""

    # Create a counter file to track call count
    counter_file = tmp_path / ".call_count"
    counter_file.write_text("0")

    # Create a script that:
    # Call 1: run_for_exit_code with debug=True (returns 1 to trigger first-call failure path)
    # Call 2: run_script_on_file with debug=False from cwd (returns 0 to trigger cwd success path)
    # Call 3: run_script_on_file with debug=True from cwd (produces output for error message)
    script = tmp_path / "test.sh"
    script.write_text(
        f"""#!/bin/bash
COUNTER_FILE="{counter_file}"
COUNT=$(cat "$COUNTER_FILE")
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

if [ "$COUNT" -eq 1 ]; then
    # First call - return 1 to trigger error path
    echo "first call output" >&2
    exit 1
elif [ "$COUNT" -eq 2 ]; then
    # Second call (local check) - return 0
    exit 0
else
    # Third call (debug run in cwd) - return 0 with output
    echo "cwd debug output" >&2
    exit 0
fi
"""
    )
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        state = ShrinkRayStateSingleFile(
            input_type=InputType.arg,
            in_place=False,
            test=[str(script)],
            filename=str(target),
            timeout=5.0,
            base="test.txt",
            parallelism=1,
            initial=b"hello",
            formatter="none",
            trivial_is_error=True,
            seed=0,
            volume=Volume.quiet,
            clang_delta_executable=None,
        )

        state.first_call = False
        state.initial_exit_code = 1

        exc = InvalidInitialExample("Test error")
        error_message = await state.build_error_message(exc)

        # Should include debug output from the cwd run
        assert "cwd debug output" in error_message
    finally:
        os.chdir(old_cwd)


async def test_volume_debug_inherits_stderr(tmp_path):
    """Test that volume=debug causes stderr to be inherited, not discarded."""
    # Create a script that writes to stderr
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho 'debug output' >&2\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.debug,  # Debug mode
        clang_delta_executable=None,
    )

    # Bypass first_call logic
    state.first_call = False
    state.initial_exit_code = 0

    # Run the script - with volume=debug, stderr should NOT be subprocess.DEVNULL
    # We can verify this by checking the kwargs would have stderr=None
    # The actual stderr output would go to the parent process's stderr
    exit_code = await state.run_for_exit_code(b"hello")
    assert exit_code == 0


# === OutputCaptureManager tests ===


def test_output_manager_allocate_and_mark_completed(tmp_path):
    """Test allocating output files and marking them completed."""
    # Use min_display_seconds=0 to test basic behavior without display window
    manager = OutputCaptureManager(output_dir=str(tmp_path), min_display_seconds=0)

    # Allocate some files
    test_id1, path1 = manager.allocate_output_file()
    test_id2, path2 = manager.allocate_output_file()

    assert test_id1 == 0
    assert test_id2 == 1
    assert path1 != path2

    # Write content to files (get_current_output only returns info for files with content)
    Path(path1).write_text("test1 output")
    Path(path2).write_text("test2 output")

    # Both should be active - get_current_output returns (path, test_id, return_code)
    output_path, test_id, return_code = manager.get_current_output()
    assert test_id == 1  # Most recent
    assert output_path == path2  # Most recent active
    assert return_code is None  # Still running

    # Mark the first one completed
    manager.mark_completed(test_id1)
    output_path, test_id, return_code = manager.get_current_output()
    assert test_id == 1  # Still have one active
    assert output_path == path2
    assert return_code is None  # Still running

    # Mark the second one completed
    manager.mark_completed(test_id2)
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2  # Most recent completed
    assert test_id == 1
    assert return_code == 0  # Completed


def test_output_manager_mark_completed_unknown_id(tmp_path):
    """Test marking an unknown test_id as completed (no-op)."""
    manager = OutputCaptureManager(output_dir=str(tmp_path))
    # Should not raise - just no-op
    manager.mark_completed(999)
    output_path, test_id, _ = manager.get_current_output()
    assert output_path is None
    assert test_id is None


def test_output_manager_get_current_output_none(tmp_path):
    """Test get_current_output when nothing allocated."""
    manager = OutputCaptureManager(output_dir=str(tmp_path))
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path is None
    assert test_id is None
    assert return_code is None


def test_output_manager_cleanup_old_files(tmp_path):
    """Test cleanup of files older than max_age."""
    manager = OutputCaptureManager(
        output_dir=str(tmp_path), max_files=100, max_age_seconds=0.1
    )

    # Allocate and complete some files
    for _ in range(5):
        test_id, path = manager.allocate_output_file()
        # Create the file so it can be deleted
        with open(path, "w") as f:
            f.write("test")
        manager.mark_completed(test_id)

    # Wait for files to age
    time.sleep(0.15)

    # Allocate and complete one more - this triggers cleanup
    test_id, path = manager.allocate_output_file()
    with open(path, "w") as f:
        f.write("test")
    manager.mark_completed(test_id)

    # Old files should have been cleaned up
    assert len(manager._completed_outputs) == 1  # Only the recent one


def test_output_manager_cleanup_excess_files(tmp_path):
    """Test cleanup of excess files beyond max_files."""
    manager = OutputCaptureManager(
        output_dir=str(tmp_path), max_files=3, max_age_seconds=3600
    )

    # Allocate and complete more than max_files
    for _ in range(5):
        test_id, path = manager.allocate_output_file()
        with open(path, "w") as f:
            f.write("test")
        manager.mark_completed(test_id)

    # Should only keep max_files
    assert len(manager._completed_outputs) == 3


def test_output_manager_cleanup_all(tmp_path):
    """Test cleanup_all removes all files."""
    manager = OutputCaptureManager(output_dir=str(tmp_path))

    # Allocate some files (some active, some completed)
    test_id1, path1 = manager.allocate_output_file()
    _, path2 = manager.allocate_output_file()  # test_id2 not used - it stays active
    with open(path1, "w") as f:
        f.write("test1")
    with open(path2, "w") as f:
        f.write("test2")

    manager.mark_completed(test_id1)  # One completed

    # Cleanup all
    manager.cleanup_all()

    # Should have no files tracked
    assert len(manager._active_outputs) == 0
    assert len(manager._completed_outputs) == 0
    # Files should be deleted
    assert not os.path.exists(path1)
    assert not os.path.exists(path2)


def test_output_manager_safe_delete_nonexistent(tmp_path):
    """Test _safe_delete doesn't crash on nonexistent files."""
    # Should not raise
    OutputCaptureManager._safe_delete(str(tmp_path / "nonexistent.log"))


def test_output_manager_active_test_takes_priority(tmp_path):
    """Test that active tests always take priority over completed tests."""
    manager = OutputCaptureManager(output_dir=str(tmp_path), min_display_seconds=0.5)

    # Allocate and complete a test
    test_id1, path1 = manager.allocate_output_file()
    with open(path1, "w") as f:
        f.write("output1")
    manager.mark_completed(test_id1)

    # Start a new test immediately
    test_id2, path2 = manager.allocate_output_file()
    with open(path2, "w") as f:
        f.write("output2")

    # Active test should take priority over recently completed
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2
    assert test_id == test_id2
    assert return_code is None  # Still running

    # Complete the second test
    manager.mark_completed(test_id2)

    # Now should show the most recently completed test
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2
    assert test_id == test_id2
    assert return_code == 0  # Completed


def test_output_manager_display_window_no_new_test(tmp_path):
    """Test display window behavior when no new test starts."""
    manager = OutputCaptureManager(
        output_dir=str(tmp_path), min_display_seconds=0.2, grace_period_seconds=0.2
    )

    # Allocate and complete a test
    test_id1, path1 = manager.allocate_output_file()
    with open(path1, "w") as f:
        f.write("output1")
    manager.mark_completed(test_id1)

    # Within display window: still shows path1, completed test
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert test_id == test_id1
    assert return_code == 0  # Completed

    # Wait for min_display_seconds but within grace period
    time.sleep(0.25)

    # Still within grace period (0.2 + 0.2 = 0.4s total), should still show completed
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert return_code == 0  # Still completed

    # Wait for grace period to expire
    time.sleep(0.2)

    # After full window: still shows path1 (fallback), still completed
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert return_code == 0


def test_output_manager_grace_period_with_new_test(tmp_path):
    """Test that new test starting during grace period is shown immediately."""
    manager = OutputCaptureManager(
        output_dir=str(tmp_path), min_display_seconds=0.15, grace_period_seconds=0.3
    )

    # Allocate and complete a test
    test_id1, path1 = manager.allocate_output_file()
    with open(path1, "w") as f:
        f.write("output1")
    manager.mark_completed(test_id1)

    # Wait until we're past min_display but within grace period
    time.sleep(0.2)

    # Should still show completed (in grace period, no active test)
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert return_code == 0  # Completed

    # Start a new test during grace period
    test_id2, path2 = manager.allocate_output_file()
    with open(path2, "w") as f:
        f.write("output2")

    # New active test should take priority immediately
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2
    assert test_id == test_id2
    assert return_code is None  # Still running


def test_output_manager_empty_active_file_shows_completed(tmp_path):
    """Test that active tests without content don't take priority over completed tests."""
    manager = OutputCaptureManager(output_dir=str(tmp_path), min_display_seconds=0.5)

    # Allocate and complete a test with output
    test_id1, path1 = manager.allocate_output_file()
    with open(path1, "w") as f:
        f.write("output1")
    manager.mark_completed(test_id1)

    # Start a new test but don't write any content
    test_id2, path2 = manager.allocate_output_file()
    # File exists but is empty (or doesn't exist yet)

    # Should still show completed test because active has no content
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert test_id == test_id1
    assert return_code == 0  # Showing completed test

    # Now write content to the active test
    with open(path2, "w") as f:
        f.write("output2")

    # Active test with content should take priority
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2
    assert test_id == test_id2
    assert return_code is None  # Still running


def test_output_manager_return_code(tmp_path):
    """Test that return codes are tracked correctly with the displayed output."""
    manager = OutputCaptureManager(output_dir=str(tmp_path))

    # No completed tests yet
    output_path, test_id, return_code = manager.get_current_output()
    assert return_code is None

    # Complete a test with return code
    test_id1, path1 = manager.allocate_output_file()
    with open(path1, "w") as f:
        f.write("output1")
    manager.mark_completed(test_id1, return_code=42)

    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path1
    assert test_id == test_id1
    assert return_code == 42

    # Complete another test with different return code
    test_id2, path2 = manager.allocate_output_file()
    with open(path2, "w") as f:
        f.write("output2")
    manager.mark_completed(test_id2, return_code=0)

    # Now shows the most recent completed test with its return code
    output_path, test_id, return_code = manager.get_current_output()
    assert output_path == path2
    assert test_id == test_id2
    assert return_code == 0


# === History integration tests ===


def test_state_with_history_disabled(tmp_path):
    """Test that history is not set up when history_enabled=False."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=False,
    )

    # History manager should not be created
    assert state.history_manager is None
    # Output manager should not be created (no TUI, no history)
    assert state.output_manager is None


def test_state_with_history_enabled_creates_output_manager(tmp_path):
    """Test that history enabled creates an output manager for capturing output."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # History manager should be created
    assert state.history_manager is not None
    # Output manager should be created for capturing output
    assert state.output_manager is not None


def test_get_last_captured_output_with_no_output_manager(tmp_path):
    """Test _get_last_captured_output returns None when output_manager is None."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=False,  # Disable history to not create output_manager
    )

    assert state._get_last_captured_output() is None


def test_get_last_captured_output_with_no_output_available(tmp_path):
    """Test _get_last_captured_output returns None when no output is available."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # Output manager exists but has no output yet
    assert state.output_manager is not None
    assert state._get_last_captured_output() is None


def test_get_last_captured_output_reads_output_file(tmp_path):
    """Test _get_last_captured_output reads the captured output file."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    assert state.output_manager is not None

    # Simulate a completed test with output
    test_id, output_path = state.output_manager.allocate_output_file()
    with open(output_path, "wb") as f:
        f.write(b"test output content")
    state.output_manager.mark_completed(test_id)

    output = state._get_last_captured_output()
    assert output == b"test output content"


def test_get_last_captured_output_handles_oserror(tmp_path, monkeypatch):
    """Test _get_last_captured_output returns None on OSError."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    assert state.output_manager is not None

    # Simulate a completed test with output
    test_id, output_path = state.output_manager.allocate_output_file()
    with open(output_path, "wb") as f:
        f.write(b"test output")
    state.output_manager.mark_completed(test_id)

    # Remove the file to cause OSError
    os.unlink(output_path)

    assert state._get_last_captured_output() is None


def test_directory_state_get_test_case_bytes_returns_none(tmp_path):
    """Test that directory state returns None for history recording."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("content")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target_dir),
        timeout=5.0,
        base="target",
        parallelism=1,
        initial={"file.txt": b"content"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # Directory mode should return None for test case bytes
    assert state._get_test_case_bytes({"file.txt": b"content"}) is None
    # But should still return something for initial bytes
    assert state._get_initial_bytes() is not None


def test_check_trivial_result_returns_error_message(tmp_path):
    """Test check_trivial_result returns error message for trivial results."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=False,
    )

    # Create a mock problem with trivial test case
    mock_problem = MagicMock()
    mock_problem.current_test_case = b""  # Empty/trivial

    error = state.check_trivial_result(mock_problem)
    assert error is not None
    assert "trivial" in error.lower()
    assert "size 0" in error


def test_check_trivial_result_returns_none_for_non_trivial(tmp_path):
    """Test check_trivial_result returns None for non-trivial results."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=False,
    )

    # Create a mock problem with non-trivial test case
    mock_problem = MagicMock()
    mock_problem.current_test_case = b"some content"

    error = state.check_trivial_result(mock_problem)
    assert error is None


def test_state_with_history_enabled_uses_existing_output_manager(tmp_path):
    """Test that history enabled uses an existing output_manager instead of creating a new one."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    # Create an existing output manager (simulating TUI mode)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing_manager = OutputCaptureManager(output_dir=str(output_dir))

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
        output_manager=existing_manager,
    )

    # Should use the provided output_manager, not create a new one
    assert state.output_manager is existing_manager
    # History manager should still be created
    assert state.history_manager is not None


@pytest.mark.trio
async def test_run_script_discards_output_in_quiet_mode_without_history(tmp_path):
    """Test that output is discarded in quiet mode without history or TUI."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho hello\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,  # Not debug mode
        clang_delta_executable=None,
        history_enabled=False,  # No history, so no output_manager
    )

    # Should succeed and discard output
    exit_code = await state.run_script_on_file(
        working=str(target),
        cwd=str(tmp_path),
        debug=False,
    )
    assert exit_code == 0


@pytest.mark.trio
async def test_volume_debug_without_history_or_output_manager(tmp_path):
    """Test debug mode inherits stderr when no output_manager is present."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho 'debug output' >&2\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.debug,  # Debug mode
        clang_delta_executable=None,
        history_enabled=False,  # No history, so no output_manager
    )

    # With history disabled and debug mode, stderr should be inherited
    # stdout goes to DEVNULL, stderr inherited
    assert state.output_manager is None
    exit_code = await state.run_script_on_file(
        working=str(target),
        cwd=str(tmp_path),
        debug=False,
    )
    assert exit_code == 0


@pytest.mark.trio
async def test_reducer_property_initializes_history(tmp_path):
    """Test that accessing reducer property initializes history when enabled."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # History manager should exist
    assert state.history_manager is not None
    # But not initialized yet (since we haven't accessed reducer)
    assert not state.history_manager.initialized

    # Access reducer property to trigger initialization
    reducer = state.reducer

    # History should now be initialized
    assert state.history_manager.initialized
    assert reducer is not None

    # Verify history directory was created
    assert os.path.isdir(state.history_manager.history_dir)
    initial_dir = os.path.join(state.history_manager.history_dir, "initial")
    assert os.path.isdir(initial_dir)


@pytest.mark.trio
async def test_reducer_property_without_history(tmp_path):
    """Test that accessing reducer property works when history is disabled."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=False,
    )

    # History manager should not exist
    assert state.history_manager is None

    # Access reducer property should work without history
    reducer = state.reducer

    # Reducer should be created successfully
    assert reducer is not None


@pytest.mark.trio
async def test_history_callback_records_reduction(tmp_path):
    """Test that the history callback records reductions when they happen."""
    script = tmp_path / "test.sh"
    # Script that always says "interesting" (exit 0)
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello world")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello world",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # Access reducer to initialize history and register callbacks
    reducer = state.reducer
    problem = reducer.target

    assert state.history_manager is not None
    assert state.history_manager.initialized

    # No reductions yet
    assert state.history_manager.reduction_counter == 0

    # Setup the problem first (required before calling is_interesting)
    await problem.setup()

    # Trigger a reduction by calling is_interesting with a smaller test case
    smaller = b"hello"  # Smaller than "hello world"
    result = await problem.is_interesting(smaller)

    # The script returns 0, so it should be interesting
    assert result is True

    # The callback should have recorded the reduction
    assert state.history_manager.reduction_counter == 1

    # Verify the reduction file exists
    reductions_dir = os.path.join(state.history_manager.history_dir, "reductions")
    assert os.path.isdir(reductions_dir)
    reduction_1 = os.path.join(reductions_dir, "0001")
    assert os.path.isdir(reduction_1)

    # Verify the content was saved
    saved_file = os.path.join(reduction_1, "test.txt")
    assert os.path.isfile(saved_file)
    with open(saved_file, "rb") as f:
        assert f.read() == smaller


@pytest.mark.trio
async def test_history_callback_skips_directory_mode(tmp_path):
    """Test that history callback gracefully skips directory mode (returns None)."""
    script = tmp_path / "test.sh"
    # Script that always says "interesting" (exit 0)
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Create a target directory with a file
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("hello world")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target_dir),
        timeout=5.0,
        base=target_dir.name,
        parallelism=1,
        initial={"file.txt": b"hello world"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        clang_delta_executable=None,
        history_enabled=True,
    )

    # Access reducer to initialize history and register callbacks
    reducer = state.reducer
    problem = reducer.target

    assert state.history_manager is not None
    assert state.history_manager.initialized

    # No reductions yet
    assert state.history_manager.reduction_counter == 0

    # Setup the problem first (required before calling is_interesting)
    await problem.setup()

    # Trigger a reduction by calling is_interesting with a smaller test case
    smaller = {"file.txt": b"hello"}  # Smaller than "hello world"
    result = await problem.is_interesting(smaller)

    # The script returns 0, so it should be interesting
    assert result is True

    # The callback should NOT have recorded the reduction because
    # _get_test_case_bytes returns None for directory mode
    assert state.history_manager.reduction_counter == 0
