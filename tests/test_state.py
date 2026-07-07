"""Tests for state management."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import trio

import shrinkray.state as state_mod
from shrinkray.adaptive_timeout import MIN_TIMEOUT, AdaptiveTimeoutPolicy
from shrinkray.cli import InputType
from shrinkray.history import deserialize_directory, serialize_directory
from shrinkray.problem import InvalidInitialExample, shortlex
from shrinkray.process import (
    _ULIMIT_FLAG,
    MEMORY_LIMIT_ENFORCEABLE,
    default_memory_limit,
)
from shrinkray.process import kill_process_group as original_kill
from shrinkray.reducer import DirectoryShrinkRay, ShrinkRay
from shrinkray.state import (
    MemoryLimitExceededOnInitial,
    OutputCaptureManager,
    ScriptRunResult,
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


# === memory limit tests ===


def test_memory_limit_exceeded_stores_used_and_limit():
    exc = MemoryLimitExceededOnInitial(used=200 * 1024**2, limit=50 * 1024**2)
    assert exc.used == 200 * 1024**2
    assert exc.limit == 50 * 1024**2


def test_memory_limit_exceeded_message_mentions_memory_and_flag():
    exc = MemoryLimitExceededOnInitial(used=200 * 1024**2, limit=50 * 1024**2)
    assert "memory" in str(exc).lower()
    assert "--memory-limit" in str(exc)


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_effective_memory_limit_disabled(simple_state, limit):
    simple_state.memory_limit = limit
    assert simple_state.effective_memory_limit(first_call=True) is None
    assert simple_state.effective_memory_limit(first_call=False) is None


def test_effective_memory_limit_first_call_is_generous(simple_state):
    simple_state.memory_limit = 1  # 1 byte configured
    # First call gets generous headroom (physical RAM), not the tiny limit,
    # so it can run and have its true peak measured.
    assert simple_state.effective_memory_limit(first_call=True) > 1
    assert simple_state.effective_memory_limit(first_call=False) == 1


def test_raise_if_initial_over_memory_raises_when_over(simple_state):
    simple_state.memory_limit = 50 * 1024**2
    with patch("shrinkray.state.peak_child_rss_bytes", return_value=100 * 1024**2):
        with pytest.raises(MemoryLimitExceededOnInitial):
            simple_state.raise_if_initial_over_memory()


def test_raise_if_initial_over_memory_ok_when_under(simple_state):
    simple_state.memory_limit = 50 * 1024**2
    with patch("shrinkray.state.peak_child_rss_bytes", return_value=1 * 1024**2):
        simple_state.raise_if_initial_over_memory()  # must not raise


@pytest.mark.parametrize("limit", [None, 0])
def test_raise_if_initial_over_memory_noop_when_disabled(simple_state, limit):
    simple_state.memory_limit = limit
    with patch("shrinkray.state.peak_child_rss_bytes", return_value=10**12):
        simple_state.raise_if_initial_over_memory()  # must not raise


def _memory_hog_state(tmp_path):
    # A test that really allocates ~200MB. With a 20MB configured limit the
    # first call runs with generous headroom, its peak RSS is measured, and
    # it is flagged. This is measurement-based, so it works even where the
    # RLIMIT_AS enforcement itself is a no-op (macOS).
    script = tmp_path / "hog.sh"
    script.write_text(
        "#!/usr/bin/env python3\nb = bytearray(200 * 1024 * 1024)\nassert b[0] == 0\n"
    )
    script.chmod(0o755)
    target = tmp_path / "target.txt"
    target.write_text("hello")

    return ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=30.0,
        base="target.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        memory_limit=20 * 1024**2,
    )


async def test_run_script_raises_when_initial_exceeds_memory(tmp_path):
    state = _memory_hog_state(tmp_path)
    # The non-debug path runs the test in a nursery, so the failure arrives
    # wrapped in an ExceptionGroup, exactly as the worker's
    # `except* InvalidInitialExample` handler expects.
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await state.run_script_on_file(working=str(state.filename), cwd=str(tmp_path))
    assert exc_info.value.subgroup(MemoryLimitExceededOnInitial) is not None


async def test_run_script_debug_raises_when_initial_exceeds_memory(tmp_path):
    state = _memory_hog_state(tmp_path)
    # The debug path runs the test directly (no nursery), so it raises the
    # exception unwrapped.
    with pytest.raises(MemoryLimitExceededOnInitial):
        await state.run_script_on_file(
            working=str(state.filename), cwd=str(tmp_path), debug=True
        )


# === ShrinkRayStateSingleFile tests ===


@pytest.fixture
def simple_state(tmp_path):
    """Create a simple state for testing."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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


# === External reducer wiring ===


def test_reducer_log_dir_none_without_history(simple_state):
    # simple_state has history_enabled=False
    assert simple_state.reducer_log_dir() is None


def test_reducer_log_dir_under_history(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    state = ShrinkRayStateSingleFile(
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
        history_enabled=True,
    )
    assert state.history_manager is not None
    log_dir = state.reducer_log_dir()
    assert log_dir == os.path.join(state.history_manager.history_dir, "reducers")


def test_new_reducer_forwards_external_reducer_settings(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    state = ShrinkRayStateSingleFile(
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
        history_enabled=False,
        external_reducers=[["my-reducer"]],
        python_reducer=False,
        restart_at_fixpoint=False,
    )
    reducer = state.reducer
    assert isinstance(reducer, ShrinkRay)
    assert reducer.external_reducers == [["my-reducer"]]
    assert reducer.python_reducer is False
    assert reducer.restart_at_fixpoint is False
    assert reducer.reducer_log_dir is None


def test_directory_new_reducer_forwards_settings(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.txt").write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="target",
        parallelism=1,
        initial={"a.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        external_reducers=[["my-reducer"]],
        python_reducer=False,
        restart_at_fixpoint=False,
    )
    reducer = state.reducer
    assert isinstance(reducer, DirectoryShrinkRay)
    assert reducer.external_reducers == [["my-reducer"]]
    assert reducer.python_reducer is False
    assert reducer.restart_at_fixpoint is False


# === ShrinkRayDirectoryState tests ===


@pytest.fixture
def directory_state(tmp_path):
    """Create a directory state for testing."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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
    # Mock try_decode to simulate truly undecodable binary data,
    # since chardet may decode arbitrary bytes in single-byte encodings
    binary_data = bytes([0x80, 0x81, 0x82])
    with patch("shrinkray.problem.try_decode", return_value=(None, "")):
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # With formatter=none, can_format should be False
    assert state.can_format is False
    result = await state.attempt_format(b"test")
    assert result == b"test"


# === run_for_result tests ===


async def test_run_for_result_returns_script_exit_code(tmp_path):
    """Test that run_for_result returns the script's exit code."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 42")
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
        history_enabled=False,
    )

    exit_code = (await state.run_for_result(b"hello")).exit_code
    assert exit_code == 42


async def test_run_for_result_with_stdin_input_type(tmp_path):
    """Test that stdin input type pipes data correctly."""
    # Script that exits 0 if stdin contains 'magic'
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\ngrep -q magic && exit 0 || exit 1")
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
        history_enabled=False,
    )

    # Should exit 0 because stdin contains 'magic'
    exit_code = (await state.run_for_result(b"magic word")).exit_code
    assert exit_code == 0

    # Should exit 1 because stdin doesn't contain 'magic'
    exit_code = (await state.run_for_result(b"other word")).exit_code
    assert exit_code == 1


# Regression tests for https://github.com/DRMacIver/shrinkray/issues/56:
# on OpenBSD, kqueue never reports a pipe's write end as writable once the
# read end is closed, so feeding test-case bytes to the script through a
# pipe deadlocks trio's stdin-feeder task whenever the test case exceeds
# the pipe buffer and the script exits without reading stdin. Test-case
# stdin must therefore be a real file descriptor, not a pipe.

STDIN_IS_REGULAR_FILE = (
    "import os, stat, sys; sys.exit(0 if stat.S_ISREG(os.fstat(0).st_mode) else 1)"
)


def _stdin_check_state(tmp_path):
    target = tmp_path / "test.txt"
    target.write_bytes(b"hello")
    return ShrinkRayStateSingleFile(
        input_type=InputType.stdin,
        in_place=False,
        test=[sys.executable, "-c", STDIN_IS_REGULAR_FILE],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )


async def test_stdin_is_a_regular_file_not_a_pipe(tmp_path):
    state = _stdin_check_state(tmp_path)
    assert (await state.run_for_result(b"hello")).exit_code == 0


async def test_stdin_is_a_regular_file_not_a_pipe_in_debug_mode(tmp_path):
    state = _stdin_check_state(tmp_path)
    assert (await state.run_for_result(b"hello", debug=True)).exit_code == 0


async def test_large_unread_stdin_does_not_deadlock(tmp_path):
    # The script never reads stdin and the test case is much larger than a
    # pipe buffer; with file-descriptor stdin there is no pipe to deadlock.
    content = b"x" * (1 << 20)
    target = tmp_path / "test.txt"
    target.write_bytes(content)
    state = ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=["true"],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=content,
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )
    assert (await state.run_for_result(content)).exit_code == 0


async def test_formatter_stdin_is_a_regular_file(simple_state):
    # The formatter gets its input the same way, so a formatter that exits
    # without draining stdin must not deadlock either.
    result = await simple_state.run_formatter_command(
        [
            sys.executable,
            "-c",
            "import os, stat, sys; assert stat.S_ISREG(os.fstat(0).st_mode); "
            "sys.stdout.write(sys.stdin.read())",
        ],
        b"hello",
    )
    assert result.returncode == 0
    assert result.stdout == b"hello"


async def test_run_for_result_in_place_mode(tmp_path):
    """Test in_place mode writes to original file location."""
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ncat "$1" | grep -q hello && exit 0 || exit 1')
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
        history_enabled=False,
    )

    # Should exit 0 because file contains 'hello'
    exit_code = (await state.run_for_result(b"hello there")).exit_code
    assert exit_code == 0


# === is_interesting tests ===


async def test_is_interesting_returns_true_for_exit_zero(tmp_path):
    """Test that is_interesting returns True when script exits 0."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    result = await state.is_interesting(b"hello")
    assert result is True
    # Also verify parallel tasks tracking worked
    assert state.parallel_tasks_running == 0


async def test_is_interesting_stores_no_output_when_none_available(tmp_path):
    """Test is_interesting handles None output when storing successful result.

    This covers the branch where output is None in the base class is_interesting.
    Uses ShrinkRayDirectoryState to test the base class method, with history
    disabled so no output manager is created.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test_dir"
    target.mkdir()
    (target / "a.txt").write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test_dir",
        parallelism=1,
        initial={"a.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,  # No history = no output manager
    )

    # Verify no output manager
    assert state.output_manager is None

    # Call is_interesting - should succeed without storing output
    result = await state.is_interesting({"a.txt": b"hello"})
    assert result is True

    # Verify no output was stored (since there was none to capture)
    assert len(state._successful_outputs) == 0


async def test_is_interesting_returns_false_for_non_zero_exit(tmp_path):
    """Test that is_interesting returns False when script exits non-zero."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 1")
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
        history_enabled=False,
    )

    result = await state.is_interesting(b"hello")
    assert result is False


# === attempt_format additional tests ===


async def test_attempt_format_with_working_formatter(tmp_path):
    """Test attempt_format returns formatted data when formatter works."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Formatter should work and return data
    result = await state.attempt_format(b"hello")
    assert result == b"hello"


async def test_attempt_format_disables_on_failure(tmp_path):
    """Test attempt_format disables formatting when formatter fails."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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
    script.write_text("#!/bin/sh\nsleep 0.1\nexit 0")
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
        history_enabled=False,
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
    """Test that first_call flag is cleared after first run_for_result call."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # First call flag should be True initially
    assert state.first_call is True

    await state.run_for_result(b"hello")

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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    problem = state.problem
    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "already maximally reduced" in captured.out.lower()


async def test_print_exit_message_reduced(tmp_path, capsys):
    """Test print_exit_message when size was reduced."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    problem = state.problem
    # Reduce the test case
    await problem.is_interesting(b"hello")
    await state.print_exit_message(problem)
    captured = capsys.readouterr()
    assert "Deleted" in captured.out


# === build_error_message tests ===


async def test_build_error_message_timeout_exceeded(tmp_path):
    """Test build_error_message with TimeoutExceededOnInitial."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    exc = TimeoutExceededOnInitial(runtime=5.5, timeout=1.0)
    message = await state.build_error_message(exc)
    assert "timeout" in message.lower()
    assert "5.5" in message or "5.50" in message


async def test_run_for_result_no_input_type_arg(tmp_path):
    """Test run_for_result with input_type that doesn't include arg."""
    script = tmp_path / "test.sh"
    # Script that exits 0 always (testing that command is called without arg)
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Should run without the file argument
    exit_code = (await state.run_for_result(b"hello")).exit_code
    assert exit_code == 0


async def test_run_for_result_in_place_not_basename(tmp_path):
    """Test run_for_result in_place mode but not basename input type."""
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ntest -f "$1" && exit 0 || exit 1')
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
        history_enabled=False,
    )

    # Should create a temporary file with unique name
    exit_code = (await state.run_for_result(b"hello world")).exit_code
    assert exit_code == 0


async def test_run_for_result_in_place_cleanup_handles_unlink_error(
    tmp_path, monkeypatch
):
    """Test that in-place cleanup handles OSError from os.unlink gracefully."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=True,
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
        history_enabled=False,
    )

    original_unlink = os.unlink

    def failing_unlink(path):
        # Only fail on the temp working files, not other unlinks
        if "test-" in str(path):
            raise OSError("permission denied")
        return original_unlink(path)

    monkeypatch.setattr(state_mod.os, "unlink", failing_unlink)
    # Should not raise despite unlink failure
    exit_code = (await state.run_for_result(b"hello world")).exit_code
    assert exit_code == 0


async def test_process_group_killed_on_cancellation(tmp_path, monkeypatch):
    """Test that the process group is killed when the task is cancelled."""
    script = tmp_path / "test.sh"
    # Script that sleeps forever
    script.write_text("#!/bin/sh\nsleep 1000")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=100.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )

    kill_called = [False]

    def tracking_kill(sp):
        kill_called[0] = True
        original_kill(sp)

    monkeypatch.setattr(state_mod, "kill_process_group", tracking_kill)
    with trio.move_on_after(0.5):
        await state.run_for_result(b"hello")

    assert kill_called[0]


async def test_cancelled_test_is_not_recorded_as_exiting_with_code_zero(tmp_path):
    """Regression test: when a test was cancelled (or timed out) before
    producing an exit code, mark_completed recorded exit code 0, so the
    TUI showed "exited with code 0" for a test that was actually killed."""
    script = tmp_path / "test.sh"
    # Script that sleeps forever
    script.write_text("#!/bin/sh\nsleep 1000")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=100.0,
        base="test.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )
    state.output_manager = OutputCaptureManager(output_dir=str(tmp_path))

    with trio.move_on_after(0.5):
        await state.run_for_result(b"hello")

    _, _, return_code = state.output_manager.get_current_output()
    assert return_code is not None
    assert return_code != 0


async def test_cleanup_when_process_never_started(tmp_path):
    """Test that cleanup works when cancelled before the process starts.

    Covers the sp=None branch in the finally block of run_script_on_file.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Cancel immediately so nursery.start never completes and sp stays None.
    # Call run_script_on_file directly because run_for_result would be
    # cancelled during write_test_case_to_file before reaching this code.
    with trio.CancelScope() as scope:
        scope.cancel()
        await state.run_script_on_file(working=str(target), cwd=str(tmp_path))


# === Additional error path tests ===


async def test_build_error_message_non_timeout_rerun_fails(tmp_path):
    """Test build_error_message when initial test fails with non-zero exit.

    Exercises the debug rerun path where the script produces a non-zero exit code.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 1")  # Always fails
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
        history_enabled=False,
    )

    # Pass a non-timeout exception to trigger the else branch
    message = await state.build_error_message(ValueError("test error"))
    assert "exit" in message.lower() or "debug" in message.lower()


async def test_build_error_message_cwd_dependent(tmp_path, monkeypatch):
    """Test build_error_message when test fails in temp dir but works locally.

    Exercises the cwd dependency detection in build_error_message.
    """
    call_count = {"value": 0}

    script = tmp_path / "test.sh"
    # Script that always succeeds
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Mock run_for_result to fail (simulating temp dir failure)
    original_run_for_result = state.run_for_result

    async def mock_run_for_result(test_case, debug=False):
        call_count["value"] += 1
        if call_count["value"] == 1:
            # First call in build_error_message should fail
            return ScriptRunResult(exit_code=1)
        return await original_run_for_result(test_case, debug)

    monkeypatch.setattr(state, "run_for_result", mock_run_for_result)

    message = await state.build_error_message(ValueError("test error"))
    # Should mention running in directory
    assert "directory" in message.lower()


async def test_print_exit_message_trivial_error(tmp_path, capsys):
    """Test print_exit_message when result is trivial and trivial_is_error=True.

    Exercises the trivial result error path in print_exit_message.
    """

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Try to run on a non-existent file
    with pytest.raises(ValueError, match="No such file"):
        await state.run_script_on_file(
            working=str(tmp_path / "nonexistent.txt"),
            debug=False,
            cwd=str(tmp_path),
        )


async def test_default_formatter_fallback(tmp_path):
    """Test default formatter when no formatter command is determined.

    Exercises the default_reformat_data fallback path in format_data.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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
    formatter.write_text("#!/bin/sh\necho 'formatted'")
    formatter.chmod(0o755)

    # Script that doesn't accept 'formatted'
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q "hello" "$1" && exit 0 || exit 1')
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
        history_enabled=False,
    )

    # Initially can_format should be True (formatter is set)
    assert state.can_format is True

    # attempt_format should try formatting, but 'formatted' won't be interesting
    # so it should set can_format to False and return original
    result = await state.attempt_format(b"hello")
    assert result == b"hello"
    assert state.can_format is False


async def test_print_exit_message_formatting_increase(tmp_path, capsys):
    """Test print_exit_message when formatting increases size.

    Exercises the formatting increase message path in print_exit_message.
    """

    # Create a formatter that adds content (increases size)
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/sh\ncat; echo 'extra content'")
    formatter.chmod(0o755)

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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


async def test_run_for_result_in_place_basename(tmp_path):
    """Test run_for_result in_place mode with basename input type.

    Exercises the in_place with basename input type path in run_for_result.
    """

    # Change to tmp_path so the script can find the file by basename
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        script = tmp_path / "test.sh"
        # Script that checks the file by basename exists in cwd
        script.write_text('#!/bin/sh\ntest -f "test.txt" && exit 0 || exit 1')
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
            history_enabled=False,
        )

        # Should write to the original filename and run the script
        exit_code = (await state.run_for_result(b"hello world")).exit_code
        assert exit_code == 0
    finally:
        os.chdir(original_cwd)


async def test_build_error_message_flaky_test(tmp_path):
    """Test build_error_message when test is flaky (different exit codes).

    Exercises the flaky test detection in build_error_message when the
    script returns different exit codes on repeated runs.
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
        f"""#!/bin/sh
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
        history_enabled=False,
    )

    # Set initial_exit_code to 0 (what it would be if initial test passed)
    # Also set first_call = False to prevent run_for_result from overwriting it
    state.initial_exit_code = 0
    state.first_call = False

    message = await state.build_error_message(ValueError("test error"))
    # Should mention flaky
    assert "flaky" in message.lower()


async def test_build_error_message_nondeterministic(tmp_path):
    """Test build_error_message when initial was non-zero but now exits 0.

    Exercises the nondeterministic behavior detection in build_error_message
    when the test now succeeds but previously failed.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")  # Always succeeds now
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
        history_enabled=False,
    )

    # Set initial_exit_code to non-zero (as if initial test returned non-zero)
    # Also set first_call = False to prevent run_for_result from overwriting it
    state.initial_exit_code = 1
    state.first_call = False

    message = await state.build_error_message(ValueError("test error"))
    # Should mention nondeterministic
    assert "nondeterministic" in message.lower()


async def test_print_exit_message_reformatted_is_interesting(tmp_path, capsys):
    """Test print_exit_message when reformatted result is interesting.

    Exercises the formatter application path in print_exit_message.
    """
    # Create a formatter that transforms content
    formatter = tmp_path / "formatter.sh"
    formatter.write_text("#!/bin/sh\necho 'formatted'")
    formatter.chmod(0o755)

    # Script accepts anything
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
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


async def test_timeout_on_first_call(tmp_path):
    """Test that TimeoutExceededOnInitial is raised when first call exceeds timeout.

    Exercises the timeout check on first call in run_for_result.
    """
    # Create a script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nsleep 0.5\nexit 0")
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
        history_enabled=False,
    )

    # First call should raise TimeoutExceededOnInitial (wrapped in ExceptionGroup by trio)
    with pytest.raises(ExceptionGroup) as exc_info:
        await state.run_for_result(b"hello")

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
    script.write_text("#!/bin/sh\nsleep 2\nexit 0")
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
        history_enabled=False,
    )

    # First call should raise TimeoutExceededOnInitial and also kill the process
    with pytest.raises(ExceptionGroup) as exc_info:
        await state.run_for_result(b"hello")

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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # This should exercise the directory cleanup code
    exit_code = (await state.run_for_result({"a.txt": b"modified"})).exit_code
    assert exit_code == 0


# === Debug mode tests ===


async def test_run_for_result_debug_mode_timeout_on_first_call(tmp_path):
    """Test timeout handling in debug mode on first call.

    Exercises the timeout check in debug mode.
    """
    # Create a script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nsleep 0.5\nexit 0")
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
        history_enabled=False,
    )

    # First call in debug mode should raise TimeoutExceededOnInitial
    with pytest.raises(TimeoutExceededOnInitial) as exc_info:
        await state.run_for_result(b"hello", debug=True)

    assert exc_info.value.timeout == 0.1
    assert exc_info.value.runtime >= 0.1
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_result_debug_mode_dynamic_timeout(tmp_path):
    """Debug mode does not enforce or adapt timeouts on the first call."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # The user-specified timeout stays unset; timeouts are chosen by the
    # adaptive policy, which the debug path does not feed.
    assert state.timeout is None
    exit_code = (await state.run_for_result(b"hello", debug=True)).exit_code
    assert exit_code == 0
    assert state.timeout is None
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_result_dynamic_timeout_non_debug(tmp_path):
    """The first call's measured runtime feeds the adaptive timeout policy."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # With no user timeout, the first call runs under the calibration
    # timeout and its runtime is recorded by the adaptive policy.
    assert state.timeout is None
    exit_code = (await state.run_for_result(b"hello", debug=False)).exit_code
    assert exit_code == 0
    assert state.timeout is None
    # The fast measured runtime pulls the adaptive timeout down from the cap.
    policy = state.timeout_policy
    assert MIN_TIMEOUT <= policy.current_timeout() < policy.cap
    # first_call should be False after this
    assert state.first_call is False


async def test_run_for_result_debug_mode_captures_stdout(tmp_path):
    """Test that debug mode captures stdout output.

    Exercises the stdout capture in debug mode.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\necho 'hello from stdout'\nexit 0")
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
        history_enabled=False,
    )

    # Run in debug mode
    exit_code = (await state.run_for_result(b"hello", debug=True)).exit_code
    assert exit_code == 0

    # Check that stdout was captured
    assert "hello from stdout" in state._last_debug_output


async def test_run_for_result_debug_mode_captures_stderr(tmp_path):
    """Test that debug mode captures stderr output.

    Exercises the stderr capture in debug mode.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\necho 'error from stderr' >&2\nexit 0")
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
        history_enabled=False,
    )

    # Run in debug mode
    exit_code = (await state.run_for_result(b"hello", debug=True)).exit_code
    assert exit_code == 0

    # Check that stderr was captured
    assert "error from stderr" in state._last_debug_output


async def test_build_error_message_for_memory_limit(simple_state):
    exc = MemoryLimitExceededOnInitial(used=200 * 1024**2, limit=50 * 1024**2)
    message = await simple_state.build_error_message(exc)
    assert "memory" in message.lower()
    assert "--memory-limit" in message


async def test_build_error_message_includes_debug_output(tmp_path):
    """Test that build_error_message includes debug output.

    Exercises the debug output inclusion in build_error_message.
    """

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\necho 'diagnostic output' >&2\nexit 1")
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
        history_enabled=False,
    )

    # First, ensure first_call is set so we can trigger the right code path
    state.first_call = False
    state.initial_exit_code = 0

    # Create an InvalidInitialExample exception
    exc = InvalidInitialExample("Test error")

    # Build the error message (this calls run_for_result with debug=True internally)
    error_message = await state.build_error_message(exc)

    # The error message should include the captured debug output
    assert "diagnostic output" in error_message


async def test_build_error_message_includes_cwd_debug_output(tmp_path):
    """Test build_error_message includes debug output from cwd run."""

    # Create a counter file to track call count
    counter_file = tmp_path / ".call_count"
    counter_file.write_text("0")

    # Create a script that:
    # Call 1: run_for_result with debug=True (returns 1 to trigger first-call failure path)
    # Call 2: run_script_on_file with debug=False from cwd (returns 0 to trigger cwd success path)
    # Call 3: run_script_on_file with debug=True from cwd (produces output for error message)
    script = tmp_path / "test.sh"
    script.write_text(
        f"""#!/bin/sh
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
            history_enabled=False,
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
    script.write_text("#!/bin/sh\necho 'debug output' >&2\nexit 0")
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
        history_enabled=False,
    )

    # Bypass first_call logic
    state.first_call = False
    state.initial_exit_code = 0

    # Run the script - with volume=debug, stderr should NOT be subprocess.DEVNULL
    # We can verify this by checking the kwargs would have stderr=None
    # The actual stderr output would go to the parent process's stderr
    exit_code = (await state.run_for_result(b"hello")).exit_code
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # History manager should not be created
    assert state.history_manager is None
    # Output manager should not be created (no TUI, no history)
    assert state.output_manager is None


def test_state_with_history_enabled_creates_output_manager(tmp_path):
    """Test that history enabled creates an output manager for capturing output."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    # History manager should be created
    assert state.history_manager is not None
    # Output manager should be created for capturing output
    assert state.output_manager is not None


def test_get_last_captured_output_with_no_output_manager(tmp_path):
    """Test _get_last_captured_output returns None when output_manager is None."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,  # Disable history to not create output_manager
    )

    assert state._get_last_captured_output() is None


def test_get_last_captured_output_with_no_output_available(tmp_path):
    """Test _get_last_captured_output returns None when no output is available."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    # Output manager exists but has no output yet
    assert state.output_manager is not None
    assert state._get_last_captured_output() is None


def test_get_last_captured_output_returns_stored_output(tmp_path):
    """Test _get_last_captured_output returns the stored _last_test_output."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    # _get_last_captured_output returns _last_test_output (set during run_script_on_file)
    assert state._get_last_captured_output() is None

    # Directly set the stored output (simulating what run_script_on_file does)
    state._last_test_output = b"test output content"

    output = state._get_last_captured_output()
    assert output == b"test output content"


async def test_run_script_on_file_handles_output_oserror(tmp_path, monkeypatch):
    """Test run_script_on_file handles OSError when reading output file."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    assert state.output_manager is not None

    # Mock open to raise OSError when reading the output file back
    original_open = open
    output_dir = state.output_manager.output_dir

    def mock_open(path, *args, **kwargs):
        # Raise OSError when trying to read (rb) an output file
        path_str = str(path)
        if path_str.startswith(output_dir) and "rb" in args:
            raise OSError("Simulated read error")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open)

    # Run the script - should complete without raising OSError
    run_result = await state.run_script_on_file(
        str(target), debug=False, cwd=str(tmp_path)
    )
    assert run_result.exit_code == 0

    # The OSError was caught, so _last_test_output should be None
    assert state._last_test_output is None


def test_directory_state_get_test_case_bytes_returns_serialized(tmp_path):
    """Test that directory state returns serialized bytes for history recording."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    # Directory mode should return serialized bytes for test case
    test_case_bytes = state._get_test_case_bytes({"file.txt": b"content"})
    assert test_case_bytes is not None
    assert isinstance(test_case_bytes, bytes)

    # Should also return something for initial bytes
    initial_bytes = state._get_initial_bytes()
    assert initial_bytes is not None

    # Both should match for same content
    assert test_case_bytes == initial_bytes


def test_check_trivial_result_returns_error_message(tmp_path):
    """Test check_trivial_result returns error message for trivial results."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
    script.write_text("#!/bin/sh\nexit 0")
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
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
    script.write_text("#!/bin/sh\necho hello\nexit 0")
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
        history_enabled=False,  # No history, so no output_manager
    )

    # Should succeed and discard output
    run_result = await state.run_script_on_file(
        working=str(target),
        cwd=str(tmp_path),
        debug=False,
    )
    assert run_result.exit_code == 0


@pytest.mark.trio
async def test_volume_debug_without_history_or_output_manager(tmp_path):
    """Test debug mode inherits stderr when no output_manager is present."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\necho 'debug output' >&2\nexit 0")
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
        history_enabled=False,  # No history, so no output_manager
    )

    # With history disabled and debug mode, stderr should be inherited
    # stdout goes to DEVNULL, stderr inherited
    assert state.output_manager is None
    run_result = await state.run_script_on_file(
        working=str(target),
        cwd=str(tmp_path),
        debug=False,
    )
    assert run_result.exit_code == 0


@pytest.mark.trio
async def test_reducer_property_initializes_history(tmp_path):
    """Test that accessing reducer property initializes history when enabled."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
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
    script.write_text("#!/bin/sh\nexit 0")
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
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
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
async def test_history_callback_records_directory_mode(tmp_path):
    """Test that history callback records reductions for directory mode."""
    script = tmp_path / "test.sh"
    # Script that always says "interesting" (exit 0)
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
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

    # The callback SHOULD have recorded the reduction for directory mode
    assert state.history_manager.reduction_counter == 1

    # Verify the directory was saved
    # For directory mode, files are saved inside target_basename subdirectory
    reductions_dir = os.path.join(
        state.history_manager.history_dir, "reductions", "0001"
    )
    assert os.path.isdir(reductions_dir)
    target_subdir = os.path.join(reductions_dir, "target")
    assert os.path.isdir(target_subdir)
    saved_file = os.path.join(target_subdir, "file.txt")
    assert os.path.isfile(saved_file)
    with open(saved_file, "rb") as f:
        assert f.read() == b"hello"


# === also-interesting tests ===


@pytest.mark.trio
async def test_is_interesting_records_also_interesting_exit_code(tmp_path):
    """Test that is_interesting records test cases with also-interesting exit code."""
    script = tmp_path / "test.sh"
    # Script that returns 101 (also-interesting code)
    script.write_text("#!/bin/sh\nexit 101")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
        also_interesting_code=101,
    )

    # Initialize the reducer to set up history
    _ = state.reducer

    assert state.history_manager is not None
    assert state.history_manager.also_interesting_counter == 0

    # Call is_interesting - should return False but record the case
    result = await state.is_interesting(b"test content")

    # Exit code 101 is not 0, so not interesting for reduction
    assert result is False

    # But it should have been recorded as also-interesting
    assert state.history_manager.also_interesting_counter == 1

    # Verify the file was saved
    also_interesting_dir = os.path.join(
        state.history_manager.history_dir, "also-interesting", "0001"
    )
    assert os.path.isdir(also_interesting_dir)
    saved_file = os.path.join(also_interesting_dir, "test.txt")
    assert os.path.isfile(saved_file)
    with open(saved_file, "rb") as f:
        assert f.read() == b"test content"


@pytest.mark.trio
async def test_also_interesting_disabled_by_default(tmp_path):
    """Test that also-interesting is disabled when code is None."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 101")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
        also_interesting_code=None,  # Disabled
    )

    # Initialize the reducer to set up history
    _ = state.reducer

    assert state.history_manager is not None
    assert state.history_manager.also_interesting_counter == 0

    # Call is_interesting
    result = await state.is_interesting(b"test content")

    # Still not interesting (exit 101 != 0)
    assert result is False

    # But nothing should be recorded since also_interesting_code is None
    assert state.history_manager.also_interesting_counter == 0


@pytest.mark.trio
async def test_also_interesting_works_without_full_history(tmp_path):
    """Test that also-interesting works even when history_enabled=False."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 101")
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
        history_enabled=False,  # History disabled
        history_base_dir=str(tmp_path),  # But history_manager is still created
        also_interesting_code=101,  # But also-interesting is set
    )

    # History manager IS created when also_interesting_code is set
    assert state.history_manager is not None
    # But it won't record reductions
    assert state.history_manager.record_reductions is False

    # Initialize the reducer to set up history
    _ = state.reducer

    # Call is_interesting
    result = await state.is_interesting(b"test content")
    assert result is False

    # Also-interesting should be recorded
    assert state.history_manager.also_interesting_counter == 1


@pytest.mark.trio
async def test_no_history_manager_when_both_disabled(tmp_path):
    """Test that no history manager is created when both history and also-interesting are disabled."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 101")
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
        history_enabled=False,  # History disabled
        also_interesting_code=None,  # Also-interesting disabled
    )

    # No history manager when both are disabled
    assert state.history_manager is None

    # Should work without error
    result = await state.is_interesting(b"test content")
    assert result is False


@pytest.mark.trio
async def test_also_interesting_different_exit_code_not_recorded(tmp_path):
    """Test that non-matching exit codes are not recorded."""
    script = tmp_path / "test.sh"
    # Script returns 1, but also-interesting is 101
    script.write_text("#!/bin/sh\nexit 1")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
        also_interesting_code=101,  # Different from script's exit code
    )

    # Initialize the reducer to set up history
    _ = state.reducer

    assert state.history_manager is not None

    result = await state.is_interesting(b"test content")

    # Not interesting (exit 1 != 0)
    assert result is False

    # Also not also-interesting (exit 1 != 101)
    assert state.history_manager.also_interesting_counter == 0


@pytest.mark.trio
async def test_also_interesting_exit_code_zero_is_interesting_not_also(tmp_path):
    """Test that exit code 0 is interesting, not also-interesting."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
        also_interesting_code=101,
    )

    # Initialize the reducer to set up history
    _ = state.reducer

    assert state.history_manager is not None

    result = await state.is_interesting(b"test content")

    # Interesting (exit 0)
    assert result is True

    # Not also-interesting (0 != 101)
    assert state.history_manager.also_interesting_counter == 0


@pytest.mark.trio
async def test_also_interesting_records_directory_mode(tmp_path):
    """Test that also-interesting records for directory mode."""
    script = tmp_path / "test.sh"
    # Script that returns 101 (also-interesting code)
    script.write_text("#!/bin/sh\nexit 101")
    script.chmod(0o755)

    # Create a target directory with a file
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target_dir),
        timeout=5.0,
        base=target_dir.name,
        parallelism=1,
        initial={"file.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=True,
        history_base_dir=str(tmp_path),
        also_interesting_code=101,
    )

    assert state.history_manager is not None

    # Call is_interesting with a test case that returns 101
    result = await state.is_interesting({"file.txt": b"test"})

    # Not interesting (exit 101 != 0)
    assert result is False

    # SHOULD record for directory mode now
    assert state.history_manager.also_interesting_counter == 1

    # Verify the directory was saved
    # For directory mode, files are saved inside target_basename subdirectory
    also_interesting_dir = os.path.join(
        state.history_manager.history_dir, "also-interesting", "0001"
    )
    assert os.path.isdir(also_interesting_dir)
    target_subdir = os.path.join(also_interesting_dir, "target")
    assert os.path.isdir(target_subdir)
    saved_file = os.path.join(target_subdir, "file.txt")
    assert os.path.isfile(saved_file)
    with open(saved_file, "rb") as f:
        assert f.read() == b"test"


@pytest.mark.trio
async def test_history_counter_never_lags_stats_reductions(tmp_path):
    """Regression: history_manager.reduction_counter must be at least as large
    as problem.stats.reductions at every scheduling point.

    The progress update loop emits reductions=stats.reductions while
    record_reduction() populates the history directory. If the callback that
    writes to the history runs after a yielding callback (like the one that
    writes the test case to disk), the emit loop can observe reductions=N
    while the Nth reduction directory doesn't exist yet. A concurrent
    restart_from(N) request then fails with "Reduction N not found".
    """
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q KEEP "$1"')
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("KEEP\nremove\n")

    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="test.txt",
        parallelism=1,
        initial=b"KEEP\nremove\n",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )

    problem = state.reducer.target
    assert state.history_manager is not None
    history_manager = state.history_manager

    # Slow down the file-write callback so any scheduler task has a clear
    # opportunity to observe intermediate state.
    original_write_impl = state.write_test_case_to_file_impl

    async def slow_write(working, test_case):
        await trio.sleep(0.05)
        await original_write_impl(working, test_case)

    state.write_test_case_to_file_impl = slow_write  # type: ignore[method-assign]

    violations: list[tuple[int, int]] = []

    async def observer():
        while True:
            r = problem.stats.reductions
            c = history_manager.reduction_counter
            if r > c:
                violations.append((r, c))
            await trio.sleep(0)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(observer)
        # A smaller interesting test case triggers a reduction.
        assert await problem.is_interesting(b"KEEP\n") is True
        # Give the observer one more tick to confirm post-reduction state.
        await trio.sleep(0)
        nursery.cancel_scope.cancel()

    assert not violations, (
        f"stats.reductions ran ahead of history.reduction_counter: {violations}"
    )


# === reset_for_restart and excluded_test_cases tests ===


@pytest.mark.trio
async def test_excluded_test_cases_rejects_matching(tmp_path):
    """Test that excluded_test_cases causes is_interesting to return False."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Normally the test case would be interesting
    assert await state.is_interesting(b"hello") is True

    # Add to exclusion set
    state.excluded_test_cases = {b"excluded_value"}

    # The excluded value should be rejected
    assert await state.is_interesting(b"excluded_value") is False

    # Other values should still work
    assert await state.is_interesting(b"hello") is True


@pytest.mark.trio
async def test_reset_for_restart_clears_reducer(tmp_path):
    """Test that reset_for_restart clears the cached reducer."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Access the reducer to cache it
    reducer1 = state.reducer

    # Reset for restart with new initial
    state.reset_for_restart(b"world", {b"excluded"})

    # Accessing reducer should return a new instance
    reducer2 = state.reducer

    # Should be different instances
    assert reducer1 is not reducer2

    # Initial should be updated
    assert state.initial == b"world"

    # Exclusion set should be set
    assert state.excluded_test_cases == {b"excluded"}


@pytest.mark.trio
async def test_reset_for_restart_without_existing_reducer(tmp_path):
    """Test reset_for_restart when reducer hasn't been accessed yet."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Reset without accessing reducer first (should not raise)
    state.reset_for_restart(b"world", {b"excluded"})

    # Verify state was updated
    assert state.initial == b"world"
    assert state.excluded_test_cases == {b"excluded"}


@pytest.mark.trio
async def test_reset_for_restart_resets_initial_exit_code(tmp_path):
    """Test that reset_for_restart resets initial_exit_code to 0.

    This is a regression test for a bug where initial_exit_code kept its
    old value after restart, causing assertion failures in build_error_message
    when the assertion `assert self.initial_exit_code not in (None, 0)` was hit.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
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
        history_enabled=False,
    )

    # Set initial_exit_code to a non-zero value (simulating an earlier run
    # that had a different exit code scenario)
    state.initial_exit_code = 1

    # Reset for restart with new initial
    state.reset_for_restart(b"world", {b"excluded"})

    # initial_exit_code should be reset to 0 since the new initial
    # (from history) is known to be interesting
    assert state.initial_exit_code == 0


def test_directory_state_set_initial_for_restart_works(tmp_path):
    """Test that directory state can deserialize and set initial for restart."""
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target_dir),
        timeout=5.0,
        base=target_dir.name,
        parallelism=1,
        initial={"file.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )

    # Serialize new content
    new_content = {"file.txt": b"world", "other.txt": b"test"}
    serialized = serialize_directory(new_content)

    # Set initial for restart
    state._set_initial_for_restart(serialized)

    # Verify initial was updated
    assert state.initial == new_content


@pytest.mark.trio
async def test_directory_state_excluded_test_cases(tmp_path):
    """Test that excluded_test_cases works for directory state.

    This tests the base class is_interesting method which uses
    _get_test_case_bytes for comparison.
    """
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("hello")

    state = ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target_dir),
        timeout=5.0,
        base="target",
        parallelism=1,
        initial={"file.txt": b"hello"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )

    # Normally the test case would be interesting
    assert await state.is_interesting({"file.txt": b"hello"}) is True

    # Create exclusion set with serialized directory content
    excluded_content = {"file.txt": b"excluded"}
    excluded_serialized = serialize_directory(excluded_content)
    state.excluded_test_cases = {excluded_serialized}

    # The excluded value should be rejected
    assert await state.is_interesting({"file.txt": b"excluded"}) is False

    # Other values should still work
    assert await state.is_interesting({"file.txt": b"hello"}) is True


def test_directory_state_serialize_deserialize_roundtrip():
    """Test that directory serialization is reversible."""
    original = {
        "file.txt": b"hello world",
        "subdir/nested.py": b"print('test')",
        "binary.bin": bytes(range(256)),  # Binary content
    }

    serialized = serialize_directory(original)
    deserialized = deserialize_directory(serialized)

    assert deserialized == original


# === Working-file cleanup tests ===


def make_in_place_state(tmp_path, filename="reduced.cpp", initial=b"aaaa"):
    """Factory for an in-place single-file state whose interestingness
    test always succeeds, for exercising temp-file handling."""
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)
    target = tmp_path / filename
    target.write_bytes(initial)
    return ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=True,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base=filename,
        parallelism=1,
        initial=initial,
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )


def working_file_leftovers(directory, filename="reduced.cpp"):
    stem, ext = os.path.splitext(filename)
    pattern = re.compile(re.escape(stem) + r"-[0-9a-f]{32}" + re.escape(ext) + r"\Z")
    return [n for n in os.listdir(directory) if pattern.match(n)]


def test_in_place_run_cleans_up_its_working_file(tmp_path):
    state = make_in_place_state(tmp_path)
    trio.run(state.run_for_result, b"aaa")
    assert working_file_leftovers(tmp_path) == []


def test_in_place_working_file_cleanup_survives_cwd_change(tmp_path, monkeypatch):
    # The temp path must be absolute so cleanup is not defeated by the
    # process's working directory changing during the test call.
    other = tmp_path / "elsewhere"
    other.mkdir()
    state = make_in_place_state(tmp_path)

    original_run = state.run_script_on_file

    async def run_then_chdir(*args, **kwargs):
        result = await original_run(*args, **kwargs)
        os.chdir(other)  # simulate something moving cwd mid-flight
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(state, "run_script_on_file", run_then_chdir)
    trio.run(state.run_for_result, b"aaa")
    assert working_file_leftovers(tmp_path) == []


def test_stale_working_files_swept_on_construction(tmp_path):
    # A leftover from a previous killed run, matching the temp pattern.
    stale = tmp_path / ("reduced-" + "a" * 32 + ".cpp")
    stale.write_bytes(b"junk")
    # An unrelated file that merely starts the same way must be kept.
    keep = tmp_path / "reduced-notahash.cpp"
    keep.write_bytes(b"keep me")

    make_in_place_state(tmp_path)

    assert not stale.exists()
    assert keep.exists()


def test_sweep_pattern_none_for_non_in_place(tmp_path):
    state = make_in_place_state(tmp_path)
    object.__setattr__(state, "in_place", False)
    assert state.stale_working_file_pattern() is None


def test_sweep_pattern_none_for_basename_mode(tmp_path):
    state = make_in_place_state(tmp_path)
    object.__setattr__(state, "input_type", InputType.basename)
    assert state.stale_working_file_pattern() is None


def test_sweep_tolerates_missing_directory(tmp_path):
    state = make_in_place_state(tmp_path)
    # Point at a directory that does not exist; sweep must not raise.
    object.__setattr__(state, "filename", str(tmp_path / "gone" / "reduced.cpp"))
    state.sweep_stale_working_files()


def make_in_place_directory_state(tmp_path):
    """Factory for an in-place directory-mode state, which writes candidate
    *directories* named ``<base>-<hex>`` next to the target."""
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.txt").write_bytes(b"aaaa")
    return ShrinkRayDirectoryState(
        input_type=InputType.arg,
        in_place=True,
        test=[str(script)],
        filename=str(target),
        timeout=5.0,
        base="target",
        parallelism=1,
        initial={"a.txt": b"aaaa"},
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
    )


def test_sweep_removes_stale_candidate_directory(tmp_path):
    # In-place directory mode leaves candidate *directories* behind on a
    # hard kill. The sweep must remove them, not just plain files.
    state = make_in_place_directory_state(tmp_path)
    stale_dir = tmp_path / ("target-" + "a" * 32)
    stale_dir.mkdir()
    (stale_dir / "leftover.txt").write_bytes(b"junk")

    state.sweep_stale_working_files()

    assert not stale_dir.exists()


def test_sweep_tolerates_unlink_failure(tmp_path, monkeypatch):
    state = make_in_place_state(tmp_path)
    # Create the stale file after construction so the constructor's own
    # sweep doesn't remove it before we exercise the failure path.
    stale = tmp_path / ("reduced-" + "b" * 32 + ".cpp")
    stale.write_bytes(b"junk")

    def boom(path):
        raise OSError("nope")

    monkeypatch.setattr(os, "unlink", boom)
    # Must swallow the error rather than propagating it.
    state.sweep_stale_working_files()


# === successful-output pruning tests ===


def test_record_history_keeps_concurrent_better_candidate(tmp_path):
    """A candidate that is interesting and sorts better than the one just
    adopted must keep its captured output.

    Under parallelism two candidates A and B can both be interesting; A is
    adopted first and its record runs, then B (which sorts better) is
    adopted. If recording A discarded B's stored output, B's own history
    entry would be written without its test output and the output for the
    now-current test case would be lost.
    """
    state = make_in_place_state(tmp_path)
    # Build the reducer so state.problem (and its sort_key) is available.
    _ = state.reducer
    state.history_manager = MagicMock()

    adopted = b"bbb"
    better = b"aa"  # sorts before (shorter than) the adopted candidate
    assert state.problem.sort_key(better) < state.problem.sort_key(adopted)
    state._successful_outputs = {adopted: b"out-A", better: b"out-B"}
    state._successful_output_keys = {
        adopted: state.problem.sort_key(adopted),
        better: state.problem.sort_key(better),
    }

    state._record_reduction_history(adopted)

    # The better, still-adoptable candidate's output survives.
    assert state._successful_outputs.get(better) == b"out-B"
    # The adopted candidate's output is retained for the LLM prompts.
    assert state._successful_outputs.get(adopted) == b"out-A"


def test_record_history_prunes_losing_candidate(tmp_path):
    """A candidate that was interesting but sorts worse than the adopted
    one can never be adopted again, so its output is pruned."""
    state = make_in_place_state(tmp_path)
    _ = state.reducer
    state.history_manager = MagicMock()

    adopted = b"aa"
    loser = b"cccc"  # sorts after (longer than) the adopted candidate
    state._successful_outputs = {adopted: b"out-A", loser: b"out-L"}
    state._successful_output_keys = {
        adopted: state.problem.sort_key(adopted),
        loser: state.problem.sort_key(loser),
    }

    state._record_reduction_history(adopted)

    assert loser not in state._successful_outputs
    assert loser not in state._successful_output_keys
    assert state._successful_outputs.get(adopted) == b"out-A"


# === adaptive timeout integration tests ===


def make_adaptive_state(
    tmp_path,
    script_body,
    *,
    timeout=5.0,
    min_timeout=0.1,
    initial=b"hello world",
    clock=None,
):
    script = tmp_path / "adaptive_test.sh"
    script.write_text(script_body)
    script.chmod(0o755)

    target = tmp_path / "adaptive_target.txt"
    target.write_bytes(initial)

    if clock is None:
        policy = AdaptiveTimeoutPolicy(user_timeout=timeout, min_timeout=min_timeout)
    else:
        policy = AdaptiveTimeoutPolicy(
            user_timeout=timeout, min_timeout=min_timeout, clock=clock
        )

    return ShrinkRayStateSingleFile(
        input_type=InputType.all,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=timeout,
        base="adaptive_target.txt",
        parallelism=1,
        initial=initial,
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        timeout_policy=policy,
    )


def test_state_creates_timeout_policy_from_user_timeout(simple_state):
    assert simple_state.timeout_policy.cap == simple_state.timeout


async def test_fast_completions_pull_timeout_down(tmp_path):
    # A generous cap so that even a heavily loaded machine (slow script
    # startup inflates the measured runtime) stays well below it.
    state = make_adaptive_state(
        tmp_path, "#!/bin/sh\nexit 0", timeout=60.0, min_timeout=0.2
    )
    result = await state.run_for_result(b"hello")
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.timeout_used is None
    # The fast run pulls the adaptive timeout down well below the cap
    # (how far depends on the measured runtime).
    policy = state.timeout_policy
    assert 0.2 <= policy.current_timeout() < policy.cap


async def test_timed_out_run_is_recorded_in_policy(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nsleep 5", min_timeout=0.1)
    policy = state.timeout_policy
    # Skip first-call calibration and adapt the timeout down so the test
    # runs quickly.
    state.first_call = False
    policy.record_completion(0.02, interesting=True)
    expected_timeout = policy.current_timeout()
    assert expected_timeout < 1.0

    result = await state.run_for_result(b"hello")
    assert result.timed_out
    assert result.exit_code != 0
    assert result.timeout_used == pytest.approx(expected_timeout)
    assert policy.recent_timeout_rate == pytest.approx(0.5)


async def test_timed_out_results_are_conditionally_cached(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nsleep 5", min_timeout=0.1)
    policy = state.timeout_policy
    state.first_call = False
    policy.record_completion(0.02, interesting=True)

    outcome = await state.check_interesting(b"hello")
    assert not outcome.interesting
    assert outcome.cache_valid is not None
    assert outcome.cache_valid()
    # Raising the timeout invalidates the cached result.
    assert policy.attempt_unstick()
    assert not outcome.cache_valid()


async def test_completed_uninteresting_results_cached_unconditionally(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nexit 1")
    outcome = await state.check_interesting(b"hello")
    assert not outcome.interesting
    assert outcome.cache_valid is None


async def test_interesting_results_cached_unconditionally(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nexit 0")
    outcome = await state.check_interesting(b"hello")
    assert outcome.interesting
    assert outcome.cache_valid is None


async def test_problem_unstick_raises_policy_timeout(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nexit 0", min_timeout=0.1)
    policy = state.timeout_policy
    state.first_call = False
    policy.record_completion(0.02, interesting=True)
    policy.record_timeout(policy.current_timeout())
    before = policy.current_timeout()

    problem = state.problem
    assert await problem.attempt_unstick()
    assert policy.current_timeout() == 2 * before

    # Repeated unsticking climbs to the cap, then gives up and reverts.
    while await problem.attempt_unstick():
        assert policy.current_timeout() <= policy.cap
    assert policy.current_timeout() == before


async def test_reduction_notes_progress_to_policy(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nexit 0")
    problem = state.problem
    with patch.object(
        state.timeout_policy,
        "note_reduction",
        wraps=state.timeout_policy.note_reduction,
    ) as note:
        assert await problem.is_interesting(b"hello")
        note.assert_called_once()


def test_reset_for_restart_resets_timeout_policy(tmp_path):
    state = make_adaptive_state(tmp_path, "#!/bin/sh\nexit 0")
    policy = state.timeout_policy
    policy.record_completion(0.02, interesting=True)
    assert policy.current_timeout() < policy.cap
    state.reset_for_restart(b"new", set())
    assert policy.current_timeout() == policy.cap


@pytest.mark.slow
async def test_adaptive_timeout_unlocks_slow_reduction(tmp_path):
    """End-to-end: a reduction whose interesting form is much slower than
    the adapted timeout is still found, because the reducer raises the
    timeout when it runs out of things to try (attempt_unstick).

    The policy is given a frozen clock so the *automatic*, wall-clock-driven
    stall exploration never fires: on a slow/loaded machine it would kick in
    unpredictably (the reduction stalling past STALL_MIN_SECONDS) and could
    exhaust itself on unrelated candidates, which made this test flaky. With
    the frozen clock the raise is driven purely by attempt_unstick, which is
    exactly the "raise before giving up" behaviour under test, and the real
    subprocess timing (the 0.4s sleep vs the real timeout kills) is
    unchanged."""
    script_body = """#!/bin/sh
content=$(cat "$1")
if [ "$content" = "hello world" ]; then exit 0; fi
if [ "$content" = "hello" ]; then sleep 0.4; exit 0; fi
exit 1
"""
    state = make_adaptive_state(
        tmp_path, script_body, timeout=10.0, min_timeout=0.15, clock=lambda: 0.0
    )
    await state.problem.setup()
    await state.reducer.run()
    assert state.problem.current_test_case == b"hello"


def _history_output_state(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\necho "still interesting"\ngrep -q hello "$1"\n')
    script.chmod(0o755)
    target = tmp_path / "test.txt"
    target.write_text("hello world")
    return ShrinkRayStateSingleFile(
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
        history_enabled=True,
        history_base_dir=str(tmp_path),
    )


async def test_adopted_reductions_keep_their_test_output(tmp_path):
    # The adopted test case's output stays available (the LLM passes put it
    # in their prompts); outputs of other interesting candidates are pruned.
    state = _history_output_state(tmp_path)
    _ = state.reducer  # registers the history callback
    problem = state.problem
    await problem.setup()
    assert await problem.is_interesting(b"hello")
    assert list(state._successful_outputs) == [b"hello"]
    output = state._successful_outputs[b"hello"]
    assert b"still interesting" in output


async def test_history_records_reductions_without_captured_output(
    tmp_path, monkeypatch
):
    state = _history_output_state(tmp_path)
    monkeypatch.setattr(state, "_get_last_captured_output", lambda: None)
    _ = state.reducer
    problem = state.problem
    await problem.setup()
    assert await problem.is_interesting(b"hello")
    assert state._successful_outputs == {}


# === Auto-disable of the default memory limit on the initial test ===
#
# Sanitizer builds (ASan/MSan/TSan) reserve tens of terabytes of virtual
# address space and abort under ANY RLIMIT_AS, so the physical-RAM default
# memory limit fails them out of the box. When the limit was not set by the
# user and the initial test fails, shrink ray re-runs it once without the cap;
# if it then passes the cap was the culprit, so the limit is disabled for the
# rest of the run.
#
# A real `ulimit`-based reproduction only works where the platform's
# memory-limit ulimit (RLIMIT_AS via `-v` on Linux, RLIMIT_DATA via `-d` on
# OpenBSD) is actually enforced and reads "unlimited" when unset — see
# _real_ulimit_reproducible below. For a deterministic, cross-platform test we
# patch the one platform-dependent piece — memory_limited_command, the wrapper
# that applies the cap — to simulate a test that aborts under any cap, while
# the real subprocess machinery (and its first-call bookkeeping) runs
# unchanged. A real end-to-end test guarded on _real_ulimit_reproducible
# follows.


def _real_ulimit_reproducible() -> bool:
    """Whether the real-ulimit end-to-end test can run on this platform.

    It needs the memory-limit ulimit to be enforced and to read "unlimited"
    when unset, so that applying the default limit is observable and removing
    it restores "unlimited". True on Linux (`-v`); false on macOS (not
    enforced) and on any platform that already imposes a default cap on the
    relevant resource (e.g. OpenBSD's data-segment limit).
    """
    if not MEMORY_LIMIT_ENFORCEABLE:
        return False
    result = subprocess.run(
        ["/bin/sh", "-c", f"ulimit {_ULIMIT_FLAG}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "unlimited"


def _fake_cap_aborts(command, memory_limit):
    """Stand-in for memory_limited_command that models a sanitizer build.

    Under any positive cap the wrapped command aborts (exit 137, as a killed
    process would); with no cap the real command runs untouched.
    """
    if memory_limit:
        return ["/bin/sh", "-c", "exit 137"]
    return command


def _memory_probe_state(tmp_path, *, script_body, memory_limit_explicit):
    script = tmp_path / "test.sh"
    script.write_text(script_body)
    script.chmod(0o755)
    target = tmp_path / "target.txt"
    target.write_text("hello")
    return ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=30.0,
        base="target.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        # Mirror the real default: an unset limit is the machine's physical RAM.
        memory_limit=default_memory_limit(),
        memory_limit_explicit=memory_limit_explicit,
    )


@pytest.mark.parametrize(
    "explicit,limit,expected",
    [
        pytest.param(False, 1024, True, id="default_limit_probes"),
        pytest.param(True, 1024, False, id="explicit_limit_left_alone"),
        pytest.param(False, None, False, id="no_limit_nothing_to_probe"),
        pytest.param(False, 0, False, id="disabled_limit_nothing_to_probe"),
    ],
)
def test_default_memory_limit_may_block_initial(
    simple_state, explicit, limit, expected
):
    simple_state.memory_limit_explicit = explicit
    simple_state.memory_limit = limit
    assert simple_state._default_memory_limit_may_block_initial() is expected


async def test_default_memory_limit_auto_disabled_when_it_blocks_initial(
    tmp_path, capsys
):
    # Passes without a cap, "aborts" under one: the sanitizer situation.
    state = _memory_probe_state(
        tmp_path, script_body="#!/bin/sh\nexit 0\n", memory_limit_explicit=False
    )
    with patch.object(state_mod, "memory_limited_command", _fake_cap_aborts):
        await state.problem.setup()  # must not raise
    # The default limit was disabled for the rest of the run.
    assert state.memory_limit is None
    # The single logical initial call is the no-limit one, which passed.
    assert state.initial_exit_code == 0
    assert state.first_call is False
    err = capsys.readouterr().err
    assert "--memory-limit" in err
    assert "sanitizer" in err.lower()


async def test_explicit_memory_limit_not_auto_disabled(tmp_path, capsys):
    limit = default_memory_limit()
    state = _memory_probe_state(
        tmp_path, script_body="#!/bin/sh\nexit 0\n", memory_limit_explicit=True
    )
    with patch.object(state_mod, "memory_limited_command", _fake_cap_aborts):
        with pytest.raises(InvalidInitialExample) as exc_info:
            await state.problem.setup()
        # The limit is untouched (no auto-disable, no probe).
        assert state.memory_limit == limit
        # The error explanation points at --memory-limit=0 as a possible fix.
        message = await state.build_error_message(exc_info.value)
    assert "--memory-limit=0" in message
    # No auto-disable warning was printed.
    assert "only passed with the" not in capsys.readouterr().err


async def test_default_memory_limit_kept_when_failure_is_genuine(tmp_path, capsys):
    limit = default_memory_limit()
    # Fails with or without a cap: a genuinely uninteresting initial test.
    state = _memory_probe_state(
        tmp_path, script_body="#!/bin/sh\nexit 3\n", memory_limit_explicit=False
    )
    with patch.object(state_mod, "memory_limited_command", _fake_cap_aborts):
        with pytest.raises(InvalidInitialExample):
            await state.problem.setup()
        # The no-limit retry also failed, so the limit is restored.
        assert state.memory_limit == limit
        # The retry's genuine exit code is what gets recorded.
        assert state.initial_exit_code == 3
        # A non-explicit limit that was ruled out does not suggest
        # --memory-limit=0 (it would be misleading).
        message = await state.build_error_message(
            InvalidInitialExample("uninteresting")
        )
    assert "--memory-limit=0" not in message
    assert "only passed with the" not in capsys.readouterr().err


async def test_interesting_initial_under_default_limit_runs_once(tmp_path, capsys):
    # A normally-interesting test must not trigger the probe or any extra run.
    counter = tmp_path / "runs"
    counter.write_text("")
    script = tmp_path / "test.sh"
    script.write_text(f"#!/bin/sh\nprintf x >> {counter}\nexit 0\n")
    script.chmod(0o755)
    target = tmp_path / "target.txt"
    target.write_text("hello")
    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=30.0,
        base="target.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        memory_limit=default_memory_limit(),
        memory_limit_explicit=False,
    )
    await state.problem.setup()
    # Exactly one run: no no-limit retry was needed.
    assert counter.read_text() == "x"
    assert state.memory_limit == default_memory_limit()
    assert "only passed with the" not in capsys.readouterr().err


@pytest.mark.skipif(
    not _real_ulimit_reproducible(),
    reason="the platform's memory-limit ulimit is not enforced or not "
    "'unlimited' when unset (e.g. macOS does not enforce it; OpenBSD caps "
    "the data segment)",
)
async def test_default_memory_limit_auto_disabled_real_ulimit(tmp_path, capsys):
    # End-to-end with a real memory cap: the test is interesting only when the
    # platform's memory-limit ulimit is unlimited, exactly reproducing the
    # sanitizer case.
    script = tmp_path / "test.sh"
    script.write_text(f'#!/bin/sh\n[ "$(ulimit {_ULIMIT_FLAG})" = unlimited ]\n')
    script.chmod(0o755)
    target = tmp_path / "target.txt"
    target.write_text("hello")
    state = ShrinkRayStateSingleFile(
        input_type=InputType.arg,
        in_place=False,
        test=[str(script)],
        filename=str(target),
        timeout=30.0,
        base="target.txt",
        parallelism=1,
        initial=b"hello",
        formatter="none",
        trivial_is_error=True,
        seed=0,
        volume=Volume.quiet,
        history_enabled=False,
        memory_limit=default_memory_limit(),
        memory_limit_explicit=False,
    )
    await state.problem.setup()  # must not raise
    assert state.memory_limit is None
    assert "--memory-limit" in capsys.readouterr().err
