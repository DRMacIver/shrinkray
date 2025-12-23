"""Tests for state management."""

import pytest

from shrinkray.cli import InputType
from shrinkray.state import (
    ShrinkRayDirectoryState,
    ShrinkRayStateSingleFile,
    TimeoutExceededOnInitial,
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
    assert "sort_key" in kwargs


def test_directory_state_size_function(directory_state):
    kwargs = directory_state.extra_problem_kwargs
    size_fn = kwargs["size"]
    test_case = {"a.txt": b"hello", "b.txt": b"world!"}
    assert size_fn(test_case) == 11  # 5 + 6


def test_directory_state_sort_key_function(directory_state):
    kwargs = directory_state.extra_problem_kwargs
    sort_key_fn = kwargs["sort_key"]

    tc1 = {"a.txt": b"hi"}
    tc2 = {"a.txt": b"hello"}
    tc3 = {"a.txt": b"hi", "b.txt": b"x"}

    # Fewer files should come first
    assert sort_key_fn(tc1) < sort_key_fn(tc3)
    # Smaller total size should come first
    assert sort_key_fn(tc1) < sort_key_fn(tc2)


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
    with pytest.raises(AssertionError):
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

    import trio

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
