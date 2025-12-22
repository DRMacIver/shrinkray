"""Tests for state management."""


import pytest

from shrinkray.cli import InputType
from shrinkray.state import (
    ShrinkRayDirectoryState,
    ShrinkRayStateSingleFile,
    TimeoutExceededOnInitial,
)
from shrinkray.work import Volume


class TestTimeoutExceededOnInitial:
    """Tests for TimeoutExceededOnInitial exception."""

    def test_stores_runtime_and_timeout(self):
        exc = TimeoutExceededOnInitial(runtime=5.5, timeout=2.0)
        assert exc.runtime == 5.5
        assert exc.timeout == 2.0

    def test_message_includes_timeout(self):
        exc = TimeoutExceededOnInitial(runtime=5.5, timeout=2.0)
        assert "2.0s" in str(exc)
        assert "timeout" in str(exc).lower()


class TestShrinkRayStateSingleFile:
    """Tests for ShrinkRayStateSingleFile class."""

    @pytest.fixture
    def simple_state(self, tmp_path):
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

    def test_creates_reducer(self, simple_state):
        reducer = simple_state.reducer
        assert reducer is not None

    def test_reducer_is_cached(self, simple_state):
        reducer1 = simple_state.reducer
        reducer2 = simple_state.reducer
        assert reducer1 is reducer2

    def test_problem_property(self, simple_state):
        problem = simple_state.problem
        assert problem is not None
        assert problem.current_test_case == b"hello world"

    async def test_is_interesting_tracks_parallel_tasks(self, simple_state):
        # Before any calls
        assert simple_state.parallel_tasks_running == 0

        # During a call, the counter should be incremented
        # (we can't easily test this without more complex setup)

    async def test_write_test_case_to_file(self, tmp_path, simple_state):
        target = tmp_path / "output.txt"
        await simple_state.write_test_case_to_file(str(target), b"test data")
        assert target.read_bytes() == b"test data"

    async def test_format_data_with_none_formatter(self, simple_state):
        # With formatter="none", format_data should return the input unchanged
        result = await simple_state.format_data(b"test data")
        assert result == b"test data"

    async def test_run_formatter_command(self, simple_state):
        # Test running a simple formatter command
        result = await simple_state.run_formatter_command(
            ["cat"], b"hello"
        )
        assert result.stdout == b"hello"
        assert result.returncode == 0


class TestShrinkRayDirectoryState:
    """Tests for ShrinkRayDirectoryState class."""

    @pytest.fixture
    def directory_state(self, tmp_path):
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

    def test_creates_reducer(self, directory_state):
        reducer = directory_state.reducer
        assert reducer is not None

    def test_extra_problem_kwargs_includes_size_and_sort_key(self, directory_state):
        kwargs = directory_state.extra_problem_kwargs
        assert "size" in kwargs
        assert "sort_key" in kwargs

    def test_size_function(self, directory_state):
        kwargs = directory_state.extra_problem_kwargs
        size_fn = kwargs["size"]
        test_case = {"a.txt": b"hello", "b.txt": b"world!"}
        assert size_fn(test_case) == 11  # 5 + 6

    def test_sort_key_function(self, directory_state):
        kwargs = directory_state.extra_problem_kwargs
        sort_key_fn = kwargs["sort_key"]

        tc1 = {"a.txt": b"hi"}
        tc2 = {"a.txt": b"hello"}
        tc3 = {"a.txt": b"hi", "b.txt": b"x"}

        # Fewer files should come first
        assert sort_key_fn(tc1) < sort_key_fn(tc3)
        # Smaller total size should come first
        assert sort_key_fn(tc1) < sort_key_fn(tc2)

    async def test_write_test_case_to_file_creates_directory(self, tmp_path, directory_state):
        target = tmp_path / "output_dir"
        test_case = {"sub/a.txt": b"content a", "b.txt": b"content b"}

        await directory_state.write_test_case_to_file(str(target), test_case)

        assert target.is_dir()
        assert (target / "sub" / "a.txt").read_bytes() == b"content a"
        assert (target / "b.txt").read_bytes() == b"content b"

    async def test_format_data_returns_none(self, directory_state):
        # Directory formatting is not implemented
        result = await directory_state.format_data({"a.txt": b"test"})
        assert result is None

    async def test_run_formatter_command_raises(self, directory_state):
        with pytest.raises(AssertionError):
            await directory_state.run_formatter_command(["cat"], {"a.txt": b"test"})


class TestAttemptFormat:
    """Tests for attempt_format method."""

    @pytest.fixture
    def state_with_formatter(self, tmp_path):
        """Create a state with a formatter that adds a newline."""
        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\nexit 0")
        script.chmod(0o755)

        target = tmp_path / "test.txt"
        target.write_text("hello")

        return ShrinkRayStateSingleFile(
            input_type=InputType.all,
            in_place=False,
            test=[str(script)],
            filename=str(target),
            timeout=5.0,
            base="test.txt",
            parallelism=1,
            initial=b"hello",
            formatter="default",
            trivial_is_error=True,
            seed=0,
            volume=Volume.quiet,
            clang_delta_executable=None,
        )

    async def test_returns_data_when_cannot_format(self, tmp_path):
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
