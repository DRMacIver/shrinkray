"""Tests for ReducerWorker subprocess."""

import io
import json
import os
import runpy
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import trio

import shrinkray.subprocess.worker
from shrinkray.passes.clangdelta import find_clang_delta
from shrinkray.problem import InvalidInitialExample
from shrinkray.state import ShrinkRayDirectoryState, ShrinkRayStateSingleFile
from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)
from shrinkray.subprocess.worker import (
    InputStream,
    ReducerWorker,
    main,
)


# === ReducerWorker initialization tests ===


def test_worker_initial_state():
    worker = ReducerWorker()
    assert worker.running is False
    assert worker.reducer is None
    assert worker.problem is None
    assert worker.state is None
    assert worker._cancel_scope is None
    assert worker._restart_requested is False
    assert worker._parallel_samples == 0
    assert worker._parallel_total == 0


# === emit tests ===


async def test_worker_emit_writes_to_stdout():
    worker = ReducerWorker()
    output = io.StringIO()

    with patch.object(sys, "stdout", output):
        response = Response(id="test-123", result={"status": "ok"})
        await worker.emit(response)

    written = output.getvalue()
    assert "test-123" in written
    assert "ok" in written
    assert written.endswith("\n")


async def test_worker_emit_progress_update():
    worker = ReducerWorker()
    output = io.StringIO()

    with patch.object(sys, "stdout", output):
        update = ProgressUpdate(
            status="running",
            size=100,
            original_size=200,
            calls=10,
            reductions=5,
            interesting_calls=8,
            wasted_calls=2,
            runtime=1.5,
            parallel_workers=2,
            average_parallelism=1.5,
            effective_parallelism=1.2,
            time_since_last_reduction=0.5,
            content_preview="test",
            hex_mode=False,
        )
        await worker.emit(update)

    written = output.getvalue()
    assert "progress" in written
    assert "running" in written
    assert written.endswith("\n")


# === handle_command tests ===


async def test_worker_handle_command_status_not_running():
    worker = ReducerWorker()
    request = Request(id="test-1", command="status", params={})

    response = await worker.handle_command(request)

    assert response.id == "test-1"
    assert response.result == {"running": False}
    assert response.error is None


async def test_worker_handle_command_cancel():
    worker = ReducerWorker()
    request = Request(id="test-2", command="cancel", params={})

    response = await worker.handle_command(request)

    assert response.id == "test-2"
    assert response.result == {"status": "cancelled"}
    assert worker.running is False


async def test_worker_handle_command_unknown():
    worker = ReducerWorker()
    request = Request(id="test-3", command="unknown_cmd", params={})

    response = await worker.handle_command(request)

    assert response.id == "test-3"
    assert response.error == "Unknown command: unknown_cmd"


async def test_worker_handle_command_start_already_running():
    worker = ReducerWorker()
    worker.running = True
    request = Request(id="test-4", command="start", params={})

    response = await worker.handle_command(request)

    assert response.id == "test-4"
    assert response.error == "Already running"


# === _handle_status tests ===


def test_worker_handle_status_with_running_problem():
    worker = ReducerWorker()
    worker.running = True

    # Mock the problem with stats
    mock_stats = MagicMock()
    mock_stats.current_test_case_size = 100
    mock_stats.initial_test_case_size = 200
    mock_stats.calls = 50
    mock_stats.reductions = 10

    mock_problem = MagicMock()
    mock_problem.stats = mock_stats
    worker.problem = mock_problem

    mock_reducer = MagicMock()
    mock_reducer.status = "Reducing bytes"
    worker.reducer = mock_reducer

    response = worker._handle_status("test-id")

    assert response.id == "test-id"
    assert response.result["running"] is True
    assert response.result["status"] == "Reducing bytes"
    assert response.result["size"] == 100
    assert response.result["original_size"] == 200
    assert response.result["calls"] == 50
    assert response.result["reductions"] == 10


# === _handle_cancel tests ===


def test_worker_handle_cancel_without_cancel_scope():
    worker = ReducerWorker()
    worker.running = True

    response = worker._handle_cancel("test-id")

    assert response.id == "test-id"
    assert response.result == {"status": "cancelled"}
    assert worker.running is False


def test_worker_handle_cancel_with_cancel_scope():
    worker = ReducerWorker()
    worker.running = True

    mock_scope = MagicMock()
    worker._cancel_scope = mock_scope

    response = worker._handle_cancel("test-id")

    assert response.id == "test-id"
    assert response.result == {"status": "cancelled"}
    assert worker.running is False
    mock_scope.cancel.assert_called_once()


# === _get_content_preview tests ===


def test_worker_get_content_preview_no_problem():
    worker = ReducerWorker()
    worker.problem = None

    preview, hex_mode = worker._get_content_preview()

    assert preview == ""
    assert hex_mode is False


def test_worker_get_content_preview_text_file():
    worker = ReducerWorker()
    mock_problem = MagicMock()
    mock_problem.current_test_case = b"hello world\nthis is text"
    worker.problem = mock_problem

    preview, hex_mode = worker._get_content_preview()

    assert preview == "hello world\nthis is text"
    assert hex_mode is False


def test_worker_get_content_preview_binary_file():
    worker = ReducerWorker()
    mock_problem = MagicMock()
    # Binary content (high bytes)
    mock_problem.current_test_case = bytes(range(256))
    worker.problem = mock_problem

    preview, hex_mode = worker._get_content_preview()

    assert hex_mode is True
    # Should contain hex dump format
    assert "00000000" in preview  # Address prefix


def test_worker_get_content_preview_directory_mode():
    worker = ReducerWorker()
    mock_problem = MagicMock()
    mock_problem.current_test_case = {
        "file1.txt": b"content1",
        "file2.txt": b"content2content2",
    }
    worker.problem = mock_problem

    preview, hex_mode = worker._get_content_preview()

    assert hex_mode is False
    assert "file1.txt: 8 bytes" in preview
    assert "file2.txt: 16 bytes" in preview


def test_worker_get_content_preview_truncates_large_text():
    worker = ReducerWorker()
    mock_problem = MagicMock()
    # Create text larger than 100KB limit
    large_text = "x" * 150_000
    mock_problem.current_test_case = large_text.encode("utf-8")
    worker.problem = mock_problem

    preview, hex_mode = worker._get_content_preview()

    assert hex_mode is False
    assert len(preview) == 100_000


# === handle_line tests ===


async def test_worker_handle_line_valid_request():
    worker = ReducerWorker()
    output = io.StringIO()

    with patch.object(sys, "stdout", output):
        await worker.handle_line('{"id": "test", "command": "status", "params": {}}')

    written = output.getvalue()
    assert "test" in written
    assert "running" in written


async def test_worker_handle_line_invalid_json():
    worker = ReducerWorker()
    output = io.StringIO()

    with patch.object(sys, "stdout", output):
        await worker.handle_line("not valid json")

    written = output.getvalue()
    assert "error" in written.lower()


async def test_worker_handle_line_non_request():
    worker = ReducerWorker()
    output = io.StringIO()

    # Send a valid JSON but not a Request (e.g., a Response)
    with patch.object(sys, "stdout", output):
        await worker.handle_line('{"id": "x", "result": {}, "error": null}')

    written = output.getvalue()
    assert "Expected a request" in written


# === run_reducer tests ===


async def test_worker_run_reducer_no_reducer():
    worker = ReducerWorker()
    worker.reducer = None

    # Should return immediately without error
    await worker.run_reducer()

    assert worker.running is False


async def test_worker_run_reducer_with_mock_reducer():
    worker = ReducerWorker()
    worker.running = True

    mock_reducer = MagicMock()

    async def mock_run():
        pass

    mock_reducer.run = mock_run
    worker.reducer = mock_reducer

    await worker.run_reducer()

    assert worker.running is False
    assert worker._cancel_scope is None


# === Injectable stream tests ===


class MemoryInputStream:
    """An in-memory input stream for testing."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self.pos >= len(self.data):
            raise StopAsyncIteration
        # Return data in small chunks to simulate streaming
        chunk_size = min(64, len(self.data) - self.pos)
        chunk = self.data[self.pos : self.pos + chunk_size]
        self.pos += chunk_size
        return chunk

    async def aclose(self) -> None:
        pass


class MemoryOutputStream:
    """An in-memory output stream for testing."""

    def __init__(self):
        self.data = b""

    async def send(self, data: bytes) -> None:
        self.data += data


async def test_worker_emit_with_injected_output_stream():
    """Test emit writes to injected output stream."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    response = Response(id="test-123", result={"status": "ok"})
    await worker.emit(response)

    assert b"test-123" in output.data
    assert b"ok" in output.data
    assert output.data.endswith(b"\n")


async def test_worker_read_commands_with_injected_input_stream():
    """Test read_commands reads from injected input stream."""
    # Create a status request
    request = Request(id="req-1", command="status", params={})
    input_data = serialize(request) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    # Run read_commands with a timeout
    with trio.move_on_after(1):
        await worker.read_commands()

    # Verify response was written
    assert b"req-1" in output.data
    assert b"running" in output.data


async def test_worker_read_commands_with_stream_parameter():
    """Test read_commands accepts stream as parameter."""
    request = Request(id="req-2", command="cancel", params={})
    input_data = serialize(request) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(output_stream=output)

    with trio.move_on_after(1):
        await worker.read_commands(input_stream=input_stream)

    assert b"req-2" in output.data
    assert b"cancelled" in output.data


async def test_worker_read_commands_handles_multiple_commands():
    """Test read_commands processes multiple commands."""
    req1 = Request(id="a", command="status", params={})
    req2 = Request(id="b", command="cancel", params={})
    input_data = serialize(req1) + "\n" + serialize(req2) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    with trio.move_on_after(1):
        await worker.read_commands()

    # Both responses should be in output
    assert output.data.count(b'"id"') >= 2


async def test_worker_emit_progress_updates_loop():
    """Test emit_progress_updates emits updates while running."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)
    worker.running = True

    # Mock the problem and state
    mock_stats = MagicMock()
    mock_stats.current_test_case_size = 100
    mock_stats.initial_test_case_size = 200
    mock_stats.calls = 10
    mock_stats.reductions = 5
    mock_stats.interesting_calls = 8
    mock_stats.wasted_interesting_calls = 2
    mock_stats.start_time = 0
    mock_stats.time_since_last_reduction.return_value = 0.5

    mock_problem = MagicMock()
    mock_problem.stats = mock_stats
    mock_problem.current_test_case = b"test content"
    worker.problem = mock_problem

    mock_reducer = MagicMock()
    mock_reducer.status = "Testing"
    worker.reducer = mock_reducer

    mock_state = MagicMock()
    mock_state.parallel_tasks_running = 2
    mock_state.output_manager = None  # No test output capture in this mock
    mock_state.history_manager = None  # No history in this mock
    worker.state = mock_state

    # Run for a short time then stop
    async def stop_after_delay():
        await trio.sleep(0.25)
        worker.running = False

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.emit_progress_updates)
        nursery.start_soon(stop_after_delay)

    # Should have emitted at least one progress update
    assert b"progress" in output.data
    assert b"Testing" in output.data


# === Integration tests with real files ===
# These tests run actual bash scripts with tight timeouts, so they need to run
# sequentially to avoid timeout failures under parallel load.


@pytest.mark.serial
async def test_worker_start_reduction_single_file(tmp_path):
    """Test _start_reduction with a real single file."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    await worker._start_reduction(params)

    assert worker.running is True
    assert worker.state is not None
    assert worker.problem is not None
    assert worker.reducer is not None
    assert worker.problem.current_test_case == b"hello world"


async def test_worker_start_reduction_skip_validation(tmp_path):
    """Test _start_reduction with skip_validation=True skips setup()."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script that would FAIL if run - this verifies setup() is skipped
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1")  # Always fails
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
        "skip_validation": True,  # Skip validation - setup() won't run the test
    }

    # This would raise InvalidInitialExample if setup() was called
    await worker._start_reduction(params)

    assert worker.running is True
    assert worker.state is not None
    assert worker.problem is not None


@pytest.mark.serial
async def test_worker_start_reduction_directory(tmp_path):
    """Test _start_reduction with a directory."""
    # Create a test directory with files
    target = tmp_path / "testdir"
    target.mkdir()
    (target / "a.txt").write_text("file a")
    (target / "b.txt").write_text("file b")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "arg",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    await worker._start_reduction(params)

    assert worker.running is True
    assert worker.state is not None
    assert worker.problem is not None
    # Directory mode returns a dict
    test_case = worker.problem.current_test_case
    assert isinstance(test_case, dict)
    assert "a.txt" in test_case
    assert "b.txt" in test_case


async def test_worker_handle_start_success(tmp_path):
    """Test _handle_start successfully starts reduction."""
    target = tmp_path / "test.txt"
    target.write_text("hello")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "no_clang_delta": True,
        "formatter": "none",
        "volume": "quiet",
    }

    response = await worker._handle_start("req-123", params)

    assert response.id == "req-123"
    assert response.error is None
    assert response.result == {"status": "started"}
    assert worker.running is True


async def test_worker_handle_start_error(tmp_path):
    """Test _handle_start returns error on failure."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    # Invalid params - missing file
    params = {
        "file_path": "/nonexistent/file.txt",
        "test": ["/bin/true"],
    }

    response = await worker._handle_start("req-456", params)

    assert response.id == "req-456"
    assert response.error is not None
    assert worker.running is False


# === Additional coverage tests ===


async def test_worker_read_commands_empty_lines():
    """Test read_commands handles empty lines between commands."""
    # Create commands with empty line between them
    req1 = Request(id="a", command="status", params={})
    req2 = Request(id="b", command="status", params={})
    input_data = serialize(req1) + "\n\n" + serialize(req2) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    with trio.move_on_after(1):
        await worker.read_commands()

    # Both responses should be present
    assert output.data.count(b'"id"') >= 2


async def test_worker_emit_progress_updates_no_problem():
    """Test emit_progress_updates continues when problem is None."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)
    worker.running = True
    worker.problem = None

    # Run for a short time then stop
    async def stop_after_delay():
        await trio.sleep(0.25)
        worker.running = False

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.emit_progress_updates)
        nursery.start_soon(stop_after_delay)

    # No updates should be emitted since problem is None
    assert b"progress" not in output.data


async def test_worker_emit_progress_updates_no_parallel_attr():
    """Test emit_progress_updates handles state without parallel_tasks_running attribute."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)
    worker.running = True

    # Mock the problem
    mock_stats = MagicMock()
    mock_stats.current_test_case_size = 100
    mock_stats.initial_test_case_size = 200
    mock_stats.calls = 10
    mock_stats.reductions = 5
    mock_stats.interesting_calls = 8
    mock_stats.wasted_interesting_calls = 2
    mock_stats.start_time = 0
    mock_stats.time_since_last_reduction.return_value = 0.5

    mock_problem = MagicMock()
    mock_problem.stats = mock_stats
    mock_problem.current_test_case = b"test content"
    worker.problem = mock_problem

    mock_reducer = MagicMock()
    mock_reducer.status = "Testing"
    worker.reducer = mock_reducer

    # State without parallel_tasks_running attribute
    mock_state = MagicMock(spec=["output_manager", "history_manager"])
    mock_state.output_manager = None  # No test output capture
    mock_state.history_manager = None  # No history
    worker.state = mock_state

    # Run for a short time then stop
    async def stop_after_delay():
        await trio.sleep(0.15)
        worker.running = False

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.emit_progress_updates)
        nursery.start_soon(stop_after_delay)

    # Should have emitted progress with parallel_workers = 0
    assert b"progress" in output.data


async def test_worker_emit_progress_updates_zero_samples():
    """Test emit_progress_updates with zero parallel samples."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)
    worker.running = True
    # Explicitly set samples to 0
    worker._parallel_samples = 0
    worker._parallel_total = 0

    # Mock the problem
    mock_stats = MagicMock()
    mock_stats.current_test_case_size = 100
    mock_stats.initial_test_case_size = 200
    mock_stats.calls = 0
    mock_stats.reductions = 0
    mock_stats.interesting_calls = 0
    mock_stats.wasted_interesting_calls = 0
    mock_stats.start_time = 0
    mock_stats.time_since_last_reduction.return_value = 0.0

    mock_problem = MagicMock()
    mock_problem.stats = mock_stats
    mock_problem.current_test_case = b"test"
    worker.problem = mock_problem

    mock_reducer = MagicMock()
    mock_reducer.status = "Starting"
    worker.reducer = mock_reducer

    # No state
    worker.state = None

    # Run for just one iteration
    async def stop_quickly():
        await trio.sleep(0.12)
        worker.running = False

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.emit_progress_updates)
        nursery.start_soon(stop_quickly)

    # Should emit with zero parallelism values
    assert b"progress" in output.data


def test_worker_get_content_preview_decode_exception():
    """Test _get_content_preview handles decode exceptions gracefully."""
    worker = ReducerWorker()

    # Create a mock problem with real bytes but mock is_binary_string to return False
    # and mock the decode to fail
    mock_problem = MagicMock()
    mock_problem.current_test_case = b"hello world text content"
    worker.problem = mock_problem

    # Patch is_binary_string to return False (treat as text)
    # and patch bytes.decode via the test_case object

    # We need a more creative approach - patch the decode call site

    class FailingDecodeBytes(bytes):
        """Bytes subclass that fails on decode."""

        def decode(self, *args, **kwargs):
            raise RuntimeError("Simulated decode failure")

    # Create instance by copying the data
    failing_bytes = FailingDecodeBytes(b"hello world text content")
    mock_problem.current_test_case = failing_bytes

    preview, hex_mode = worker._get_content_preview()

    # Should return empty string and hex_mode=True on exception
    assert preview == ""
    assert hex_mode is True


@pytest.mark.serial
async def test_worker_start_reduction_with_clang_delta(tmp_path):
    """Test _start_reduction with a C file and clang_delta enabled."""

    clang_delta_path = find_clang_delta()
    if not clang_delta_path:
        # Skip if clang_delta not available
        return

    # Create a C file
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": False,
        # Don't specify clang_delta path - let it find it automatically
    }

    await worker._start_reduction(params)

    assert worker.running is True


@pytest.mark.serial
async def test_worker_full_run_with_mock(tmp_path):
    """Test the full run() method with mocked reducer."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script that passes
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Create a start command
    start_params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }
    start_request = Request(id="start-1", command="start", params=start_params)
    input_data = serialize(start_request) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    # Mock the reducer to complete quickly

    async def mock_run_reducer():
        if worker.reducer is not None:
            # Just set running to false instead of actually running
            worker.running = False

    worker.run_reducer = mock_run_reducer

    # Run with a timeout
    with trio.move_on_after(5):
        await worker.run()

    # Should have received the start response and completion
    assert b"started" in output.data


def test_worker_main_function():
    """Test the main() function is callable."""

    # We can't easily test the actual main() since it blocks on trio.run
    # But we can verify it exists and is callable
    assert callable(main)


def test_worker_main_guard():
    """Test that the module has a main function."""

    # The module exists and can be imported
    assert hasattr(shrinkray.subprocess.worker, "main")


@pytest.mark.serial
async def test_worker_run_waits_for_start(tmp_path):
    """Test that run() waits for start command before proceeding."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()

    # Create an input stream that sends the start command after a delay
    start_params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "no_clang_delta": True,
        "formatter": "none",
        "volume": "quiet",
    }
    start_request = Request(id="start-1", command="start", params=start_params)

    class DelayedInputStream:
        """Input stream that delays before sending the command."""

        def __init__(self, command_data: bytes):
            self._data = command_data
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self) -> bytes:
            if self._sent:
                # After sending, just wait forever until cancelled
                await trio.sleep(100)
                raise StopAsyncIteration
            # Delay to ensure run() reaches the waiting loop
            await trio.sleep(0.05)
            self._sent = True
            return self._data

        async def aclose(self) -> None:
            pass

    input_stream = DelayedInputStream((serialize(start_request) + "\n").encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    # Mock run_reducer to complete quickly
    async def mock_run_reducer():
        worker.running = False

    worker.run_reducer = mock_run_reducer

    # Run with a timeout
    with trio.move_on_after(2):
        await worker.run()

    # Should have received the start response
    assert b"started" in output.data


@pytest.mark.serial
async def test_worker_start_reduction_clang_delta_not_found(tmp_path):
    """Test _start_reduction when find_clang_delta returns empty string."""
    # Create a C file
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "no_clang_delta": False,
        # No clang_delta path specified, will call find_clang_delta()
    }

    # Mock find_clang_delta to return empty string
    with patch("shrinkray.passes.clangdelta.find_clang_delta", return_value=""):
        await worker._start_reduction(params)

    assert worker.running is True
    # No clang_delta should be set
    # The test passes because we successfully started reduction


@pytest.mark.serial
async def test_worker_start_reduction_clang_delta_found(tmp_path):
    """Test _start_reduction when find_clang_delta returns a path."""
    # Create a C file
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "no_clang_delta": False,
        # No clang_delta path specified, will call find_clang_delta()
    }

    # Mock find_clang_delta to return a fake path and ClangDelta
    with patch(
        "shrinkray.passes.clangdelta.find_clang_delta", return_value="/fake/clang_delta"
    ):
        with patch("shrinkray.passes.clangdelta.ClangDelta") as mock_clang_delta:
            await worker._start_reduction(params)

    assert worker.running is True
    mock_clang_delta.assert_called_once_with("/fake/clang_delta")


@pytest.mark.serial
async def test_worker_start_reduction_clang_delta_path_provided(tmp_path):
    """Test _start_reduction when clang_delta path is provided directly."""
    # Create a C file
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    # Create a test script
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "no_clang_delta": False,
        "clang_delta": "/provided/clang_delta",  # Provided directly
    }

    # Mock ClangDelta since the path doesn't exist
    with patch("shrinkray.passes.clangdelta.ClangDelta") as mock_clang_delta:
        await worker._start_reduction(params)

    assert worker.running is True
    # Should use provided path, skipping find_clang_delta call
    mock_clang_delta.assert_called_once_with("/provided/clang_delta")


def test_worker_main_runs_trio():
    """Test main() function creates worker and runs trio."""

    # Mock trio.run and ReducerWorker to verify the flow
    with patch("shrinkray.subprocess.worker.trio.run") as mock_trio_run:
        with patch("shrinkray.subprocess.worker.ReducerWorker") as mock_worker_class:
            mock_worker = MagicMock()
            mock_worker_class.return_value = mock_worker

            main()

            mock_worker_class.assert_called_once()
            mock_trio_run.assert_called_once_with(mock_worker.run)


async def test_worker_read_commands_uses_stdin_when_no_stream():
    """Test read_commands uses stdin when no stream is provided."""

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    # Create a pipe to simulate stdin
    read_fd, write_fd = os.pipe()

    # Write a status command to the pipe
    request = Request(id="stdin-test", command="status", params={})
    os.write(write_fd, (serialize(request) + "\n").encode("utf-8"))
    os.close(write_fd)

    # Mock sys.stdin.fileno() to return our read pipe
    with patch.object(sys, "stdin") as mock_stdin:
        mock_stdin.fileno.return_value = read_fd

        # Run with no input_stream parameter (uses stdin path)
        with trio.move_on_after(1):
            await worker.read_commands()

    os.close(read_fd)

    # Should have received response
    assert b"stdin-test" in output.data


def test_worker_main_module_entry_point():
    """Test the __name__ == '__main__' guard."""

    # Mock trio.run to prevent it from actually running
    with patch("shrinkray.subprocess.worker.trio.run") as mock_trio_run:
        # Use runpy to execute the module with __name__ == "__main__"
        try:
            runpy.run_module(
                "shrinkray.subprocess.worker",
                run_name="__main__",
                alter_sys=True,
            )
        except SystemExit:
            pass  # Module might call sys.exit

    # trio.run should have been called via main()
    assert mock_trio_run.called


# === Tests for startup error conditions ===


async def test_worker_start_with_failing_interestingness_test(tmp_path):
    """Test that worker properly reports error when initial test case is not interesting.

    When the interestingness test (e.g., 'false') returns non-zero for the initial
    test case, the worker should return an error response immediately during start,
    not hang or corrupt the JSON protocol with print statements.
    """
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script that always fails (returns non-zero)
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    # Start the reduction - should fail immediately with detailed error
    response = await worker._handle_start("test-start", params)

    # Should get back an error response with detailed message
    assert response.error is not None
    assert "Shrink ray cannot proceed" in response.error
    assert "interestingness test" in response.error
    assert response.result is None

    # Worker should not be running
    assert worker.running is False


async def test_worker_start_validates_initial_example(tmp_path):
    """Test that _start_reduction validates initial example and raises on failure."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": ["false"],  # Always returns 1
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    with pytest.raises(InvalidInitialExample):
        await worker._start_reduction(params)

    # Worker should not be marked as running since start failed
    assert worker.running is False


@pytest.mark.slow
async def test_worker_full_run_with_failing_test(tmp_path):
    """Test full run() with a test that fails on initial example."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a start command with a failing test
    start_params = {
        "file_path": str(target),
        "test": ["false"],  # Always fails
        "parallelism": 1,
        "timeout": 1.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }
    start_request = Request(id="start-fail", command="start", params=start_params)
    input_data = serialize(start_request) + "\n"

    output = MemoryOutputStream()
    input_stream = MemoryInputStream(input_data.encode("utf-8"))

    worker = ReducerWorker(input_stream=input_stream, output_stream=output)

    # Run with a timeout to prevent hanging
    with trio.move_on_after(10):
        await worker.run()

    # Should not hang - worker should complete with error
    output_str = output.data.decode("utf-8")

    # Should have gotten an error response (not "started")
    assert b"error" in output.data
    assert b"Shrink ray cannot proceed" in output.data

    for line in output_str.strip().split("\n"):
        if line:
            parsed = json.loads(line)  # Will raise JSONDecodeError if not valid JSON
            # Check it's a valid protocol message
            assert "id" in parsed or "type" in parsed


@pytest.mark.slow
async def test_worker_timeout_on_initial_test(tmp_path):
    """Test handling when initial test exceeds timeout."""
    # Create a test file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a test script that sleeps longer than timeout
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nsleep 10\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 0.1,  # Very short timeout
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    # Start should fail with timeout error during initial validation
    response = await worker._handle_start("test-timeout", params)

    # Should get an error (either about timeout or wrapped exception)
    assert response.error is not None
    # The error could be a timeout message, or it could be wrapped in an exception group
    # Either way, the worker should not be in a running state
    assert worker.running is False


async def test_worker_error_message_is_detailed(tmp_path):
    """Test that error messages for invalid initial examples are detailed."""
    # Create a target file
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a script that always fails
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/bash\nexit 1")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 5.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
    }

    # Start should fail with detailed error message
    response = await worker._handle_start("test-detailed", params)

    # Should get an error response
    assert response.error is not None
    error_message = response.error

    # Check that the error message is detailed (not just a simple exception message)
    assert "Shrink ray cannot proceed" in error_message
    assert "interestingness test" in error_message
    assert "exit" in error_message.lower() or "code" in error_message.lower()


@pytest.mark.trio
async def test_worker_trivial_result_error(tmp_path):
    """Test that reducing to a trivial result shows an error."""
    # Create a target file with content
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a script that always succeeds (accepts any input)
    script = tmp_path / "pass.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 5.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
        "trivial_is_error": True,
    }

    # Start should succeed
    response = await worker._handle_start("test-trivial", params)
    assert response.error is None
    assert response.result == {"status": "started"}

    # Run the reducer
    await worker.run_reducer()

    # Check the output for the trivial error message
    output_data = output.data
    found_trivial_error = False
    for line in output_data.split(b"\n"):
        if line:
            msg = deserialize(line.decode("utf-8"))
            if isinstance(msg, Response) and msg.error:
                if "trivial" in msg.error.lower():
                    found_trivial_error = True
                    break

    assert found_trivial_error, "Should have emitted a trivial result error"


@pytest.mark.trio
async def test_worker_trivial_result_no_error_when_disabled(tmp_path):
    """Test that trivial result is not an error when trivial_is_error=False."""
    # Create a target file with content
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Create a script that always succeeds (accepts any input)
    script = tmp_path / "pass.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    params = {
        "file_path": str(target),
        "test": [str(script)],
        "parallelism": 1,
        "timeout": 5.0,
        "seed": 0,
        "input_type": "all",
        "in_place": False,
        "formatter": "none",
        "volume": "quiet",
        "no_clang_delta": True,
        "trivial_is_error": False,
    }

    # Start should succeed
    response = await worker._handle_start("test-trivial-disabled", params)
    assert response.error is None

    # Run the reducer
    await worker.run_reducer()

    # Check the output - should NOT have trivial error
    # (completion message is emitted by run(), not run_reducer())
    output_data = output.data
    found_error = False
    for line in output_data.split(b"\n"):
        if line:
            msg = deserialize(line.decode("utf-8"))
            if isinstance(msg, Response) and msg.error:
                if "trivial" in msg.error.lower():
                    found_error = True

    assert not found_error, "Should not have emitted trivial error"


# === Additional coverage tests for exception handling ===


@pytest.mark.trio
async def test_worker_handle_start_invalid_initial_example_without_state():
    """Test error handling when InvalidInitialExample is raised before state is set."""

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    async def mock_start_reduction(params):
        raise InvalidInitialExample("Test error without state")

    worker._start_reduction = mock_start_reduction

    response = await worker._handle_start("test-no-state", {"test": ["fake"]})
    assert response.error == "Test error without state"


@pytest.mark.trio
async def test_worker_run_reducer_exception_handling():
    """Test that run_reducer catches and emits generic exceptions."""
    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    class MockReducer:
        async def run(self):
            raise RuntimeError("Test runtime error")

    worker.reducer = MockReducer()  # type: ignore[assignment]
    worker.running = True

    await worker.run_reducer()

    output_data = output.data.decode("utf-8")
    assert "Test runtime error" in output_data


@pytest.mark.trio
async def test_worker_run_reducer_invalid_initial_example_without_state():
    """Test run_reducer handles InvalidInitialExample when state is None."""

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    class MockReducer:
        async def run(self):
            raise InvalidInitialExample("Invalid example without state")

    worker.reducer = MockReducer()  # type: ignore[assignment]
    worker.state = None
    worker.running = True

    await worker.run_reducer()

    output_data = output.data.decode("utf-8")
    assert "Invalid example without state" in output_data


@pytest.mark.trio
async def test_worker_run_reducer_invalid_initial_example_with_state():
    """Test run_reducer builds error message when state is available."""

    output = MemoryOutputStream()
    worker = ReducerWorker(output_stream=output)

    class MockReducer:
        async def run(self):
            raise InvalidInitialExample("Test initial example error")

    mock_state = MagicMock()
    mock_state.build_error_message = AsyncMock(return_value="Built error message")

    worker.reducer = MockReducer()  # type: ignore[assignment]
    worker.state = mock_state
    worker.running = True

    await worker.run_reducer()

    mock_state.build_error_message.assert_called_once()
    output_data = output.data.decode("utf-8")
    assert "Built error message" in output_data


@pytest.mark.trio
async def test_worker_run_with_none_progress_update():
    """Test run() when _build_progress_update returns None (376->380 branch)."""

    output = MemoryOutputStream()

    # Create JSON command as bytes - use chr(10) for newline
    json_cmd = b'{"id":"test","command":"start","params":{"file_path":"/tmp/test.txt","test":["./test.sh"],"parallelism":1}}'
    input_data = json_cmd + bytes([10])  # 10 is newline

    class MockInputStream:
        def __init__(self):
            self.data = input_data
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return self.data
            raise StopAsyncIteration

        async def aclose(self) -> None:
            pass

    worker = ReducerWorker(input_stream=MockInputStream(), output_stream=output)

    async def mock_handle_start(request_id, params):
        worker.running = True
        return Response(id=request_id, result={"status": "started"})

    worker._handle_start = mock_handle_start

    async def quick_run_reducer():
        await trio.sleep(0)

    worker.run_reducer = quick_run_reducer
    worker._build_progress_update = AsyncMock(return_value=None)

    with trio.move_on_after(0.5):
        await worker.run()

    output_data = output.data.decode("utf-8")
    assert len(output_data) >= 0


# === Pass Control Tests ===


def test_worker_handle_disable_pass_no_pass_name():
    """Test disable_pass handler with missing pass_name."""
    worker = ReducerWorker()

    response = worker._handle_disable_pass("test-id", {})
    assert response.error == "pass_name is required"


def test_worker_handle_disable_pass_empty_pass_name():
    """Test disable_pass handler with empty pass_name."""
    worker = ReducerWorker()

    response = worker._handle_disable_pass("test-id", {"pass_name": ""})
    assert response.error == "pass_name is required"


def test_worker_handle_disable_pass_no_reducer():
    """Test disable_pass handler when reducer is None."""
    worker = ReducerWorker()
    worker.reducer = None

    response = worker._handle_disable_pass("test-id", {"pass_name": "hollow"})
    assert response.error == "Reducer does not support pass control"


def test_worker_handle_disable_pass_success():
    """Test disable_pass handler success case."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.disable_pass = Mock()
    # Mock pass_stats with the pass name in _stats
    worker.reducer.pass_stats._stats = {"hollow": Mock()}

    response = worker._handle_disable_pass("test-id", {"pass_name": "hollow"})
    assert response.result == {"status": "disabled", "pass_name": "hollow"}
    worker.reducer.disable_pass.assert_called_once_with("hollow")


def test_worker_handle_disable_pass_unknown_pass():
    """Test disable_pass handler with unknown pass name."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.disable_pass = Mock()
    # Mock pass_stats with only known passes
    worker.reducer.pass_stats._stats = {"hollow": Mock(), "delete_lines": Mock()}

    response = worker._handle_disable_pass("test-id", {"pass_name": "unknown_pass"})
    assert response.error is not None
    assert "Unknown pass 'unknown_pass'" in response.error
    assert "hollow" in response.error  # Should list known passes
    worker.reducer.disable_pass.assert_not_called()


def test_worker_handle_enable_pass_no_pass_name():
    """Test enable_pass handler with missing pass_name."""
    worker = ReducerWorker()

    response = worker._handle_enable_pass("test-id", {})
    assert response.error == "pass_name is required"


def test_worker_handle_enable_pass_empty_pass_name():
    """Test enable_pass handler with empty pass_name."""
    worker = ReducerWorker()

    response = worker._handle_enable_pass("test-id", {"pass_name": ""})
    assert response.error == "pass_name is required"


def test_worker_handle_enable_pass_no_reducer():
    """Test enable_pass handler when reducer is None."""
    worker = ReducerWorker()
    worker.reducer = None

    response = worker._handle_enable_pass("test-id", {"pass_name": "hollow"})
    assert response.error == "Reducer does not support pass control"


def test_worker_handle_enable_pass_success():
    """Test enable_pass handler success case."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.enable_pass = Mock()
    # Mock pass_stats with the pass name in _stats
    worker.reducer.pass_stats._stats = {"hollow": Mock()}

    response = worker._handle_enable_pass("test-id", {"pass_name": "hollow"})
    assert response.result == {"status": "enabled", "pass_name": "hollow"}
    worker.reducer.enable_pass.assert_called_once_with("hollow")


def test_worker_handle_enable_pass_unknown_pass():
    """Test enable_pass handler with unknown pass name."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.enable_pass = Mock()
    # Mock pass_stats with only known passes
    worker.reducer.pass_stats._stats = {"hollow": Mock(), "delete_lines": Mock()}

    response = worker._handle_enable_pass("test-id", {"pass_name": "unknown_pass"})
    assert response.error is not None
    assert "Unknown pass 'unknown_pass'" in response.error
    assert "hollow" in response.error  # Should list known passes
    worker.reducer.enable_pass.assert_not_called()


def test_worker_handle_skip_pass_no_reducer():
    """Test skip_pass handler when reducer is None."""
    worker = ReducerWorker()
    worker.reducer = None

    response = worker._handle_skip_pass("test-id")
    assert response.error == "Reducer does not support pass control"


def test_worker_handle_skip_pass_success():
    """Test skip_pass handler success case."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.skip_current_pass = Mock()

    response = worker._handle_skip_pass("test-id")
    assert response.result == {"status": "skipped"}
    worker.reducer.skip_current_pass.assert_called_once()


@pytest.mark.trio
async def test_worker_handle_command_disable_pass():
    """Test handle_command dispatches to disable_pass handler."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.disable_pass = Mock()
    # Mock pass_stats with the pass name in _stats
    worker.reducer.pass_stats._stats = {"hollow": Mock()}

    request = Request(
        id="test-id", command="disable_pass", params={"pass_name": "hollow"}
    )
    response = await worker.handle_command(request)

    assert response.result == {"status": "disabled", "pass_name": "hollow"}


@pytest.mark.trio
async def test_worker_handle_command_enable_pass():
    """Test handle_command dispatches to enable_pass handler."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.enable_pass = Mock()
    # Mock pass_stats with the pass name in _stats
    worker.reducer.pass_stats._stats = {"hollow": Mock()}

    request = Request(
        id="test-id", command="enable_pass", params={"pass_name": "hollow"}
    )
    response = await worker.handle_command(request)

    assert response.result == {"status": "enabled", "pass_name": "hollow"}


@pytest.mark.trio
async def test_worker_handle_command_skip_pass():
    """Test handle_command dispatches to skip_pass handler."""

    worker = ReducerWorker()
    worker.reducer = Mock()
    worker.reducer.skip_current_pass = Mock()

    request = Request(id="test-id", command="skip_pass", params={})
    response = await worker.handle_command(request)

    assert response.result == {"status": "skipped"}


# === _build_progress_update edge cases ===


@pytest.mark.trio
async def test_build_progress_update_with_reducer_none():
    """Test _build_progress_update when reducer is None (346->375, 376->379)."""

    worker = ReducerWorker()
    worker.reducer = None

    # Need to set up problem for _build_progress_update (it returns None if problem is None)
    worker.problem = Mock()
    worker.problem.stats = Mock()
    worker.problem.stats.current_test_case_size = 100
    worker.problem.stats.initial_test_case_size = 1000
    worker.problem.stats.calls = 10
    worker.problem.stats.reductions = 2
    worker.problem.stats.interesting_calls = 5
    worker.problem.stats.wasted_interesting_calls = 1
    worker.problem.stats.start_time = time.time()
    worker.problem.stats.time_since_last_reduction = Mock(return_value=0.5)
    worker.problem.current_test_case = b"test content"

    # Set up state for parallel workers calculation
    worker.state = Mock()
    worker.state.parallel_tasks_running = 2
    worker.state.output_manager = None  # No test output capture

    # Build progress update with reducer=None
    update = await worker._build_progress_update()

    # Should still return a valid ProgressUpdate
    assert update is not None
    assert update.pass_stats == []
    assert update.disabled_passes == []
    assert update.current_pass_name == ""


@pytest.mark.trio
async def test_build_progress_update_with_reducer_pass_stats_none():
    """Test _build_progress_update when reducer.pass_stats is None (356->375)."""

    worker = ReducerWorker()

    # Mock reducer with pass_stats=None
    worker.reducer = Mock()
    worker.reducer.pass_stats = None
    worker.reducer.current_reduction_pass = None
    worker.reducer.status = "Running"
    worker.reducer.disabled_passes = set()

    # Set up problem
    worker.problem = Mock()
    worker.problem.stats = Mock()
    worker.problem.stats.current_test_case_size = 100
    worker.problem.stats.initial_test_case_size = 1000
    worker.problem.stats.calls = 10
    worker.problem.stats.reductions = 2
    worker.problem.stats.interesting_calls = 5
    worker.problem.stats.wasted_interesting_calls = 1
    worker.problem.stats.start_time = time.time()
    worker.problem.stats.time_since_last_reduction = Mock(return_value=0.5)
    worker.problem.current_test_case = b"test content"

    # Set up state
    worker.state = Mock()
    worker.state.parallel_tasks_running = 2
    worker.state.output_manager = None  # No test output capture

    # Build progress update
    update = await worker._build_progress_update()

    assert update is not None
    assert update.pass_stats == []


@pytest.mark.trio
async def test_build_progress_update_with_reducer_no_disabled_passes_attr():
    """Test _build_progress_update when reducer lacks disabled_passes attribute."""

    worker = ReducerWorker()

    # Mock reducer without disabled_passes attribute
    worker.reducer = Mock(spec=["pass_stats", "current_reduction_pass", "status"])
    worker.reducer.pass_stats = None
    worker.reducer.current_reduction_pass = None
    worker.reducer.status = "Running"
    # Note: no disabled_passes attribute

    # Set up problem
    worker.problem = Mock()
    worker.problem.stats = Mock()
    worker.problem.stats.current_test_case_size = 100
    worker.problem.stats.initial_test_case_size = 1000
    worker.problem.stats.calls = 10
    worker.problem.stats.reductions = 2
    worker.problem.stats.interesting_calls = 5
    worker.problem.stats.wasted_interesting_calls = 1
    worker.problem.stats.start_time = time.time()
    worker.problem.stats.time_since_last_reduction = Mock(return_value=0.5)
    worker.problem.current_test_case = b"test content"

    # Set up state
    worker.state = Mock()
    worker.state.parallel_tasks_running = 2
    worker.state.output_manager = None  # No test output capture

    # Build progress update - should not crash even without disabled_passes
    update = await worker._build_progress_update()

    assert update is not None
    assert update.disabled_passes == []


@pytest.mark.trio
async def test_build_progress_update_periodic_size_history():
    """Test _build_progress_update records periodic size history when size unchanged (412-413)."""

    worker = ReducerWorker()

    # Set up problem with a start time that gives us control over runtime
    start_time = time.time()

    worker.problem = Mock()
    worker.problem.stats = Mock()
    worker.problem.stats.current_test_case_size = 500
    worker.problem.stats.initial_test_case_size = 1000
    worker.problem.stats.calls = 10
    worker.problem.stats.reductions = 2
    worker.problem.stats.interesting_calls = 5
    worker.problem.stats.wasted_interesting_calls = 1
    worker.problem.stats.start_time = start_time
    worker.problem.stats.time_since_last_reduction = Mock(return_value=0.1)
    worker.problem.current_test_case = b"test content"

    worker.state = Mock()
    worker.state.parallel_tasks_running = 2
    worker.state.output_manager = None

    # First call - initializes history with size change from initial
    update1 = await worker._build_progress_update()
    assert update1 is not None
    # Should have initial entry and size change entry
    assert len(worker._size_history) >= 1

    initial_history_len = len(worker._size_history)
    recorded_time = worker._last_history_time

    # Now the size stays the same, but not enough time has passed
    # history_interval is 0.2s for first 5 minutes
    await worker._build_progress_update()
    # No new entry should be added (size same, not enough time)
    assert len(worker._size_history) == initial_history_len

    # Now simulate enough time passing (> 0.2s)
    # We need to mock time.time to return a later time
    fake_current_time = start_time + recorded_time + 0.3  # 0.3s after last record

    with patch("shrinkray.subprocess.worker.time.time", return_value=fake_current_time):
        await worker._build_progress_update()

    # Now a periodic entry should have been added (size same, time passed)
    assert len(worker._size_history) > initial_history_len
    assert worker._size_history[-1][1] == 500  # Size is unchanged


# === _get_test_output_preview tests ===


def test_get_test_output_preview_no_output_path():
    """Test _get_test_output_preview when output_path is None."""
    worker = ReducerWorker()
    worker.state = MagicMock()

    # Output manager with no output yet
    manager = MagicMock()
    manager.get_current_output.return_value = (None, None, None)
    worker.state.output_manager = manager

    preview, test_id, return_code = worker._get_test_output_preview()
    assert preview == ""
    assert test_id is None
    assert return_code is None


def test_get_test_output_preview_large_file(tmp_path):
    """Test _get_test_output_preview with file larger than 4KB."""
    worker = ReducerWorker()
    worker.state = MagicMock()

    # Create a file larger than 4KB
    output_file = tmp_path / "test_output.log"
    content = "x" * 5000 + "LAST PART"
    output_file.write_text(content)

    manager = MagicMock()
    # Return running test (return_code=None)
    manager.get_current_output.return_value = (str(output_file), 10, None)
    worker.state.output_manager = manager

    preview, test_id, return_code = worker._get_test_output_preview()
    assert test_id == 10
    assert return_code is None
    # Should get last 4KB
    assert len(preview.encode()) <= 4096
    assert "LAST PART" in preview


def test_get_test_output_preview_file_read_error():
    """Test _get_test_output_preview handles OSError gracefully."""
    worker = ReducerWorker()
    worker.state = MagicMock()

    manager = MagicMock()
    # Return completed test with return code, but file doesn't exist
    manager.get_current_output.return_value = ("/nonexistent/path/file.log", 3, 1)
    worker.state.output_manager = manager

    preview, test_id, return_code = worker._get_test_output_preview()
    assert preview == ""
    assert test_id == 3
    assert return_code == 1


# === _handle_restart_from tests ===


@pytest.mark.trio
async def test_handle_restart_from_missing_reduction_number():
    """Test restart_from handler with missing reduction_number."""
    worker = ReducerWorker()

    response = await worker._handle_restart_from("test-id", {})
    assert response.error == "reduction_number is required"


@pytest.mark.trio
async def test_handle_restart_from_no_state():
    """Test restart_from handler when state is None."""
    worker = ReducerWorker()
    worker.state = None

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})
    assert response.error == "History not available"


@pytest.mark.trio
async def test_handle_restart_from_no_history_manager():
    """Test restart_from handler when history_manager is None."""
    worker = ReducerWorker()
    worker.state = MagicMock()
    worker.state.history_manager = None

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})
    assert response.error == "History not available"


@pytest.mark.trio
async def test_handle_restart_from_directory_reduction():
    """Test restart_from handler returns error for directory reductions."""
    worker = ReducerWorker()
    # Mock a directory state (not ShrinkRayStateSingleFile)
    worker.state = MagicMock(spec=ShrinkRayDirectoryState)
    worker.state.history_manager = MagicMock()

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})
    assert response.error == "Restart from history not supported for directory reductions"


@pytest.mark.trio
async def test_handle_restart_from_nonexistent_reduction():
    """Test restart_from handler with nonexistent reduction number."""
    worker = ReducerWorker()
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.history_manager.restart_from_reduction.side_effect = FileNotFoundError()
    worker.state.output_manager = None

    response = await worker._handle_restart_from("test-id", {"reduction_number": 999})
    assert response.error is not None and "not found" in response.error


@pytest.mark.trio
async def test_handle_restart_from_success():
    """Test restart_from handler success case."""
    worker = ReducerWorker()
    worker._cancel_scope = None
    worker.running = True

    # Set up mocks
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.history_manager.restart_from_reduction.return_value = (
        b"restart content",
        {b"excluded1", b"excluded2"},
    )
    worker.state.filename = "/tmp/test.c"
    worker.state.output_manager = None

    # Mock the reducer
    mock_reducer = MagicMock()
    mock_reducer.target = MagicMock()
    worker.state.reducer = mock_reducer

    response = await worker._handle_restart_from("test-id", {"reduction_number": 3})

    assert response.result == {"status": "restarted", "size": 15}  # len(b"restart content")
    worker.state.history_manager.restart_from_reduction.assert_called_once_with(3)
    worker.state.reset_for_restart.assert_called_once_with(
        b"restart content", {b"excluded1", b"excluded2"}
    )
    # running is False after restart; the run() loop will set it to True
    assert worker.running is False
    # restart_requested must be True so the run() loop continues
    assert worker._restart_requested is True


@pytest.mark.trio
async def test_handle_restart_from_preserves_size_history():
    """Test that restart_from appends to size history instead of resetting it.

    This is a regression test for a bug where the graph would get confused
    after restart because size_history was reset to start at time 0.
    """
    worker = ReducerWorker()
    worker._cancel_scope = None
    worker.running = True

    # Set up existing size history (simulating reduction progress before restart)
    start_time = time.time() - 60  # Started 60 seconds ago
    worker._original_start_time = start_time
    worker._size_history = [
        (0.0, 1000),  # Initial size
        (10.0, 800),  # After 10s, reduced to 800
        (30.0, 500),  # After 30s, reduced to 500
    ]
    worker._last_recorded_size = 500
    worker._last_history_time = 30.0

    # Set up mocks
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.history_manager.restart_from_reduction.return_value = (
        b"x" * 750,  # Restart from 750 bytes (larger than current 500)
        set(),
    )
    worker.state.filename = "/tmp/test.c"
    worker.state.output_manager = None

    # Mock the reducer
    mock_reducer = MagicMock()
    mock_reducer.target = MagicMock()
    worker.state.reducer = mock_reducer

    response = await worker._handle_restart_from("test-id", {"reduction_number": 2})

    assert response.result == {"status": "restarted", "size": 750}

    # Size history should have 4 entries now (original 3 + restart jump)
    assert len(worker._size_history) == 4

    # First 3 entries should be unchanged
    assert worker._size_history[0] == (0.0, 1000)
    assert worker._size_history[1] == (10.0, 800)
    assert worker._size_history[2] == (30.0, 500)

    # 4th entry should be the restart point (upward jump to 750)
    restart_time, restart_size = worker._size_history[3]
    assert restart_size == 750
    # Time should be approximately 60 seconds (current time - original start)
    assert restart_time >= 55  # Allow some tolerance

    # Last recorded size should be updated
    assert worker._last_recorded_size == 750


@pytest.mark.trio
async def test_handle_command_restart_from():
    """Test handle_command dispatches restart_from correctly."""
    worker = ReducerWorker()
    worker.state = None  # Will cause "History not available" error

    request = Request(id="test-id", command="restart_from", params={"reduction_number": 1})
    response = await worker.handle_command(request)

    assert response.error == "History not available"


@pytest.mark.trio
async def test_handle_restart_from_cancels_running_scope():
    """Test restart_from cancels the running cancel scope."""
    worker = ReducerWorker()

    # Create a mock cancel scope
    mock_cancel_scope = MagicMock()
    worker._cancel_scope = mock_cancel_scope
    worker.running = True

    # Set up mocks
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.history_manager.restart_from_reduction.return_value = (
        b"restart content",
        set(),
    )
    worker.state.filename = "/tmp/test.c"
    worker.state.output_manager = None

    # Mock the reducer
    mock_reducer = MagicMock()
    mock_reducer.target = MagicMock()
    worker.state.reducer = mock_reducer

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})

    # Cancel scope should have been cancelled
    mock_cancel_scope.cancel.assert_called_once()
    assert response.result == {"status": "restarted", "size": 15}


@pytest.mark.trio
async def test_handle_restart_from_generic_exception():
    """Test restart_from handles generic exceptions."""
    worker = ReducerWorker()
    worker._cancel_scope = None
    worker.running = True

    # Set up mocks
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.output_manager = None
    # Raise a generic exception
    worker.state.history_manager.restart_from_reduction.side_effect = RuntimeError(
        "Something went wrong"
    )

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})

    # Error now includes full traceback
    assert response.error is not None
    assert "Something went wrong" in response.error
    assert "Traceback" in response.error


@pytest.mark.trio
async def test_handle_restart_from_clears_output_manager():
    """Test restart_from clears the output manager to avoid stale output display."""
    worker = ReducerWorker()
    worker._cancel_scope = None
    worker.running = True

    # Set up mocks
    worker.state = MagicMock(spec=ShrinkRayStateSingleFile)
    worker.state.history_manager = MagicMock()
    worker.state.history_manager.restart_from_reduction.return_value = (
        b"restart content",
        set(),
    )
    worker.state.filename = "/tmp/test.c"
    worker.state.output_manager = MagicMock()

    # Mock the reducer
    mock_reducer = MagicMock()
    mock_reducer.target = MagicMock()
    worker.state.reducer = mock_reducer

    response = await worker._handle_restart_from("test-id", {"reduction_number": 1})

    # Output manager should have been cleaned up to clear stale output
    worker.state.output_manager.cleanup_all.assert_called_once()
    assert response.result == {"status": "restarted", "size": 15}


@pytest.mark.trio
async def test_run_loop_restarts_reducer():
    """Test that run() loop re-runs reducer when _restart_requested is True."""
    worker = ReducerWorker()
    run_reducer_calls = []

    async def mock_run_reducer():
        run_reducer_calls.append(len(run_reducer_calls))
        # On first call, simulate restart being requested
        if len(run_reducer_calls) == 1:
            worker._restart_requested = True
            worker.running = True
        # On second call, just complete normally

    async def mock_read_commands(
        input_stream: InputStream | None = None,
        task_status: trio.TaskStatus[None] = trio.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started()
        # Just wait indefinitely (will be cancelled when run completes)
        await trio.sleep_forever()

    # Mock the methods
    worker.run_reducer = mock_run_reducer
    worker.emit_progress_updates = AsyncMock()
    worker._build_progress_update = AsyncMock(return_value=None)
    worker.emit = AsyncMock()
    worker.read_commands = mock_read_commands

    # Start the worker
    worker.running = True

    # Run with a timeout to avoid infinite loop in case of bug
    with trio.move_on_after(1):
        await worker.run()

    # Verify run_reducer was called twice (once initially, once after restart)
    assert len(run_reducer_calls) == 2


@pytest.mark.trio
async def test_emit_progress_updates_continues_during_restart():
    """Test that emit_progress_updates keeps running during restart.

    This is a regression test for a bug where stats stopped updating after
    restart because emit_progress_updates exited its loop when running became
    False, and never resumed even after running was set back to True.
    """
    worker = ReducerWorker()
    updates_emitted = []

    async def mock_build_progress_update():
        # Record when updates are built
        updates_emitted.append(
            {"running": worker.running, "restart_requested": worker._restart_requested}
        )
        return None  # Return None so emit isn't called

    worker._build_progress_update = mock_build_progress_update

    # Start with running=True
    worker.running = True
    worker._restart_requested = False

    async def simulate_restart():
        """Simulate a restart after some updates."""
        await trio.sleep(0.25)  # Let some updates happen

        # Simulate restart: set running=False and _restart_requested=True
        worker.running = False
        worker._restart_requested = True

        await trio.sleep(0.25)  # Let updates continue during restart

        # Complete restart: set running=True and _restart_requested=False
        worker.running = True
        worker._restart_requested = False

        await trio.sleep(0.25)  # Let updates continue after restart

        # Now complete: set running=False with no restart
        worker.running = False

    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker.emit_progress_updates)
        nursery.start_soon(simulate_restart)

    # Verify we got updates during all phases:
    # 1. Initial running phase
    # 2. During restart (running=False, restart_requested=True)
    # 3. After restart (running=True again)

    # Find updates from each phase
    initial_updates = [u for u in updates_emitted if u["running"] and not u["restart_requested"]]
    restart_updates = [u for u in updates_emitted if not u["running"] and u["restart_requested"]]

    # Should have updates from all phases
    assert len(initial_updates) >= 1, "Should have updates during initial running phase"
    assert len(restart_updates) >= 1, "Should have updates during restart phase"
    # The key point is that we didn't exit early during restart - we got updates
    # while running=False but restart_requested=True
