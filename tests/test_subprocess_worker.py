"""Tests for ReducerWorker subprocess."""

import io
import sys
from unittest.mock import MagicMock, patch

import trio

from shrinkray.subprocess.protocol import ProgressUpdate, Request, Response, serialize
from shrinkray.subprocess.worker import ReducerWorker


# === ReducerWorker initialization tests ===


def test_worker_initial_state():
    worker = ReducerWorker()
    assert worker.running is False
    assert worker.reducer is None
    assert worker.problem is None
    assert worker.state is None
    assert worker._cancel_scope is None
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
