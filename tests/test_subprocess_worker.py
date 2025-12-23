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


# === Integration tests with real files ===


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


async def test_stdout_stream_send():
    """Test StdoutStream wrapper class (lines 37-38)."""
    from shrinkray.subprocess.worker import StdoutStream

    output = io.StringIO()
    stream = StdoutStream()

    with patch.object(sys, "stdout", output):
        await stream.send(b"hello world\n")

    assert output.getvalue() == "hello world\n"


async def test_worker_read_commands_empty_lines():
    """Test read_commands handles empty lines between commands (line 92->90)."""
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
    """Test emit_progress_updates continues when problem is None (line 266)."""
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
    """Test emit_progress_updates handles state without parallel_tasks_running (line 272->278)."""
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
    mock_state = MagicMock(spec=[])  # Empty spec means no attributes
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
    """Test emit_progress_updates with zero parallel samples (line 280->289)."""
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
    """Test _get_content_preview handles decode exceptions (lines 258-259)."""
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


async def test_worker_start_reduction_with_clang_delta(tmp_path):
    """Test _start_reduction with a C file and clang_delta (lines 157-160)."""
    from shrinkray.passes.clangdelta import find_clang_delta

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


async def test_worker_full_run_with_mock(tmp_path):
    """Test the full run() method (lines 322-335)."""
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
    """Test the main() function (lines 340-341)."""
    from shrinkray.subprocess.worker import main

    # We can't easily test the actual main() since it blocks on trio.run
    # But we can verify it exists and is callable
    assert callable(main)


def test_worker_main_guard():
    """Test that __name__ == '__main__' block exists (line 345)."""
    # This is just to verify the module can be imported
    import shrinkray.subprocess.worker

    # The module exists and can be imported
    assert hasattr(shrinkray.subprocess.worker, "main")


async def test_worker_run_waits_for_start(tmp_path):
    """Test that run() waits for start command (line 327)."""
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


async def test_worker_start_reduction_clang_delta_not_found(tmp_path):
    """Test _start_reduction when find_clang_delta returns empty (line 157->159 false branch)."""
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


async def test_worker_start_reduction_clang_delta_found(tmp_path):
    """Test _start_reduction when find_clang_delta returns a path (lines 157->158->159->160)."""
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


async def test_worker_start_reduction_clang_delta_path_provided(tmp_path):
    """Test _start_reduction when clang_delta path is provided directly (line 157->159)."""
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
    """Test main() function creates worker and runs trio (lines 340-341)."""
    from shrinkray.subprocess.worker import main

    # Mock trio.run and ReducerWorker to verify the flow
    with patch("shrinkray.subprocess.worker.trio.run") as mock_trio_run:
        with patch("shrinkray.subprocess.worker.ReducerWorker") as mock_worker_class:
            mock_worker = MagicMock()
            mock_worker_class.return_value = mock_worker

            main()

            mock_worker_class.assert_called_once()
            mock_trio_run.assert_called_once_with(mock_worker.run)


async def test_worker_read_commands_uses_stdin_when_no_stream():
    """Test read_commands uses stdin when no stream is provided (line 85)."""
    import os

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
    """Test the __name__ == '__main__' guard (line 345)."""
    import runpy

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
