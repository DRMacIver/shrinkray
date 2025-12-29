"""Worker subprocess that runs the reducer with trio and communicates via JSON protocol."""

import os
import shutil
import sys
import tempfile
import time
import traceback
from contextlib import aclosing
from typing import Any, Protocol

import trio
from binaryornot.helpers import is_binary_string

from shrinkray.problem import InvalidInitialExample
from shrinkray.state import ShrinkRayStateSingleFile
from shrinkray.subprocess.protocol import (
    PassStatsData,
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


class InputStream(Protocol):
    """Protocol for async input streams."""

    def __aiter__(self) -> "InputStream": ...
    async def __anext__(self) -> bytes | bytearray: ...
    async def aclose(self) -> None: ...


class OutputStream(Protocol):
    """Protocol for output streams."""

    async def send(self, data: bytes) -> None: ...


class ReducerWorker:
    """Runs the reducer in a subprocess with JSON protocol communication."""

    def __init__(
        self,
        input_stream: InputStream | None = None,
        output_stream: OutputStream | None = None,
    ):
        self.running = False
        self.reducer = None
        self.problem = None
        self.state = None
        self._cancel_scope: trio.CancelScope | None = None
        self._restart_requested = False
        # Parallelism tracking
        self._parallel_samples = 0
        self._parallel_total = 0
        # I/O streams - None means use stdin/stdout
        self._input_stream = input_stream
        self._output_stream = output_stream
        # Output directory for test output capture (cleaned up on shutdown)
        self._output_dir: str | None = None
        # Size history for graphing: list of (runtime_seconds, size) tuples
        self._size_history: list[tuple[float, int]] = []
        self._last_sent_history_index: int = 0
        self._last_recorded_size: int = 0
        self._last_history_time: float = 0.0

    async def emit(self, msg: Response | ProgressUpdate) -> None:
        """Write a message to the output stream."""
        line = serialize(msg) + "\n"
        if self._output_stream is not None:
            await self._output_stream.send(line.encode("utf-8"))
        else:
            sys.stdout.write(line)
            sys.stdout.flush()

    async def read_commands(
        self,
        input_stream: InputStream | None = None,
        task_status: trio.TaskStatus[None] = trio.TASK_STATUS_IGNORED,
    ) -> None:
        """Read commands from input stream and dispatch them."""
        task_status.started()

        # Use provided stream, or instance stream, or default to stdin
        stream: InputStream
        if input_stream is not None:
            stream = input_stream
        elif self._input_stream is not None:
            stream = self._input_stream
        else:
            stream = trio.lowlevel.FdStream(os.dup(sys.stdin.fileno()))

        buffer = b""
        async with aclosing(stream) as aiter:
            async for chunk in aiter:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        await self.handle_line(line.decode("utf-8"))

    async def handle_line(self, line: str) -> None:
        """Handle a single command line."""
        try:
            request = deserialize(line)
            if not isinstance(request, Request):
                await self.emit(Response(id="", error="Expected a request"))
                return
            response = await self.handle_command(request)
            await self.emit(response)
        except Exception as e:
            traceback.print_exc()
            await self.emit(Response(id="", error=str(e)))

    async def handle_command(self, request: Request) -> Response:
        """Handle a command request and return a response."""
        match request.command:
            case "start":
                return await self._handle_start(request.id, request.params)
            case "status":
                return self._handle_status(request.id)
            case "cancel":
                return self._handle_cancel(request.id)
            case "disable_pass":
                return self._handle_disable_pass(request.id, request.params)
            case "enable_pass":
                return self._handle_enable_pass(request.id, request.params)
            case "skip_pass":
                return self._handle_skip_pass(request.id)
            case "restart_from":
                return await self._handle_restart_from(request.id, request.params)
            case _:
                return Response(
                    id=request.id, error=f"Unknown command: {request.command}"
                )

    async def _handle_start(self, request_id: str, params: dict) -> Response:
        """Start the reduction process."""
        if self.running:
            return Response(id=request_id, error="Already running")

        try:
            await self._start_reduction(params)
            return Response(id=request_id, result={"status": "started"})
        except* InvalidInitialExample as excs:
            assert len(excs.exceptions) == 1
            (e,) = excs.exceptions
            # Build a detailed error message for invalid initial examples
            if self.state is not None:
                error_message = await self.state.build_error_message(e)
            else:
                error_message = str(e)
        except* Exception as e:
            traceback.print_exc()
            error_message = str(e.exceptions[0])
        return Response(id=request_id, error=error_message)

    async def _start_reduction(self, params: dict) -> None:
        """Initialize and start the reduction."""
        from shrinkray.cli import InputType
        from shrinkray.passes.clangdelta import (
            C_FILE_EXTENSIONS,
            ClangDelta,
            find_clang_delta,
        )
        from shrinkray.state import (
            OutputCaptureManager,
            ShrinkRayDirectoryState,
            ShrinkRayStateSingleFile,
        )
        from shrinkray.work import Volume

        filename = params["file_path"]
        test = params["test"]
        parallelism = params.get("parallelism", os.cpu_count() or 1)
        timeout = params.get("timeout")  # None means dynamic timeout
        seed = params.get("seed", 0)
        input_type = InputType[params.get("input_type", "all")]
        in_place = params.get("in_place", False)
        formatter = params.get("formatter", "default")
        volume = Volume[params.get("volume", "normal")]
        no_clang_delta = params.get("no_clang_delta", False)
        clang_delta_path = params.get("clang_delta", "")
        trivial_is_error = params.get("trivial_is_error", True)
        skip_validation = params.get("skip_validation", False)
        history_enabled = params.get("history_enabled", True)
        also_interesting_code = params.get("also_interesting_code")

        clang_delta_executable = None
        if os.path.splitext(filename)[1] in C_FILE_EXTENSIONS and not no_clang_delta:
            if not clang_delta_path:
                clang_delta_path = find_clang_delta()
            if clang_delta_path:
                clang_delta_executable = ClangDelta(clang_delta_path)

        state_kwargs: dict[str, Any] = {
            "input_type": input_type,
            "in_place": in_place,
            "test": test,
            "timeout": timeout,
            "base": os.path.basename(filename),
            "parallelism": parallelism,
            "filename": filename,
            "formatter": formatter,
            "trivial_is_error": trivial_is_error,
            "seed": seed,
            "volume": volume,
            "clang_delta_executable": clang_delta_executable,
            "history_enabled": history_enabled,
            "also_interesting_code": also_interesting_code,
        }

        if os.path.isdir(filename):
            files = [os.path.join(d, f) for d, _, fs in os.walk(filename) for f in fs]
            initial = {}
            for f in files:
                with open(f, "rb") as i:
                    initial[os.path.relpath(f, filename)] = i.read()
            self.state = ShrinkRayDirectoryState(initial=initial, **state_kwargs)
        else:
            with open(filename, "rb") as reader:
                initial = reader.read()
            self.state = ShrinkRayStateSingleFile(initial=initial, **state_kwargs)

        # Create output manager for test output capture (always enabled for TUI)
        self._output_dir = tempfile.mkdtemp(prefix="shrinkray-output-")
        self.state.output_manager = OutputCaptureManager(output_dir=self._output_dir)

        self.problem = self.state.problem
        self.reducer = self.state.reducer

        # Validate initial example before starting - this will raise
        # InvalidInitialExample if the initial test case fails.
        # Skip if validation was already done by the caller (e.g., main()).
        if not skip_validation:
            await self.problem.setup()

        self.running = True

    def _handle_status(self, request_id: str) -> Response:
        """Get current status."""
        if not self.running or self.problem is None:
            return Response(id=request_id, result={"running": False})

        stats = self.problem.stats
        return Response(
            id=request_id,
            result={
                "running": True,
                "status": self.reducer.status if self.reducer else "",
                "size": stats.current_test_case_size,
                "original_size": stats.initial_test_case_size,
                "calls": stats.calls,
                "reductions": stats.reductions,
            },
        )

    def _handle_cancel(self, request_id: str) -> Response:
        """Cancel the reduction."""
        if self._cancel_scope is not None:
            self._cancel_scope.cancel()
        self.running = False
        return Response(id=request_id, result={"status": "cancelled"})

    def _get_known_pass_names(self) -> set[str]:
        """Get the set of known pass names from pass stats."""
        if self.reducer is None or self.reducer.pass_stats is None:
            return set()
        return set(self.reducer.pass_stats._stats.keys())

    def _handle_disable_pass(self, request_id: str, params: dict) -> Response:
        """Disable a reduction pass by name."""
        pass_name = params.get("pass_name", "")
        if not pass_name:
            return Response(id=request_id, error="pass_name is required")

        known_passes = self._get_known_pass_names()
        if known_passes and pass_name not in known_passes:
            return Response(
                id=request_id,
                error=f"Unknown pass '{pass_name}'. Known passes: {sorted(known_passes)}",
            )

        if self.reducer is not None and hasattr(self.reducer, "disable_pass"):
            self.reducer.disable_pass(pass_name)
            return Response(
                id=request_id, result={"status": "disabled", "pass_name": pass_name}
            )
        return Response(id=request_id, error="Reducer does not support pass control")

    def _handle_enable_pass(self, request_id: str, params: dict) -> Response:
        """Enable a previously disabled reduction pass."""
        pass_name = params.get("pass_name", "")
        if not pass_name:
            return Response(id=request_id, error="pass_name is required")

        known_passes = self._get_known_pass_names()
        if known_passes and pass_name not in known_passes:
            return Response(
                id=request_id,
                error=f"Unknown pass '{pass_name}'. Known passes: {sorted(known_passes)}",
            )

        if self.reducer is not None and hasattr(self.reducer, "enable_pass"):
            self.reducer.enable_pass(pass_name)
            return Response(
                id=request_id, result={"status": "enabled", "pass_name": pass_name}
            )
        return Response(id=request_id, error="Reducer does not support pass control")

    def _handle_skip_pass(self, request_id: str) -> Response:
        """Skip the currently running pass."""
        if self.reducer is not None and hasattr(self.reducer, "skip_current_pass"):
            self.reducer.skip_current_pass()
            return Response(id=request_id, result={"status": "skipped"})
        return Response(id=request_id, error="Reducer does not support pass control")

    async def _handle_restart_from(
        self, request_id: str, params: dict
    ) -> Response:
        """Restart reduction from a specific history point.

        This moves all reductions after the specified point to also-interesting,
        resets the current test case to that point, and modifies the
        interestingness test to reject previously reduced values.
        """
        reduction_number = params.get("reduction_number")
        if reduction_number is None:
            return Response(id=request_id, error="reduction_number is required")

        if self.state is None or self.state.history_manager is None:
            return Response(id=request_id, error="History not available")

        # Restart only works with single-file reductions
        if not isinstance(self.state, ShrinkRayStateSingleFile):
            return Response(
                id=request_id,
                error="Restart from history not supported for directory reductions",
            )

        # Cancel current reduction if running
        if self._cancel_scope is not None:
            self._cancel_scope.cancel()
        self.running = False

        try:
            # Clear old test output to avoid showing stale output from before restart
            if self.state.output_manager is not None:
                self.state.output_manager.cleanup_all()

            # Get restart data from history manager
            new_test_case, excluded_set = (
                self.state.history_manager.restart_from_reduction(reduction_number)
            )

            # Reset state with new initial and exclusions
            self.state.reset_for_restart(new_test_case, excluded_set)

            # Write new test case to file
            await self.state.write_test_case_to_file(
                self.state.filename, new_test_case
            )

            # Get fresh reducer (will create new problem)
            self.reducer = self.state.reducer
            self.problem = self.reducer.target

            # Reset size history for progress tracking
            self._size_history = [(0.0, len(new_test_case))]
            self._last_sent_history_index = 0

            # Signal that we need to restart the reduction loop
            self._restart_requested = True
            self.running = True
            return Response(
                id=request_id,
                result={"status": "restarted", "size": len(new_test_case)},
            )
        except FileNotFoundError:
            return Response(
                id=request_id, error=f"Reduction {reduction_number} not found"
            )
        except Exception as e:
            traceback.print_exc()
            return Response(id=request_id, error=str(e))

    def _get_test_output_preview(self) -> tuple[str, int | None, int | None]:
        """Get preview of current test output, test ID, and return code.

        Returns (content, test_id, return_code) where:
        - content: the last 4KB of the output file
        - test_id: the test ID being displayed
        - return_code: None if test is still running, otherwise the exit code
        """
        if self.state is None or self.state.output_manager is None:
            return "", None, None

        output_path, test_id, return_code = (
            self.state.output_manager.get_current_output()
        )

        if output_path is None:
            return "", None, None

        # Read last 4KB of file
        try:
            with open(output_path, "rb") as f:
                f.seek(0, 2)  # Seek to end
                size = f.tell()
                if size > 4096:
                    f.seek(-4096, 2)
                else:
                    f.seek(0)
                data = f.read()
            return (
                data.decode("utf-8", errors="replace"),
                test_id,
                return_code,
            )
        except OSError:
            return "", test_id, return_code

    def _get_content_preview(self) -> tuple[str, bool]:
        """Get a preview of the current test case content."""
        if self.problem is None:
            return "", False

        test_case = self.problem.current_test_case

        # Handle directory mode
        if isinstance(test_case, dict):
            lines = []
            for name, content in sorted(test_case.items()):
                size = len(content)
                lines.append(f"{name}: {size} bytes")
            return "\n".join(lines[:50]), False

        # Handle single file mode
        hex_mode = is_binary_string(
            test_case[:1024] if len(test_case) > 1024 else test_case
        )

        if hex_mode:
            # Show hex dump for binary files
            preview_bytes = test_case[:512]
            lines = []
            for i in range(0, len(preview_bytes), 16):
                chunk = preview_bytes[i : i + 16]
                hex_part = " ".join(f"{b:02x}" for b in chunk)
                ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                lines.append(f"{i:08x}  {hex_part:<48}  {ascii_part}")
            return "\n".join(lines), True
        else:
            # Show text for text files
            # Send up to 100KB of content - the TUI will handle display truncation
            try:
                text = test_case.decode("utf-8", errors="replace")
                # Truncate by character count to handle files with very long lines
                max_chars = 100_000
                if len(text) > max_chars:
                    text = text[:max_chars]
                return text, False
            except Exception:
                return "", True

    async def _build_progress_update(self) -> ProgressUpdate | None:
        """Build a progress update from current state."""
        if self.problem is None:
            return None

        stats = self.problem.stats
        runtime = time.time() - stats.start_time
        current_size = stats.current_test_case_size

        # Record size history when size changes or periodically
        # Use 200ms interval for first 5 minutes, then 1s (ticks are at 1-minute intervals)
        history_interval = 1.0 if runtime >= 300 else 0.2

        if not self._size_history:
            # First sample: record initial size at time 0
            self._size_history.append((0.0, stats.initial_test_case_size))
            self._last_recorded_size = stats.initial_test_case_size
            self._last_history_time = 0.0

        if current_size != self._last_recorded_size:
            # Size changed - always record
            self._size_history.append((runtime, current_size))
            self._last_recorded_size = current_size
            self._last_history_time = runtime
        elif runtime - self._last_history_time >= history_interval:
            # No size change but interval passed - record periodic update
            self._size_history.append((runtime, current_size))
            self._last_history_time = runtime

        content_preview, hex_mode = self._get_content_preview()

        # Get parallel workers count and track average
        parallel_workers = 0
        if self.state is not None and hasattr(self.state, "parallel_tasks_running"):
            parallel_workers = self.state.parallel_tasks_running
            self._parallel_samples += 1
            self._parallel_total += parallel_workers

        # Calculate parallelism stats
        average_parallelism = 0.0
        effective_parallelism = 0.0
        if self._parallel_samples > 0:
            average_parallelism = self._parallel_total / self._parallel_samples
            wasteage = (
                stats.wasted_interesting_calls / stats.calls if stats.calls > 0 else 0.0
            )
            effective_parallelism = average_parallelism * (1.0 - wasteage)

        # Collect pass statistics in run order (only those with test evaluations)
        pass_stats_list = []
        current_pass_name = ""
        if self.reducer is not None:
            # Get the currently running pass name
            current_pass = self.reducer.current_reduction_pass
            if current_pass is not None:
                current_pass_name = getattr(current_pass, "__name__", "")

            # Get all stats in the order they were first run
            pass_stats = self.reducer.pass_stats
            if pass_stats is not None:
                all_stats = pass_stats.get_stats_in_order()

                # Only include passes that have made at least one test evaluation
                pass_stats_list = [
                    PassStatsData(
                        pass_name=ps.pass_name,
                        bytes_deleted=ps.bytes_deleted,
                        run_count=ps.run_count,
                        test_evaluations=ps.test_evaluations,
                        successful_reductions=ps.successful_reductions,
                        success_rate=ps.success_rate,
                    )
                    for ps in all_stats
                    if ps.test_evaluations > 0
                ]

        # Get disabled passes
        if self.reducer is not None and hasattr(self.reducer, "disabled_passes"):
            disabled_passes = list(self.reducer.disabled_passes)
        else:
            disabled_passes = []

        # Get test output preview
        test_output_preview, active_test_id, last_return_code = (
            self._get_test_output_preview()
        )

        # Get new size history entries since last update
        new_entries = self._size_history[self._last_sent_history_index :]
        self._last_sent_history_index = len(self._size_history)

        # Get history directory info for history explorer
        history_dir: str | None = None
        target_basename = ""
        if self.state is not None and self.state.history_manager is not None:
            history_dir = self.state.history_manager.history_dir
            target_basename = self.state.history_manager.target_basename

        return ProgressUpdate(
            status=self.reducer.status if self.reducer else "",
            size=stats.current_test_case_size,
            original_size=stats.initial_test_case_size,
            calls=stats.calls,
            reductions=stats.reductions,
            interesting_calls=stats.interesting_calls,
            wasted_calls=stats.wasted_interesting_calls,
            runtime=runtime,
            parallel_workers=parallel_workers,
            average_parallelism=average_parallelism,
            effective_parallelism=effective_parallelism,
            time_since_last_reduction=stats.time_since_last_reduction(),
            content_preview=content_preview,
            hex_mode=hex_mode,
            pass_stats=pass_stats_list,
            current_pass_name=current_pass_name,
            disabled_passes=disabled_passes,
            test_output_preview=test_output_preview,
            active_test_id=active_test_id,
            last_test_return_code=last_return_code,
            new_size_history=new_entries,
            history_dir=history_dir,
            target_basename=target_basename,
        )

    async def emit_progress_updates(self) -> None:
        """Periodically emit progress updates."""
        # Emit initial progress update immediately
        update = await self._build_progress_update()
        if update is not None:
            await self.emit(update)

        while self.running:
            await trio.sleep(0.1)
            update = await self._build_progress_update()
            if update is not None:
                await self.emit(update)

    async def run_reducer(self) -> None:
        """Run the reducer."""
        if self.reducer is None:
            return

        try:
            with trio.CancelScope() as scope:
                self._cancel_scope = scope
                await self.reducer.run()

            # Check for trivial result after successful completion
            if self.state is not None and self.problem is not None:
                trivial_error = self.state.check_trivial_result(self.problem)
                if trivial_error:
                    await self.emit(Response(id="", error=trivial_error))
        except* InvalidInitialExample as excs:
            assert len(excs.exceptions) == 1
            (e,) = excs.exceptions
            # Build a detailed error message for invalid initial examples
            if self.state is not None:
                error_message = await self.state.build_error_message(e)
            else:
                error_message = str(e)
            await self.emit(Response(id="", error=error_message))
        except* Exception as e:
            # Catch any other exception during reduction and emit as error
            traceback.print_exc()
            await self.emit(Response(id="", error=str(e.exceptions[0])))
        finally:
            self._cancel_scope = None
            self.running = False

    async def run(self) -> None:
        """Main entry point for the worker."""
        try:
            async with trio.open_nursery() as nursery:
                await nursery.start(self.read_commands)

                # Wait for start command
                while not self.running:
                    await trio.sleep(0.01)

                # Start progress updates
                nursery.start_soon(self.emit_progress_updates)

                # Run reducer, looping if restart is requested
                while True:
                    self._restart_requested = False
                    await self.run_reducer()

                    # Check if we should restart
                    if not self._restart_requested:
                        break

                    # Wait for restart to be fully set up
                    while not self.running:
                        await trio.sleep(0.01)

                # Emit final progress update before completion
                final_update = await self._build_progress_update()
                if final_update is not None:
                    await self.emit(final_update)

                # Signal completion
                await self.emit(Response(id="", result={"status": "completed"}))
                nursery.cancel_scope.cancel()
        finally:
            # Clean up test output files and temp directory
            if self.state is not None and self.state.output_manager is not None:
                self.state.output_manager.cleanup_all()
            if self._output_dir is not None and os.path.isdir(self._output_dir):
                shutil.rmtree(self._output_dir, ignore_errors=True)


def main() -> None:
    """Entry point for the worker subprocess."""
    worker = ReducerWorker()
    trio.run(worker.run)


if __name__ == "__main__":
    main()
