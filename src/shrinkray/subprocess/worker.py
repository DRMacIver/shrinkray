"""Worker subprocess that runs the reducer with trio and communicates via JSON protocol."""

import os
import sys
import time
import traceback
from contextlib import aclosing
from typing import Any, Protocol

import trio
from binaryornot.helpers import is_binary_string

from shrinkray.problem import InvalidInitialExample
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


class StdoutStream:
    """Wrapper around sys.stdout for the OutputStream protocol."""

    async def send(self, data: bytes) -> None:
        sys.stdout.write(data.decode("utf-8"))
        sys.stdout.flush()


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
        # Parallelism tracking
        self._parallel_samples = 0
        self._parallel_total = 0
        # I/O streams - None means use stdin/stdout
        self._input_stream = input_stream
        self._output_stream = output_stream

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

        return ProgressUpdate(
            status=self.reducer.status if self.reducer else "",
            size=stats.current_test_case_size,
            original_size=stats.initial_test_case_size,
            calls=stats.calls,
            reductions=stats.reductions,
            interesting_calls=stats.interesting_calls,
            wasted_calls=stats.wasted_interesting_calls,
            runtime=time.time() - stats.start_time,
            parallel_workers=parallel_workers,
            average_parallelism=average_parallelism,
            effective_parallelism=effective_parallelism,
            time_since_last_reduction=stats.time_since_last_reduction(),
            content_preview=content_preview,
            hex_mode=hex_mode,
            pass_stats=pass_stats_list,
            current_pass_name=current_pass_name,
            disabled_passes=disabled_passes,
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
        async with trio.open_nursery() as nursery:
            await nursery.start(self.read_commands)

            # Wait for start command
            while not self.running:
                await trio.sleep(0.01)

            # Start progress updates and reducer
            nursery.start_soon(self.emit_progress_updates)
            await self.run_reducer()

            # Emit final progress update before completion
            final_update = await self._build_progress_update()
            if final_update is not None:
                await self.emit(final_update)

            # Signal completion
            await self.emit(Response(id="", result={"status": "completed"}))
            nursery.cancel_scope.cancel()


def main() -> None:
    """Entry point for the worker subprocess."""
    worker = ReducerWorker()
    trio.run(worker.run)


if __name__ == "__main__":
    main()
