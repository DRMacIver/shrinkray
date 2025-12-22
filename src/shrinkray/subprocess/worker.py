"""Worker subprocess that runs the reducer with trio and communicates via JSON protocol."""

import os
import sys
import time
from typing import Any

import trio
from binaryornot.helpers import is_binary_string

from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


class ReducerWorker:
    """Runs the reducer in a subprocess with JSON protocol communication."""

    def __init__(self):
        self.running = False
        self.reducer = None
        self.problem = None
        self.state = None
        self._cancel_scope: trio.CancelScope | None = None
        # Parallelism tracking
        self._parallel_samples = 0
        self._parallel_total = 0

    def emit(self, msg: Response | ProgressUpdate) -> None:
        """Write a message to stdout."""
        line = serialize(msg) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    async def read_commands(self, task_status=trio.TASK_STATUS_IGNORED) -> None:
        """Read commands from stdin and dispatch them."""
        task_status.started()
        stdin = trio.lowlevel.FdStream(os.dup(sys.stdin.fileno()))
        buffer = b""
        async for chunk in stdin:
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
                self.emit(Response(id="", error="Expected a request"))
                return
            response = await self.handle_command(request)
            self.emit(response)
        except Exception as e:
            self.emit(Response(id="", error=str(e)))

    async def handle_command(self, request: Request) -> Response:
        """Handle a command request and return a response."""
        command = request.command
        params = request.params

        if command == "start":
            return await self._handle_start(request.id, params)
        elif command == "status":
            return self._handle_status(request.id)
        elif command == "cancel":
            return self._handle_cancel(request.id)
        else:
            return Response(id=request.id, error=f"Unknown command: {command}")

    async def _handle_start(self, request_id: str, params: dict) -> Response:
        """Start the reduction process."""
        if self.running:
            return Response(id=request_id, error="Already running")

        try:
            await self._start_reduction(params)
            return Response(id=request_id, result={"status": "started"})
        except Exception as e:
            return Response(id=request_id, error=str(e))

    async def _start_reduction(self, params: dict) -> None:
        """Initialize and start the reduction."""
        # Import here to avoid circular imports
        from shrinkray.__main__ import (
            C_FILE_EXTENSIONS,
            ClangDelta,
            InputType,
            ShrinkRayDirectoryState,
            ShrinkRayStateSingleFile,
            find_clang_delta,
        )
        from shrinkray.work import Volume

        filename = params["file_path"]
        test = params["test"]
        parallelism = params.get("parallelism", os.cpu_count() or 1)
        timeout = params.get("timeout", 1.0)
        seed = params.get("seed", 0)
        input_type = InputType[params.get("input_type", "all")]
        in_place = params.get("in_place", False)
        formatter = params.get("formatter", "default")
        volume = Volume[params.get("volume", "normal")]
        no_clang_delta = params.get("no_clang_delta", False)
        clang_delta_path = params.get("clang_delta", "")

        clang_delta_executable = None
        if os.path.splitext(filename)[1] in C_FILE_EXTENSIONS and not no_clang_delta:
            if not clang_delta_path:
                clang_delta_path = find_clang_delta()
            if clang_delta_path:
                clang_delta_executable = ClangDelta(clang_delta_path)

        state_kwargs: dict[str, Any] = dict(
            input_type=input_type,
            in_place=in_place,
            test=test,
            timeout=timeout,
            base=os.path.basename(filename),
            parallelism=parallelism,
            filename=filename,
            formatter=formatter,
            trivial_is_error=False,
            seed=seed,
            volume=volume,
            clang_delta_executable=clang_delta_executable,
        )

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

    async def emit_progress_updates(self) -> None:
        """Periodically emit progress updates."""
        while self.running:
            await trio.sleep(0.1)
            if self.problem is None:
                continue
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
                    stats.wasted_interesting_calls / stats.calls
                    if stats.calls > 0
                    else 0.0
                )
                effective_parallelism = average_parallelism * (1.0 - wasteage)

            update = ProgressUpdate(
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
            )
            self.emit(update)

    async def run_reducer(self) -> None:
        """Run the reducer."""
        if self.reducer is None:
            return

        try:
            with trio.CancelScope() as scope:
                self._cancel_scope = scope
                await self.reducer.run()
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

            # Signal completion
            self.emit(Response(id="", result={"status": "completed"}))
            nursery.cancel_scope.cancel()


def main() -> None:
    """Entry point for the worker subprocess."""
    worker = ReducerWorker()
    trio.run(worker.run)


if __name__ == "__main__":
    main()
