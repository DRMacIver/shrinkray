"""Worker subprocess that runs the reducer with trio and communicates via JSON protocol."""

import os
import sys
from typing import Any

import trio

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
            ShrinkRayStateSingleFile,
            ShrinkRayDirectoryState,
            InputType,
            ClangDelta,
            find_clang_delta,
            C_FILE_EXTENSIONS,
        )
        from shrinkray.work import Volume
        import shutil

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

    async def emit_progress_updates(self) -> None:
        """Periodically emit progress updates."""
        while self.running:
            await trio.sleep(0.1)
            if self.problem is None:
                continue
            stats = self.problem.stats
            update = ProgressUpdate(
                status=self.reducer.status if self.reducer else "",
                size=stats.current_test_case_size,
                original_size=stats.initial_test_case_size,
                calls=stats.calls,
                reductions=stats.reductions,
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
