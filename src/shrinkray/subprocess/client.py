"""Client for communicating with the reducer subprocess."""

import asyncio
import os
import sys
import tempfile
import traceback
import uuid
from collections.abc import AsyncGenerator
from typing import IO, Any

from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


class SubprocessClient:
    """Client for communicating with the reducer subprocess via JSON protocol."""

    def __init__(self, debug_mode: bool = False):
        self._process: asyncio.subprocess.Process | None = None
        self._pending_responses: dict[str, asyncio.Future[Response]] = {}
        self._progress_queue: asyncio.Queue[ProgressUpdate] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._completed = False
        self._error_message: str | None = None
        self._debug_mode = debug_mode
        self._stderr_log_file: IO[str] | None = None
        self._stderr_log_path: str | None = None

    async def start(self) -> None:
        """Launch the subprocess."""
        # Log subprocess stderr to a temp file for debugging.
        # This captures bootstrap errors before history is set up.
        # Once the worker starts with history enabled, it redirects its own
        # stderr to the per-run history directory.
        fd, self._stderr_log_path = tempfile.mkstemp(
            prefix="shrinkray-stderr-",
            suffix=".log",
        )
        self._stderr_log_file = os.fdopen(fd, "w", encoding="utf-8")

        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "shrinkray.subprocess.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._stderr_log_file,
        )
        self._reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        """Read and dispatch messages from subprocess stdout."""
        if self._process is None or self._process.stdout is None:
            return

        buffer = b""
        while True:
            try:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        await self._handle_message(line.decode("utf-8"))
            except Exception:
                traceback.print_exc()
                break

    async def _handle_message(self, line: str) -> None:
        """Handle a message from the subprocess."""
        try:
            msg = deserialize(line)
        except Exception:
            traceback.print_exc()
            return

        if isinstance(msg, ProgressUpdate):
            await self._progress_queue.put(msg)
        elif isinstance(msg, Response):
            # Check for completion or error signal (unsolicited responses with empty id)
            if msg.id == "":
                if msg.result and msg.result.get("status") == "completed":
                    self._completed = True
                    # Wake up any pending futures
                    for future in self._pending_responses.values():
                        if not future.done():
                            future.set_exception(Exception("Subprocess completed"))
                elif msg.error:
                    self._completed = True
                    self._error_message = msg.error
                    # Wake up any pending futures with the error
                    for future in self._pending_responses.values():
                        if not future.done():
                            future.set_exception(Exception(msg.error))
                return

            # Match response to pending request
            if msg.id in self._pending_responses:
                future = self._pending_responses.pop(msg.id)
                if not future.done():
                    future.set_result(msg)

    async def send_command(
        self, command: str, params: dict[str, Any] | None = None
    ) -> Response:
        """Send a command to the subprocess and wait for response."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Subprocess not started")

        request_id = str(uuid.uuid4())
        request = Request(id=request_id, command=command, params=params or {})

        # Create future for response
        future: asyncio.Future[Response] = asyncio.get_event_loop().create_future()
        self._pending_responses[request_id] = future

        # Send request
        line = serialize(request) + "\n"
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

        # Wait for response
        try:
            return await future
        except Exception:
            self._pending_responses.pop(request_id, None)
            raise

    async def start_reduction(
        self,
        file_path: str,
        test: list[str],
        parallelism: int | None = None,
        timeout: float | None = None,
        seed: int = 0,
        input_type: str = "all",
        in_place: bool = False,
        formatter: str = "default",
        volume: str = "normal",
        no_clang_delta: bool = False,
        clang_delta: str = "",
        trivial_is_error: bool = True,
        skip_validation: bool = False,
        history_enabled: bool = True,
        also_interesting_code: int | None = None,
    ) -> Response:
        """Start the reduction process."""
        params: dict[str, Any] = {
            "file_path": file_path,
            "test": test,
            "seed": seed,
            "input_type": input_type,
            "in_place": in_place,
            "formatter": formatter,
            "volume": volume,
            "no_clang_delta": no_clang_delta,
            "clang_delta": clang_delta,
            "trivial_is_error": trivial_is_error,
            "skip_validation": skip_validation,
            "history_enabled": history_enabled,
            "also_interesting_code": also_interesting_code,
        }
        if parallelism is not None:
            params["parallelism"] = parallelism
        if timeout is not None:
            params["timeout"] = timeout
        return await self.send_command("start", params)

    async def get_status(self) -> Response:
        """Get current reduction status."""
        return await self.send_command("status")

    async def cancel(self) -> Response:
        """Cancel the reduction."""
        if self._completed:
            return Response(id="", result={"status": "already_completed"})
        if self._process is None or self._process.returncode is not None:
            return Response(id="", result={"status": "process_exited"})
        try:
            return await self.send_command("cancel")
        except Exception:
            return Response(id="", result={"status": "cancelled"})

    async def disable_pass(self, pass_name: str) -> Response:
        """Disable a reduction pass by name."""
        if self._completed:
            return Response(id="", result={"status": "already_completed"})
        try:
            return await self.send_command("disable_pass", {"pass_name": pass_name})
        except Exception:
            traceback.print_exc()
            return Response(id="", error="Failed to disable pass")

    async def enable_pass(self, pass_name: str) -> Response:
        """Enable a previously disabled reduction pass."""
        if self._completed:
            return Response(id="", result={"status": "already_completed"})
        try:
            return await self.send_command("enable_pass", {"pass_name": pass_name})
        except Exception:
            traceback.print_exc()
            return Response(id="", error="Failed to enable pass")

    async def skip_current_pass(self) -> Response:
        """Skip the currently running pass."""
        if self._completed:
            return Response(id="", result={"status": "already_completed"})
        try:
            return await self.send_command("skip_pass")
        except Exception:
            traceback.print_exc()
            return Response(id="", error="Failed to skip pass")

    async def restart_from(self, reduction_number: int) -> Response:
        """Restart reduction from a specific history point.

        This moves all reductions after the specified point to also-interesting,
        resets the current test case to that point, and continues reduction
        from there, rejecting previously reduced values.

        Args:
            reduction_number: The reduction entry number to restart from
                (e.g., 3 for reduction 0003)
        """
        if self._completed:
            return Response(id="", error="Reduction already completed")
        try:
            return await self.send_command(
                "restart_from", {"reduction_number": reduction_number}
            )
        except Exception:
            traceback.print_exc()
            return Response(id="", error="Failed to send restart command")

    async def get_progress_updates(self) -> AsyncGenerator[ProgressUpdate, None]:
        """Yield progress updates as they arrive."""
        while not self._completed:
            try:
                update = await asyncio.wait_for(self._progress_queue.get(), timeout=0.5)
                yield update
            except TimeoutError:
                continue

    @property
    def is_completed(self) -> bool:
        """Check if the reduction has completed."""
        return self._completed

    @property
    def error_message(self) -> str | None:
        """Get the error message if the subprocess failed."""
        return self._error_message

    async def close(self) -> None:
        """Close the subprocess."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception:
                    traceback.print_exc()
            # Only terminate if still running
            if self._process.returncode is None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    pass  # Process already exited

        # Close and remove the stderr log file
        if self._stderr_log_file is not None:
            try:
                self._stderr_log_file.close()
            except Exception:
                pass
        if self._stderr_log_path is not None:
            try:
                os.unlink(self._stderr_log_path)
            except Exception:
                pass

    async def __aenter__(self) -> "SubprocessClient":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
