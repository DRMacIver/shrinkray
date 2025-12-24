"""Client for communicating with the reducer subprocess."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


class SubprocessClient:
    """Client for communicating with the reducer subprocess via JSON protocol."""

    def __init__(self):
        self._process: asyncio.subprocess.Process | None = None
        self._pending_responses: dict[str, asyncio.Future[Response]] = {}
        self._progress_queue: asyncio.Queue[ProgressUpdate] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._completed = False
        self._error_message: str | None = None

    async def start(self) -> None:
        """Launch the subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "shrinkray.subprocess.worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_output())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """Read and discard stderr to prevent subprocess from blocking."""
        if self._process is None or self._process.stderr is None:
            return

        while True:
            try:
                chunk = await self._process.stderr.read(4096)
                if not chunk:
                    break
            except Exception:
                break

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
                break

    async def _handle_message(self, line: str) -> None:
        """Handle a message from the subprocess."""
        try:
            msg = deserialize(line)
        except Exception:
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
        timeout: float = 1.0,
        seed: int = 0,
        input_type: str = "all",
        in_place: bool = False,
        formatter: str = "default",
        volume: str = "normal",
        no_clang_delta: bool = False,
        clang_delta: str = "",
        trivial_is_error: bool = True,
    ) -> Response:
        """Start the reduction process."""
        params = {
            "file_path": file_path,
            "test": test,
            "timeout": timeout,
            "seed": seed,
            "input_type": input_type,
            "in_place": in_place,
            "formatter": formatter,
            "volume": volume,
            "no_clang_delta": no_clang_delta,
            "clang_delta": clang_delta,
            "trivial_is_error": trivial_is_error,
        }
        if parallelism is not None:
            params["parallelism"] = parallelism
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

    async def get_progress_updates(self) -> AsyncIterator[ProgressUpdate]:
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

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
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

    async def __aenter__(self) -> "SubprocessClient":
        await self.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
