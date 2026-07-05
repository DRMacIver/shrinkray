"""Integration tests for the single-process reducer/TUI architecture."""

import asyncio
import json
import os
import shlex
import signal
import socket
import threading
import time

import pytest
import trio

from shrinkray.interp.host import (
    create_tui_interpreter,
    run_with_tui_interpreter,
)
from shrinkray.interp.protocol import Request, Response, deserialize, serialize
from shrinkray.interp.worker import ReducerWorker
from shrinkray.tui import run_tui_in_interpreter
from tests.test_interp_worker import MemoryOutputStream


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# === Subinterpreter import safety ===


@pytest.mark.parametrize(
    "module",
    [
        pytest.param("shrinkray.tui", id="tui"),
        pytest.param("shrinkray.interp.client", id="client"),
        pytest.param("shrinkray.interp.signals", id="signals"),
    ],
)
def test_module_imports_in_subinterpreter(module):
    """Modules the TUI entry needs must stay loadable in a subinterpreter.

    Interpreter.call imports the entry's module inside the
    subinterpreter, so shrinkray.tui must never grow a (transitive)
    import of a native module that refuses to load there (libcst,
    tree-sitter, black, ...).
    """
    interp = create_tui_interpreter()
    try:
        interp.exec(f"import {module}")
    finally:
        interp.close()


# === Worker shutdown behaviour ===


class ClosableInputStream:
    """An input stream that blocks until closed, then signals EOF."""

    def __init__(self, initial: bytes):
        self._chunks: list[bytes] = [initial] if initial else []
        self._closed = trio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        await self._closed.wait()
        raise StopAsyncIteration

    def close(self) -> None:
        self._closed.set()

    async def aclose(self) -> None:
        pass


@pytest.mark.slow
async def test_eof_kills_running_interestingness_tests(tmp_path):
    """Regression test: quitting the TUI closes the worker's command
    stream. The worker must unwind the reduction, killing the process
    groups of in-flight interestingness tests, which otherwise keep
    running indefinitely."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # The interestingness test records its PID and then hangs.
    pid_file = tmp_path / "test_pids"
    script = tmp_path / "test.sh"
    script.write_text(
        f"#!/bin/sh\necho $$ >> {shlex.quote(str(pid_file))}\nsleep 1000\n"
    )
    script.chmod(0o755)

    start_request = Request(
        id="start-1",
        command="start",
        params={
            "file_path": str(target),
            "test": [str(script)],
            "parallelism": 1,
            "timeout": 100.0,
            "seed": 0,
            "input_type": "all",
            "in_place": False,
            "formatter": "none",
            "volume": "quiet",
            "history_enabled": False,
            # The TUI always sends skip_validation (main() validates
            # before the worker starts). Without it the start handler
            # would run the hanging test inline and the worker could
            # not see our EOF until that call timed out.
            "skip_validation": True,
        },
    )

    input_stream = ClosableInputStream((serialize(start_request) + "\n").encode("utf-8"))
    worker = ReducerWorker(input_stream=input_stream, output_stream=MemoryOutputStream())

    test_pid: int | None = None
    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(worker.run)

            # Wait for the interestingness test to be running.
            with trio.fail_after(30):
                while not pid_file.exists() or not pid_file.read_text().strip():
                    await trio.sleep(0.05)
            test_pid = int(pid_file.read_text().split()[0])
            assert _pid_alive(test_pid)

            # Quit the way the TUI does: close the command stream.
            input_stream.close()

        # The nursery exiting means worker.run() has finished its cleanup.
        deadline = time.time() + 10
        while time.time() < deadline and _pid_alive(test_pid):
            await trio.sleep(0.05)
        assert not _pid_alive(test_pid), (
            f"Interestingness test (pid {test_pid}) was orphaned"
        )
    finally:
        # Best effort: never leak the sleeping test process even on failure.
        if test_pid is not None:
            try:
                os.killpg(os.getpgid(test_pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


# === Full single-process run ===


@pytest.mark.slow
def test_full_reduction_through_the_host(tmp_path):
    """Drive a real reduction end to end through the subinterpreter host.

    The TUI entry runs headless in the subinterpreter while the real
    worker reduces a file in the main interpreter.
    """
    target = tmp_path / "test.txt"
    target.write_text("hello world\ngoodbye world\n")

    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q hello "$1"\n')
    script.chmod(0o755)

    exit_code = run_with_tui_interpreter(
        run_tui_in_interpreter,
        {
            "file_path": str(target),
            "test": [str(script)],
            "parallelism": 2,
            "timeout": 30.0,
            "input_type": "arg",
            "formatter": "none",
            "volume": "quiet",
            "trivial_is_error": False,
            "exit_on_completion": True,
            "history_enabled": False,
            "headless": True,
        },
    )

    assert exit_code == 0
    result = target.read_text()
    assert "hello" in result
    assert len(result) < len("hello world\ngoodbye world\n")


def test_run_tui_in_interpreter_runs_headless_against_a_worker(tmp_path):
    """Exercise the subinterpreter entry point in the main interpreter.

    The entry is normally only executed inside the subinterpreter,
    where the coverage tracer cannot see it; here it runs directly (it
    only touches fds and the signal shim, which work in any
    interpreter), headless, against a scripted worker on the other end
    of the socketpair.
    """
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    client_sock, worker_sock = socket.socketpair()
    signal_read_fd, signal_write_fd = os.pipe()

    def fake_worker() -> None:
        async def run() -> None:
            reader, writer = await asyncio.open_connection(sock=worker_sock)
            request = deserialize((await reader.readline()).decode("utf-8"))
            assert isinstance(request, Request)
            assert request.command == "start"
            for response in [
                Response(id=request.id, result={"status": "started"}),
                Response(id="", result={"status": "completed"}),
            ]:
                writer.write((serialize(response) + "\n").encode("utf-8"))
                await writer.drain()
            # Wait for the TUI to close its end.
            await reader.read()
            writer.close()

        asyncio.run(run())

    worker_thread = threading.Thread(target=fake_worker)
    worker_thread.start()
    try:
        exit_code = run_tui_in_interpreter(
            client_sock.fileno(),
            signal_read_fd,
            signal_write_fd,
            json.dumps(
                {
                    "file_path": str(target),
                    "test": ["true"],
                    "volume": "quiet",
                    "trivial_is_error": False,
                    "exit_on_completion": True,
                    "history_enabled": False,
                    "headless": True,
                }
            ),
        )
    finally:
        # The entry works on a dup of the socket; the worker sees EOF
        # only once this copy is closed too (in the real host,
        # TuiThread does this after the entry returns).
        client_sock.close()
        worker_thread.join(timeout=30)
        os.close(signal_read_fd)
        os.close(signal_write_fd)

    assert exit_code == 0
    assert not worker_thread.is_alive()


def test_start_params_round_trip_as_json():
    """The host passes TUI params as JSON; everything main() sends must
    survive the round trip (including float('inf') timeouts)."""
    params = {
        "file_path": "x",
        "test": ["./t.sh"],
        "timeout": float("inf"),
        "parallelism": None,
        "external_reducers": [["creduce"]],
    }
    decoded = json.loads(json.dumps(params))
    assert decoded["timeout"] == float("inf")
    assert decoded["parallelism"] is None
    assert decoded["external_reducers"] == [["creduce"]]
