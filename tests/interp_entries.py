"""Entry callables for the subinterpreter host tests.

Interpreter.call imports the callable's module inside the
subinterpreter, so this module must only (transitively) import modules
that can load there. In particular it must not import
shrinkray.interp.host or shrinkray.interp.worker, whose dependencies
include native modules (tree-sitter, libcst) that refuse to load in
subinterpreters. This mirrors the constraint on the real TUI entry
point in shrinkray.tui.

Interpreter.call reconstructs the called function without its module
globals for any functions defined in its body, so entries delegate to
module-level helpers (which run with their real globals) instead of
defining anything nested.

These functions do not contribute coverage (the coverage tracer only
sees the main interpreter), so they are kept trivial.
"""

import _interpreters
import asyncio
import json
import os
import socket

from shrinkray.interp.client import WorkerClient


def entry_returns_exit_code(
    sock_fd: int, signal_read_fd: int, signal_write_fd: int, params_json: str
) -> int:
    sock = socket.socket(fileno=os.dup(sock_fd))
    sock.close()
    return json.loads(params_json)["exit_code"]


def entry_raises(
    sock_fd: int, signal_read_fd: int, signal_write_fd: int, params_json: str
) -> int:
    raise ValueError("tui exploded")


def entry_records_interpreter(
    sock_fd: int, signal_read_fd: int, signal_write_fd: int, params_json: str
) -> int:
    sock = socket.socket(fileno=os.dup(sock_fd))
    sock.close()
    interp_id, _ = _interpreters.get_current()
    main_id, _ = _interpreters.get_main()
    return 0 if interp_id != main_id else 1


async def _status_roundtrip(sock: socket.socket) -> int:
    client = WorkerClient(sock)
    await client.start()
    response = await client.get_status()
    assert response.result == {"running": False}
    await client.close()
    return 7


def entry_sends_start_and_quits(
    sock_fd: int, signal_read_fd: int, signal_write_fd: int, params_json: str
) -> int:
    """Minimal stand-in for the TUI: issue a status command, then quit."""
    sock = socket.socket(fileno=os.dup(sock_fd))
    return asyncio.run(_status_roundtrip(sock))


def simple_entry(value: int) -> int:
    return value * 2


def failing_entry() -> int:
    raise RuntimeError("nope")
