"""Single-process host for the reducer and the TUI.

The trio-based reducer worker runs in the main interpreter (several of
its dependencies have native modules that cannot load in
subinterpreters), while the asyncio-based textual TUI runs in an
isolated subinterpreter with its own GIL. They communicate over a
socketpair using the JSON line protocol from
``shrinkray.interp.protocol``, exactly as the old worker-subprocess
architecture did over pipes.

Lifecycle: the host starts the TUI entry on a helper thread and runs
the worker in the main thread (which is required both for trio's
signal handling and for forwarding signals to the TUI). Shutdown is
EOF-driven from either direction: the TUI closing its socket tells the
worker to unwind (killing any in-flight interestingness tests), and
the worker's socket closing tells the TUI the reducer has gone away.
"""

import _interpreters
import json
import os
import signal
import socket
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent import interpreters
from contextlib import contextmanager
from typing import Any, TextIO

import trio

from shrinkray.interp.signals import SignalForwarder
from shrinkray.interp.worker import ReducerWorker


# Signals the TUI needs to observe: terminal resizes and job-control
# suspend/resume. They can only be caught in the main interpreter, so
# the host forwards them over a pipe (see shrinkray.interp.signals).
FORWARDED_SIGNALS = (signal.SIGWINCH, signal.SIGTSTP, signal.SIGCONT)


def create_tui_interpreter() -> interpreters.Interpreter:
    """Create the subinterpreter that the TUI runs in.

    ``concurrent.interpreters.create()`` offers no way to customise the
    interpreter's config, and its default (isolated) config disallows
    daemon threads, which textual's terminal writer thread needs. So
    create the interpreter through the low-level module with daemon
    threads allowed and wrap the result in the high-level API.
    """
    config = _interpreters.new_config("isolated", allow_daemon_threads=True)
    interp_id = _interpreters.create(config, reqrefs=True)
    interp = interpreters.Interpreter(interp_id, _ownsref=True)
    # A fresh interpreter computes sys.path from the site defaults
    # alone. Mirror this interpreter's sys.path so that modules resolve
    # identically on both sides (e.g. running from a checkout, or with
    # PYTHONPATH manipulated by the test runner).
    interp.exec(f"import sys\nsys.path = {json.dumps(sys.path)}")
    return interp


class TuiThread:
    """Runs a callable in a subinterpreter on a dedicated thread.

    The callable's return value is stored in ``result``; an exception
    (including failures to even start the callable, such as an import
    error in its module) is stored in ``error``.

    Any sockets in ``close_after`` are closed once the call has
    finished, however it finished. The worker relies on this: EOF on
    its command socket is its signal to shut down, so the host's copy
    of the TUI end must be closed even if the entry never ran.
    """

    def __init__(
        self,
        interp: interpreters.Interpreter,
        entry: Callable[..., int],
        args: Sequence[Any],
        close_after: Sequence[socket.socket] = (),
    ) -> None:
        self._interp = interp
        self._entry = entry
        self._args = tuple(args)
        self._close_after = list(close_after)
        self.result: int | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="shrinkray-tui")

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    def _run(self) -> None:
        try:
            self.result = self._interp.call(self._entry, *self._args)
        except BaseException as e:
            self.error = e
        finally:
            for sock in self._close_after:
                sock.close()


@contextmanager
def redirected_stdio(log_file: TextIO) -> Iterator[None]:
    """Point this interpreter's stdout and stderr at a log file.

    The TUI owns the terminal, so anything the reducer side prints
    would corrupt its display. The subinterpreter has its own sys
    module and is unaffected.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


class SocketSendStream:
    """Adapts a trio stream to the worker's OutputStream protocol.

    The worker emits messages from several tasks at once (progress
    updates, command responses, completion), and trio streams forbid
    concurrent send_all calls, so sends are serialised with a lock —
    which also keeps each protocol line intact on the wire.
    """

    def __init__(self, stream: trio.SocketStream) -> None:
        self._stream = stream
        self._lock = trio.Lock()

    async def send(self, data: bytes) -> None:
        try:
            async with self._lock:
                await self._stream.send_all(data)
        except (trio.ClosedResourceError, trio.BrokenResourceError):
            # The TUI has quit: it closed its end of the socket (or the
            # worker's command reader closed the shared stream on EOF)
            # while a progress update was still being emitted. Nobody is
            # listening, so dropping the message is correct — raising
            # would escape the worker's nursery and turn a clean quit
            # into a crash.
            pass


async def _run_worker(worker_sock: socket.socket) -> None:
    """Run the reducer worker over its end of the socketpair."""
    stream = trio.SocketStream(trio.socket.from_stdlib_socket(worker_sock.dup()))
    async with stream:
        worker = ReducerWorker(
            input_stream=stream,
            output_stream=SocketSendStream(stream),
        )
        await worker.run()


def run_with_tui_interpreter(entry: Callable[..., int], params: dict[str, Any]) -> int:
    """Run the reducer worker here and ``entry`` in a TUI subinterpreter.

    ``entry`` must be a module-level callable; it is invoked in the
    subinterpreter as ``entry(sock_fd, signal_read_fd, signal_write_fd,
    params_json)`` where ``sock_fd`` is the TUI's end of the worker
    socketpair (which it must ``os.dup`` before wrapping, since the
    host retains ownership of the original), the signal fds carry
    forwarded signals for a SignalShim, and ``params_json`` is
    ``params`` as JSON (Interpreter.call only accepts shareable
    argument types, which excludes dicts). Returns the entry's return
    value, or raises its exception (or the worker's, if the worker
    failed).

    Must be called from the main thread: both signal forwarding and the
    worker's own signal handling need it.
    """
    worker_sock, ui_sock = socket.socketpair()
    signal_read_fd, signal_write_fd = os.pipe()
    interp = create_tui_interpreter()

    # Capture any stray output from the reducer side while the TUI has
    # the terminal. This mainly catches bootstrap errors: once a
    # reduction starts with history enabled, the worker redirects its
    # stderr to the per-run history directory.
    log_fd, log_path = tempfile.mkstemp(prefix="shrinkray-log-", suffix=".log")
    log_file = os.fdopen(log_fd, "w", encoding="utf-8")

    thread = TuiThread(
        interp,
        entry,
        (ui_sock.fileno(), signal_read_fd, signal_write_fd, json.dumps(params)),
        close_after=[ui_sock],
    )
    try:
        with redirected_stdio(log_file):
            with SignalForwarder(signal_write_fd, FORWARDED_SIGNALS):
                thread.start()
                try:
                    trio.run(_run_worker, worker_sock)
                finally:
                    # If the worker failed, closing its socket is what
                    # tells the TUI to exit; without it the join below
                    # would deadlock.
                    worker_sock.close()
                    thread.join()
    finally:
        os.close(signal_read_fd)
        os.close(signal_write_fd)
        interp.close()
        log_file.close()
        os.unlink(log_path)

    if thread.error is not None:
        raise thread.error
    return thread.result or 0
