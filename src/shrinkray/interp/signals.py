"""Signal forwarding between the main interpreter and the TUI subinterpreter.

Python only allows OS signal handlers to be installed in the main
thread of the main interpreter, but the textual TUI (which runs in a
subinterpreter) needs to react to signals: SIGWINCH for terminal
resizes and SIGTSTP/SIGCONT for job-control suspend and resume.

The two halves here bridge that gap over a pipe:

- ``SignalForwarder`` runs in the main interpreter. It installs real
  handlers for the forwarded signals that write the signal number as a
  single byte to the pipe.

- ``SignalShim`` runs in the subinterpreter. It replaces
  ``signal.signal`` with a recorder (so that textual's handler
  registrations succeed instead of raising ValueError) and runs a
  thread that reads signal numbers from the pipe and invokes the
  recorded handlers.
"""

import os
import signal
import threading
import traceback
from collections.abc import Iterable
from types import FrameType
from typing import Any


# Written to the pipe to shut down the shim's dispatch thread. Signal 0
# is not a real signal, so it can never collide with a forwarded one.
STOP_BYTE = 0


def _stop_current_process() -> None:
    """Emulate the default SIGTSTP disposition: stop the whole process."""
    os.kill(os.getpid(), signal.SIGSTOP)


class SignalShim:
    """Subinterpreter-side signal handling.

    Records handlers registered via ``signal.signal`` and dispatches
    signal numbers read from ``read_fd`` to them. ``stop()`` writes a
    sentinel to ``write_fd`` to terminate the dispatch thread and
    restores ``signal.signal``. The shim borrows the fds; closing them
    is the caller's responsibility.
    """

    def __init__(self, read_fd: int, write_fd: int) -> None:
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._handlers: dict[int, Any] = {}
        self._original_signal: Any = None
        self._thread: threading.Thread | None = None

    @property
    def thread_for_testing(self) -> threading.Thread:
        assert self._thread is not None
        return self._thread

    def install(self) -> None:
        """Replace signal.signal and start the dispatch thread."""
        if self._thread is not None:
            raise RuntimeError("Signal shim is already installed")
        self._original_signal = signal.signal
        signal.signal = self._record_handler
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="shrinkray-signal-shim",
            daemon=True,
        )
        self._thread.start()

    def _record_handler(self, signum: int, handler: Any) -> Any:
        previous = self._handlers.get(signum, signal.SIG_DFL)
        self._handlers[signum] = handler
        return previous

    def _dispatch_loop(self) -> None:
        while True:
            data = os.read(self._read_fd, 1)
            if not data or data[0] == STOP_BYTE:
                break
            self._dispatch(data[0])

    def _dispatch(self, signum: int) -> None:
        handler = self._handlers.get(signum)
        if callable(handler):
            try:
                frame: FrameType | None = None
                handler(signum, frame)
            except Exception:
                # A misbehaving handler must not kill signal dispatch
                # for the rest of the run.
                traceback.print_exc()
        elif handler in (None, signal.SIG_DFL) and signum == signal.SIGTSTP:
            # The host forwards SIGTSTP instead of letting the default
            # disposition stop the process, so if the TUI has not
            # registered a handler we have to stop the process ourselves.
            _stop_current_process()

    def stop(self) -> None:
        """Terminate the dispatch thread and restore signal.signal."""
        if self._thread is None:
            return
        try:
            os.write(self._write_fd, bytes([STOP_BYTE]))
        except OSError:
            # The pipe is already closed, which also terminates the
            # dispatch thread (it sees EOF).
            pass
        self._thread.join()
        self._thread = None
        signal.signal = self._original_signal
        self._original_signal = None


class SignalForwarder:
    """Main-interpreter-side signal forwarding.

    A context manager that installs handlers for ``signums`` which
    write each received signal's number to ``write_fd``, restoring the
    previous handlers on exit.
    """

    def __init__(self, write_fd: int, signums: Iterable[int]) -> None:
        self._write_fd = write_fd
        self._signums = list(signums)
        self._previous: dict[int, Any] = {}

    def _forward(self, signum: int, frame: FrameType | None) -> None:
        os.write(self._write_fd, bytes([signum]))

    def __enter__(self) -> SignalForwarder:
        for signum in self._signums:
            self._previous[signum] = signal.signal(signum, self._forward)
        return self

    def __exit__(self, *args: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()
