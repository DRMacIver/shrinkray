"""Tests for signal forwarding between the main interpreter and the TUI
subinterpreter.

These tests run entirely in the main interpreter: SignalShim only
manipulates the signal module's ``signal`` attribute and a pipe-reading
thread, so its behaviour is identical in a subinterpreter.
"""

import os
import signal
import threading
from unittest.mock import patch

import pytest

from shrinkray.interp.signals import (
    STOP_BYTE,
    SignalForwarder,
    SignalShim,
    _stop_current_process,
)


@pytest.fixture
def shim_pipe():
    read_fd, write_fd = os.pipe()
    yield read_fd, write_fd
    for fd in (read_fd, write_fd):
        try:
            os.close(fd)
        except OSError:
            pass


@pytest.fixture
def shim(shim_pipe):
    read_fd, write_fd = shim_pipe
    shim = SignalShim(read_fd, write_fd)
    real_signal = signal.signal
    shim.install()
    try:
        yield shim
    finally:
        shim.stop()
        assert signal.signal is real_signal


# === Handler recording ===


def test_shim_records_handler_and_returns_previous(shim):
    def handler(signum, frame):
        pass

    previous = signal.signal(signal.SIGWINCH, handler)
    assert previous is signal.SIG_DFL

    def handler2(signum, frame):
        pass

    previous = signal.signal(signal.SIGWINCH, handler2)
    assert previous is handler


def test_shim_does_not_touch_real_signal_handlers(shim):
    real_handler = signal.getsignal(signal.SIGWINCH)

    signal.signal(signal.SIGWINCH, lambda signum, frame: None)
    assert signal.getsignal(signal.SIGWINCH) is real_handler


def test_shim_install_is_not_reentrant(shim):
    with pytest.raises(RuntimeError, match="already installed"):
        shim.install()


# === Dispatch ===


def _wait_for(event: threading.Event) -> None:
    assert event.wait(timeout=5), "handler was never called"


def test_shim_dispatches_forwarded_signal_to_recorded_handler(shim, shim_pipe):
    _, write_fd = shim_pipe
    called = threading.Event()
    seen = []

    def handler(signum, frame):
        seen.append((signum, frame))
        called.set()

    signal.signal(signal.SIGWINCH, handler)
    os.write(write_fd, bytes([signal.SIGWINCH]))
    _wait_for(called)
    assert seen == [(signal.SIGWINCH, None)]


def test_shim_ignores_signal_with_no_recorded_handler(shim, shim_pipe):
    _, write_fd = shim_pipe
    called = threading.Event()

    def handler(signum, frame):
        called.set()

    # A handler for a different signal must not be invoked.
    signal.signal(signal.SIGWINCH, handler)
    os.write(write_fd, bytes([signal.SIGUSR1]))
    # Follow with the one we do handle so we can tell dispatch happened.
    os.write(write_fd, bytes([signal.SIGWINCH]))
    _wait_for(called)


def test_shim_ignores_sig_ign_and_sig_dfl_handlers(shim, shim_pipe):
    _, write_fd = shim_pipe
    called = threading.Event()

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR2, signal.SIG_DFL)
    os.write(write_fd, bytes([signal.SIGUSR1]))
    os.write(write_fd, bytes([signal.SIGUSR2]))

    def handler(signum, frame):
        called.set()

    signal.signal(signal.SIGWINCH, handler)
    os.write(write_fd, bytes([signal.SIGWINCH]))
    _wait_for(called)


def test_stop_current_process_sends_sigstop_to_self():
    with patch("shrinkray.interp.signals.os.kill") as kill:
        _stop_current_process()
    kill.assert_called_once_with(os.getpid(), signal.SIGSTOP)


def test_shim_emulates_default_stop_for_unhandled_sigtstp(shim, shim_pipe):
    _, write_fd = shim_pipe
    stopped = threading.Event()

    with patch(
        "shrinkray.interp.signals._stop_current_process",
        side_effect=lambda: stopped.set(),
    ):
        os.write(write_fd, bytes([signal.SIGTSTP]))
        _wait_for(stopped)


def test_shim_handler_exceptions_do_not_kill_dispatch(shim, shim_pipe):
    _, write_fd = shim_pipe
    called = threading.Event()

    def bad_handler(signum, frame):
        raise ValueError("boom")

    def good_handler(signum, frame):
        called.set()

    signal.signal(signal.SIGUSR1, bad_handler)
    signal.signal(signal.SIGWINCH, good_handler)
    os.write(write_fd, bytes([signal.SIGUSR1]))
    os.write(write_fd, bytes([signal.SIGWINCH]))
    _wait_for(called)


# === Shutdown ===


def test_shim_stop_is_idempotent(shim_pipe):
    read_fd, write_fd = shim_pipe
    shim = SignalShim(read_fd, write_fd)
    shim.install()
    shim.stop()
    shim.stop()


def test_shim_stop_without_install_is_a_noop(shim_pipe):
    read_fd, write_fd = shim_pipe
    shim = SignalShim(read_fd, write_fd)
    shim.stop()


def test_shim_thread_exits_on_eof(shim_pipe):
    read_fd, write_fd = shim_pipe
    shim = SignalShim(read_fd, write_fd)
    real_signal = signal.signal
    shim.install()
    try:
        os.close(write_fd)
        assert shim._thread is not None
        shim.thread_for_testing.join(timeout=5)
        assert not shim.thread_for_testing.is_alive()
    finally:
        shim.stop()
    assert signal.signal is real_signal


def test_stop_byte_is_not_a_valid_signal():
    # The stop sentinel must never collide with a forwardable signal
    # number; 0 is not a valid signal.
    assert STOP_BYTE == 0


# === Host-side forwarding ===


def test_forwarder_forwards_signal_as_byte(shim_pipe):
    read_fd, write_fd = shim_pipe
    with SignalForwarder(write_fd, [signal.SIGUSR1]):
        os.kill(os.getpid(), signal.SIGUSR1)
        data = os.read(read_fd, 1)
    assert data == bytes([signal.SIGUSR1])


def test_forwarder_restores_previous_handlers(shim_pipe):
    _, write_fd = shim_pipe
    seen = []

    def previous(signum, frame):
        seen.append(signum)

    old = signal.signal(signal.SIGUSR1, previous)
    try:
        with SignalForwarder(write_fd, [signal.SIGUSR1]):
            assert signal.getsignal(signal.SIGUSR1) is not previous
        assert signal.getsignal(signal.SIGUSR1) is previous
        os.kill(os.getpid(), signal.SIGUSR1)
        assert seen == [signal.SIGUSR1]
    finally:
        signal.signal(signal.SIGUSR1, old)
