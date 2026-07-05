"""Tests for the single-process host that runs the reducer in the main
interpreter and the TUI entry in a subinterpreter."""

import socket
import sys
import threading
from unittest.mock import patch

import pytest

from shrinkray.interp.host import (
    TuiThread,
    create_tui_interpreter,
    redirected_stdio,
    run_with_tui_interpreter,
)
from tests.interp_entries import (
    entry_raises,
    entry_records_interpreter,
    entry_returns_exit_code,
    entry_sends_start_and_quits,
    failing_entry,
    simple_entry,
)


# === create_tui_interpreter ===


def test_tui_interpreter_allows_daemon_threads():
    interp = create_tui_interpreter()
    try:
        interp.exec(
            "import threading\n"
            "t = threading.Thread(target=lambda: None, daemon=True)\n"
            "t.start()\n"
            "t.join()\n"
        )
    finally:
        interp.close()


def test_tui_interpreter_is_isolated_from_main():
    interp = create_tui_interpreter()
    try:
        interp.prepare_main(marker=1)
        # A subinterpreter has its own module state.
        sys.modules.setdefault("_shrinkray_host_test_marker", sys)
        interp.exec(
            "import sys\nassert '_shrinkray_host_test_marker' not in sys.modules"
        )
    finally:
        del sys.modules["_shrinkray_host_test_marker"]
        interp.close()


# === TuiThread ===


def test_tui_thread_returns_result():
    interp = create_tui_interpreter()
    try:
        thread = TuiThread(interp, simple_entry, (21,))
        thread.start()
        thread.join()
        assert thread.result == 42
        assert thread.error is None
    finally:
        interp.close()


def test_tui_thread_captures_errors():
    interp = create_tui_interpreter()
    try:
        thread = TuiThread(interp, failing_entry, ())
        thread.start()
        thread.join()
        assert thread.result is None
        assert thread.error is not None
        assert "nope" in str(thread.error)
    finally:
        interp.close()


def test_tui_thread_closes_socket_after_entry():
    interp = create_tui_interpreter()
    try:
        a, b = socket.socketpair()
        thread = TuiThread(interp, simple_entry, (21,), close_after=[a])
        thread.start()
        thread.join()
        # The other end sees EOF once the thread has closed its socket.
        b.settimeout(5)
        assert b.recv(1) == b""
        b.close()
    finally:
        interp.close()


# === redirected_stdio ===


def test_redirected_stdio_swaps_and_restores_streams(tmp_path):
    log_path = tmp_path / "log.txt"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirected_stdio(log_file):
            assert sys.stdout is log_file
            assert sys.stderr is log_file
            print("captured")
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert "captured" in log_path.read_text()


def test_redirected_stdio_restores_streams_on_error(tmp_path):
    log_path = tmp_path / "log.txt"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, "w", encoding="utf-8") as log_file:
        with pytest.raises(ValueError):
            with redirected_stdio(log_file):
                raise ValueError("boom")
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


# === run_with_tui_interpreter ===


def test_run_returns_tui_exit_code():
    assert (run_with_tui_interpreter(entry_returns_exit_code, {"exit_code": 3})) == 3


def test_run_returns_zero_exit_code():
    assert (run_with_tui_interpreter(entry_returns_exit_code, {"exit_code": 0})) == 0


def test_run_entry_executes_in_a_subinterpreter():
    assert run_with_tui_interpreter(entry_records_interpreter, {}) == 0


def test_run_worker_answers_commands_from_the_tui():
    assert run_with_tui_interpreter(entry_sends_start_and_quits, {}) == 7


def test_run_raises_tui_errors_after_cleanup():
    with pytest.raises(Exception, match="tui exploded"):
        run_with_tui_interpreter(entry_raises, {})


def test_run_unblocks_when_worker_crashes():
    async def explode(self) -> None:
        raise RuntimeError("worker exploded")

    with patch("shrinkray.interp.host.ReducerWorker.run", explode):
        with pytest.raises(BaseException, match="worker exploded"):
            run_with_tui_interpreter(entry_sends_start_and_quits, {})


def test_run_restores_stdio():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    run_with_tui_interpreter(entry_returns_exit_code, {"exit_code": 0})
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr


def test_run_leaves_no_extra_threads():
    before = threading.active_count()
    run_with_tui_interpreter(entry_returns_exit_code, {"exit_code": 0})
    assert threading.active_count() <= before
