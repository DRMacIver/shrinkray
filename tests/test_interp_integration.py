"""Integration tests for subprocess communication."""

import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time

import pytest

from shrinkray.interp import client, worker
from shrinkray.interp.client import SubprocessClient


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# === Worker module tests ===


def test_worker_can_be_imported():
    """Test that the worker module can be imported."""

    assert hasattr(worker, "ReducerWorker")
    assert hasattr(worker, "main")


def test_worker_module_runs_as_main():
    """Test that the worker module can be executed as a subprocess."""
    # Start the worker process
    proc = subprocess.Popen(
        [sys.executable, "-m", "shrinkray.interp.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Send a status command
    command = b'{"id":"test-1","command":"status","params":{}}\n'
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(command)
    proc.stdin.flush()

    ready, _, _ = select.select([proc.stdout], [], [], 5.0)
    if ready:
        response = proc.stdout.readline()
        assert b'"id":"test-1"' in response
        assert (
            b'"running":false' in response.lower()
            or b'"running": false' in response.lower()
        )

    # Clean up
    proc.terminate()
    proc.wait(timeout=5)


# === Client module tests ===


def test_client_can_be_imported():
    """Test that the client module can be imported."""

    assert hasattr(client, "SubprocessClient")


def test_subprocess_client_has_expected_methods():
    """Test that SubprocessClient has the expected interface."""

    client = SubprocessClient()
    assert hasattr(client, "start")
    assert hasattr(client, "close")
    assert hasattr(client, "send_command")
    assert hasattr(client, "start_reduction")
    assert hasattr(client, "get_status")
    assert hasattr(client, "cancel")
    assert hasattr(client, "get_progress_updates")


# === Protocol with worker tests ===


def test_worker_handles_unknown_command():
    """Test that the worker returns an error for unknown commands."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "shrinkray.interp.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Send an unknown command
    command = b'{"id":"test-2","command":"unknown_command","params":{}}\n'
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(command)
    proc.stdin.flush()

    ready, _, _ = select.select([proc.stdout], [], [], 5.0)
    if ready:
        response = proc.stdout.readline()
        assert b'"id":"test-2"' in response
        assert b'"error"' in response
        assert b"Unknown command" in response

    # Clean up
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.slow
def test_worker_handles_malformed_json():
    """Test that the worker handles malformed JSON gracefully."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "shrinkray.interp.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Send malformed JSON
    command = b"not valid json\n"
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(command)
    proc.stdin.flush()

    # The worker should not crash - send a valid command after
    valid_command = b'{"id":"test-3","command":"status","params":{}}\n'
    proc.stdin.write(valid_command)
    proc.stdin.flush()

    responses = []
    for _ in range(2):  # Try to read up to 2 responses
        ready, _, _ = select.select([proc.stdout], [], [], 2.0)
        if ready:
            response = proc.stdout.readline()
            if response:
                responses.append(response)

    # Should have at least one response (the error and/or the status)
    assert len(responses) >= 1

    # Clean up
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.slow
def test_sigterm_kills_running_interestingness_tests(tmp_path):
    """Regression test: quitting the TUI sends SIGTERM to the worker
    (SubprocessClient.close). The worker used to die without unwinding,
    orphaning the process groups of in-flight interestingness tests,
    which kept running indefinitely."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # The interestingness test records its PID and then hangs.
    pid_file = tmp_path / "test_pids"
    script = tmp_path / "test.sh"
    script.write_text(
        f"#!/bin/sh\necho $$ >> {shlex.quote(str(pid_file))}\nsleep 1000\n"
    )
    script.chmod(0o755)

    start_command = {
        "id": "start-1",
        "command": "start",
        "params": {
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
        },
    }

    proc = subprocess.Popen(
        [sys.executable, "-m", "shrinkray.interp.worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(start_command) + "\n").encode("utf-8"))
        proc.stdin.flush()

        # Wait for the interestingness test to be running.
        deadline = time.time() + 30
        while time.time() < deadline and not pid_file.exists():
            time.sleep(0.05)
        assert pid_file.exists(), "Interestingness test never started"
        test_pid = int(pid_file.read_text().split()[0])
        assert _pid_alive(test_pid)

        # Quit the way SubprocessClient.close() does.
        proc.terminate()
        proc.wait(timeout=10)

        # The hanging test process must be killed by the worker's cleanup.
        deadline = time.time() + 10
        while time.time() < deadline and _pid_alive(test_pid):
            time.sleep(0.05)
        assert not _pid_alive(test_pid), (
            f"Interestingness test (pid {test_pid}) was orphaned"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        # Best effort: never leak the sleeping test process even on failure.
        try:
            os.killpg(os.getpgid(int(pid_file.read_text().split()[0])), signal.SIGKILL)
        except (OSError, ValueError, FileNotFoundError, ProcessLookupError):
            pass
