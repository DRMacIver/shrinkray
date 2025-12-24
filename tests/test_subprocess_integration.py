"""Integration tests for subprocess communication."""

import subprocess
import sys

import pytest


# === Worker module tests ===


def test_worker_can_be_imported():
    """Test that the worker module can be imported."""
    from shrinkray.subprocess import worker

    assert hasattr(worker, "ReducerWorker")
    assert hasattr(worker, "main")


def test_worker_module_runs_as_main():
    """Test that the worker module can be executed as a subprocess."""
    # Start the worker process
    proc = subprocess.Popen(
        [sys.executable, "-m", "shrinkray.subprocess.worker"],
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

    # Read response with timeout
    import select

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
    from shrinkray.subprocess import client

    assert hasattr(client, "SubprocessClient")


def test_subprocess_client_has_expected_methods():
    """Test that SubprocessClient has the expected interface."""
    from shrinkray.subprocess import SubprocessClient

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
        [sys.executable, "-m", "shrinkray.subprocess.worker"],
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

    # Read response with timeout
    import select

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
        [sys.executable, "-m", "shrinkray.subprocess.worker"],
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

    # Read responses with timeout
    import select

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
