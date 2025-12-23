"""Tests for process management utilities."""

import os
import signal
import subprocess
import sys

import trio

from shrinkray.process import interrupt_wait_and_kill, signal_group


# === signal_group tests ===


async def test_signal_group_sends_signal_to_process_group():
    # Start a process in its own process group
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "import time; time.sleep(100)"],
        preexec_fn=os.setsid,
    )
    try:
        # Process should be running
        assert sp.poll() is None

        # Send SIGTERM to the group
        signal_group(sp, signal.SIGTERM)

        # Wait for process to exit
        with trio.move_on_after(5):
            await sp.wait()

        assert sp.returncode is not None
    finally:
        if sp.returncode is None:
            sp.kill()
            await sp.wait()


# === interrupt_wait_and_kill tests ===


async def test_interrupt_wait_and_kill_does_nothing_if_already_exited():
    # Start a process that exits immediately
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "pass"],
        preexec_fn=os.setsid,
    )
    # Wait for it to exit
    await sp.wait()
    assert sp.returncode is not None

    # Should not raise
    await interrupt_wait_and_kill(sp)


async def test_interrupt_wait_and_kill_kills_process_ignoring_sigint():
    # Start a process that ignores SIGINT
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGINT, lambda *a: None); time.sleep(100)",
        ],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
    )

    # Give it a moment to set up the signal handler
    await trio.sleep(0.1)

    # This should eventually kill it
    await interrupt_wait_and_kill(sp, delay=0.05)

    assert sp.returncode is not None


async def test_interrupt_wait_and_kill_handles_fast_exit_after_sigint():
    # Start a process that exits quickly after SIGINT
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGINT, lambda *a: exit(0)); time.sleep(100)",
        ],
        preexec_fn=os.setsid,
    )

    await trio.sleep(0.05)
    await interrupt_wait_and_kill(sp, delay=0.05)

    assert sp.returncode is not None
    assert sp.returncode == 0


async def test_interrupt_wait_and_kill_closes_pipes_before_signaling():
    # Start a process with stdout pipe
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); sys.stdout.flush(); import time; time.sleep(100)",
        ],
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )

    # Give it a moment to print
    await trio.sleep(0.1)

    # Should close pipes and kill
    await interrupt_wait_and_kill(sp, delay=0.05)

    assert sp.returncode is not None


async def test_interrupt_wait_and_kill_handles_process_lookup_error_on_sigkill():
    """Test that ProcessLookupError is handled when sending SIGKILL.

    This tests lines 40-41: the process exits between checking returncode
    and sending SIGKILL.
    """
    from unittest.mock import MagicMock, patch, AsyncMock

    # Create a mock process
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    # returncode is None initially, then None at line 37 check,
    # but process disappears before SIGKILL
    poll_returns = iter([None] * 11)  # All poll() calls return None
    mock_sp.poll.side_effect = lambda: next(poll_returns)

    # Track when SIGKILL is attempted
    sigkill_attempted = [False]

    # returncode property: None initially, then 0 after SIGKILL is attempted
    # (process exits between the check and the kill)
    def get_returncode():
        if sigkill_attempted[0]:
            return 0  # Process has exited
        return None

    type(mock_sp).returncode = property(lambda self: get_returncode())

    mock_sp.wait = AsyncMock()

    # signal_group should work for SIGINT but raise ProcessLookupError for SIGKILL
    def mock_signal_group(sp, sig):
        if sig == signal.SIGKILL:
            sigkill_attempted[0] = True
            raise ProcessLookupError("No such process")
        # Don't actually send signal to avoid affecting real processes

    with patch("shrinkray.process.signal_group", mock_signal_group):
        # Should not raise despite ProcessLookupError
        await interrupt_wait_and_kill(mock_sp, delay=0.001)


async def test_interrupt_wait_and_kill_raises_on_unkillable_process():
    """Test that ValueError is raised when process cannot be killed.

    This tests line 47: the process persists after all kill attempts.
    """
    from unittest.mock import MagicMock, patch, AsyncMock
    import pytest

    # Create a mock process that never dies
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    # Always return None for poll (process never exits from SIGINT)
    mock_sp.poll.return_value = None

    # returncode is always None (process never dies)
    type(mock_sp).returncode = property(lambda self: None)

    mock_sp.wait = AsyncMock()

    # signal_group does nothing (signal is ignored)
    with patch("shrinkray.process.signal_group"):
        with pytest.raises(ValueError, match="Could not kill subprocess"):
            await interrupt_wait_and_kill(mock_sp, delay=0.001)


async def test_interrupt_wait_and_kill_skips_sigkill_if_process_exits_after_poll_loop():
    """Test that SIGKILL is skipped if process exits after poll loop completes.

    This tests the branch at line 37 where returncode becomes non-None
    after all poll() calls return None but before checking returncode at line 37.
    """
    from unittest.mock import MagicMock, patch, AsyncMock

    # Create a mock process
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    # poll() always returns None during the loop (so loop completes all iterations)
    mock_sp.poll.return_value = None

    # Track when we're past the poll loop
    poll_loop_done = [False]

    # returncode: None during poll loop, 0 after (simulating process exiting
    # after the loop but before the check at line 37)
    original_poll = mock_sp.poll
    call_count = [0]

    def counting_poll():
        call_count[0] += 1
        if call_count[0] >= 10:
            poll_loop_done[0] = True
        return None

    mock_sp.poll.side_effect = counting_poll

    def get_returncode():
        if poll_loop_done[0]:
            return 0  # Process exited after the poll loop
        return None

    type(mock_sp).returncode = property(lambda self: get_returncode())

    mock_sp.wait = AsyncMock()

    sigkill_sent = [False]

    def mock_signal_group(sp, sig):
        if sig == signal.SIGKILL:
            sigkill_sent[0] = True

    with patch("shrinkray.process.signal_group", mock_signal_group):
        await interrupt_wait_and_kill(mock_sp, delay=0.001)

    # SIGKILL should not have been sent since returncode became non-None
    # after the poll loop, skipping the SIGKILL branch at line 37
    assert not sigkill_sent[0]
