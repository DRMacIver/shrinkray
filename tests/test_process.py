"""Tests for process management utilities."""

import os
import signal
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import trio

from shrinkray.process import (
    MEMORY_LIMIT_ENFORCEABLE,
    _close_pipes_sync,
    default_memory_limit,
    interrupt_wait_and_kill,
    kill_process_group,
    memory_limited_command,
    parse_memory_limit,
    peak_child_rss_bytes,
    signal_group,
)


# === memory limit parsing tests ===


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1024", 1024),
        ("512M", 512 * 1024**2),
        ("8G", 8 * 1024**3),
        ("1.5G", int(1.5 * 1024**3)),
        ("2T", 2 * 1024**4),
        ("4K", 4 * 1024),
        ("  8G  ", 8 * 1024**3),  # surrounding whitespace
        ("8g", 8 * 1024**3),  # lowercase suffix
        ("0", None),  # zero disables
        ("-5", None),  # negative disables
        ("none", None),
        ("OFF", None),
        ("disabled", None),
        ("unlimited", None),
    ],
)
def test_parse_memory_limit(value, expected):
    assert parse_memory_limit(value) == expected


@pytest.mark.parametrize("value", ["abc", "12x", "G", ""])
def test_parse_memory_limit_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_memory_limit(value)


def test_default_memory_limit_is_positive():
    assert default_memory_limit() > 0


def test_default_memory_limit_falls_back_when_unavailable():
    with patch("shrinkray.process.os.sysconf", side_effect=ValueError):
        assert default_memory_limit() == 8 * 1024**3


def test_default_memory_limit_falls_back_on_nonsense_values():
    with patch("shrinkray.process.os.sysconf", return_value=0):
        assert default_memory_limit() == 8 * 1024**3


# === memory_limited_command tests ===
#
# The memory limit is applied by a shell prefix in the child rather than
# a preexec_fn: preexec_fn forces subprocess to fork, and forking a
# process that hosts the TUI subinterpreter crashes the child.


@pytest.mark.parametrize("limit", [None, 0, -1])
def test_memory_limited_command_passes_through_when_disabled(limit):
    command = ["./test.sh", "arg"]
    assert memory_limited_command(command, limit) is command


def test_memory_limited_command_wraps_with_ulimit():
    wrapped = memory_limited_command(["./test.sh", "arg"], 4 * 1024**3)
    assert wrapped[0] == "/bin/sh"
    assert "ulimit" in wrapped[2]
    assert str(4 * 1024**3 // 1024) in wrapped[2]
    assert wrapped[-2:] == ["./test.sh", "arg"]


def test_memory_limited_command_preserves_arguments_exactly():
    # Arguments must survive the shell wrapper byte-for-byte, including
    # whitespace and quoting characters.
    tricky = ["echo", "a b", "it's", '"quoted"', "$HOME", "*"]
    wrapped = memory_limited_command(tricky, 8 * 1024**3)
    result = subprocess.run(wrapped, capture_output=True, check=True)
    assert result.stdout == b'a b it\'s "quoted" $HOME *\n'


def test_memory_limited_command_runs_in_new_process_group():
    # The wrapper execs the real command, so the child keeps the pid it
    # was spawned with and start_new_session still makes it the leader
    # of its own session.
    wrapped = memory_limited_command(["/bin/sh", "-c", "echo $$"], 8 * 1024**3)
    result = subprocess.run(
        wrapped, capture_output=True, check=True, start_new_session=True
    )
    assert result.stdout.strip().isdigit()


@pytest.mark.skipif(
    not MEMORY_LIMIT_ENFORCEABLE, reason="platform does not enforce RLIMIT_AS"
)
def test_memory_limited_command_enforces_the_limit():
    allocate = [
        sys.executable,
        "-c",
        "x = bytearray(1024 * 1024 * 1024); print('allocated')",
    ]
    unlimited = subprocess.run(
        memory_limited_command(allocate, None), capture_output=True
    )
    assert unlimited.returncode == 0, unlimited.stderr
    limited = subprocess.run(
        memory_limited_command(allocate, 256 * 1024 * 1024), capture_output=True
    )
    assert limited.returncode != 0


# === peak_child_rss_bytes tests ===


def test_peak_child_rss_bytes_is_nonnegative():
    assert peak_child_rss_bytes() >= 0


def test_peak_child_rss_bytes_normalises_linux_kib():
    fake = MagicMock(ru_maxrss=2048)
    with (
        patch("shrinkray.process.sys.platform", "linux"),
        patch("shrinkray.process.resource.getrusage", return_value=fake),
    ):
        assert peak_child_rss_bytes() == 2048 * 1024


def test_peak_child_rss_bytes_uses_bytes_on_macos():
    fake = MagicMock(ru_maxrss=2048)
    with (
        patch("shrinkray.process.sys.platform", "darwin"),
        patch("shrinkray.process.resource.getrusage", return_value=fake),
    ):
        assert peak_child_rss_bytes() == 2048


# === signal_group tests ===


async def test_signal_group_sends_signal_to_process_group():
    # Start a process in its own process group
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "import time; time.sleep(100)"],
        start_new_session=True,
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


async def test_signal_group_refuses_own_process_group():
    # Start a process WITHOUT setsid, so it shares our process group and
    # is not a group leader. signal_group must not signal our group in
    # that case: the child's pid names no process group at all, so killpg
    # fails with ESRCH (SIGCONT is used as it is harmless if sent).
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "import time; time.sleep(100)"],
    )
    try:
        assert os.getpgid(sp.pid) == os.getpgrp()
        with pytest.raises(ProcessLookupError):
            signal_group(sp, signal.SIGCONT)
    finally:
        sp.kill()
        await sp.wait()


# === interrupt_wait_and_kill tests ===


async def test_interrupt_wait_and_kill_does_nothing_if_already_exited():
    # Start a process that exits immediately
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    # Wait for it to exit
    await sp.wait()
    assert sp.returncode is not None

    # Should not raise
    await interrupt_wait_and_kill(sp)


@pytest.mark.slow
async def test_interrupt_wait_and_kill_kills_process_ignoring_sigint():
    # Start a process that ignores SIGINT
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGINT, lambda *a: None); time.sleep(100)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
    )

    # Give it a moment to set up the signal handler
    await trio.sleep(0.1)

    # This should eventually kill it
    await interrupt_wait_and_kill(sp, delay=0.05)

    assert sp.returncode is not None


async def test_interrupt_wait_and_kill_handles_fast_exit_after_sigint(tmp_path):
    # Start a process that exits quickly after SIGINT. It touches a file
    # once its signal handler is installed so we don't race the setup.
    ready_marker = tmp_path / "ready"
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import pathlib, signal, time; "
            "signal.signal(signal.SIGINT, lambda *a: exit(0)); "
            f"pathlib.Path({str(ready_marker)!r}).touch(); "
            "time.sleep(100)",
        ],
        start_new_session=True,
    )

    while not ready_marker.exists():
        await trio.sleep(0.01)
    await interrupt_wait_and_kill(sp, delay=0.05)

    assert sp.returncode == 0


async def test_interrupt_wait_and_kill_tolerates_eperm_for_exited_group():
    # On macOS, killpg raises EPERM (not ESRCH) when the target group's
    # only member exits between interrupt_wait_and_kill closing its pipes
    # and the signal being sent - exactly what the external reducer's
    # persistent subprocess does, since it exits on stdin EOF. The kill
    # sequence must fall through and reap rather than crash.
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        start_new_session=True,
        stdin=subprocess.PIPE,
    )
    with patch(
        "shrinkray.process.os.killpg",
        side_effect=PermissionError(1, "Operation not permitted"),
    ):
        await interrupt_wait_and_kill(sp, delay=0.5)
    assert sp.returncode == 0


async def test_interrupt_wait_and_kill_raises_when_signals_cannot_be_sent():
    # If killpg reports EPERM while the process is genuinely still alive,
    # nothing can be killed and the failure must be loud.
    sp = await trio.lowlevel.open_process(
        [sys.executable, "-c", "import time; time.sleep(100)"],
        start_new_session=True,
    )
    try:
        with patch(
            "shrinkray.process.os.killpg",
            side_effect=PermissionError(1, "Operation not permitted"),
        ):
            with pytest.raises(ValueError, match="Could not kill subprocess"):
                await interrupt_wait_and_kill(sp, delay=0.01)
    finally:
        sp.kill()
        await sp.wait()


async def test_interrupt_wait_and_kill_closes_pipes_before_signaling():
    # Start a process with stdout pipe
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import sys; print('hello'); sys.stdout.flush(); import time; time.sleep(100)",
        ],
        start_new_session=True,
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

    The process exits between checking returncode and sending SIGKILL.
    """
    # Create a mock process
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    # returncode is None initially, then None at returncode check,
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

    The process persists after all kill attempts.
    """
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

    The returncode becomes non-None after all poll() calls return None but
    before checking returncode for the SIGKILL decision.
    """
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
    # after the loop but before the returncode check)
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
    # after the poll loop, skipping the SIGKILL branch
    assert not sigkill_sent[0]


# === _close_pipes_sync tests ===


def test_close_pipes_sync_closes_all_pipes():
    mock_sp = MagicMock()
    mock_sp.stdout = MagicMock()
    mock_sp.stdout.fileno.return_value = 10
    mock_sp.stderr = MagicMock()
    mock_sp.stderr.fileno.return_value = 11
    mock_sp.stdin = MagicMock()
    mock_sp.stdin.fileno.return_value = 12

    with patch("shrinkray.process.os.close") as mock_close:
        _close_pipes_sync(mock_sp)

    assert mock_close.call_count == 3
    mock_close.assert_any_call(10)
    mock_close.assert_any_call(11)
    mock_close.assert_any_call(12)


def test_close_pipes_sync_skips_none_pipes():
    mock_sp = MagicMock()
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    # Should not raise
    _close_pipes_sync(mock_sp)


def test_close_pipes_sync_skips_pipes_without_fileno():
    """Test that pipes without a fileno() method are silently skipped."""
    mock_sp = MagicMock()
    pipe_without_fileno = object()  # Has no fileno attribute
    mock_sp.stdout = pipe_without_fileno
    mock_sp.stderr = None
    mock_sp.stdin = None

    # Should not raise
    _close_pipes_sync(mock_sp)


def test_close_pipes_sync_handles_oserror():
    mock_sp = MagicMock()
    mock_sp.stdout = MagicMock()
    mock_sp.stdout.fileno.return_value = 10
    mock_sp.stderr = None
    mock_sp.stdin = None

    with patch("shrinkray.process.os.close", side_effect=OSError):
        # Should not raise even if os.close fails
        _close_pipes_sync(mock_sp)


# === kill_process_group tests ===


def test_kill_process_group_sends_sigkill_to_group():
    """Test that kill_process_group sends SIGKILL to the process group."""
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    with patch("shrinkray.process.os.killpg") as mock_killpg:
        kill_process_group(mock_sp)

    mock_killpg.assert_called_once_with(12345, signal.SIGKILL)


def test_kill_process_group_closes_pipes_before_killing():
    """Test that pipes are closed before sending SIGKILL."""
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout.fileno.return_value = 10
    mock_sp.stderr.fileno.return_value = 11
    mock_sp.stdin.fileno.return_value = 12

    call_order = []

    def tracking_close(fd):
        call_order.append(f"close_{fd}")

    def tracking_killpg(pid, sig):
        call_order.append("killpg")

    with (
        patch("shrinkray.process.os.close", side_effect=tracking_close),
        patch("shrinkray.process.os.killpg", side_effect=tracking_killpg),
    ):
        kill_process_group(mock_sp)

    assert call_order.index("killpg") > call_order.index("close_10")


def test_kill_process_group_handles_process_lookup_error():
    """Test that ProcessLookupError from killpg is handled gracefully."""
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    with patch("shrinkray.process.os.killpg", side_effect=ProcessLookupError):
        # Should not raise
        kill_process_group(mock_sp)


def test_kill_process_group_handles_permission_error():
    """Test that PermissionError from killpg is handled gracefully."""
    mock_sp = MagicMock()
    mock_sp.pid = 12345
    mock_sp.stdout = None
    mock_sp.stderr = None
    mock_sp.stdin = None

    with patch("shrinkray.process.os.killpg", side_effect=PermissionError):
        # Should not raise
        kill_process_group(mock_sp)


async def test_kill_process_group_with_real_process():
    """Integration test: kill_process_group kills a real subprocess."""
    sp = await trio.lowlevel.open_process(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(100)",
        ],
        start_new_session=True,
    )

    assert sp.poll() is None

    kill_process_group(sp)

    # Process should be dead
    with trio.move_on_after(1):
        await sp.wait()
    assert sp.returncode is not None
