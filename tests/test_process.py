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
