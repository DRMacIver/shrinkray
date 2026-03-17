"""Process management utilities for shrink ray."""

import os
import random
import signal

import trio


def signal_group(sp: "trio.Process", sig: int) -> None:
    """Send a signal to a process group."""
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, sig)


def _close_pipes_sync(sp: "trio.Process") -> None:
    """Close all pipes on a process synchronously.

    Trio process pipes are FdStream instances which have a fileno() method,
    but the type annotations declare them as abstract SendStream/ReceiveStream.
    We use getattr to access fileno() without upsetting the type checker.
    """
    for pipe in [sp.stdout, sp.stderr, sp.stdin]:
        if pipe is None:
            continue
        fileno_fn = getattr(pipe, "fileno", None)
        if fileno_fn is not None:
            try:
                os.close(fileno_fn())
            except OSError:
                pass


def kill_process_group(sp: "trio.Process") -> None:
    """Synchronously kill a process group.

    Sends SIGKILL to the entire process group to clean up child processes.
    This is needed because Trio only kills the direct child on cancellation,
    but shell scripts often fork children that continue running.

    Always attempts to kill the group even if the group leader (sp) has
    already exited, because child processes in the group may still be alive.
    """
    _close_pipes_sync(sp)
    try:
        os.killpg(sp.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def interrupt_wait_and_kill(sp: "trio.Process", delay: float = 0.1) -> None:
    """Interrupt a process, wait for it to exit, and kill it if necessary."""
    await trio.lowlevel.checkpoint()
    if sp.returncode is None:
        try:
            # In case the subprocess forked. Python might hang if you don't close
            # all pipes.
            for pipe in [sp.stdout, sp.stderr, sp.stdin]:
                if pipe:
                    await pipe.aclose()
            signal_group(sp, signal.SIGINT)
            for n in range(10):
                if sp.poll() is not None:
                    return
                await trio.sleep(delay * 1.5**n * random.random())
        except ProcessLookupError:  # pragma: no cover
            # This is incredibly hard to trigger reliably, because it only happens
            # if the process exits at exactly the wrong time.
            pass

        if sp.returncode is None:
            try:
                signal_group(sp, signal.SIGKILL)
            except ProcessLookupError:
                pass

        with trio.move_on_after(delay):
            await sp.wait()

        if sp.returncode is None:
            raise ValueError(
                f"Could not kill subprocess with pid {sp.pid}. Something has gone seriously wrong."
            )
