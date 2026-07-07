"""Process management utilities for shrink ray."""

import os
import random
import resource
import signal
import sys

import trio


# RLIMIT_AS (address-space) memory limiting is reliably enforced on Linux
# but not on macOS, where setrlimit(RLIMIT_AS, ...) refuses to set a real
# limit and the kernel would not honour it. We still attempt it (it is
# harmless when it fails), but only advertise enforcement where it works,
# so we can warn the user rather than give a false sense of protection.
MEMORY_LIMIT_ENFORCEABLE = sys.platform != "darwin"

# OpenBSD has no RLIMIT_AS at all; RLIMIT_DATA there covers malloc'd
# memory (the kernel accounts it against the data segment), so it serves
# the same runaway-test protection.
MEMORY_RLIMIT = getattr(resource, "RLIMIT_AS", resource.RLIMIT_DATA)

_MEMORY_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_memory_limit(value: str) -> int | None:
    """Parse a ``--memory-limit`` value into a byte count.

    Accepts a plain byte count or a value with a binary K/M/G/T suffix
    (e.g. ``512M``, ``8G``, ``1.5G``). A value of zero or less, or one of
    ``none``/``off``/``disabled``/``unlimited``, disables the limit and
    returns ``None``. Raises ``ValueError`` on anything unparseable.
    """
    text = value.strip().upper()
    if text in ("NONE", "OFF", "DISABLED", "UNLIMITED"):
        return None
    multiplier = 1
    if text and text[-1] in _MEMORY_UNITS:
        multiplier = _MEMORY_UNITS[text[-1]]
        text = text[:-1].strip()
    try:
        amount = float(text)
    except ValueError:
        raise ValueError(f"Invalid memory limit: {value!r}")
    limit = int(amount * multiplier)
    if limit <= 0:
        return None
    return limit


def default_memory_limit() -> int:
    """A generous default oracle memory limit: the machine's physical RAM.

    A single interestingness test using more memory than the whole machine
    has is pathological, so this is a safety net rather than a tight bound.
    Falls back to 8 GiB when the physical size cannot be determined.
    """
    fallback = 8 * 1024**3
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return fallback
    if pages <= 0 or page_size <= 0:
        return fallback
    return pages * page_size


# ulimit flag matching MEMORY_RLIMIT: -v sets RLIMIT_AS; on OpenBSD,
# which has no RLIMIT_AS, -d sets RLIMIT_DATA (see MEMORY_RLIMIT above).
_ULIMIT_FLAG = "-v" if hasattr(resource, "RLIMIT_AS") else "-d"


def memory_limited_command(command: list[str], memory_limit: int | None) -> list[str]:
    """Wrap an interestingness-test command to cap its memory use.

    When a memory limit is set, the command is prefixed with a shell
    that applies ``ulimit`` and then execs the real command, so a
    runaway test cannot exhaust host memory. The limit cannot be set
    with a ``preexec_fn``: that forces subprocess to fork, and forking
    a process that hosts the TUI subinterpreter crashes the child, so
    tests are spawned with ``start_new_session=True`` (which uses
    posix_spawn) instead. A failure to set the limit (e.g. on macOS,
    which rejects ``RLIMIT_AS``) is ignored so the child still runs.
    """
    if memory_limit is None or memory_limit <= 0:
        return command
    kib = memory_limit // 1024
    return [
        "/bin/sh",
        "-c",
        f'ulimit {_ULIMIT_FLAG} {kib} 2>/dev/null; exec "$@"',
        "sh",
        *command,
    ]


def peak_child_rss_bytes() -> int:
    """Peak resident memory of reaped child processes, in bytes.

    ``getrusage`` reports ``ru_maxrss`` in bytes on macOS/BSD and in
    kibibytes on Linux; this normalises both to bytes.
    """
    ru_maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        return ru_maxrss
    return ru_maxrss * 1024


def signal_group(sp: "trio.Process", sig: int) -> None:
    """Send a signal to the process group led by sp.

    Test subprocesses are started with start_new_session=True (setsid),
    so the child leads its own process group and its pid names that group. The
    group is deliberately not looked up with getpgid: on OpenBSD that
    fails with EPERM for processes in a different session, and using the
    pid directly also cannot name shrink-ray's own group by mistake (that
    would signal shrink-ray itself): shrink-ray's group leader predates
    the child, so their pids cannot collide, and for a child that somehow
    skipped setsid, killpg fails with ESRCH instead.
    """
    os.killpg(sp.pid, sig)


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
        except (ProcessLookupError, PermissionError):
            # The group can be gone if the process exits at exactly the
            # wrong time. macOS reports that as EPERM rather than ESRCH
            # when the group's only member has exited but is unreaped;
            # the sp.wait() below then reaps it.
            pass

        if sp.returncode is None:
            try:
                signal_group(sp, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        with trio.move_on_after(delay):
            await sp.wait()

        if sp.returncode is None:
            raise ValueError(
                f"Could not kill subprocess with pid {sp.pid}. Something has gone seriously wrong."
            )
