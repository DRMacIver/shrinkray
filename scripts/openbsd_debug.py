"""Diagnostics for the OpenBSD hang in issue #56.

On OpenBSD, shrinkray's reduction stalls on the very first interestingness
call, which goes through trio.run_process. Initial validation (which uses
subprocess.run in a thread) works fine. The prime suspect is trio's
kqueue-based child-exit detection (EVFILT_PROC / NOTE_EXIT), so this script
probes each layer independently:

  kqueue_basic              raw kqueue NOTE_EXIT on a plain child
  kqueue_setsid             raw kqueue NOTE_EXIT on a child that calls setsid()
  kqueue_late               registering after the child has already exited
  kqueue_pipe_write_drained raw kqueue EVFILT_WRITE on a pipe that drains
  kqueue_pipe_write_widowed raw kqueue EVFILT_WRITE on a pipe whose read end
                            closes (what trio's stdin feeder relies on)
  trio_plain                trio.run_process on a short-lived child
  trio_setsid               trio.run_process with shrinkray's preexec_fn
  trio_stdin_unread         big stdin the child never reads
  trio_stdin_read           big stdin the child consumes
  trio_stdin_fd             stdin passed as a file descriptor (fix pattern)
  trio_shrinkray_like       the spawn pattern from run_script_on_file BEFORE
                            the fix (nursery.start + setsid + piped stdin);
                            expected to hang on OpenBSD
  trio_shrinkray_fixed      the spawn pattern from run_script_on_file AFTER
                            the fix (stdin is a file descriptor)

Run `python3 scripts/openbsd_debug.py` to run every test, each in its own
subprocess with a hard kill timeout so one hang can't block the rest.
Run `python3 scripts/openbsd_debug.py <name>` to run a single test inline.

Each test returns (ok, detail).
"""

import os
import select
import subprocess
import sys
import time

TIMEOUT = 15.0

Result = tuple[bool, str]


def _kqueue_wait_for_exit(child: "subprocess.Popen[bytes]") -> Result:
    """Register NOTE_EXIT for child on a fresh kqueue and wait for it."""
    kq = select.kqueue()
    event = select.kevent(
        child.pid,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
        fflags=select.KQ_NOTE_EXIT,
    )
    try:
        kq.control([event], 0)
    except ProcessLookupError:
        return True, "kevent registration raised ProcessLookupError (already exited)"
    start = time.time()
    got = kq.control(None, 1, TIMEOUT)
    elapsed = time.time() - start
    if got:
        return True, f"NOTE_EXIT delivered after {elapsed:.3f}s: {got[0]!r}"
    return False, f"no kevent within {elapsed:.1f}s; child poll={child.poll()!r}"


def _fill_pipe(wfd: int) -> int:
    """Write to wfd (non-blocking) until the pipe buffer is full."""
    os.set_blocking(wfd, False)
    written = 0
    try:
        while True:
            written += os.write(wfd, b"x" * 65536)
    except BlockingIOError:
        return written


def test_kqueue_pipe_write_drained() -> Result:
    """EVFILT_WRITE on a full pipe must fire when the reader drains it."""
    rfd, wfd = os.pipe()
    _fill_pipe(wfd)
    kq = select.kqueue()
    kq.control(
        [select.kevent(wfd, filter=select.KQ_FILTER_WRITE, flags=select.KQ_EV_ADD)],
        0,
    )
    os.read(rfd, 1 << 22)
    got = kq.control(None, 1, TIMEOUT)
    if got:
        return True, f"EVFILT_WRITE fired after drain: {got[0]!r}"
    return False, f"no EVFILT_WRITE within {TIMEOUT}s after draining the pipe"


def test_kqueue_pipe_write_widowed() -> Result:
    """EVFILT_WRITE on a full pipe must fire when the read end is closed.

    This is what trio's stdin-feeder task relies on to learn that the
    child exited without reading its input.
    """
    rfd, wfd = os.pipe()
    _fill_pipe(wfd)
    kq = select.kqueue()
    kq.control(
        [select.kevent(wfd, filter=select.KQ_FILTER_WRITE, flags=select.KQ_EV_ADD)],
        0,
    )
    os.close(rfd)
    start = time.time()
    got = kq.control(None, 1, TIMEOUT)
    elapsed = time.time() - start
    if got:
        return True, f"EVFILT_WRITE fired after {elapsed:.3f}s: {got[0]!r}"
    return False, f"no EVFILT_WRITE within {elapsed:.1f}s after closing read end"


def test_trio_stdin_unread() -> Result:
    """run_process where the child ignores a bigger-than-pipe-buffer stdin."""
    import trio

    async def main() -> Result:
        with trio.move_on_after(TIMEOUT):
            completed = await trio.run_process(
                ["sleep", "0.3"],
                check=False,
                stdin=b"x" * 4_000_000,
                capture_stdout=True,
                capture_stderr=True,
            )
            return True, f"run_process returned {completed.returncode}"
        return False, f"run_process with unread stdin hung for {TIMEOUT}s"

    return trio.run(main)


def test_trio_stdin_fd() -> Result:
    """The fix pattern: stdin is a real file descriptor, not fed via pipe."""
    import trio

    async def main() -> Result:
        path = os.path.join(os.getcwd(), "words")
        fd = os.open(path, os.O_RDONLY)
        try:
            with trio.move_on_after(TIMEOUT):
                completed = await trio.run_process(
                    ["sleep", "0.3"],
                    check=False,
                    stdin=fd,
                    capture_stdout=True,
                    capture_stderr=True,
                    preexec_fn=os.setsid,
                )
                return True, f"run_process returned {completed.returncode}"
            return False, f"run_process with fd stdin hung for {TIMEOUT}s"
        finally:
            os.close(fd)

    return trio.run(main)


def test_trio_stdin_read() -> Result:
    """run_process where the child consumes a bigger-than-pipe-buffer stdin."""
    import trio

    async def main() -> Result:
        with trio.move_on_after(TIMEOUT):
            completed = await trio.run_process(
                ["sh", "-c", "cat > /dev/null"],
                check=False,
                stdin=b"x" * 4_000_000,
                capture_stdout=True,
                capture_stderr=True,
            )
            return True, f"run_process returned {completed.returncode}"
        return False, f"run_process with consumed stdin hung for {TIMEOUT}s"

    return trio.run(main)


def test_kqueue_basic() -> Result:
    child = subprocess.Popen(["sleep", "0.3"])
    return _kqueue_wait_for_exit(child)


def test_kqueue_setsid() -> Result:
    child = subprocess.Popen(["sleep", "0.3"], preexec_fn=os.setsid)
    return _kqueue_wait_for_exit(child)


def test_kqueue_late() -> Result:
    child = subprocess.Popen(["true"])
    time.sleep(1.0)
    # Child is a zombie now (not yet reaped). Trio relies on registration
    # either succeeding with an immediate event or raising ProcessLookupError.
    return _kqueue_wait_for_exit(child)


def test_trio_plain() -> Result:
    import trio

    async def main() -> Result:
        with trio.move_on_after(TIMEOUT):
            completed = await trio.run_process(
                ["sleep", "0.3"],
                capture_stdout=True,
                capture_stderr=True,
                check=False,
            )
            return True, f"run_process returned {completed.returncode}"
        return False, f"trio.run_process hung for {TIMEOUT}s"

    return trio.run(main)


def test_trio_setsid() -> Result:
    import trio

    async def main() -> Result:
        with trio.move_on_after(TIMEOUT):
            completed = await trio.run_process(
                ["sleep", "0.3"],
                capture_stdout=True,
                capture_stderr=True,
                check=False,
                preexec_fn=os.setsid,
            )
            return True, f"run_process returned {completed.returncode}"
        return False, f"trio.run_process with setsid hung for {TIMEOUT}s"

    return trio.run(main)


def test_trio_shrinkray_like() -> Result:
    import trio

    async def main() -> Result:
        result: Result = False, "nursery exited without running the test"
        async with trio.open_nursery() as nursery:

            def start_process(task_status=trio.TASK_STATUS_IGNORED):  # type: ignore[no-untyped-def]
                return trio.run_process(
                    ["sh", "-c", "grep -c bug words"],
                    check=False,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid,
                    stdin=b"bugbear\n" * 300_000,  # >> pipe buffer, never read
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    task_status=task_status,
                )

            sp = await nursery.start(start_process)
            start = time.time()
            with trio.move_on_after(TIMEOUT) as scope:
                await sp.wait()
            elapsed = time.time() - start
            if scope.cancelled_caught:
                result = (
                    False,
                    f"sp.wait() hung for {elapsed:.1f}s"
                    f" (Popen.poll()={sp._proc.poll()!r})",
                )
                sp.kill()
            else:
                result = True, f"sp.wait() returned {sp.returncode} after {elapsed:.3f}s"
        return result

    return trio.run(main)


def test_trio_shrinkray_fixed() -> Result:
    """run_script_on_file's pattern after the fix: fd stdin, no pipe."""
    import trio

    async def main() -> Result:
        result: Result = False, "nursery exited without running the test"
        async with trio.open_nursery() as nursery:

            def start_process(stdin, task_status=trio.TASK_STATUS_IGNORED):  # type: ignore[no-untyped-def]
                return trio.run_process(
                    ["sh", "-c", "grep -c bug words"],
                    check=False,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid,
                    stdin=stdin,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    task_status=task_status,
                )

            with open(os.path.join(os.getcwd(), "words"), "rb") as stdin:
                sp = await nursery.start(start_process, stdin)
            start = time.time()
            with trio.move_on_after(TIMEOUT) as scope:
                await sp.wait()
            elapsed = time.time() - start
            if scope.cancelled_caught:
                result = False, f"sp.wait() hung for {elapsed:.1f}s"
                sp.kill()
            else:
                result = True, f"sp.wait() returned {sp.returncode} after {elapsed:.3f}s"
        return result

    return trio.run(main)


TESTS = {
    name.removeprefix("test_"): fn
    for name, fn in sorted(globals().items())
    if name.startswith("test_")
}


def run_all() -> int:
    print(f"uname: {os.uname()}")
    print(f"python: {sys.version}")
    try:
        import trio

        print(f"trio: {trio.__version__}")
    except ImportError as e:
        print(f"trio: not importable ({e})")
    failures = 0
    for name in TESTS:
        print(f"\n=== {name} ===", flush=True)
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), name])
        try:
            code = proc.wait(timeout=TIMEOUT * 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"HARD TIMEOUT: {name} did not finish in {TIMEOUT * 2}s", flush=True)
            failures += 1
            continue
        if code != 0:
            failures += 1
    print(f"\n{failures} failing test(s) out of {len(TESTS)}")
    return 1 if failures else 0


def main() -> int:
    if len(sys.argv) == 1:
        return run_all()
    name = sys.argv[1]
    if name not in TESTS:
        print(f"unknown test {name!r}; available: {', '.join(TESTS)}")
        return 2
    # A "words" file for the shrinkray-like test's grep to scan.
    with open(os.path.join(os.getcwd(), "words"), "w") as f:
        f.write("bugseed\nbugweed\nbugwort\n" * 10000)  # > pipe buffer
    ok, detail = TESTS[name]()
    print(f"{'PASS' if ok else 'FAIL'}: {detail}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
