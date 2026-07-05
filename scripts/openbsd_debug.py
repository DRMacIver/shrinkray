"""Diagnostics for the OpenBSD hang in issue #56.

On OpenBSD, shrinkray's reduction stalls on the very first interestingness
call, which goes through trio.run_process. Initial validation (which uses
subprocess.run in a thread) works fine. The prime suspect is trio's
kqueue-based child-exit detection (EVFILT_PROC / NOTE_EXIT), so this script
probes each layer independently:

  kqueue_basic          raw kqueue NOTE_EXIT on a plain child
  kqueue_setsid         raw kqueue NOTE_EXIT on a child that calls setsid()
  kqueue_late           registering after the child has already exited
  trio_plain            trio.run_process on a short-lived child
  trio_setsid           trio.run_process with shrinkray's preexec_fn
  trio_shrinkray_like   the exact spawn pattern from state.run_script_on_file
                        (nursery.start + setsid + big stdin + DEVNULL output)

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
        f.write("bugseed\nbugweed\nbugwort\n" * 1000)
    ok, detail = TESTS[name]()
    print(f"{'PASS' if ok else 'FAIL'}: {detail}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
