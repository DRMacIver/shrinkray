"""External reducers: reduction passes backed by a subprocess.

An external reducer is a subprocess that shrink ray drives over the protocol in
:mod:`shrinkray.reducers.protocol`. This module provides the shrink-ray side:

- :func:`drive_external_reducer` runs the protocol over a pair of streams,
  evaluating the reducer's queries with ``problem.is_interesting`` and feeding
  the results back. It is independent of process management so it can be tested
  directly.
- :func:`external_reducer` wraps a command line as a :data:`ReductionPass`,
  launching the subprocess, redirecting its stderr to a log file, and killing it
  on completion, timeout, or cancellation.
"""

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import trio

from shrinkray.passes.definitions import ReductionPass
from shrinkray.problem import ReductionProblem
from shrinkray.process import interrupt_wait_and_kill
from shrinkray.reducers.protocol import (
    LineReader,
    decode_query,
    encode_feedback,
)


# If the reducer emits no query for this long, we assume it has wedged and
# terminate it. Overridable per reducer.
DEFAULT_REDUCER_TIMEOUT = 60.0


async def drive_external_reducer(
    problem: ReductionProblem[bytes],
    *,
    send_stream: trio.abc.SendStream,
    recv_stream: trio.abc.ReceiveStream,
    timeout: float = DEFAULT_REDUCER_TIMEOUT,
    parallelism: int = 1,
) -> None:
    """Run the shrink-ray side of the external reducer protocol.

    Sends the initial test case, then reads candidate queries from
    ``recv_stream`` and answers them on ``send_stream``, running interestingness
    tests up to ``parallelism`` at a time. Returns when the reducer closes
    ``recv_stream`` (EOF) or when no query arrives for ``timeout`` seconds.
    """
    reader = LineReader(recv_stream)
    send_lock = trio.Lock()
    # The current test case as last advertised to the reducer. Whenever the real
    # current test case moves ahead of this, we push an unsolicited update.
    last_advertised = problem.current_test_case

    async def send_feedback(content: bytes, interesting: bool) -> None:
        try:
            await send_stream.send_all(encode_feedback(content, interesting))
        except (trio.BrokenResourceError, trio.ClosedResourceError):
            # The reducer exited while we were replying; nothing more to say.
            pass

    # Handshake: tell the reducer what it is starting from.
    await send_feedback(problem.current_test_case, True)

    # A semaphore (not a CapacityLimiter) because it is acquired by the reader
    # task but released by the handler task, and semaphore tokens are not bound
    # to the acquiring task.
    slots = trio.Semaphore(max(parallelism, 1))

    async def handle(candidate: bytes) -> None:
        nonlocal last_advertised
        try:
            interesting = await problem.is_interesting(candidate)
            async with send_lock:
                await send_feedback(candidate, interesting)
                current = problem.current_test_case
                if current != last_advertised:
                    last_advertised = current
                    await send_feedback(current, True)
        finally:
            slots.release()

    async with trio.open_nursery() as nursery:
        while True:
            # Acquire a slot before reading so that a flood of queries can't
            # outrun our workers: once all slots are busy the pipe fills and the
            # reducer blocks on its next send.
            await slots.acquire()
            line: bytes | None = None
            with trio.move_on_after(timeout) as scope:
                line = await reader.readline()
            if scope.cancelled_caught:
                slots.release()
                break
            if line is None:
                slots.release()
                break
            if not line.strip():
                slots.release()
                continue
            try:
                candidate = decode_query(line)
            except ValueError:
                # Malformed line: ignore it rather than killing the reducer.
                slots.release()
                continue
            nursery.start_soon(handle, candidate)
        nursery.cancel_scope.cancel()


async def _terminate(proc: trio.Process) -> None:
    """Shut down the reducer subprocess, killing its process group if needed."""
    with trio.CancelScope(shield=True):
        await interrupt_wait_and_kill(proc)


def external_reducer(
    command: Sequence[str],
    *,
    log_file: str | os.PathLike[str] | None = None,
    timeout: float = DEFAULT_REDUCER_TIMEOUT,
    name: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> ReductionPass[bytes]:
    """Build a reduction pass that reduces via an external reducer subprocess.

    Args:
        command: The command (argv) to launch the reducer.
        log_file: Path to append the reducer's stderr to. If None, stderr is
            discarded.
        timeout: Seconds without a query before the reducer is terminated.
        name: Name for the pass (used in stats/status). Defaults to the command
            basename.
        extra_env: Extra environment variables for the subprocess.
    """
    command = list(command)
    reducer_name = name or f"external:{os.path.basename(command[0])}"

    async def run_external(problem: ReductionProblem[bytes]) -> None:
        parallelism = problem.work.parallelism
        env = dict(os.environ)
        env["SHRINKRAY_REDUCER_PARALLELISM"] = str(parallelism)
        if extra_env is not None:
            env.update(extra_env)

        if log_file is not None:
            stderr_ctx: Any = open(log_file, "ab")
        else:
            stderr_ctx = open(os.devnull, "ab")

        with stderr_ctx as stderr:
            proc = await trio.lowlevel.open_process(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr.fileno(),
                env=env,
                preexec_fn=os.setsid,
            )
            try:
                assert proc.stdin is not None
                assert proc.stdout is not None
                await drive_external_reducer(
                    problem,
                    send_stream=proc.stdin,
                    recv_stream=proc.stdout,
                    timeout=timeout,
                    parallelism=parallelism,
                )
            finally:
                await _terminate(proc)

    run_external.__name__ = reducer_name
    return run_external
