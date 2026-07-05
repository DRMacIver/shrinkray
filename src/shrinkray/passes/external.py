"""External reducers: reduction passes backed by a subprocess.

An external reducer is a subprocess that shrink ray drives over the protocol in
:mod:`shrinkray.reducers.protocol`. This module provides the shrink-ray side:

- :func:`drive_external_reducer` runs one reduce request over a pair of streams,
  evaluating the reducer's queries with ``problem.is_interesting`` and feeding
  the results back. It is independent of process management so it can be tested
  directly.
- :class:`ExternalReducerPass` wraps a command line as a reduction pass. It
  keeps the subprocess alive across invocations (so expensive startup happens
  once), redirects its stderr to a log file, and cleans it up on completion,
  timeout, or cancellation.
- :func:`external_reducer` is a thin constructor for :class:`ExternalReducerPass`.
"""

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import IO

import trio

from shrinkray.problem import ReductionProblem
from shrinkray.process import interrupt_wait_and_kill
from shrinkray.reducers.protocol import (
    Idle,
    LineReader,
    encode_feedback,
    parse_from_reducer,
)


# If the reducer emits nothing for this long while working, we assume it has
# wedged and terminate it. Overridable per reducer.
DEFAULT_REDUCER_TIMEOUT = 60.0


async def drive_external_reducer(
    problem: ReductionProblem[bytes],
    *,
    send_stream: trio.abc.SendStream,
    reader: LineReader,
    timeout: float = DEFAULT_REDUCER_TIMEOUT,
    parallelism: int = 1,
) -> bool:
    """Reduce the current test case with an external reducer, once.

    Hands the reducer the current test case (as a feedback message, which -
    matching no query it made - it takes as a fresh test case to reduce), then
    answers its queries (running interestingness tests up to ``parallelism`` at a
    time) until it reports idle. ``reader`` reads the reducer's output and is
    reused across calls so buffered bytes are not lost.

    Returns True if the reducer went idle (and is still alive), or False if it
    exited (EOF) or timed out.
    """
    send_lock = trio.Lock()

    async def send(data: bytes) -> None:
        async with send_lock:
            try:
                await send_stream.send_all(data)
            except (trio.BrokenResourceError, trio.ClosedResourceError):
                # The reducer exited; nothing more to say.
                pass

    await send(encode_feedback(problem.current_test_case, True))

    # A semaphore (not a CapacityLimiter) because it is acquired by the reader
    # loop but released by the handler task, and semaphore tokens are not bound
    # to the acquiring task.
    slots = trio.Semaphore(max(parallelism, 1))

    async def handle(candidate: bytes) -> None:
        try:
            interesting = await problem.is_interesting(candidate)
            await send(encode_feedback(candidate, interesting))
        finally:
            slots.release()

    idle = False
    async with trio.open_nursery() as nursery:
        while True:
            # Acquire a slot before reading so a flood of queries can't outrun
            # our workers: once all slots are busy the pipe fills and the reducer
            # blocks on its next send.
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
                message = parse_from_reducer(line)
            except ValueError:
                slots.release()
                continue
            if isinstance(message, Idle):
                slots.release()
                idle = True
                break
            nursery.start_soon(handle, message.content)
        nursery.cancel_scope.cancel()
    return idle


async def _terminate(proc: trio.Process) -> None:
    """Shut down the reducer subprocess, killing its process group if needed."""
    with trio.CancelScope(shield=True):
        await interrupt_wait_and_kill(proc)


class ExternalReducerPass:
    """A reduction pass that reduces via a persistent external reducer subprocess.

    The subprocess is launched on first use and kept alive across pass
    invocations, so per-launch startup cost (such as importing libcst) is paid
    once. It is torn down when the reducer exits, when a call is cancelled or
    errors, or when :meth:`aclose` is called (by the reducer at the end of a
    run).
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        log_file: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_REDUCER_TIMEOUT,
        name: str | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self._command = list(command)
        self._log_file = log_file
        self._timeout = timeout
        self._extra_env = extra_env
        self.__name__ = name or f"external:{os.path.basename(self._command[0])}"
        self._proc: trio.Process | None = None
        self._reader: LineReader | None = None
        self._stderr_file: IO[bytes] | None = None

    async def _launch(self, parallelism: int) -> None:
        env = dict(os.environ)
        env["SHRINKRAY_REDUCER_PARALLELISM"] = str(parallelism)
        if self._extra_env is not None:
            env.update(self._extra_env)

        target = self._log_file if self._log_file is not None else os.devnull
        stderr_file: IO[bytes] = open(target, "ab")
        try:
            proc = await trio.lowlevel.open_process(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file.fileno(),
                env=env,
                preexec_fn=os.setsid,
            )
        except BaseException:
            stderr_file.close()
            raise
        assert proc.stdout is not None
        self._proc = proc
        self._reader = LineReader(proc.stdout)
        self._stderr_file = stderr_file

    async def __call__(self, problem: ReductionProblem[bytes]) -> None:
        parallelism = problem.work.parallelism
        try:
            if self._proc is None:
                await self._launch(parallelism)
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._reader is not None
            alive = await drive_external_reducer(
                problem,
                send_stream=self._proc.stdin,
                reader=self._reader,
                timeout=self._timeout,
                parallelism=parallelism,
            )
            if not alive:
                await self._shutdown()
        except BaseException:
            await self._shutdown()
            raise

    async def _shutdown(self) -> None:
        if self._proc is not None:
            await _terminate(self._proc)
            self._proc = None
            self._reader = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    async def aclose(self) -> None:
        """Terminate the reducer subprocess if it is still running."""
        await self._shutdown()


def external_reducer(
    command: Sequence[str],
    *,
    log_file: str | os.PathLike[str] | None = None,
    timeout: float = DEFAULT_REDUCER_TIMEOUT,
    name: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> ExternalReducerPass:
    """Build a reduction pass that reduces via an external reducer subprocess.

    Args:
        command: The command (argv) to launch the reducer.
        log_file: Path to append the reducer's stderr to. If None, stderr is
            discarded.
        timeout: Seconds without output from a working reducer before it is
            terminated.
        name: Name for the pass (used in stats/status). Defaults to the command
            basename.
        extra_env: Extra environment variables for the subprocess.
    """
    return ExternalReducerPass(
        command,
        log_file=log_file,
        timeout=timeout,
        name=name,
        extra_env=extra_env,
    )
