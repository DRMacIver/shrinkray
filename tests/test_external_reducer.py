"""In-process tests for the external reducer machinery.

These connect the reducer side (``run_reducer`` / ``RemoteReductionProblem``)
to the shrink-ray side (``drive_external_reducer``) over trio memory streams,
with no subprocess involved. This exercises the whole protocol quickly.
"""

import os
import runpy
import sys
from collections.abc import Callable, Iterable
from unittest.mock import patch

import pytest
import trio
from trio.testing import memory_stream_one_way_pair

from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.external import drive_external_reducer, external_reducer
from shrinkray.passes.python import PYTHON_PASSES, python_reducer_command
from shrinkray.problem import BasicReductionProblem, sort_key_for_initial
from shrinkray.reducers import python as python_reducer_module
from shrinkray.reducers.driver import run_reducer
from shrinkray.reducers.protocol import (
    LineReader,
    decode_feedback,
    encode_feedback,
    encode_idle,
    encode_query,
)
from shrinkray.work import WorkContext
from tests.helpers import reduce_with


def make_basic_problem(
    initial: bytes, is_interesting: Callable[[bytes], bool], parallelism: int = 1
) -> BasicReductionProblem[bytes]:
    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    return BasicReductionProblem(
        initial=initial,
        is_interesting=acondition,
        work=WorkContext(parallelism=parallelism),
        sort_key=sort_key_for_initial(initial),
    )


def reduce_externally(
    passes: Iterable[ReductionPass[bytes]],
    initial: bytes,
    is_interesting: Callable[[bytes], bool],
    parallelism: int = 1,
) -> bytes:
    """Synchronous wrapper around :func:`run_connected`."""
    return trio.run(run_connected, passes, initial, is_interesting, parallelism)


async def run_connected(
    passes: Iterable[ReductionPass[bytes]],
    initial: bytes,
    is_interesting: Callable[[bytes], bool],
    parallelism: int = 1,
) -> bytes:
    """Reduce ``initial`` by driving ``passes`` as an external reducer.

    Returns the shrink-ray side's final test case. The reducer runs the given
    passes against a RemoteReductionProblem; the shrink-ray side answers with a
    real BasicReductionProblem wrapping ``is_interesting``.
    """
    passes = list(passes)

    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=initial,
        is_interesting=acondition,
        work=WorkContext(parallelism=parallelism),
        sort_key=sort_key_for_initial(initial),
    )

    # feedback/reduce: shrink ray -> reducer; queries/idle: reducer -> shrink ray
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    reader = LineReader(query_recv)

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _reducer() -> None:
            await run_reducer(
                passes,
                stdin_stream=feedback_recv,
                stdout_stream=query_send,
                parallelism=parallelism,
            )
            # Signal EOF so the shrink-ray side stops waiting for queries.
            await query_send.aclose()

        @nursery.start_soon
        async def _shrinkray() -> None:
            # One reduce request drives the reducer to a fixpoint.
            alive = await drive_external_reducer(
                problem,
                send_stream=feedback_send,
                reader=reader,
                timeout=30.0,
                parallelism=parallelism,
            )
            assert alive  # the reducer reported idle, staying alive
            # Close the request stream so the reducer exits.
            await feedback_send.aclose()

    return problem.current_test_case


ANNOTATED = b"""
def has_an_annotation(x: list[int]) -> list[int]:
    y: list[int] = list(reversed(x))
    return x + y
"""


@pytest.mark.parametrize("parallelism", [1, 2])
def test_matches_in_process_reduction(parallelism: int) -> None:
    """Driving PYTHON_PASSES externally matches running them in-process."""

    def is_interesting(x: bytes) -> bool:
        return b"return" in x

    expected = reduce_with(
        PYTHON_PASSES, ANNOTATED, is_interesting, parallelism=parallelism
    )
    got = reduce_externally(
        PYTHON_PASSES, ANNOTATED, is_interesting, parallelism=parallelism
    )
    assert got == expected


def test_reduces_annotations_away() -> None:
    """A trivially-true predicate strips annotations and stubs the body."""
    got = reduce_externally(PYTHON_PASSES, ANNOTATED, lambda x: True)
    assert b"list[int]" not in got  # annotations gone
    assert b"..." in got  # body stubbed out


def test_reducer_with_no_progress_terminates() -> None:
    """If nothing is interesting below the initial, the reducer still exits."""
    initial = b"x = 1\n"
    got = reduce_externally(PYTHON_PASSES, initial, lambda x: x == initial)
    assert got == initial


# === Real subprocess: the built-in Python reducer ===


def reduce_with_real_subprocess(
    initial: bytes,
    is_interesting: Callable[[bytes], bool],
    parallelism: int = 1,
    tmp_path: str | None = None,
) -> bytes:
    """Reduce ``initial`` by launching the real python reducer subprocess."""

    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    log_file = os.path.join(tmp_path, "reducer.log") if tmp_path else None
    reducer_pass = external_reducer(python_reducer_command(), log_file=log_file)

    async def run() -> bytes:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
            sort_key=sort_key_for_initial(initial),
        )
        try:
            await reducer_pass(problem)
            return problem.current_test_case
        finally:
            # The reducer stays alive between calls, so tear it down explicitly.
            await reducer_pass.aclose()

    return trio.run(run)


@pytest.mark.slow
@pytest.mark.parametrize("parallelism", [1, 2])
def test_real_python_reducer_subprocess(parallelism: int, tmp_path) -> None:
    """The built-in Python reducer, run as a real subprocess, reduces Python."""

    def is_interesting(x: bytes) -> bool:
        return b"return" in x

    expected = reduce_with(PYTHON_PASSES, ANNOTATED, is_interesting)
    got = reduce_with_real_subprocess(
        ANNOTATED, is_interesting, parallelism=parallelism, tmp_path=str(tmp_path)
    )
    assert got == expected


@pytest.mark.slow
def test_real_python_reducer_writes_log(tmp_path) -> None:
    """The reducer's stderr is captured to the given log file."""
    log_file = tmp_path / "reducer.log"
    reduce_with_real_subprocess(
        ANNOTATED, lambda x: b"return" in x, tmp_path=str(tmp_path)
    )
    # The log file is created even if the reducer logs nothing.
    assert log_file.exists()


# === reducers.python serve()/main() coverage ===


def test_python_reducer_serve_over_pipes() -> None:
    """serve() wires stdin/stdout fds up to run_reducer and returns at EOF."""
    # shrink ray -> reducer (its stdin); reducer -> shrink ray (its stdout)
    feedback_read, feedback_write = os.pipe()
    query_read, query_write = os.pipe()

    # Send one test case, then close, so the reducer sees EOF and exits.
    os.write(feedback_write, encode_feedback(b"x = 1\n", True))
    os.close(feedback_write)

    def fake_dup(fd: int) -> int:
        return {0: feedback_read, 1: query_write}[fd]

    async def run() -> None:
        with patch.object(python_reducer_module.os, "dup", fake_dup):
            await python_reducer_module.serve(parallelism=1, seed=0)

    trio.run(run)
    # Drain whatever the reducer wrote so nothing is left dangling.
    os.close(query_read)


def test_python_reducer_main_reads_env_and_runs() -> None:
    """main() reads parallelism/seed from the environment and runs serve()."""
    seen: dict[str, int] = {}

    async def fake_serve(parallelism: int, seed: int) -> None:
        seen["parallelism"] = parallelism
        seen["seed"] = seed

    env = {"SHRINKRAY_REDUCER_PARALLELISM": "3", "SHRINKRAY_REDUCER_SEED": "7"}
    with (
        patch.dict(os.environ, env),
        patch.object(python_reducer_module, "serve", fake_serve),
    ):
        python_reducer_module.main()
    assert seen == {"parallelism": 3, "seed": 7}


def test_python_reducer_main_defaults() -> None:
    """main() defaults to parallelism 1 and seed 0 with no environment."""
    seen: dict[str, int] = {}

    async def fake_serve(parallelism: int, seed: int) -> None:
        seen["parallelism"] = parallelism
        seen["seed"] = seed

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(python_reducer_module, "serve", fake_serve),
    ):
        python_reducer_module.main()
    assert seen == {"parallelism": 1, "seed": 0}


def test_python_reducer_module_entry_point() -> None:
    """Running the module as __main__ invokes main()."""
    with patch("shrinkray.reducers.python.trio.run") as mock_trio_run:
        runpy.run_module(
            "shrinkray.reducers.python", run_name="__main__", alter_sys=True
        )
    assert mock_trio_run.called


# === drive_external_reducer edge cases (no subprocess) ===


async def test_drive_ignores_blank_and_malformed_queries() -> None:
    """Blank and malformed lines are skipped; valid queries are answered; idle
    ends the request with the reducer still alive."""
    problem = make_basic_problem(b"hello world\n", lambda x: b"h" in x)

    fb_send, fb_recv = memory_stream_one_way_pair()
    q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)

    result: list[bool] = []

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _shrinkray() -> None:
            alive = await drive_external_reducer(
                problem,
                send_stream=fb_send,
                reader=reader,
                timeout=5.0,
                parallelism=1,
            )
            result.append(alive)

        @nursery.start_soon
        async def _reducer() -> None:
            fb_reader = LineReader(fb_recv)
            first_line = await fb_reader.readline()
            assert first_line is not None
            # The first message is the current test case, handed over as feedback.
            content, interesting = decode_feedback(first_line)
            assert content == b"hello world\n"
            assert interesting is True
            await q_send.send_all(b"\n")  # blank: skipped
            await q_send.send_all(b"garbage not json\n")  # malformed: skipped
            await q_send.send_all(encode_query(b"h\n"))  # valid
            line = await fb_reader.readline()
            assert line is not None
            content, interesting = decode_feedback(line)
            assert content == b"h\n"
            assert interesting is True
            await q_send.send_all(encode_idle())  # reducer done for now

    assert result == [True]


async def test_drive_returns_false_on_timeout() -> None:
    """If the reducer sends nothing within the timeout, drive returns False."""
    problem = make_basic_problem(b"hello\n", lambda x: True)

    fb_send, _fb_recv = memory_stream_one_way_pair()
    _q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)

    with trio.fail_after(5):
        alive = await drive_external_reducer(
            problem,
            send_stream=fb_send,
            reader=reader,
            timeout=0.05,
            parallelism=1,
        )
    assert alive is False


class _BlockingSendStream(trio.abc.SendStream):
    """A send stream whose send_all never completes (the peer never reads)."""

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await trio.sleep_forever()

    async def wait_send_all_might_not_block(self) -> None:
        await trio.lowlevel.checkpoint()

    async def aclose(self) -> None:
        await trio.lowlevel.checkpoint()


class _CountingBlockSendStream(trio.abc.SendStream):
    """Completes the first ``allow`` sends, then blocks forever."""

    def __init__(self, allow: int) -> None:
        self._allow = allow

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        if self._allow > 0:
            self._allow -= 1
            await trio.lowlevel.checkpoint()
            return
        await trio.sleep_forever()

    async def wait_send_all_might_not_block(self) -> None:
        await trio.lowlevel.checkpoint()

    async def aclose(self) -> None:
        await trio.lowlevel.checkpoint()


async def test_drive_returns_false_when_initial_send_wedges(autojump_clock) -> None:
    """A reducer that never reads its stdin can't hang the initial send forever.

    With a test case bigger than the pipe buffer and a peer that never reads,
    the initial send would block indefinitely. It is now bounded by the timeout,
    so drive gives up and returns False instead of hanging the whole reduction.
    """
    problem = make_basic_problem(b"x" * 1_000_000, lambda x: True)
    _q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)
    with trio.fail_after(1000):
        alive = await drive_external_reducer(
            problem,
            send_stream=_BlockingSendStream(),
            reader=reader,
            timeout=60.0,
            parallelism=1,
        )
    assert alive is False


async def test_drive_returns_false_when_handler_send_wedges(autojump_clock) -> None:
    """A pipelining reducer that floods queries but never reads our answers can't
    deadlock the driver: handler sends are bounded, so a wedged send tears the
    reduction down instead of blocking forever with the reader stuck acquiring a
    slot (the three-way mutual-flood deadlock)."""
    problem = make_basic_problem(b"hello\n", lambda x: True, parallelism=2)
    send_stream = _CountingBlockSendStream(allow=1)  # initial ok; answers wedge
    q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)
    # Flood more queries than there are slots so the reader blocks acquiring one.
    for i in range(5):
        await q_send.send_all(encode_query(f"cand{i}\n".encode()))
    with trio.fail_after(1000):
        alive = await drive_external_reducer(
            problem,
            send_stream=send_stream,
            reader=reader,
            timeout=60.0,
            parallelism=2,
        )
    assert alive is False


async def test_drive_not_killed_while_query_outstanding(autojump_clock) -> None:
    """A healthy reducer waiting for our (slow) answer is not treated as idle.

    With parallelism >= 2 and one outstanding query, the reducer legitimately
    stays silent until we answer. If the interestingness test is slower than the
    idle timeout, the driver must not fire the idle timeout and kill a healthy
    reducer mid-test.
    """

    async def slow(x: bytes) -> bool:
        await trio.sleep(1.5)  # slower than the 0.5s idle timeout
        return b"h" in x

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=b"hello\n",
        is_interesting=slow,
        work=WorkContext(parallelism=2),
        sort_key=sort_key_for_initial(b"hello\n"),
    )
    fb_send, fb_recv = memory_stream_one_way_pair()
    q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)
    result: list[bool] = []

    with trio.fail_after(1000):
        async with trio.open_nursery() as nursery:

            @nursery.start_soon
            async def _shrinkray() -> None:
                result.append(
                    await drive_external_reducer(
                        problem,
                        send_stream=fb_send,
                        reader=reader,
                        timeout=0.5,
                        parallelism=2,
                    )
                )

            @nursery.start_soon
            async def _reducer() -> None:
                fb_reader = LineReader(fb_recv)
                first = await fb_reader.readline()
                assert first is not None  # initial test case
                await q_send.send_all(encode_query(b"h\n"))
                answer = await fb_reader.readline()
                assert answer is not None
                content, interesting = decode_feedback(answer)
                assert content == b"h\n"
                assert interesting is True
                await q_send.send_all(encode_idle())

    assert result == [True]


async def test_drive_terminates_on_endless_garbage(autojump_clock) -> None:
    """A reducer that only spews malformed lines is still eventually terminated.

    Malformed input is not useful activity, so it must not keep refreshing the
    idle timeout; otherwise a garbage-spewing reducer would stay alive forever.
    """
    problem = make_basic_problem(b"hello\n", lambda x: True)
    fb_send, _fb_recv = memory_stream_one_way_pair()
    q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _garbage() -> None:
            while True:
                await q_send.send_all(b"garbage not json\n")
                await trio.sleep(0.05)

        @nursery.start_soon
        async def _drive() -> None:
            with trio.fail_after(50):
                alive = await drive_external_reducer(
                    problem,
                    send_stream=fb_send,
                    reader=reader,
                    timeout=1.0,
                    parallelism=1,
                )
            assert alive is False
            nursery.cancel_scope.cancel()


async def test_drive_returns_false_and_tolerates_broken_send() -> None:
    """A broken send stream is swallowed and EOF returns False."""
    problem = make_basic_problem(b"hello\n", lambda x: True)

    fb_send, fb_recv = memory_stream_one_way_pair()
    q_send, q_recv = memory_stream_one_way_pair()
    reader = LineReader(q_recv)
    await fb_recv.aclose()  # sending the reduce request will now break
    await q_send.aclose()  # reducer output stream is at EOF

    alive = await drive_external_reducer(
        problem,
        send_stream=fb_send,
        reader=reader,
        timeout=5.0,
        parallelism=1,
    )
    assert alive is False


# === external_reducer factory (fast subprocesses) ===


async def test_external_reducer_launches_and_terminates() -> None:
    """A reducer that reads its first request and exits is handled cleanly."""
    problem = make_basic_problem(b"hello\n", lambda x: True)
    command = [sys.executable, "-c", "import sys; sys.stdin.readline()"]
    reducer_pass = external_reducer(command, log_file=None)
    try:
        with trio.fail_after(30):
            await reducer_pass(problem)
    finally:
        await reducer_pass.aclose()


async def test_external_reducer_passes_extra_env(tmp_path) -> None:
    """extra_env is added to the reducer subprocess's environment."""
    problem = make_basic_problem(b"hello\n", lambda x: True)
    log_file = tmp_path / "reducer.log"
    command = [
        sys.executable,
        "-c",
        "import sys, os; sys.stderr.write(os.environ['SHRINKRAY_TEST_VAR']); "
        "sys.stdin.readline()",
    ]
    reducer_pass = external_reducer(
        command, log_file=str(log_file), extra_env={"SHRINKRAY_TEST_VAR": "custom"}
    )
    with trio.fail_after(30):
        await reducer_pass(problem)
    assert b"custom" in log_file.read_bytes()


async def test_external_reducer_writes_stderr_to_log(tmp_path) -> None:
    """The reducer's stderr is appended to the log file."""
    problem = make_basic_problem(b"hello\n", lambda x: True)
    log_file = tmp_path / "reducer.log"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('hello from reducer'); sys.stdin.readline()",
    ]
    reducer_pass = external_reducer(command, log_file=str(log_file))
    with trio.fail_after(30):
        await reducer_pass(problem)
    assert b"hello from reducer" in log_file.read_bytes()


# A trivial persistent reducer: for each test case it is handed it immediately
# reports idle (and never queries), staying alive until its stdin closes.
_IDLE_REDUCER = (
    "import sys, json\n"
    "for line in sys.stdin:\n"
    "    line = line.strip()\n"
    "    if not line:\n"
    "        continue\n"
    "    obj = json.loads(line)\n"
    "    if 'content' in obj:\n"
    "        sys.stdout.write('{\"idle\": true}\\n'); sys.stdout.flush()\n"
)


async def test_external_reducer_reuses_persistent_subprocess() -> None:
    """A reducer that goes idle stays alive and is reused on the next call."""
    problem = make_basic_problem(b"hello\n", lambda x: True)
    reducer_pass = external_reducer([sys.executable, "-c", _IDLE_REDUCER])
    try:
        with trio.fail_after(30):
            await reducer_pass(problem)
            first_proc = reducer_pass._proc
            assert first_proc is not None  # stayed alive after going idle
            await reducer_pass(problem)
            assert reducer_pass._proc is first_proc  # reused, not relaunched
    finally:
        await reducer_pass.aclose()
    assert reducer_pass._proc is None  # torn down by aclose


async def test_external_reducer_launch_failure_propagates() -> None:
    """If the reducer command cannot be launched, the error propagates."""
    problem = make_basic_problem(b"hello\n", lambda x: True)
    reducer_pass = external_reducer(["/nonexistent/shrinkray-reducer-xyz"])
    with pytest.raises(OSError):
        await reducer_pass(problem)


def test_external_reducer_default_name_from_command() -> None:
    reducer_pass = external_reducer([sys.executable, "-c", "pass"])
    assert reducer_pass.__name__ == f"external:{os.path.basename(sys.executable)}"


def test_external_reducer_custom_name() -> None:
    reducer_pass = external_reducer([sys.executable], name="my-reducer")
    assert reducer_pass.__name__ == "my-reducer"
