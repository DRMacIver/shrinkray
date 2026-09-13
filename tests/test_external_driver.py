"""Unit tests for the external reducer driver (RemoteReductionProblem)."""

import pytest
import trio
from trio.testing import memory_stream_one_way_pair, wait_all_tasks_blocked

from shrinkray.problem import (
    ReductionProblem,
    default_cache_key,
    sort_key_for_initial,
)
from shrinkray.reducers.driver import RemoteReductionProblem, run_reducer
from shrinkray.reducers.protocol import (
    Idle,
    LineReader,
    Query,
    decode_query,
    encode_feedback,
    parse_from_reducer,
)
from shrinkray.work import WorkContext


class CollectingSendStream(trio.abc.SendStream):
    """A minimal send stream that records everything sent to it."""

    def __init__(self) -> None:
        self.sent = bytearray()

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        await trio.lowlevel.checkpoint()
        self.sent.extend(data)

    async def wait_send_all_might_not_block(self) -> None:
        await trio.lowlevel.checkpoint()

    async def aclose(self) -> None:
        await trio.lowlevel.checkpoint()


def make_problem(initial: bytes = b"hello world\n") -> RemoteReductionProblem:
    return RemoteReductionProblem(
        initial,
        send_stream=CollectingSendStream(),
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(initial),
    )


# === Basic properties ===


def test_properties() -> None:
    problem = make_problem(b"abc")
    assert problem.current_test_case == b"abc"
    assert problem.size(b"abcd") == 4
    assert problem.display(b"abc") == repr(b"abc")
    assert problem.stats.initial_test_case_size == 3
    assert problem.sort_key(b"a") < problem.sort_key(b"ab")


def test_handle_feedback_unmatched_returns_false() -> None:
    problem = make_problem(b"hello world\n")
    # No outstanding query matches this content, so it is not a reply.
    assert problem.handle_feedback(b"hi\n", True) is False
    assert problem.current_test_case == b"hello world\n"


def test_set_current_sets_authoritatively() -> None:
    problem = make_problem(b"hello world\n")
    problem.set_current(b"hi\n")  # smaller
    assert problem.current_test_case == b"hi\n"
    # Even a larger value is adopted: shrink ray is telling us what to reduce.
    problem.set_current(b"a much larger value\n")
    assert problem.current_test_case == b"a much larger value\n"
    assert problem.stats.current_test_case_size == len(b"a much larger value\n")


# === is_interesting ===


async def test_is_interesting_returns_true_for_current() -> None:
    problem = make_problem(b"abc")
    # No query is sent for the current test case.
    assert await problem.is_interesting(b"abc") is True
    assert problem._send_stream.sent == b""  # type: ignore[attr-defined]


async def test_is_interesting_awaits_feedback() -> None:
    problem = make_problem(b"hello world\n")

    async with trio.open_nursery() as nursery:
        results: list[bool] = []

        @nursery.start_soon
        async def _() -> None:
            results.append(await problem.is_interesting(b"smaller\n"))

        await wait_all_tasks_blocked()
        # The query was sent, and the call is now blocked awaiting feedback.
        assert decode_query(bytes(problem._send_stream.sent)) == b"smaller\n"  # type: ignore[attr-defined]
        # The reply matches the outstanding query and adopts the candidate.
        assert problem.handle_feedback(b"smaller\n", True) is True

    assert results == [True]
    assert problem.current_test_case == b"smaller\n"
    assert problem.stats.reductions == 1
    assert problem.stats.current_test_case_size == len(b"smaller\n")


async def test_is_interesting_caches_results() -> None:
    problem = make_problem(b"hello world\n")

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _() -> None:
            assert await problem.is_interesting(b"nope-not-smaller-enough\n") is False

        await wait_all_tasks_blocked()
        problem.handle_feedback(b"nope-not-smaller-enough\n", False)

    # A second call returns the cached result without sending another query.
    sent_before = bytes(problem._send_stream.sent)  # type: ignore[attr-defined]
    assert await problem.is_interesting(b"nope-not-smaller-enough\n") is False
    assert bytes(problem._send_stream.sent) == sent_before  # type: ignore[attr-defined]


async def test_is_interesting_cache_keys_are_hashed() -> None:
    """The interestingness cache keys on a short content hash, not full bytes.

    A long reduction tests many distinct multi-MB candidates; keying the cache
    on the full candidate bytes would accumulate all of them in the persistent
    subprocess. The parent process caches short digests for exactly this reason.
    """
    problem = make_problem(b"hello world\n")
    big = b"x" * 100_000 + b"\n"

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _() -> None:
            await problem.is_interesting(big)

        await wait_all_tasks_blocked()
        problem.handle_feedback(big, False)

    # The full candidate is not retained as a key; only its short digest is.
    assert big not in problem._cache
    assert list(problem._cache) == [default_cache_key(big)]


async def test_is_interesting_returns_false_when_closed() -> None:
    problem = make_problem(b"hello world\n")
    problem.close()
    assert await problem.is_interesting(b"anything\n") is False


async def test_is_interesting_tolerates_broken_send() -> None:
    """A broken send mid-query unwinds cleanly instead of crashing the pass.

    If shrink ray tears down the pipe while a query is being sent, ``send_all``
    raises BrokenResourceError. Rather than let that propagate through the
    running pass and crash the reducer subprocess, is_interesting treats it like
    shutdown: it unwinds through close(), returning not-interesting.
    """

    class BrokenSendStream(trio.abc.SendStream):
        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            raise trio.BrokenResourceError

        async def wait_send_all_might_not_block(self) -> None:
            await trio.lowlevel.checkpoint()

        async def aclose(self) -> None:
            await trio.lowlevel.checkpoint()

    problem = RemoteReductionProblem(
        b"hello world\n",
        send_stream=BrokenSendStream(),
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(b"hello world\n"),
    )
    assert await problem.is_interesting(b"smaller\n") is False
    # The connection is now treated as closed, so further queries short-circuit.
    assert problem._closed is True
    assert await problem.is_interesting(b"another\n") is False


async def test_duplicate_concurrent_queries_all_resolve() -> None:
    problem = make_problem(b"hello world\n")

    async with trio.open_nursery() as nursery:
        results: list[bool] = []

        for _ in range(2):

            @nursery.start_soon
            async def query_task() -> None:
                results.append(await problem.is_interesting(b"dup\n"))

        await wait_all_tasks_blocked()
        # Two feedback messages resolve the two outstanding duplicate queries.
        problem.handle_feedback(b"dup\n", False)
        await wait_all_tasks_blocked()
        problem.handle_feedback(b"dup\n", False)

    assert results == [False, False]


async def test_close_fails_outstanding_waiters() -> None:
    problem = make_problem(b"hello world\n")

    async with trio.open_nursery() as nursery:
        results: list[bool] = []

        @nursery.start_soon
        async def _() -> None:
            results.append(await problem.is_interesting(b"pending\n"))

        await wait_all_tasks_blocked()
        problem.close()

    assert results == [False]


# === run_reducer edges ===


async def test_run_reducer_returns_on_empty_handshake() -> None:
    """If shrink ray closes the stream before the handshake, we just return."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    _query_send, query_recv = memory_stream_one_way_pair()
    await feedback_send.aclose()  # immediate EOF, no handshake

    await run_reducer([], stdin_stream=feedback_recv, stdout_stream=_query_send)
    await query_recv.aclose()


async def test_run_reducer_reduces_first_message_then_exits() -> None:
    """The first message is the initial test case; with nothing to reduce the
    reducer reports idle and then exits when the stream closes."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    reader = LineReader(query_recv)

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _reducer() -> None:
            await run_reducer([], stdin_stream=feedback_recv, stdout_stream=query_send)
            await query_send.aclose()

        @nursery.start_soon
        async def _shrinkray() -> None:
            await feedback_send.send_all(encode_feedback(b"x = 1\n", True))
            idle_line = await reader.readline()
            assert idle_line is not None
            assert isinstance(parse_from_reducer(idle_line), Idle)
            await feedback_send.aclose()


async def test_run_reducer_returns_on_malformed_first_message() -> None:
    """A malformed first message ends the reducer cleanly."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    await feedback_send.send_all(b"garbage not json\n")
    await feedback_send.aclose()

    await run_reducer([], stdin_stream=feedback_recv, stdout_stream=query_send)
    await query_recv.aclose()


async def test_run_reducer_handles_multiple_test_cases() -> None:
    """The reducer stays alive and reduces again on each new test case."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    reader = LineReader(query_recv)

    async def chop_last_byte(problem: ReductionProblem[bytes]) -> None:
        current = problem.current_test_case
        if len(current) > 1:
            await problem.is_interesting(current[:-1])

    async def one_session(test_case: bytes) -> bytes:
        """Hand over a test case, answer its single query (False), await idle."""
        await feedback_send.send_all(encode_feedback(test_case, True))
        query_line = await reader.readline()
        assert query_line is not None
        query = parse_from_reducer(query_line)
        assert isinstance(query, Query)
        await feedback_send.send_all(encode_feedback(query.content, False))
        idle_line = await reader.readline()
        assert idle_line is not None
        assert isinstance(parse_from_reducer(idle_line), Idle)
        return query.content

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _reducer() -> None:
            await run_reducer(
                [chop_last_byte],
                stdin_stream=feedback_recv,
                stdout_stream=query_send,
            )
            await query_send.aclose()

        @nursery.start_soon
        async def _shrinkray() -> None:
            # First test case (consumed as the initial).
            assert await one_session(b"abcd") == b"abc"
            # A malformed message between test cases is ignored.
            await feedback_send.send_all(b"junk\n")
            # A second, different test case handed over while the reducer is idle.
            assert await one_session(b"xy") == b"x"
            await feedback_send.aclose()


async def test_send_idle_tolerates_broken_stream() -> None:
    """send_idle swallows a broken send stream rather than raising."""

    class BrokenStream(trio.abc.SendStream):
        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            raise trio.BrokenResourceError

        async def wait_send_all_might_not_block(self) -> None:
            await trio.lowlevel.checkpoint()

        async def aclose(self) -> None:
            await trio.lowlevel.checkpoint()

    problem = RemoteReductionProblem(
        b"x",
        send_stream=BrokenStream(),
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(b"x"),
    )
    await problem.send_idle()  # must not raise


async def test_run_reducer_skips_blank_lines() -> None:
    """Blank lines are ignored by the reducer's message reader."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()

    async def one_shot(problem: ReductionProblem[bytes]) -> None:
        await problem.is_interesting(b"y = 2\n")

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def _reducer() -> None:
            await run_reducer(
                [one_shot],
                stdin_stream=feedback_recv,
                stdout_stream=query_send,
            )
            await query_send.aclose()

        @nursery.start_soon
        async def _shrinkray() -> None:
            reader = LineReader(query_recv)
            await feedback_send.send_all(encode_feedback(b"x = 1\n", True))
            await feedback_send.send_all(b"\n")  # blank line: should be skipped
            query_line = await reader.readline()
            assert query_line is not None
            query = parse_from_reducer(query_line)
            assert isinstance(query, Query)
            await feedback_send.send_all(encode_feedback(query.content, False))
            idle_line = await reader.readline()
            assert idle_line is not None
            assert isinstance(parse_from_reducer(idle_line), Idle)
            await feedback_send.aclose()


@pytest.mark.parametrize("first_verdict", [False, True])
async def test_restart_rechecks_cached_verdicts(first_verdict) -> None:
    initial = b"hello world\n"
    problem = make_problem(initial)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(problem.is_interesting, b"x\n")
        await wait_all_tasks_blocked()
        assert problem.handle_feedback(b"x\n", first_verdict)

    problem.set_current(initial)
    results = []
    async with trio.open_nursery() as nursery:

        async def query():
            results.append(await problem.is_interesting(b"x\n"))

        nursery.start_soon(query)
        await wait_all_tasks_blocked()
        # The parent's policy may have changed or backtracked since the last
        # request. It must decide this proposal again, including former hits.
        assert problem.handle_feedback(b"x\n", not first_verdict)

    assert results == [not first_verdict]
    assert problem.current_test_case == (initial if first_verdict else b"x\n")


async def test_cancelled_query_waiting_to_send_leaves_no_phantom_reply() -> None:
    problem = make_problem()
    scope = trio.CancelScope()
    await problem._send_lock.acquire()
    async with trio.open_nursery() as nursery:

        async def cancelled_query():
            with scope:
                await problem.is_interesting(b"x")

        nursery.start_soon(cancelled_query)
        await wait_all_tasks_blocked()
        scope.cancel()
    problem._send_lock.release()
    assert not problem._waiters

    results = []
    async with trio.open_nursery() as nursery:

        async def retry():
            results.append(await problem.is_interesting(b"x"))

        nursery.start_soon(retry)
        await wait_all_tasks_blocked()
        assert problem.handle_feedback(b"x", False)
    assert results == [False]


async def test_connection_closed_while_query_waits_for_send_lock() -> None:
    problem = make_problem()
    results = []
    await problem._send_lock.acquire()

    async def query():
        results.append(await problem.is_interesting(b"x"))

    async with trio.open_nursery() as nursery:
        nursery.start_soon(query)
        await wait_all_tasks_blocked()
        problem.close()
        problem._send_lock.release()
    assert results == [False]
    assert not problem._waiters


async def test_cancelled_partial_query_closes_the_connection() -> None:
    class PartialSendStream(CollectingSendStream):
        closed = False

        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            self.sent.extend(data[:1])
            await trio.sleep_forever()

        async def aclose(self) -> None:
            self.closed = True
            await trio.lowlevel.checkpoint()

    stream = PartialSendStream()
    problem = RemoteReductionProblem(
        b"hello world",
        send_stream=stream,
        work=WorkContext(parallelism=1),
        sort_key=sort_key_for_initial(b"hello world"),
    )
    async with trio.open_nursery() as nursery:
        nursery.start_soon(problem.is_interesting, b"x")
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()

    assert stream.closed
    assert problem._closed
    assert not problem._waiters
    assert not await problem.is_interesting(b"y")


async def test_cancelled_sent_query_still_consumes_its_own_reply() -> None:
    problem = make_problem()
    scope = trio.CancelScope()
    async with trio.open_nursery() as nursery:

        async def first():
            with scope:
                await problem.is_interesting(b"x")

        nursery.start_soon(first)
        await wait_all_tasks_blocked()
        scope.cancel()

    results = []
    async with trio.open_nursery() as nursery:

        async def second():
            results.append(await problem.is_interesting(b"x"))

        nursery.start_soon(second)
        await wait_all_tasks_blocked()
        assert problem.handle_feedback(b"x", False)
        await wait_all_tasks_blocked()
        assert not results
        assert problem.handle_feedback(b"x", True)

    assert results == [True]
    assert not problem._waiters


@pytest.mark.parametrize("send_query", [False, True])
async def test_broken_output_ends_session_without_waiting_for_input(
    autojump_clock,
    send_query,
) -> None:
    class BrokenSendStream(CollectingSendStream):
        async def send_all(self, data: bytes | bytearray | memoryview) -> None:
            raise trio.BrokenResourceError

    feedback_send, feedback_recv = memory_stream_one_way_pair()
    await feedback_send.send_all(encode_feedback(b"hello world", True))

    async def propose(problem):
        await problem.is_interesting(b"x")

    # In a subprocess, closing the duplicated output fd does not close the
    # original stdout fd. The parent may therefore still await output/EOF and
    # cannot be relied on to close its input before this session can terminate.
    with trio.fail_after(1):
        await run_reducer(
            [propose] if send_query else [],
            stdin_stream=feedback_recv,
            stdout_stream=BrokenSendStream(),
        )
    await feedback_send.aclose()
