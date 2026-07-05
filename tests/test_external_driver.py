"""Unit tests for the external reducer driver (RemoteReductionProblem)."""

import trio
from trio.testing import memory_stream_one_way_pair, wait_all_tasks_blocked

from shrinkray.problem import ReductionProblem, sort_key_for_initial
from shrinkray.reducers.driver import RemoteReductionProblem, run_reducer
from shrinkray.reducers.protocol import (
    Idle,
    LineReader,
    Query,
    decode_query,
    encode_feedback,
    encode_reduce,
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


def test_handle_feedback_adopts_smaller_interesting() -> None:
    problem = make_problem(b"hello world\n")
    problem.handle_feedback(b"hi\n", True)
    assert problem.current_test_case == b"hi\n"
    assert problem.stats.reductions == 1
    assert problem.stats.current_test_case_size == 3


def test_handle_feedback_ignores_larger() -> None:
    problem = make_problem(b"hi\n")
    problem.handle_feedback(b"a much longer thing\n", True)
    assert problem.current_test_case == b"hi\n"


def test_handle_feedback_ignores_uninteresting() -> None:
    problem = make_problem(b"hello world\n")
    problem.handle_feedback(b"hi\n", False)
    assert problem.current_test_case == b"hello world\n"


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
        problem.handle_feedback(b"smaller\n", True)

    assert results == [True]
    assert problem.current_test_case == b"smaller\n"


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


async def test_is_interesting_returns_false_when_closed() -> None:
    problem = make_problem(b"hello world\n")
    problem.close()
    assert await problem.is_interesting(b"anything\n") is False


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


async def test_run_reducer_ignores_non_reduce_first_message() -> None:
    """A first message that is not a reduce request ends the reducer."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()

    # Send feedback (not a reduce request) as the first message.
    await feedback_send.send_all(encode_feedback(b"x = 1\n", True))
    await feedback_send.aclose()

    await run_reducer([], stdin_stream=feedback_recv, stdout_stream=query_send)
    await query_recv.aclose()


async def test_run_reducer_returns_on_malformed_first_message() -> None:
    """A malformed first message ends the reducer cleanly."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    await feedback_send.send_all(b"garbage not json\n")
    await feedback_send.aclose()

    await run_reducer([], stdin_stream=feedback_recv, stdout_stream=query_send)
    await query_recv.aclose()


async def test_run_reducer_handles_multiple_reduce_requests() -> None:
    """The reducer stays alive and reduces again on each reduce request."""
    feedback_send, feedback_recv = memory_stream_one_way_pair()
    query_send, query_recv = memory_stream_one_way_pair()
    reader = LineReader(query_recv)

    async def chop_last_byte(problem: ReductionProblem[bytes]) -> None:
        current = problem.current_test_case
        if len(current) > 1:
            await problem.is_interesting(current[:-1])

    async def one_session(request_content: bytes) -> bytes:
        """Send a reduce request, answer its single query (False), await idle."""
        await feedback_send.send_all(encode_reduce(request_content))
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
            # First reduce request.
            assert await one_session(b"abcd") == b"abc"
            # A malformed message between requests is ignored.
            await feedback_send.send_all(b"junk\n")
            # Second reduce request with a different (smaller) current.
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
            await feedback_send.send_all(encode_reduce(b"x = 1\n"))
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
