"""Helpers for writing external reducers in Python.

This module provides :class:`RemoteReductionProblem`, a
:class:`~shrinkray.problem.ReductionProblem` whose ``is_interesting`` is
answered by shrink ray over the external-reducer protocol, and
:func:`run_reducer`, which wires a list of ordinary reduction passes up to that
problem and drives them to a fixed point.

The upshot is that a pass written against ``ReductionProblem`` (such as the
libcst-based Python passes) can run unchanged inside an external reducer
subprocess: only the problem it runs against changes.
"""

from collections import deque
from collections.abc import Callable, Iterable
from random import Random
from typing import Any

import trio

from shrinkray.passes.definitions import ReductionPass
from shrinkray.problem import (
    ReductionProblem,
    ReductionStats,
    sort_key_for_initial,
)
from shrinkray.reducers.protocol import (
    LineReader,
    decode_feedback,
    encode_query,
)
from shrinkray.work import WorkContext


class RemoteReductionProblem(ReductionProblem[bytes]):
    """A reduction problem whose interestingness test lives in another process.

    Candidates are sent to shrink ray as query messages; results arrive as
    feedback messages, which :meth:`handle_feedback` dispatches back to the
    awaiting :meth:`is_interesting` call. The current test case is kept in sync
    by adopting any feedback content that is interesting and smaller than what
    we currently hold.
    """

    def __init__(
        self,
        initial: bytes,
        *,
        send_stream: trio.abc.SendStream,
        work: WorkContext,
        sort_key: Callable[[bytes], Any],
    ) -> None:
        super().__init__(work=work)
        self.__current = initial
        self.__sort_key = sort_key
        self._send_stream = send_stream
        self._send_lock = trio.Lock()
        # Outstanding queries, keyed by content. Each value is a queue of
        # (event, result-slot) pairs so that duplicate concurrent queries for
        # the same content are each resolved exactly once, in order.
        self._waiters: dict[bytes, deque[tuple[trio.Event, list[bool]]]] = {}
        self._cache: dict[bytes, bool] = {}
        self._closed = False
        self._stats = ReductionStats(
            initial_test_case_size=len(initial),
            current_test_case_size=len(initial),
        )

    @property
    def current_test_case(self) -> bytes:
        return self.__current

    @property
    def stats(self) -> ReductionStats:
        return self._stats

    def sort_key(self, test_case: bytes) -> Any:
        return self.__sort_key(test_case)

    def size(self, test_case: bytes) -> int:
        return len(test_case)

    def display(self, value: bytes) -> str:
        return repr(value)

    def _consider(self, content: bytes, interesting: bool) -> None:
        """Adopt ``content`` as the current test case if it improves on it."""
        if interesting and self.__sort_key(content) < self.__sort_key(self.__current):
            self.__current = content
            self._stats.current_test_case_size = len(content)
            self._stats.reductions += 1

    def handle_feedback(self, content: bytes, interesting: bool) -> None:
        """Process one feedback message from shrink ray.

        Updates the current test case and resolves a matching outstanding query
        (if any). Feedback with no matching query is an unsolicited update to
        the current test case.
        """
        self._consider(content, interesting)
        queue = self._waiters.get(content)
        if queue:
            event, slot = queue.popleft()
            slot.append(interesting)
            event.set()
            if not queue:
                del self._waiters[content]

    def close(self) -> None:
        """Mark the connection closed and fail all outstanding queries.

        Called when shrink ray closes the feedback stream. Any ``is_interesting``
        call still awaiting a result is resolved as not interesting so it can
        unwind rather than hang forever.
        """
        self._closed = True
        for queue in self._waiters.values():
            # Every queued waiter is unresolved (handle_feedback removes a waiter
            # the moment it resolves it), so each still has an empty result slot.
            for event, slot in queue:
                slot.append(False)
                event.set()
        self._waiters.clear()

    async def is_interesting(self, test_case: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        if test_case == self.__current:
            return True
        try:
            return self._cache[test_case]
        except KeyError:
            pass
        if self._closed:
            return False

        event = trio.Event()
        slot: list[bool] = []
        self._waiters.setdefault(test_case, deque()).append((event, slot))
        async with self._send_lock:
            await self._send_stream.send_all(encode_query(test_case))

        self._stats.calls += 1
        await event.wait()
        result = slot[0]
        self._cache[test_case] = result
        if result:
            self._stats.interesting_calls += 1
        return result


async def run_passes_to_fixpoint(
    problem: ReductionProblem[bytes],
    passes: Iterable[ReductionPass[bytes]],
) -> None:
    """Run each pass in turn, repeating until a full round makes no progress."""
    passes = list(passes)
    while True:
        prev = problem.current_test_case
        for reduction_pass in passes:
            await reduction_pass(problem)
        if problem.current_test_case == prev:
            return


async def run_reducer(
    passes: Iterable[ReductionPass[bytes]],
    *,
    stdin_stream: trio.abc.ReceiveStream,
    stdout_stream: trio.abc.SendStream,
    parallelism: int = 1,
    seed: int = 0,
) -> None:
    """Drive ``passes`` as an external reducer over the given streams.

    Reads the handshake (the initial test case) from ``stdin_stream``, then runs
    the passes to a fixed point while a background task reads feedback. Returns
    when the passes finish or shrink ray closes ``stdin_stream``.
    """
    passes = list(passes)
    reader = LineReader(stdin_stream)

    handshake = await reader.readline()
    if handshake is None:
        # Shrink ray closed the connection before sending anything.
        return
    initial, _ = decode_feedback(handshake)

    work = WorkContext(parallelism=parallelism, random=Random(seed))
    problem = RemoteReductionProblem(
        initial,
        send_stream=stdout_stream,
        work=work,
        sort_key=sort_key_for_initial(initial),
    )

    async with trio.open_nursery() as nursery:

        @nursery.start_soon
        async def read_feedback() -> None:
            while True:
                line = await reader.readline()
                if line is None:
                    break
                if line.strip():
                    content, interesting = decode_feedback(line)
                    problem.handle_feedback(content, interesting)
            # Shrink ray is gone: unblock any pending queries and stop.
            problem.close()
            nursery.cancel_scope.cancel()

        await run_passes_to_fixpoint(problem, passes)
        nursery.cancel_scope.cancel()
