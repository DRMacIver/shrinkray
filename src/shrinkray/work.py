"""Parallelism coordination and work management.

This module provides WorkContext, which manages parallel execution of
reduction work using Trio. The key insight is that test-case reduction
benefits from parallelism even though results are sequential: we can
speculatively test multiple candidates and keep the first success.

Key concepts:
- Lazy evaluation: Don't compute results until needed
- Backpressure: Limit in-flight work to avoid memory exhaustion
- Order preservation: Results come out in input order for reproducibility
"""

import heapq
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing, asynccontextmanager
from enum import IntEnum
from itertools import islice
from random import Random
from typing import TypeVar

import trio


class Volume(IntEnum):
    """Logging verbosity levels."""

    quiet = 0
    normal = 1
    verbose = 2
    debug = 3


S = TypeVar("S")
T = TypeVar("T")


class WorkContext:
    """Coordinates parallel work execution for reduction.

    WorkContext provides methods for parallel map, filter, and search
    operations that are tailored for test-case reduction:

    - map(): Lazy parallel map with backpressure
    - filter(): Parallel filter, yielding matching items
    - find_first_value(): Find first item satisfying a predicate
    - find_large_integer(): Binary search for largest valid integer

    The parallelism is speculative: multiple candidates are tested
    concurrently, but only one result "wins". This trades CPU for
    wall-clock time, which is usually worthwhile when interestingness
    tests are slow.
    """

    def __init__(
        self,
        random: Random | None = None,
        parallelism: int = 1,
        volume: Volume = Volume.normal,
    ):
        self.random = random or Random(0)
        self.parallelism = parallelism
        self.volume = volume
        self.last_ticked = float("-inf")

    @asynccontextmanager
    async def map(self, ls: Sequence[T], f: Callable[[T], Awaitable[S]]):
        """Lazy parallel map.

        Does a reasonable amount of fine tuning so that it doesn't race
        ahead of the current point of iteration and will generallly have
        prefetched at most as many values as you've already read. This
        is especially important for its use in implementing `find_first`,
        which we want to avoid doing redundant work when there are lots of
        reduction opportunities.
        """

        async with trio.open_nursery() as nursery:
            send, receive = trio.open_memory_channel(self.parallelism + 1)

            @nursery.start_soon
            async def do_map():
                if self.parallelism > 1:
                    it = iter(ls)

                    for x in it:
                        await send.send(await f(x))
                        break
                    else:
                        send.close()
                        return

                    n = 2
                    while True:
                        values = list(islice(it, n))
                        if not values:
                            send.close()
                            return

                        async with parallel_map(
                            values, f, parallelism=min(self.parallelism, n)
                        ) as result:
                            async with aclosing(result) as aiter:
                                async for v in aiter:
                                    await send.send(v)

                        n *= 2
                else:
                    for x in ls:
                        await send.send(await f(x))
                    send.close()

            yield receive

    @asynccontextmanager
    async def filter(self, ls: Sequence[T], f: Callable[[T], Awaitable[bool]]):
        async def apply(x: T) -> tuple[T, bool]:
            return (x, await f(x))

        async with trio.open_nursery() as nursery:
            send, receive = trio.open_memory_channel(float("inf"))

            @nursery.start_soon
            async def _():
                async with self.map(ls, apply) as results:
                    async with aclosing(results) as aiter:
                        async for x, v in aiter:
                            if v:
                                await send.send(x)
                    send.close()

            yield receive
            nursery.cancel_scope.cancel()

    async def find_first_value(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[bool]]
    ) -> T:
        """Returns the first element of `ls` that satisfies `f`, or
        raises `NotFound` if no such element exists.

        Will run in parallel if parallelism is enabled.
        """
        async with self.filter(ls, f) as filtered:
            async with aclosing(filtered) as aiter:
                async for x in aiter:
                    return x
        raise NotFound()

    async def find_large_integer(self, f: Callable[[int], Awaitable[bool]]) -> int:
        """Finds a (hopefully large) integer n such that f(n) is True and f(n + 1)
        is False. Runs in O(log(n)).

        f(0) is assumed to be True and will not be checked. May not terminate unless
        f(n) is False for all sufficiently large n.
        """
        # We first do a linear scan over the small numbers and only start to do
        # anything intelligent if f(4) is true. This is because it's very hard to
        # win big when the result is small. If the result is 0 and we try 2 first
        # then we've done twice as much work as we needed to!
        for i in range(1, 5):
            if not await f(i):
                return i - 1

        # We now know that f(4) is true. We want to find some number for which
        # f(n) is *not* true.
        # lo is the largest number for which we know that f(lo) is true.
        lo = 4

        # Exponential probe upwards until we find some value hi such that f(hi)
        # is not true. Subsequently we maintain the invariant that hi is the
        # smallest number for which we know that f(hi) is not true.
        hi = 5
        while await f(hi):
            lo = hi
            hi *= 2

        # Now binary search until lo + 1 = hi. At that point we have f(lo) and not
        # f(lo + 1), as desired.
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if await f(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def warn(self, msg: str) -> None:
        self.report(msg, Volume.normal)

    def note(self, msg: str) -> None:
        self.report(msg, Volume.normal)

    def debug(self, msg: str) -> None:
        self.report(msg, Volume.debug)

    def report(self, msg: str, level: Volume) -> None:
        return


class NotFound(Exception):
    pass


@asynccontextmanager
async def parallel_map(
    ls: Sequence[T],
    f: Callable[[T], Awaitable[S]],
    parallelism: int,
):
    send_out_values, receive_out_values = trio.open_memory_channel(parallelism)

    work = list(enumerate(ls))
    work.reverse()

    result_heap = []

    async with trio.open_nursery() as nursery:
        results_ready = trio.Event()

        for _ in range(parallelism):

            @nursery.start_soon
            async def do_work():
                while work:
                    i, x = work.pop()
                    result = await f(x)
                    heapq.heappush(result_heap, (i, result))
                    results_ready.set()

        @nursery.start_soon
        async def consolidate() -> None:
            i = 0

            while work or result_heap:
                while not result_heap:
                    await results_ready.wait()
                assert result_heap
                j, x = result_heap[0]
                if j == i:
                    await send_out_values.send(x)
                    i = j + 1
                    heapq.heappop(result_heap)
                else:
                    await results_ready.wait()
            send_out_values.close()

        yield receive_out_values
        nursery.cancel_scope.cancel()
