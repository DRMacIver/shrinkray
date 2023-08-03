import heapq
import sys
from contextlib import AsyncExitStack
from contextlib import aclosing
from contextlib import asynccontextmanager
from enum import IntEnum
from itertools import islice
from random import Random
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import TypeVar

import trio


class Volume(IntEnum):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3


S = TypeVar("S")
T = TypeVar("T")


class WorkContext:
    """A grab bag of useful tools for 'doing work'. Manages randomness,
    logging, concurrency."""

    def __init__(
        self,
        random: Optional[Random] = None,
        parallelism: int = 1,
        volume: Volume = Volume.normal,
    ):
        self.random = random
        self.parallelism = parallelism
        self.volume = volume

    async def map(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[S]]
    ) -> AsyncIterator[S]:
        if self.parallelism > 1:
            it = iter(ls)

            yield await f(next(it))

            n = 2
            while True:
                values = list(islice(it, n))
                if not values:
                    return

                async with parallel_map(
                    values, f, parallelism=min(self.parallelism, n)
                ) as result:
                    async for v in result:
                        yield v

                n *= 2
        else:
            for x in ls:
                yield await f(x)

    async def filter(self, ls: Sequence[T], f: Callable[[T], bool]) -> AsyncIterator[T]:
        async def apply(x: T) -> (T, bool):
            return (x, await f(x))

        async with aclosing(self.map(ls, apply)) as results:
            async for x, v in results:
                if v:
                    yield x

    async def find_first_value(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[bool]]
    ) -> T:
        """Returns the first element of `ls` that satisfies `f`, or
        raises `NotFound` if no such element exists.

        Will run in parallel if parallelism is enabled.
        """
        async with aclosing(self.filter(ls, f)) as filtered:
            async for x in filtered:
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
        # f(lo + 1), as desired..
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
        if self.volume >= level:
            print(msg, file=sys.stderr)


class NotFound(Exception):
    pass


@asynccontextmanager
async def parallel_map(
    ls: Sequence[T], f: Callable[[T], S], parallelism: int
) -> AsyncIterator[S]:
    async with trio.open_nursery() as nursery:
        send_ls_values, receive_ls_values = trio.open_memory_channel(
            max_buffer_size=parallelism * 2
        )

        async def queue_producer():
            for i, x in enumerate(ls):
                await send_ls_values.send((i, x))
            await send_ls_values.aclose()

        nursery.start_soon(queue_producer)

        send_computed_values, receive_computed_values = trio.open_memory_channel(
            max_buffer_size=parallelism * 2
        )

        async def worker(i):
            while True:
                try:
                    i, x = await receive_ls_values.receive()
                except trio.EndOfChannel:
                    await send_computed_values.send(None)
                    return
                result = await f(x)
                await send_computed_values.send((i, result))

        for worker_n in range(parallelism):
            nursery.start_soon(worker, worker_n)

        send_out_values, receive_out_values = trio.open_memory_channel(parallelism * 2)

        async def consolidate():
            i = 0
            completed = 0

            heap = []

            while completed < parallelism:
                value = await receive_computed_values.receive()
                if value is None:
                    completed += 1
                    continue
                heapq.heappush(heap, value)

                while heap:
                    j, x = heap[0]
                    if j == i:
                        try:
                            await send_out_values.send(x)
                        except trio.BrokenResourceError:
                            return
                        heapq.heappop(heap)
                        i += 1
                    else:
                        break
            await send_out_values.aclose()

        nursery.start_soon(consolidate)

        try:
            yield receive_out_values
        finally:
            await receive_out_values.aclose()
