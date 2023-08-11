import heapq
import sys
from contextlib import aclosing, asynccontextmanager
from enum import IntEnum
from itertools import islice
from random import Random
import time
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Sequence,
    TypeVar,
)

import trio
from attr import define
from tqdm import tqdm


class Volume(IntEnum):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3


S = TypeVar("S")
T = TypeVar("T")


TICK_FREQUENCY = 0.05


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
        self.__progress_bars: dict[int, ProgressBar] = {}
        self.last_ticked = float("-inf")

    async def map(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[S]]
    ) -> AsyncGenerator[S, None]:
        """Lazy parallel map.

        Does a reasonable amount of fine tuning so that it doesn't race
        ahead of the current point of iteration and will generallly have
        prefetched at most as many values as you've already read. This
        is especially important for its use in implementing `find_first`,
        which we want to avoid doing redundant work when there are lots of
        reduction opportunities.
        """
        if self.parallelism > 1:
            it = iter(ls)

            for x in it:
                yield await f(x)
                break
            else:
                return

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

    async def filter(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[bool]]
    ) -> AsyncGenerator[T, None]:
        async def apply(x: T) -> tuple[T, bool]:
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
        if self.volume >= level:
            tqdm.write(msg, file=sys.stderr)

    @asynccontextmanager
    async def pb(
        self,
        current: Callable[[], int],
        total: Callable[[], int],
        **kwargs: Any,
    ) -> "AsyncIterator[None]":
        if self.volume < Volume.normal:
            yield
        else:
            i = len(self.__progress_bars)
            while i in self.__progress_bars:
                i += 1
            self.__progress_bars[i] = ProgressBar(
                bar=tqdm(total=total(), **kwargs),
                total=total,
                current=current,
            )
            self.tick()
            try:
                yield
            finally:
                self.__progress_bars.pop(i).bar.close()

    def tick(self) -> None:
        now = time.time()
        if now <= self.last_ticked + TICK_FREQUENCY:
            return
        self.last_ticked = now
        for pb in self.__progress_bars.values():
            pb.bar.total = pb.total()
            pb.bar.update(pb.current() - pb.bar.n)
            pb.bar.refresh()


class NotFound(Exception):
    pass


@asynccontextmanager
async def parallel_map(
    ls: Sequence[T], f: Callable[[T], Awaitable[S]], parallelism: int
) -> AsyncGenerator[trio.MemoryReceiveChannel[S], None]:
    async with trio.open_nursery() as nursery:
        send_ls_values, receive_ls_values = trio.open_memory_channel(  # type: ignore
            max_buffer_size=parallelism * 2
        )

        async def queue_producer() -> None:
            for i, x in enumerate(ls):
                await send_ls_values.send((i, x))
            await send_ls_values.aclose()

        nursery.start_soon(queue_producer)

        send_computed_values, receive_computed_values = trio.open_memory_channel(  # type: ignore
            max_buffer_size=parallelism * 2
        )

        async def worker(i: int) -> None:
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

        send_out_values, receive_out_values = trio.open_memory_channel(parallelism * 2)  # type: ignore

        async def consolidate() -> None:
            i = 0
            completed = 0

            heap: list[tuple[int, S]] = []

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

        async with aclosing(receive_out_values):
            yield receive_out_values


@define
class ProgressBar:
    bar: "tqdm[None]"
    total: Callable[[], int]
    current: Callable[[], int]
