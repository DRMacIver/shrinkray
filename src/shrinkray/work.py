from enum import IntEnum
from itertools import islice
from random import Random
import sys
from typing import Awaitable, Callable, Optional, Sequence, TypeVar


class Volume(IntEnum):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3


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
        assert self.parallelism <= 1  # TODO: Make parallelism work on trio

    async def find_first_value(
        self, ls: Sequence[T], f: Callable[[T], Awaitable[bool]]
    ) -> T:
        """Returns the first element of `ls` that satisfies `f`, or
        raises `NotFound` if no such element exists.

        Will run in parallel if parallelism is enabled.
        """
        if not ls:
            raise NotFound()
        if self.parallelism > 1:
            assert False
        else:
            for x in ls:
                if await f(x):
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

    def debug(self, msg: str) -> None:
        self.report(msg, Volume.debug)

    def report(self, msg: str, level: Volume) -> None:
        if self.volume >= level:
            print(msg, file=sys.stderr)


class NotFound(Exception):
    pass
