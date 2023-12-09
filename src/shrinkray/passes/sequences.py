from typing import Any, Generic, Iterable, Sequence, TypeVar, cast

import trio

from shrinkray.problem import ReductionProblem
from shrinkray.reducer import ReductionPass
from shrinkray.work import NotFound

Seq = TypeVar("Seq", bound=Sequence[Any])


async def single_forward_delete(problem: ReductionProblem[Seq]) -> None:
    test_case = problem.current_test_case

    if not test_case:
        await trio.lowlevel.checkpoint()
        return

    def deleted(j: int, k: int) -> Seq:
        return test_case[:j] + test_case[k:]  # type: ignore

    async def can_delete(j: int, k: int) -> bool:
        return await problem.is_interesting(deleted(j, k))

    i = 0

    while i < len(test_case):
        try:
            i = await problem.work.find_first_value(
                range(i, len(test_case)), lambda j: can_delete(j, j + 1)
            )
        except NotFound:
            break

        test_case = deleted(i, i + 1)

        async def delete_k(k: int) -> bool:
            if i + k > len(test_case):
                await trio.lowlevel.checkpoint()
                return False
            return await can_delete(i, i + k)

        k = await problem.work.find_large_integer(delete_k)
        test_case = deleted(i, i + k)

        i += 1


async def single_backward_delete(problem: ReductionProblem[Seq]) -> None:
    test_case = problem.current_test_case
    if not test_case:
        await trio.lowlevel.checkpoint()
        return

    def deleted(j: int, k: int) -> Seq:
        return test_case[:j] + test_case[k:]  # type: ignore

    async def can_delete(j: int, k: int) -> bool:
        return await problem.is_interesting(deleted(j, k))

    i = len(test_case) - 1

    while i >= 0:
        try:
            i = await problem.work.find_first_value(
                range(i, -1, -1), lambda j: can_delete(j, j + 1)
            )
        except NotFound:
            break

        test_case = deleted(i, i + 1)

        async def delete_k(k: int) -> bool:
            if k > i:
                await trio.lowlevel.checkpoint()
                return False
            return await can_delete(i - k, i)

        k = await problem.work.find_large_integer(delete_k)
        test_case = deleted(i - k, i)
        i -= k + 1


class ElementDeleter(Generic[Seq]):
    def __init__(self, problem: ReductionProblem[Seq]):
        self.problem = problem
        self.initial = problem.current_test_case
        self.claimed = set()
        self.deleted = frozenset()
        self.indices = range(len(self.initial))
        element_index = {}

        for i, x in enumerate(self.initial):
            element_index.setdefault(x, set()).add(i)

        for k, v in element_index.items():
            element_index[k] = frozenset(v)

        self.element_index = element_index

    async def try_delete_region(self, start: int, end: int) -> bool:
        if start < 0 or end > len(self.initial):
            await trio.lowlevel.checkpoint()
            return False

        to_delete = frozenset(range(start, end))

        return await self.try_delete_set(to_delete)

    async def try_delete_set(self, to_delete: frozenset[int]) -> bool:
        iters = 0
        while True:
            new_deletions = self.deleted | to_delete

            iters += 1
            assert iters <= 100

            if to_delete.issubset(self.deleted):
                await trio.lowlevel.checkpoint()
                return True

            attempt = cast(
                Seq,
                type(self.initial)(
                    [s for i, s in enumerate(self.initial) if i not in new_deletions]
                ),
            )
            if attempt == self.problem.current_test_case:
                return True
            if not await self.problem.is_interesting(attempt):
                return False
            if self.deleted.issubset(new_deletions):
                self.deleted = new_deletions
                return True
            # Sleep for a bit to give whichever other task is making progress
            # that's stomping on ours time to do its thing.
            await trio.sleep(self.problem.work.random.expovariate(1))

    async def try_delete_element(self, i: int) -> bool:
        if i in self.claimed or i in self.deleted:
            self.claimed.add(i)
            await trio.lowlevel.checkpoint()
            return False
        self.claimed.add(i)
        if await self.try_delete_region(i, i + 1):
            others = self.element_index[self.initial[i]]
            if len(others) > 1:
                await self.try_delete_set(others)
            return True
        return False

    async def sweep_forward(self, i: int) -> None:
        k = await self.problem.work.find_large_integer(
            lambda k: self.try_delete_region(i, i + k)
        )
        self.claimed.add(i + k)

    async def sweep_backward(self, i: int) -> None:
        k = await self.problem.work.find_large_integer(
            lambda k: self.try_delete_region(i - k, i + 1)
        )
        self.claimed.add(i - k - 1)

    async def delete_forwards(self):
        for i in self.indices:
            if await self.try_delete_element(i):
                await self.sweep_forward(i)

    async def delete_backwards(self):
        for i in reversed(self.indices):
            if await self.try_delete_element(i):
                await self.sweep_backward(i)

    async def delete_randomly(self):
        indices = list(self.indices)
        self.problem.work.random.shuffle(indices)
        for i in indices:
            if await self.try_delete_element(i):
                await self.sweep_backward(i)
                await self.sweep_forward(i)

    async def run(self):
        async with trio.open_nursery() as nursery:
            for _ in range(self.problem.work.parallelism):
                nursery.start_soon(self.delete_backwards)
                nursery.start_soon(self.delete_forwards)
                nursery.start_soon(self.delete_randomly)


async def delete_elements(problem: ReductionProblem[Seq]):
    await ElementDeleter(problem).run()


def sequence_passes(
    problem: ReductionProblem[Seq],
) -> Iterable[ReductionPass[Seq]]:
    yield delete_elements
