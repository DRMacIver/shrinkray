from typing import Any, Generic, Iterable, Sequence, TypeVar, cast

import trio

from shrinkray.problem import ReductionProblem
from shrinkray.passes.definitions import ReductionPass
from shrinkray.work import NotFound

Seq = TypeVar("Seq", bound=Sequence[Any])


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
            nursery.start_soon(self.delete_backwards)
            nursery.start_soon(self.delete_forwards)
            for _ in range(self.problem.work.parallelism):
                nursery.start_soon(self.delete_randomly)


async def delete_elements(problem: ReductionProblem[Seq]):
    await ElementDeleter(problem).run()


def merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized = []
    for start, end in sorted(map(tuple, intervals)):
        if normalized and normalized[-1][-1] >= start:
            normalized[-1][-1] = max(normalized[-1][-1], end)
        else:
            normalized.append([start, end])
    return list(map(tuple, normalized))


def with_deletions(target: Seq, deletions: list[tuple[int, int]]) -> Seq:
    result = []
    prev = 0
    total_deleted = 0
    for start, end in deletions:
        total_deleted += end - start
        result.extend(target[prev:start])
        prev = end
    result.extend(target[prev:])
    assert len(result) + total_deleted == len(target)
    return type(target)(result)


class SimpleCutTarget(Generic[Seq]):
    def __init__(self, problem: ReductionProblem[Seq]):
        self.problem = problem
        self.target = problem.current_test_case

        self.applied = []
        self.generation = 0
        self.current_merge_attempts = 0

    async def try_merge(self, intervals: list[tuple[int, int]]) -> bool:
        iters = 0
        while self.current_merge_attempts > 0:
            iters += 1
            if iters == 1:
                await trio.lowlevel.checkpoint()
            else:
                await trio.sleep(0.05)

        trying_to_merge = False
        try:
            while True:
                generation_at_start = self.generation

                merged = merged_intervals(self.applied + intervals)
                if merged == self.applied:
                    return True
                attempt = with_deletions(self.target, merged)

                succeeded = await self.problem.is_interesting(attempt)

                if not succeeded:
                    return False

                if self.generation == generation_at_start:
                    self.generation += 1
                    self.applied = merged
                    return True
                if not trying_to_merge:
                    trying_to_merge = True
                    self.current_merge_attempts += 1
        finally:
            if trying_to_merge:
                self.current_merge_attempts -= 1
                assert self.current_merge_attempts >= 0


def block_deletion(min_block, max_block):
    async def apply(problem: ReductionProblem[Seq]):
        async with trio.open_nursery() as nursery:
            send_blocks, receive_blocks = trio.open_memory_channel(1000)

            cutter = SimpleCutTarget(problem)

            @nursery.start_soon
            async def _():
                cuts = [
                    (i, i + block_size)
                    for block_size in range(min_block, max_block + 1)
                    for i in range(len(problem.current_test_case) - block_size)
                ]
                while cuts:
                    i = problem.work.random.randrange(0, len(cuts))
                    cuts[i], cuts[-1] = cuts[-1], cuts[i]
                    await send_blocks.send(cuts.pop())

                send_blocks.close()

            for _ in range(problem.work.parallelism):

                @nursery.start_soon
                async def _():
                    while True:
                        try:
                            cut = await receive_blocks.receive()
                        except trio.EndOfChannel:
                            break
                        await cutter.try_merge([cut])

    apply.__name__ = f"block_deletion({min_block}, {max_block})"
    return apply
