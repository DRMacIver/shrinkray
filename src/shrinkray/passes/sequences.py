from collections import defaultdict
from typing import Any, Generic, Sequence, TypeVar

import trio

from shrinkray.passes.patching import Cuts, apply_patches
from shrinkray.problem import ReductionProblem

Seq = TypeVar("Seq", bound=Sequence[Any])


async def delete_elements(problem: ReductionProblem[Seq]):
    await apply_patches(
        problem, Cuts(), [[(i, i + 1)] for i in range(len(problem.current_test_case))]
    )


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
        n = len(problem.current_test_case)
        blocks = [
            [(i, i + block_size)]
            for block_size in range(min_block, max_block + 1)
            for offset in range(block_size)
            for i in range(offset, n, block_size)
            if i + block_size <= n
        ]
        await apply_patches(problem, Cuts(), blocks)

    apply.__name__ = f"block_deletion({min_block}, {max_block})"
    return apply


async def delete_duplicates(problem: ReductionProblem[Seq]):
    index = defaultdict(list)

    for i, c in enumerate(problem.current_test_case):
        index[c].append(i)

    cuts = []

    for ix in index.values():
        if len(ix) > 1:
            cuts.append([(i, i + 1) for i in ix])
    await apply_patches(problem, Cuts(), cuts)
