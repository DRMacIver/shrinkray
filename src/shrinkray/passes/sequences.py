from collections import defaultdict
from typing import Any, Sequence, TypeVar

from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.patching import CutPatch, Cuts, apply_patches
from shrinkray.problem import ReductionProblem

Seq = TypeVar("Seq", bound=Sequence[Any])


async def delete_elements(problem: ReductionProblem[Seq]) -> None:
    await apply_patches(
        problem, Cuts(), [[(i, i + 1)] for i in range(len(problem.current_test_case))]
    )


def merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized: list[list[int]] = []
    for start, end in sorted(map(tuple, intervals)):
        if normalized and normalized[-1][-1] >= start:
            normalized[-1][-1] = max(normalized[-1][-1], end)
        else:
            normalized.append([start, end])
    return list(map(tuple, normalized))  # type: ignore


def with_deletions(target: Seq, deletions: list[tuple[int, int]]) -> Seq:
    result: list[Any] = []
    prev = 0
    total_deleted = 0
    for start, end in deletions:
        total_deleted += end - start
        result.extend(target[prev:start])
        prev = end
    result.extend(target[prev:])
    assert len(result) + total_deleted == len(target)
    return type(target)(result)  # type: ignore


def block_deletion(min_block: int, max_block: int) -> ReductionPass[Seq]:
    async def apply(problem: ReductionProblem[Seq]) -> None:
        n = len(problem.current_test_case)
        if n <= min_block:
            return
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


async def delete_duplicates(problem: ReductionProblem[Seq]) -> None:
    index: dict[int, list[int]] = defaultdict(list)

    for i, c in enumerate(problem.current_test_case):
        index[c].append(i)

    cuts: list[CutPatch] = []

    for ix in index.values():
        if len(ix) > 1:
            cuts.append([(i, i + 1) for i in ix])
    await apply_patches(problem, Cuts(), cuts)
