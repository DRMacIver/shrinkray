from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from random import Random
from typing import Any, Generic, Iterable, Sequence, TypeVar, cast

import trio

from shrinkray.problem import ReductionProblem

Seq = TypeVar("Seq", bound=Sequence[Any])
T = TypeVar("T")

PatchType = TypeVar("PatchType")
TargetType = TypeVar("TargetType")


class Conflict(Exception):
    pass


class Patches(Generic[PatchType, TargetType], ABC):
    @abstractproperty
    def empty(self) -> PatchType:
        ...

    @abstractmethod
    def combine(self, *patches: PatchType) -> PatchType:
        ...

    @abstractmethod
    def apply(self, patch: PatchType, target: TargetType) -> TargetType:
        ...

    @abstractmethod
    def size(self, patch: PatchType) -> int:
        ...


class PatchApplicationSharedState(Generic[PatchType, TargetType]):
    def __init__(
        self,
        problem: ReductionProblem[TargetType],
        patch_info: Patches[PatchType, TargetType],
        patches: Iterable[PatchType],
    ):
        self.base = problem.current_test_case
        self.patches = list(patches)
        self.patch_info = patch_info
        self.problem = problem

        self.current_patch = patch_info.empty

        self.claimed: set[int] = set()
        self.concurrent_merge_attempts = 0
        self.inflight_patch_size = 0

        self.pending_patches: list[PatchType] = []
        self.running_tasks = 0
        self.started_tasks = 0


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class Completed(Exception):
    pass


class PatchApplicationTask(Generic[PatchType, TargetType]):
    def __init__(
        self,
        shared_state: PatchApplicationSharedState[PatchType, TargetType],
    ):
        self.shared_state = shared_state
        self.sequential_failures = 0

    async def run(self, patch_indices: Iterable[int]) -> None:
        state = self.shared_state
        state.running_tasks += 1
        state.started_tasks += 1
        try:
            work = state.problem.work
            for start in patch_indices:
                if start in state.claimed:
                    continue
                if await self.try_apply_patch(state.patches[start]):
                    end = start + 1
                    end += await work.find_large_integer(
                        lambda k: self.try_apply_range(start, end + k)
                    )
                    start -= await work.find_large_integer(
                        lambda k: self.try_apply_range(start - k, end)
                    )
                    state.claimed.add(start - 1)
                    state.claimed.add(end)
        finally:
            state.running_tasks -= 1

    async def try_apply_range(self, start: int, end: int) -> bool:
        if start < 0 or end > len(self.shared_state.patches):
            return False

        state = self.shared_state
        all_patches = state.patches
        patches = [all_patches[i] for i in range(start, end) if i not in state.claimed]

        if not patches:
            await trio.lowlevel.checkpoint()
            return True

        try:
            patch = self.shared_state.patch_info.combine(
                *patches,
            )
        except Conflict:
            await trio.lowlevel.checkpoint()
            return False

        if await self.try_apply_patch(patch):
            state.claimed.update(range(start, end))
            return True

        return False

    async def try_apply_patch(self, patch: PatchType) -> bool:
        state = self.shared_state
        patch_info: Patches[PatchType, TargetType] = state.patch_info

        await trio.lowlevel.checkpoint()
        prev_patch = state.current_patch

        try:
            merged = patch_info.combine(prev_patch, patch)
        except Conflict:
            return False
        if merged == state.current_patch:
            return True
        attempt = patch_info.apply(
            merged,
            state.base,
        )

        prob = state.problem

        if prob.sort_key(attempt) >= prob.sort_key(prob.current_test_case):
            return False

        succeeded = await prob.is_interesting(attempt)

        if not succeeded:
            self.sequential_failures += 1
            return False

        self.sequential_failures = 0

        if state.current_patch == prev_patch:
            state.current_patch = merged
            return True
        else:
            state.pending_patches.append(patch)
            while state.pending_patches:
                await trio.sleep(0.01)

            try:
                return (
                    patch_info.combine(state.current_patch, patch)
                    == state.current_patch
                )
            except Conflict:
                return False


async def apply_patches(
    problem: ReductionProblem[TargetType],
    patch_info: Patches[PatchType, TargetType],
    patches: Iterable[PatchType],
) -> None:
    patches = sorted(patches, key=patch_info.size)
    try:
        full_patch = patch_info.combine(*patches)
    except Conflict:
        pass
    else:
        initial = problem.current_test_case
        all_applied = patch_info.apply(full_patch, initial)
        if all_applied == initial or await problem.is_interesting(all_applied):
            return

    state = PatchApplicationSharedState(problem, patch_info, patches)
    rnd = problem.work.random

    async with trio.open_nursery() as nursery:
        indices = range(len(patches))
        long_order = sorted(
            indices, key=lambda i: patch_info.size(patches[i]), reverse=True
        )

        # It's in some sense clearly correct to always try the patches in order
        # from largest to smallest. Instead we don't do that, we spend some of
        # our time trying patches in random order. This is to avoid stalls. We
        # will eventually want to try all the long patches, but it may well be
        # the case that long patches mostly don't work, and if that happens we
        # want to make sure we're trying some calls that do if there's a
        # reasonable percentage of patches that would actually work.
        for _ in range(2):
            nursery.start_soon(
                PatchApplicationTask(
                    shared_state=state,
                ).run,
                lazy_shuffle(indices, rnd),
            )

        for _ in range(
            min(problem.work.parallelism, max(2, problem.work.parallelism - 2))
        ):
            nursery.start_soon(
                PatchApplicationTask(
                    shared_state=state,
                ).run,
                long_order,
            )

        while state.started_tasks == 0:
            await trio.sleep(0.01)

        @nursery.start_soon
        async def _() -> None:
            while state.running_tasks > 0 or state.pending_patches:
                if not state.pending_patches:
                    await trio.sleep(0.01)
                    continue

                async def can_apply(k: int) -> bool:
                    if k > len(state.pending_patches):
                        return False

                    patches_to_apply = state.pending_patches[:k]

                    while True:
                        prev_patch = state.current_patch
                        try:
                            combined = patch_info.combine(prev_patch, *patches_to_apply)
                        except Conflict:
                            return False
                        attempt = patch_info.apply(combined, state.base)
                        if await state.problem.is_interesting(attempt):
                            if state.current_patch == prev_patch:
                                state.current_patch = combined
                                return True
                            else:
                                await trio.lowlevel.checkpoint()
                        else:
                            return False

                k = await problem.work.find_large_integer(can_apply)
                del state.pending_patches[: k + 1]


class LazyMutableRange:
    def __init__(self, n: int):
        self.__size = n
        self.__mask: dict[int, int] = {}

    def __getitem__(self, i: int) -> int:
        return self.__mask.get(i, i)

    def __setitem__(self, i: int, v: int) -> None:
        self.__mask[i] = v

    def __len__(self) -> int:
        return self.__size

    def pop(self) -> int:
        i = len(self) - 1
        result = self[i]
        self.__size = i
        self.__mask.pop(i, None)
        return result


def lazy_shuffle(seq: Sequence[T], rnd: Random) -> Iterable[T]:
    indices = LazyMutableRange(len(seq))
    while indices:
        j = len(indices) - 1
        i = rnd.randrange(0, len(indices))
        indices[i], indices[j] = indices[j], indices[i]
        yield seq[indices.pop()]


CutPatch = list[tuple[int, int]]


class Cuts(Patches[CutPatch, Seq]):
    @property
    def empty(self) -> CutPatch:
        return []

    def combine(self, *patches: CutPatch) -> CutPatch:
        all_cuts: CutPatch = []
        for p in patches:
            all_cuts.extend(p)
        all_cuts.sort()
        normalized: list[list[int]] = []
        for start, end in all_cuts:
            if normalized and normalized[-1][-1] >= start:
                normalized[-1][-1] = max(normalized[-1][-1], end)
            else:
                normalized.append([start, end])
        return [cast(tuple[int, int], tuple(x)) for x in normalized]

    def apply(self, patch: CutPatch, target: Seq) -> Seq:
        result: list[Any] = []
        prev = 0
        total_deleted = 0
        for start, end in patch:
            total_deleted += end - start
            result.extend(target[prev:start])
            prev = end
        result.extend(target[prev:])
        assert len(result) + total_deleted == len(target)
        return type(target)(result)  # type: ignore

    def size(self, patch: CutPatch) -> int:
        return sum(v - u for u, v in patch)
