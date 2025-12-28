from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar, cast

import trio

from shrinkray.problem import ReductionProblem


Seq = TypeVar("Seq", bound=Sequence[Any])


class Conflict(Exception):
    pass


class Patches[PatchType, TargetType](ABC):
    @property
    @abstractmethod
    def empty(self) -> PatchType: ...

    @abstractmethod
    def combine(self, *patches: PatchType) -> PatchType: ...

    @abstractmethod
    def apply(self, patch: PatchType, target: TargetType) -> TargetType: ...

    @abstractmethod
    def size(self, patch: PatchType) -> int: ...


class SetPatches[T, TargetType](Patches[frozenset[T], TargetType]):
    def __init__(self, apply: Callable[[frozenset[T], TargetType], TargetType]):
        self.__apply = apply

    @property
    def empty(self):
        return frozenset()

    def combine(self, *patches: frozenset[T]) -> frozenset[T]:
        result = set()
        for p in patches:
            result.update(p)
        return frozenset(result)

    def apply(self, patch: frozenset[T], target: TargetType) -> TargetType:
        return self.__apply(patch, target)

    def size(self, patch: frozenset[T]) -> int:
        return len(patch)


class PatchApplier[PatchType, TargetType]:
    def __init__(
        self,
        patches: Patches[PatchType, TargetType],
        problem: ReductionProblem[TargetType],
    ):
        self.__patches = patches
        self.__problem = problem

        self.__tick = 0
        self.__merge_queue = []
        self.__merge_lock = trio.Lock()

        self.__current_patch = self.__patches.empty
        self.__initial_test_case = problem.current_test_case

    async def __possibly_become_merge_master(self):
        try:
            self.__merge_lock.acquire_nowait()
        except trio.WouldBlock:
            return False
        try:
            while self.__merge_queue:
                base_patch = self.__current_patch
                to_merge = len(self.__merge_queue)

                async def can_merge(k):
                    # find_large_integer doubles each time, and
                    # if we call it then we know that can_merge(to_merge)
                    # is False, so we should never hit this.
                    assert k <= 2 * to_merge
                    try:
                        attempted_patch = self.__patches.combine(
                            base_patch,
                            *[p for _, p, _ in self.__merge_queue[:k]],
                        )
                    except Conflict:
                        return False
                    with_patch_applied = self.__patches.apply(
                        attempted_patch, self.__initial_test_case
                    )
                    if await self.__problem.is_reduction(with_patch_applied):
                        self.__current_patch = attempted_patch
                        return True
                    else:
                        return False

                if await can_merge(to_merge):
                    merged = to_merge
                else:
                    merged = await self.__problem.work.find_large_integer(can_merge)

                for _, _, send_result in self.__merge_queue[:merged]:
                    send_result.send_nowait(True)

                assert merged <= to_merge
                if merged < to_merge:
                    self.__merge_queue[merged][-1].send_nowait(False)
                    del self.__merge_queue[: merged + 1]
                else:
                    del self.__merge_queue[:to_merge]
        finally:
            self.__merge_lock.release()

        return True

    async def try_apply_patch(self, patch: PatchType) -> bool:
        initial_patch = self.__current_patch
        try:
            combined_patch = self.__patches.combine(initial_patch, patch)
        except Conflict:
            return False
        if combined_patch == self.__current_patch:
            return True
        with_patch_applied = self.__patches.apply(
            combined_patch, self.__initial_test_case
        )
        if with_patch_applied == self.__problem.current_test_case:
            return True
        if not await self.__problem.is_interesting(with_patch_applied):
            return False
        send_merge_result, receive_merge_result = trio.open_memory_channel(1)

        sort_key = (self.__tick, self.__problem.sort_key(with_patch_applied))
        self.__tick += 1

        self.__merge_queue.append((sort_key, patch, send_merge_result))

        # If nobody else is merging the queue, that's our job now. This will
        # run until the queue is fully cleared, including the job we just
        # put on it.
        if await self.__possibly_become_merge_master():
            # This should always have been populated during the merge step we just
            # performed, so we use a nowait here to ensure it doesn't hang on a
            # bug.
            return receive_merge_result.receive_nowait()
        else:
            # Wait to clear to merge queue.
            return await receive_merge_result.receive()


async def apply_patches[PatchType, TargetType](
    problem: ReductionProblem[TargetType],
    patch_info: Patches[PatchType, TargetType],
    patches: Iterable[PatchType],
) -> None:
    try:
        if await problem.is_interesting(
            patch_info.apply(patch_info.combine(*patches), problem.current_test_case)
        ):
            return
    except Conflict:
        pass

    applier = PatchApplier(patch_info, problem)

    send_patches, receive_patches = trio.open_memory_channel(float("inf"))

    patches = list(patches)
    problem.work.random.shuffle(patches)
    patches.sort(key=patch_info.size, reverse=True)
    for patch in patches:
        send_patches.send_nowait(patch)
    send_patches.close()

    async with trio.open_nursery() as nursery:
        for _i in range(problem.work.parallelism):

            @nursery.start_soon
            async def worker() -> None:
                while True:
                    try:
                        patch = await receive_patches.receive()
                    except trio.EndOfChannel:
                        break
                    await applier.try_apply_patch(patch)


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
