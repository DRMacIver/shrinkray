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


class PatchApplier(Generic[PatchType, TargetType], ABC):
    def __init__(
        self,
        patches: Patches[PatchType, TargetType],
        problem: ReductionProblem[TargetType],
    ):
        self.__patches = patches
        self.__problem = problem

        self.__is_merging = False
        self.__merge_queue = []
        self.__merge_lock = trio.Lock()

        self.__current_patch = self.__patches.empty
        self.__initial_test_case = problem.current_test_case

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
        if not await self.__problem.is_interesting(with_patch_applied):
            return False
        send_merge_result, receive_merge_result = trio.open_memory_channel(1)
        self.__merge_queue.append((patch, send_merge_result))

        async with self.__merge_lock:
            if (
                self.__current_patch == initial_patch
                and len(self.__merge_queue) == 1
                and self.__merge_queue[0][0] == patch
            ):
                self.__current_patch = combined_patch
                self.__merge_queue.clear()
                return True

            while self.__merge_queue:
                base_patch = self.__current_patch
                to_merge = len(self.__merge_queue)

                async def can_merge(k):
                    if k > to_merge:
                        return False
                    try:
                        attempted_patch = self.__patches.combine(
                            base_patch, *[p for p, _ in self.__merge_queue[:k]]
                        )
                    except Conflict:
                        return False
                    if attempted_patch == base_patch:
                        return True
                    with_patch_applied = self.__patches.apply(
                        combined_patch, self.__initial_test_case
                    )
                    if await self.__problem.is_interesting(with_patch_applied):
                        self.__current_patch = attempted_patch
                        return True
                    else:
                        return False

                if await can_merge(to_merge):
                    merged = to_merge
                else:
                    merged = await self.__problem.work.find_large_integer(can_merge)

                for _, send_result in self.__merge_queue[:merged]:
                    send_result.send_nowait(True)

                assert merged <= to_merge
                if merged < to_merge:
                    self.__merge_queue[merged][1].send_nowait(False)
                    del self.__merge_queue[: merged + 1]
                else:
                    del self.__merge_queue[:to_merge]

        # This should always have been populated during the previous merge,
        # either by us or someone else merging.
        return receive_merge_result.receive_nowait()


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class Completed(Exception):
    pass


async def apply_patches(
    problem: ReductionProblem[TargetType],
    patch_info: Patches[PatchType, TargetType],
    patches: Iterable[PatchType],
) -> None:
    applier = PatchApplier(patch_info, problem)

    send_patches, receive_patches = trio.open_memory_channel(float("inf"))

    patches = list(patches)
    problem.work.random.shuffle(patches)
    for patch in patches:
        send_patches.send_nowait(patch)
    send_patches.close()

    async with trio.open_nursery() as nursery:
        for _ in range(problem.work.parallelism):

            @nursery.start_soon
            async def _():
                while True:
                    try:
                        patch = await receive_patches.receive()
                    except trio.EndOfChannel:
                        break
                    await applier.try_apply_patch(patch)


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
