from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import sys
from typing import Any, Generic, Iterable, Sequence, TypeVar
import trio

from shrinkray.problem import ReductionProblem

Seq = TypeVar("Seq", bound=Sequence[Any])

PatchType = TypeVar("PatchType")
TargetType = TypeVar("TargetType")


class Conflict(Exception):
    pass


class Patches(Generic[PatchType, TargetType], ABC):
    @abstractproperty
    def empty(self):
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

        self.claimed = set()
        self.remaining_starts = LazyMutableRange(len(self.patches))
        self.concurrent_merge_attempts = 0


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class Completed(Exception):
    pass


class PatchApplicationTask(Generic[PatchType, TargetType]):
    def __init__(
        self,
        shared_state: PatchApplicationSharedState[PatchType, TargetType],
        index: int,
        direction: Direction,
    ):
        if index >= len(shared_state.patches) or index < 0:
            raise ValueError(
                f"Index {index} out of range [0, {len(shared_state.patches)}]"
            )
        self.shared_state = shared_state
        self.index = index
        self.direction = direction

        self.sequential_failures = 0

    def bounce(self):
        self.sequential_failures = 0
        state = self.shared_state
        starts = state.remaining_starts
        random = state.problem.work.random
        while starts:
            i = random.randrange(0, len(starts))
            j = len(starts) - 1
            starts[i], starts[j] = starts[j], starts[i]
            start = starts[j]
            if start in state.claimed:
                starts.pop()
                continue
            self.index = start
            if self.index - 1 in state.claimed or random.randint(0, 1):
                self.direction = Direction.RIGHT
            else:
                self.direction = Direction.LEFT
            return

        raise Completed()

    async def run(self):
        try:
            while True:
                state = self.shared_state
                work = state.problem.work
                if (
                    self.index in state.claimed
                    or self.index < 0
                    or self.index >= len(state.patches)
                    or self.sequential_failures >= 100
                ):
                    self.bounce()

                state.claimed.add(self.index)
                await self.try_apply_patch(state.patches[self.index])
                match self.direction:
                    case Direction.LEFT:
                        k = await work.find_large_integer(
                            lambda k: self.try_apply_range(
                                self.index - k, self.index + 1
                            )
                        )
                        state.claimed.update(
                            range(max(0, self.index - k - 1), self.index + 1)
                        )
                        self.index -= k + 1
                    case Direction.RIGHT:
                        k = await work.find_large_integer(
                            lambda k: self.try_apply_range(self.index, self.index + k)
                        )
                        state.claimed.update(range(self.index, self.index + k + 1))
                        self.index += k + 1
        except Completed:
            return

    async def try_apply_range(self, start: int, end: int) -> bool:
        if start < 0 or end > len(self.shared_state.patches):
            return False

        try:
            patch = self.shared_state.patch_info.combine(
                self.shared_state.current_patch, *self.shared_state.patches[start:end]
            )
        except Conflict:
            await trio.lowlevel.checkpoint()
            return False

        if patch == self.shared_state.current_patch:
            await trio.lowlevel.checkpoint()
            return True

        return await self.try_apply_patch(patch)

    async def try_apply_patch(self, patch: PatchType) -> bool:
        print(self.sequential_failures, file=sys.stderr)
        state = self.shared_state
        patch_info: Patches[PatchType, TargetType] = state.patch_info

        # If we are likely to get a successful call to the interestingness test,
        # and some other task is currently trying to apply a patch that it thinks
        # will work, we pause here until it's made its attempt, so as to avoid
        # doing a lot of duplicated work where we repeatedly stomp on each other's
        # toes. If however we don't think this is likely to succeed, we try it
        # anyway so as to get some book keeping done in the background.
        if self.sequential_failures <= 10:
            iters = 0
            while state.concurrent_merge_attempts > 0:
                iters += 1
                if iters == 1:
                    await trio.lowlevel.checkpoint()
                else:
                    await trio.sleep(0.05)

        trying_to_merge = False
        try:
            while True:
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

                succeeded = await state.problem.is_interesting(attempt)

                if not succeeded:
                    self.sequential_failures += 1
                    return False

                self.sequential_failures = 0

                if state.current_patch == prev_patch:
                    state.current_patch = merged
                    return True
                if not trying_to_merge:
                    trying_to_merge = True
                    state.concurrent_merge_attempts += 1
        finally:
            if trying_to_merge:
                state.concurrent_merge_attempts -= 1
                assert state.concurrent_merge_attempts >= 0


async def apply_patches(
    problem: ReductionProblem[TargetType],
    patch_info: Patches[PatchType, TargetType],
    patches: Iterable[PatchType],
):
    patches = sorted(patches, key=patch_info.size)
    state = PatchApplicationSharedState(problem, patch_info, patches)
    n = len(state.patches)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(
            PatchApplicationTask(
                shared_state=state,
                index=0,
                direction=Direction.RIGHT,
            ).run
        )
        nursery.start_soon(
            PatchApplicationTask(
                shared_state=state,
                index=n - 1,
                direction=Direction.LEFT,
            ).run
        )

        rnd = problem.work.random
        for _ in range(max(1, problem.work.parallelism - 2)):
            nursery.start_soon(
                PatchApplicationTask(
                    shared_state=state,
                    index=rnd.randrange(0, n),
                    direction=rnd.choice((Direction.LEFT, Direction.RIGHT)),
                ).run
            )

    assert len(state.remaining_starts) == 0


class LazyMutableRange:
    def __init__(self, n):
        self.__size = n
        self.__mask = {}

    def __getitem__(self, i):
        return self.__mask.get(i, i)

    def __setitem__(self, i, v):
        self.__mask[i] = v

    def __len__(self):
        return self.__size

    def pop(self):
        i = len(self) - 1
        result = self[i]
        self.__size = i
        self.__mask.pop(i, None)
        return result


CutPatch = list[tuple[int, int]]


class Cuts(Patches[CutPatch, Seq]):
    @property
    def empty(self):
        return []

    def combine(self, *patches: CutPatch) -> CutPatch:
        all_cuts = []
        for p in patches:
            all_cuts.extend(p)
        all_cuts.sort()
        normalized = []
        for start, end in all_cuts:
            if normalized and normalized[-1][-1] >= start:
                normalized[-1][-1] = max(normalized[-1][-1], end)
            else:
                normalized.append([start, end])
        return list(map(tuple, normalized))

    def apply(self, patch: CutPatch, target: TargetType) -> TargetType:
        result = []
        prev = 0
        total_deleted = 0
        for start, end in patch:
            total_deleted += end - start
            result.extend(target[prev:start])
            prev = end
        result.extend(target[prev:])
        assert len(result) + total_deleted == len(target)
        return type(target)(result)

    def size(self, patch: CutPatch) -> int:
        return sum(v - u for u, v in patch)
