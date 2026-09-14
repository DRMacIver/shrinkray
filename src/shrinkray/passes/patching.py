from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from itertools import chain
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
    """Applies patches to `initial_test_case` (by default the problem's
    current test case at construction) and keeps the set of patches known
    to be jointly interesting, so that candidates build on each other.

    Patches are computed against a particular test case, so the applier
    must be bound to that one even if the problem has since moved on."""

    def __init__(
        self,
        patches: Patches[PatchType, TargetType],
        problem: ReductionProblem[TargetType],
        initial_test_case: TargetType | None = None,
    ):
        self.__patches = patches
        self.__problem = problem

        self.__merge_queue: list[tuple[PatchType, trio.MemorySendChannel[bool]]] = []
        self.__merge_lock = trio.Lock()

        # The patches found jointly interesting so far, which candidates
        # build on. It may run ahead of the current test case (a candidate
        # built on an older set can be adopted after a merge); the next
        # candidate built on it puts the merged patches back.
        self.__current_patch = self.__patches.empty
        self.__initial_test_case = (
            problem.current_test_case
            if initial_test_case is None
            else initial_test_case
        )

    async def __possibly_become_merge_master(self):
        try:
            self.__merge_lock.acquire_nowait()
        except trio.WouldBlock:
            return False
        merged_so_far = 0
        try:
            while self.__merge_queue:
                # More patches can come in while we're merging and it ends
                # up fiddly and unreliable if we try to merge those as part
                # of this round, so we leave them for the next one.
                base_patch = self.__current_patch
                to_merge = len(self.__merge_queue)
                merged_so_far = 0

                async def can_merge(k: int) -> bool:
                    nonlocal merged_so_far
                    try:
                        attempted_patch = self.__patches.combine(
                            base_patch,
                            *[p for p, _ in self.__merge_queue[:k]],
                        )
                        with_patch_applied = self.__patches.apply(
                            attempted_patch, self.__initial_test_case
                        )
                    except Conflict:
                        return False
                    stats = self.__problem.stats
                    calls_before = stats.calls
                    stats.merge_probes += 1
                    try:
                        if await self.__problem.is_reduction(with_patch_applied):
                            self.__current_patch = attempted_patch
                            merged_so_far = max(merged_so_far, k)
                            return True
                        return False
                    finally:
                        stats.merge_probe_calls += stats.calls - calls_before

                async def can_merge_shorter(k: int) -> bool:
                    # The full prefix has just failed; the search for a
                    # shorter one need not probe it again.
                    return k < to_merge and await can_merge(k)

                if await can_merge(to_merge):
                    merged = to_merge
                else:
                    merged = await self.__problem.work.find_large_integer(
                        can_merge_shorter
                    )

                assert merged <= to_merge

                for _, send_result in self.__merge_queue[:merged]:
                    send_result.send_nowait(True)

                if merged < to_merge:
                    self.__merge_queue[merged][1].send_nowait(False)
                    del self.__merge_queue[: merged + 1]
                else:
                    del self.__merge_queue[:to_merge]
        finally:
            # If we were cancelled mid-merge, tasks already queued would
            # otherwise wait forever for a result that no one is going to
            # send. The patches a probe of this round merged are applied;
            # the rest are not, and a later pass can still retry them.
            for i, (_, send_result) in enumerate(self.__merge_queue):
                send_result.send_nowait(i < merged_so_far)
            del self.__merge_queue[:]
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
        try:
            with_patch_applied = self.__patches.apply(
                combined_patch, self.__initial_test_case
            )
        except Conflict:
            return False
        if with_patch_applied == self.__problem.current_test_case:
            return True
        if not await self.__problem.is_interesting(with_patch_applied):
            return False
        send_merge_result, receive_merge_result = trio.open_memory_channel[bool](1)

        self.__merge_queue.append((patch, send_merge_result))

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


# If a pass tries this many candidate patches without a single successful
# reduction, give up on the rest rather than churning through them: a pass
# that cannot make any progress on this input is very unlikely to start.
# The bound scales with the input size (so large inputs get a proportionate
# try) with a floor (so small inputs are explored fully). Because
# apply_patches shuffles patches, a pass that *can* help almost always
# lands an early success and so runs to completion.
MIN_PATCH_ATTEMPTS = 250
EARLY_ABORT_SIZE_FACTOR = 3

# How many patches a worker may take from the queue between explicit
# scheduler checkpoints.
SCHEDULER_CHECKPOINT_INTERVAL = 64


async def apply_patches[PatchType, TargetType](
    problem: ReductionProblem[TargetType],
    patch_info: Patches[PatchType, TargetType],
    patches: Iterable[PatchType],
    early_abort: bool = False,
) -> None:
    """Try to apply `patches`, adopting any that reduce the test case.

    `early_abort` is for passes whose candidates are interchangeable and
    same-length (e.g. byte lowering): with it, a pass that makes no progress
    at all within a size-scaled budget gives up instead of grinding through
    every remaining candidate. It must NOT be set for deletion passes, whose
    useful patches can be sparse and would then be skipped, losing size.
    """
    # Shortcut: if applying every patch at once works, there's nothing to
    # merge. The shortcut counts as succeeding whenever the attempt changed
    # the current test case (through a view the adopted parse can differ
    # from `combined`); the individual patches were computed against the
    # old test case and must not be applied to the new one. An interesting
    # result that changed nothing (it can sort above the current test case)
    # falls through to trying the patches individually.
    patches = list(patches)
    before = problem.current_test_case
    try:
        combined = patch_info.apply(patch_info.combine(*patches), before)
        if combined == before or (
            await problem.is_interesting(combined)
            and problem.current_test_case != before
        ):
            return
    except Conflict:
        pass

    # The current test case may have moved on during the shortcut (a
    # backtrack under nondeterminism handling); the patches describe
    # `before` and are applied to it.
    applier = PatchApplier(patch_info, problem, initial_test_case=before)

    problem.work.random.shuffle(patches)
    patches.sort(key=patch_info.size, reverse=True)
    # Workers pull from this shared iterator. Advancing it never awaits, so
    # under trio's cooperative scheduling each patch goes to exactly one
    # worker without any locking.
    queue = iter(enumerate(patches))

    give_up_after = max(
        MIN_PATCH_ATTEMPTS, EARLY_ABORT_SIZE_FACTOR * problem.current_size
    )
    any_success = False

    async with trio.open_nursery() as nursery:
        for _i in range(problem.work.parallelism):

            @nursery.start_soon
            async def worker() -> None:
                nonlocal any_success
                for attempted, (i, patch) in enumerate(queue):
                    # Conflicting or redundant patches are rejected without
                    # ever awaiting, so a worker could otherwise run through
                    # a long streak of them without yielding. Check in with
                    # the scheduler periodically, which also bounds how long
                    # a cancellation goes unnoticed.
                    if attempted % SCHEDULER_CHECKPOINT_INTERVAL == 0:
                        await trio.lowlevel.checkpoint()
                    # The give-up decision is by queue position, not by a
                    # count of completed attempts: which patches fall inside
                    # the budget must not depend on parallelism or
                    # scheduling, so a pass attempts the same candidates at
                    # every parallelism level when nothing is succeeding.
                    if early_abort and not any_success and i >= give_up_after:
                        return
                    if await applier.try_apply_patch(patch):
                        any_success = True


ReplacementPatch = tuple[tuple[int, int, bytes], ...]


class Replacements(Patches[ReplacementPatch, bytes]):
    """Patches that replace byte ranges with new contents.

    A patch is a sorted tuple of (start, end, replacement) triples.
    Unlike Cuts, overlapping edits cannot be merged meaningfully, so
    combining patches with overlapping ranges raises Conflict (exact
    duplicates are fine and are deduplicated).
    """

    @property
    def empty(self) -> ReplacementPatch:
        return ()

    def combine(self, *patches: ReplacementPatch) -> ReplacementPatch:
        merged = sorted(set().union(*patches))
        for (u1, v1, _), (u2, _, _) in zip(merged, merged[1:], strict=False):
            if u2 < v1 or u1 == u2:
                raise Conflict()
        return tuple(merged)

    def apply(self, patch: ReplacementPatch, target: bytes) -> bytes:
        parts = []
        prev = 0
        for start, end, replacement in patch:
            parts.append(target[prev:start])
            parts.append(replacement)
            prev = end
        parts.append(target[prev:])
        return b"".join(parts)

    def size(self, patch: ReplacementPatch) -> int:
        return sum(
            (end - start) - len(replacement) for start, end, replacement in patch
        )


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
        normalized: CutPatch = []
        for start, end in all_cuts:
            if normalized and normalized[-1][1] >= start:
                previous_start, previous_end = normalized[-1]
                if end > previous_end:
                    normalized[-1] = (previous_start, end)
            else:
                normalized.append((start, end))
        return normalized

    def apply(self, patch: CutPatch, target: Seq) -> Seq:
        kept: list[Sequence[Any]] = []
        prev = 0
        total_deleted = 0
        for start, end in patch:
            total_deleted += end - start
            kept.append(target[prev:start])
            prev = end
        kept.append(target[prev:])
        if isinstance(target, bytes):
            # Joining slices copies each kept byte once; the generic path
            # below would iterate them one int at a time.
            result = b"".join(cast(list[bytes], kept))
        else:
            # Every sequence type Cuts is applied to (list, tuple, ...) is
            # constructible from an iterable of its elements, which the
            # Sequence bound on Seq cannot express.
            construct = cast(Callable[[Iterable[Any]], Seq], type(target))
            result = construct(chain.from_iterable(kept))
        assert len(result) + total_deleted == len(target)
        return cast(Seq, result)

    def size(self, patch: CutPatch) -> int:
        return sum(v - u for u, v in patch)
