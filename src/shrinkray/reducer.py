from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

import attrs
import trio
from attrs import define

from shrinkray.passes.bytes import (
    Split,
    Tokenize,
    debracket,
    delete_byte_spans,
    hollow,
    lexeme_based_deletions,
    lift_braces,
    line_sorter,
    lower_bytes,
    lower_individual_bytes,
    remove_indents,
    remove_whitespace,
    replace_space_with_newlines,
    short_deletions,
    standard_substitutions,
)
from shrinkray.passes.clangdelta import (
    C_FILE_EXTENSIONS,
    ClangDelta,
    clang_delta_pumps,
)
from shrinkray.passes.definitions import (
    Format,
    ReductionPass,
    ReductionPump,
    compose,
)
from shrinkray.passes.genericlanguages import (
    combine_expressions,
    cut_comment_like_things,
    merge_adjacent_strings,
    normalize_identifiers,
    reduce_integer_literals,
    replace_falsey_with_zero,
    simplify_brackets,
)
from shrinkray.passes.json import JSON, JSON_PASSES
from shrinkray.passes.patching import PatchApplier, Patches
from shrinkray.passes.python import PYTHON_PASSES, is_python
from shrinkray.passes.sat import SAT_PASSES, DimacsCNF
from shrinkray.passes.sequences import block_deletion, delete_duplicates
from shrinkray.problem import ReductionProblem, ReductionStats, shortlex


@define
class Reducer[T](ABC):
    target: ReductionProblem[T]

    # Optional pass statistics tracking (implemented by ShrinkRay)
    pass_stats: "PassStatsTracker | None" = attrs.field(default=None, init=False)
    # Optional current pass tracking (implemented by ShrinkRay)
    current_reduction_pass: "ReductionPass[T] | None" = attrs.field(default=None, init=False)

    @contextmanager
    def backtrack(self, restart: T) -> Generator[None, None, None]:
        current = self.target
        try:
            self.target = self.target.backtrack(restart)
            yield
        finally:
            self.target = current

    @abstractmethod
    async def run(self) -> None: ...

    @property
    def status(self) -> str:
        return ""

    @property
    def disabled_passes(self) -> set[str]:
        """Set of disabled pass names. Override in subclasses for pass control."""
        return set()

    def disable_pass(self, pass_name: str) -> None:  # noqa: B027
        """Disable a pass by name. Override in subclasses for pass control."""

    def enable_pass(self, pass_name: str) -> None:  # noqa: B027
        """Enable a pass by name. Override in subclasses for pass control."""

    def skip_current_pass(self) -> None:  # noqa: B027
        """Skip the currently running pass. Override in subclasses for pass control."""


@define
class BasicReducer[T](Reducer[T]):
    reduction_passes: Iterable[ReductionPass[T]]
    pumps: Iterable[ReductionPump[T]] = ()
    _status: str = "Starting up"

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value

    async def run_pass(self, rp: ReductionPass[T]) -> None:
        await rp(self.target)

    async def run(self) -> None:
        await self.target.setup()

        while True:
            prev = self.target.current_test_case
            for rp in self.reduction_passes:
                self.status = f"Running reduction pass {rp.__name__}"
                await self.run_pass(rp)
            for pump in self.pumps:
                self.status = f"Pumping with {pump.__name__}"
                pumped = await pump(self.target)
                if pumped != self.target.current_test_case:
                    with self.backtrack(pumped):
                        for rp in self.reduction_passes:
                            self.status = f"Running reduction pass {rp.__name__} under pump {pump.__name__}"
                            await self.run_pass(rp)
            if prev == self.target.current_test_case:
                return


class RestartPass(Exception):
    pass


@define
class PassStats:
    """Statistics for a single reduction pass."""

    pass_name: str
    bytes_deleted: int = 0
    non_size_reductions: int = 0
    call_count: int = 0
    test_evaluations: int = 0
    successful_reductions: int = 0
    order: int = 0  # Track insertion order for stable display

    @property
    def success_rate(self) -> float:
        """Percentage of test evaluations that led to a reduction."""
        if self.test_evaluations == 0:
            return 0.0
        return (self.successful_reductions / self.test_evaluations) * 100.0


@define
class PassStatsTracker:
    """Tracks statistics for all reduction passes."""

    _stats: dict[str, PassStats] = attrs.Factory(dict)
    _next_order: int = 0

    def get_or_create(self, pass_name: str) -> PassStats:
        if pass_name not in self._stats:
            self._stats[pass_name] = PassStats(
                pass_name=pass_name, order=self._next_order
            )
            self._next_order += 1
        return self._stats[pass_name]

    def get_stats_in_order(self) -> list[PassStats]:
        """Get stats in the order passes were first run."""
        return sorted(self._stats.values(), key=lambda s: s.order)


class SkipPass(Exception):
    """Raised to skip the current pass."""

    pass


@define
class ShrinkRay(Reducer[bytes]):
    clang_delta: ClangDelta | None = None

    current_pump: ReductionPump[bytes] | None = None

    unlocked_ok_passes: bool = False
    pass_stats: PassStatsTracker | None = attrs.Factory(PassStatsTracker)

    # Pass control: disabled passes and skip functionality
    disabled_passes: set[str] = attrs.Factory(set)
    _skip_requested: bool = attrs.field(default=False, init=False)
    _current_pass_scope: "trio.CancelScope | None" = attrs.field(default=None, init=False)
    _passes_were_skipped: bool = attrs.field(default=False, init=False)

    def disable_pass(self, pass_name: str) -> None:
        """Disable a pass by name. If it's currently running, skip it."""
        self.disabled_passes.add(pass_name)
        # If this pass is currently running, skip it
        if (
            self.current_reduction_pass is not None
            and self.current_reduction_pass.__name__ == pass_name
        ):
            self.skip_current_pass()

    def enable_pass(self, pass_name: str) -> None:
        """Enable a previously disabled pass."""
        self.disabled_passes.discard(pass_name)

    def is_pass_disabled(self, pass_name: str) -> bool:
        """Check if a pass is disabled."""
        return pass_name in self.disabled_passes

    def skip_current_pass(self) -> None:
        """Request to skip the currently running pass."""
        self._skip_requested = True
        if self._current_pass_scope is not None:
            self._current_pass_scope.cancel()

    initial_cuts: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            cut_comment_like_things,
            hollow,
            compose(Split(b"\n"), delete_duplicates),
            compose(Split(b"\n"), block_deletion(10, 100)),
            lift_braces,
            remove_indents,
            remove_whitespace,
        ]
    )

    great_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            compose(Split(b"\n"), delete_duplicates),
            compose(Split(b"\n"), block_deletion(1, 10)),
            compose(Split(b";"), block_deletion(1, 10)),
            remove_indents,
            hollow,
            lift_braces,
            debracket,
        ]
    )

    ok_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            delete_byte_spans,
            compose(Split(b"\n"), block_deletion(11, 20)),
            remove_indents,
            remove_whitespace,
            compose(Tokenize(), block_deletion(1, 20)),
            reduce_integer_literals,
            replace_falsey_with_zero,
            combine_expressions,
            merge_adjacent_strings,
            lexeme_based_deletions,
            short_deletions,
            normalize_identifiers,
            line_sorter,
        ]
    )

    last_ditch_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            compose(Split(b"\n"), block_deletion(21, 100)),
            replace_space_with_newlines,
            delete_byte_spans,
            lower_bytes,
            lower_individual_bytes,
            simplify_brackets,
            standard_substitutions,
            # This is in last ditch because it's probably not useful
            # to run it more than once.
            cut_comment_like_things,
        ]
    )

    def __attrs_post_init__(self) -> None:
        if is_python(self.target.current_test_case):
            self.great_passes.extend(PYTHON_PASSES)
            self.initial_cuts.extend(PYTHON_PASSES)
        self.register_format_specific_pass(JSON, JSON_PASSES)
        self.register_format_specific_pass(
            DimacsCNF,
            SAT_PASSES,
        )

    def register_format_specific_pass[T](
        self, format: Format[bytes, T], passes: Iterable[ReductionPass[T]]
    ):
        if format.is_valid(self.target.current_test_case):
            composed = [compose(format, p) for p in passes]
            self.great_passes.extend(composed)
            self.initial_cuts.extend(composed)

    @property
    def pumps(self) -> Iterable[ReductionPump[bytes]]:
        if self.clang_delta is None:
            return ()
        else:
            return clang_delta_pumps(self.clang_delta)

    @property
    def status(self) -> str:
        if self.current_pump is None:
            if self.current_reduction_pass is not None:
                return f"Running reduction pass {self.current_reduction_pass.__name__}"
            else:
                return "Selecting reduction pass"
        else:
            if self.current_reduction_pass is not None:
                return f"Running reduction pass {self.current_reduction_pass.__name__} under pump {self.current_pump.__name__}"
            else:
                return f"Running reduction pump {self.current_pump.__name__}"

    async def run_pass(self, rp: ReductionPass[bytes]) -> None:
        pass_name = rp.__name__

        # Skip if pass is disabled
        if self.is_pass_disabled(pass_name):
            return

        try:
            assert self.current_reduction_pass is None
            self.current_reduction_pass = rp
            self._skip_requested = False

            # Get or create stats entry for this pass
            assert self.pass_stats is not None  # Always set by Factory
            stats_entry = self.pass_stats.get_or_create(pass_name)
            stats_entry.call_count += 1

            # Set current pass stats on the problem for real-time updates
            self.target.current_pass_stats = stats_entry

            # Run the pass with a cancel scope that can be externally cancelled
            with trio.CancelScope() as scope:
                self._current_pass_scope = scope
                await rp(self.target)

            # If the pass was cancelled/skipped, mark that passes were skipped
            if scope.cancelled_caught:
                self._passes_were_skipped = True

        finally:
            self.current_reduction_pass = None
            self.target.current_pass_stats = None
            self._current_pass_scope = None
            self._skip_requested = False

    async def pump(self, rp: ReductionPump[bytes]) -> None:
        try:
            assert self.current_pump is None
            self.current_pump = rp
            pumped = await rp(self.target)
            current = self.target.current_test_case
            if pumped == current:
                return
            with self.backtrack(pumped):
                for f in [
                    self.run_great_passes,
                    self.run_ok_passes,
                    self.run_last_ditch_passes,
                ]:
                    await f()
                    if self.target.sort_key(
                        self.target.current_test_case
                    ) < self.target.sort_key(current):
                        break

        finally:
            self.current_pump = None

    async def run_great_passes(self) -> None:
        current = self.great_passes
        while True:
            prev = self.target.current_test_case
            successful = []
            for rp in current:
                size = self.target.current_size
                await self.run_pass(rp)
                if self.target.current_size < size:
                    successful.append(rp)
            if self.target.current_test_case == prev:
                if len(current) == len(self.great_passes):
                    break
                else:
                    current = self.great_passes
            elif not successful:
                current = self.great_passes
            else:
                current = successful

    async def run_ok_passes(self) -> None:
        for rp in self.ok_passes:
            await self.run_pass(rp)

    async def run_last_ditch_passes(self) -> None:
        for rp in self.last_ditch_passes:
            await self.run_pass(rp)

    async def run_some_passes(self) -> None:
        prev = self.target.current_test_case
        await self.run_great_passes()
        if prev != self.target.current_test_case and not self.unlocked_ok_passes:
            return
        self.unlocked_ok_passes = True
        await self.run_ok_passes()
        if prev != self.target.current_test_case:
            return
        await self.run_last_ditch_passes()

    async def initial_cut(self) -> None:
        while True:
            prev = self.target.current_size
            for rp in self.initial_cuts:
                async with trio.open_nursery() as nursery:

                    @nursery.start_soon
                    async def _() -> None:
                        """
                        Watcher task that cancels the current reduction pass as
                        soon as it stops looking like a good idea to keep running
                        it. Current criteria:

                        1. If it's been more than 5s since the last successful reduction.
                        2. If the reduction rate of the task has dropped under 50% of its
                           best so far.
                        """
                        iters = 0
                        initial_size = self.target.current_size
                        best_reduction_rate: float | None = None

                        while True:
                            iters += 1
                            deleted = initial_size - self.target.current_size

                            current = self.target.current_test_case
                            await trio.sleep(5)
                            rate = deleted / iters

                            if (
                                best_reduction_rate is None
                                or rate > best_reduction_rate
                            ):
                                best_reduction_rate = rate

                            assert best_reduction_rate is not None

                            if (
                                rate < 0.5 * best_reduction_rate
                                or current == self.target.current_test_case
                            ):
                                nursery.cancel_scope.cancel()
                                break

                    await self.run_pass(rp)
                    nursery.cancel_scope.cancel()
            if self.target.current_size >= 0.99 * prev:
                return

    async def run(self) -> None:
        await self.target.setup()

        if await self.target.is_interesting(b""):
            return

        prev = 0
        for c in [0, 1, ord(b"\n"), ord(b"0"), ord(b"z"), 255]:
            if await self.target.is_interesting(bytes([c])):
                for i in range(c):
                    if await self.target.is_interesting(bytes([i])):
                        break
                return

        await self.initial_cut()

        while True:
            # Reset skip tracking for this iteration
            self._passes_were_skipped = False

            prev = self.target.current_test_case
            await self.run_some_passes()
            if self.target.current_test_case != prev:
                continue
            for pump in self.pumps:
                await self.pump(pump)
            if self.target.current_test_case == prev:
                # Only terminate if no passes were skipped
                # If passes were skipped, we need another full run to be sure
                if not self._passes_were_skipped:
                    break


class UpdateKeys(Patches[dict[str, bytes], dict[str, bytes]]):
    @property
    def empty(self) -> dict[str, bytes]:
        return {}

    def combine(self, *patches: dict[str, bytes]) -> dict[str, bytes]:
        result = {}
        for p in patches:
            for k, v in p.items():
                result[k] = v
        return result

    def apply(
        self, patch: dict[str, bytes], target: dict[str, bytes]
    ) -> dict[str, bytes]:
        result = target.copy()
        result.update(patch)
        return result

    def size(self, patch: dict[str, bytes]) -> int:
        return len(patch)


class KeyProblem(ReductionProblem[bytes]):
    def __init__(
        self,
        base_problem: ReductionProblem[dict[str, bytes]],
        applier: PatchApplier[dict[str, bytes], dict[str, bytes]],
        key: str,
    ):
        super().__init__(work=base_problem.work)
        self.base_problem = base_problem
        self.applier = applier
        self.key = key

    @property
    def current_test_case(self) -> bytes:
        return self.base_problem.current_test_case[self.key]

    @property
    def stats(self) -> ReductionStats:
        return self.base_problem.stats

    async def is_interesting(self, test_case: bytes) -> bool:
        return await self.applier.try_apply_patch({self.key: test_case})

    def size(self, test_case: bytes) -> int:
        return len(test_case)

    def sort_key(self, test_case: bytes) -> Any:
        return shortlex(test_case)

    def display(self, value: bytes) -> str:
        return repr(value)


@define
class DirectoryShrinkRay(Reducer[dict[str, bytes]]):
    clang_delta: ClangDelta | None = None

    async def run(self):
        prev = None
        while prev != self.target.current_test_case:
            prev = self.target.current_test_case
            await self.delete_keys()
            await self.shrink_values()

    async def delete_keys(self):
        target = self.target.current_test_case
        keys = list(target.keys())
        keys.sort(key=lambda k: (shortlex(target[k]), shortlex(k)), reverse=True)
        for k in keys:
            attempt = self.target.current_test_case.copy()
            del attempt[k]
            await self.target.is_interesting(attempt)

    async def shrink_values(self):
        async with trio.open_nursery() as nursery:
            applier = PatchApplier(patches=UpdateKeys(), problem=self.target)
            for k in self.target.current_test_case.keys():
                key_problem = KeyProblem(
                    base_problem=self.target,
                    applier=applier,
                    key=k,
                )
                if self.clang_delta is not None and any(
                    k.endswith(s) for s in C_FILE_EXTENSIONS
                ):
                    clang_delta = self.clang_delta
                else:
                    clang_delta = None

                key_shrinkray = ShrinkRay(
                    clang_delta=clang_delta,
                    target=key_problem,
                )
                nursery.start_soon(key_shrinkray.run)
