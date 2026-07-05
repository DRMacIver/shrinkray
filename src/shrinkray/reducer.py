import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

import attrs
import trio
from attrs import define

from shrinkray.history import sanitize_for_filename
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
from shrinkray.passes.cpp import (
    C_FILE_EXTENSIONS,
    CPP_PASSES,
    CPP_PUMPS,
)
from shrinkray.passes.definitions import (
    Format,
    ReductionPass,
    ReductionPump,
    compose,
)
from shrinkray.passes.external import ExternalReducerPass, external_reducer
from shrinkray.passes.genericlanguages import (
    combine_expressions,
    cut_comment_like_things,
    merge_adjacent_strings,
    normalize_identifiers,
    reduce_integer_literals,
    replace_falsey_with_zero,
    replace_identifiers_with_zero,
    simplify_brackets,
)
from shrinkray.passes.json import JSON, JSON_PASSES
from shrinkray.passes.patching import PatchApplier, Patches
from shrinkray.passes.python import is_python, python_reducer_command
from shrinkray.passes.sat import SAT_PASSES, DimacsCNF
from shrinkray.passes.sequences import block_deletion, delete_duplicates
from shrinkray.passes.treesitter import language_for_filename, treesitter_passes
from shrinkray.problem import (
    ReductionProblem,
    ReductionStats,
    shortlex,
    sort_key_for_initial,
)


@define
class Reducer[T](ABC):
    target: ReductionProblem[T]

    # Optional pass statistics tracking (implemented by ShrinkRay)
    pass_stats: "PassStatsTracker | None" = attrs.field(default=None, init=False)
    # Optional current pass tracking (implemented by ShrinkRay)
    current_reduction_pass: "ReductionPass[T] | None" = attrs.field(
        default=None, init=False
    )

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
class PassStats:
    """Statistics for a single reduction pass."""

    pass_name: str
    bytes_deleted: int = 0
    run_count: int = 0
    test_evaluations: int = 0
    successful_reductions: int = 0

    @property
    def success_rate(self) -> float:
        """Percentage of test evaluations that led to a reduction."""
        if self.test_evaluations == 0:
            return 0.0
        return (self.successful_reductions / self.test_evaluations) * 100.0


@define
class PassStatsTracker:
    """Tracks statistics for all reduction passes.

    Python 3.7+ dicts maintain insertion order, so stats are returned
    in the order passes were first run.
    """

    _stats: dict[str, PassStats] = attrs.Factory(dict)

    def get_or_create(self, pass_name: str) -> PassStats:
        if pass_name not in self._stats:
            self._stats[pass_name] = PassStats(pass_name=pass_name)
        return self._stats[pass_name]

    def get_stats_in_order(self) -> list[PassStats]:
        """Get stats in the order passes were first run."""
        return list(self._stats.values())


@define
class ShrinkRay(Reducer[bytes]):
    # Enables the C/C++ specific passes and pumps. Set when the test
    # case's file name suggests it's C or C++.
    enable_cpp_passes: bool = False

    # Enables grammar-aware passes for the named tree-sitter language.
    # Set from the test case's file extension.
    treesitter_language: str | None = None

    # Commands (argv lists) for user-specified external reducers, run as
    # reduction passes. See shrinkray.passes.external.
    external_reducers: list[list[str]] = attrs.Factory(list)

    # Whether to run the built-in libcst Python reducer (as an external
    # reducer) when the test case looks like Python.
    python_reducer: bool = True

    # Directory to write external reducer stderr logs to, or None to discard.
    reducer_log_dir: str | None = None

    # The external reducer passes built for this reducer, kept so their
    # subprocesses can be torn down when the run finishes.
    _external_reducer_passes: list[ExternalReducerPass] = attrs.field(
        factory=list, init=False
    )

    current_pump: ReductionPump[bytes] | None = None

    unlocked_ok_passes: bool = False
    pass_stats: PassStatsTracker | None = attrs.Factory(PassStatsTracker)

    # Pass control: disabled passes and skip functionality
    disabled_passes: set[str] = attrs.Factory(set)
    _skip_requested: bool = attrs.field(default=False, init=False)
    _current_pass_scope: "trio.CancelScope | None" = attrs.field(
        default=None, init=False
    )
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
            reduce_integer_literals,
            replace_falsey_with_zero,
            replace_identifiers_with_zero,
            combine_expressions,
            merge_adjacent_strings,
            lexeme_based_deletions,
            normalize_identifiers,
            line_sorter,
        ]
    )

    last_ditch_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            # Fine-grained token block deletion generates a very large
            # candidate set with a low success rate; benchmarking shows it
            # is much cheaper run late, on already-reduced test cases.
            compose(Tokenize(), block_deletion(1, 20)),
            compose(Split(b"\n"), block_deletion(21, 100)),
            replace_space_with_newlines,
            delete_byte_spans,
            simplify_brackets,
            standard_substitutions,
            # This is in last ditch because it's probably not useful
            # to run it more than once.
            cut_comment_like_things,
        ]
    )

    # Expensive passes with very low success rates, mostly valuable for
    # normalising the final result rather than shrinking it. They only run
    # once everything else has converged: benchmarking showed they consumed
    # a large fraction of all interestingness calls when interleaved with
    # the productive passes, for almost no reductions.
    polish_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            short_deletions,
            lower_bytes,
            lower_individual_bytes,
        ]
    )

    # Number of consecutive failed calls after which a pass on probation
    # (one whose previous completed run made no progress) is abandoned for
    # now. Abandoned passes are recorded in incomplete_passes and re-run
    # without a budget before the reducer finishes, so this only affects
    # when their work happens, not whether it does.
    probation_budget: int = 25

    # Passes whose previous completed run made no progress. Such a pass
    # gets only probation_budget consecutive failures on its next run.
    pass_probation: dict[str, bool] = attrs.Factory(dict)

    # Passes whose most recent run was cut short by the probation budget,
    # keyed by name. They must be re-run to completion before finishing.
    incomplete_passes: dict[str, ReductionPass[bytes]] = attrs.Factory(dict)

    # For each pass that ran to completion without making progress, the
    # test case it ran against. Re-running a pass on an identical test
    # case is a deterministic no-op (randomness only affects the order in
    # which candidates are tried, not which candidates are generated), so
    # such runs are skipped entirely.
    pass_fingerprints: dict[str, bytes] = attrs.Factory(dict)

    def _log_file_for(self, name: str) -> str | None:
        """Path for an external reducer's stderr log, or None to discard."""
        if self.reducer_log_dir is None:
            return None
        os.makedirs(self.reducer_log_dir, exist_ok=True)
        return os.path.join(self.reducer_log_dir, f"reducer-{name}.log")

    def build_external_reducer_passes(self) -> list[ExternalReducerPass]:
        """Build the external reducer passes for the current test case.

        This includes the built-in Python reducer (when enabled and the test
        case looks like Python) followed by any user-specified reducers.
        """
        passes: list[ExternalReducerPass] = []
        if self.python_reducer and is_python(self.target.current_test_case):
            passes.append(
                external_reducer(
                    python_reducer_command(),
                    name="python",
                    log_file=self._log_file_for("python"),
                )
            )
        for i, command in enumerate(self.external_reducers):
            name = f"reduce-with-{i}"
            passes.append(
                external_reducer(
                    command,
                    name=name,
                    log_file=self._log_file_for(name),
                )
            )
        return passes

    async def close_external_reducers(self) -> None:
        """Tear down any external reducer subprocesses started during the run."""
        with trio.CancelScope(shield=True):
            for reducer_pass in self._external_reducer_passes:
                await reducer_pass.aclose()

    def __attrs_post_init__(self) -> None:
        external_passes = self.build_external_reducer_passes()
        self._external_reducer_passes = external_passes
        self.great_passes.extend(external_passes)
        self.initial_cuts.extend(external_passes)
        if self.enable_cpp_passes:
            self.great_passes.extend(CPP_PASSES)
            self.initial_cuts.extend(CPP_PASSES)
        if self.treesitter_language is not None:
            passes = treesitter_passes(self.treesitter_language)
            self.great_passes.extend(passes)
            self.initial_cuts.extend(passes)
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
        if self.enable_cpp_passes:
            return CPP_PUMPS
        else:
            return ()

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

    async def run_pass(
        self, rp: ReductionPass[bytes], *, budgeted: bool = True
    ) -> None:
        pass_name = rp.__name__

        # Skip if pass is disabled
        if self.is_pass_disabled(pass_name):
            return

        problem = self.target

        # A pass that ran to completion without making progress cannot make
        # progress on an identical test case, so skip it for free.
        if self.pass_fingerprints.get(pass_name) == problem.current_test_case:
            return

        use_budget = budgeted and self.pass_probation.get(pass_name, False)
        scope = trio.CancelScope()
        last_seen = problem.current_test_case
        consecutive_failures = 0
        budget_exhausted = False
        made_progress = False

        def monitor() -> None:
            nonlocal last_seen, consecutive_failures, budget_exhausted
            nonlocal made_progress
            current = problem.current_test_case
            if current is not last_seen:
                last_seen = current
                made_progress = True
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if use_budget and consecutive_failures > self.probation_budget:
                    budget_exhausted = True
                    scope.cancel()

        try:
            assert self.current_reduction_pass is None
            self.current_reduction_pass = rp
            self._skip_requested = False

            # Get or create stats entry for this pass
            assert self.pass_stats is not None  # Always set by Factory
            stats_entry = self.pass_stats.get_or_create(pass_name)
            stats_entry.run_count += 1

            # Set current pass stats on the problem for real-time updates
            problem.current_pass_stats = stats_entry
            problem.pass_call_monitor = monitor

            # Run the pass with a cancel scope that can be externally cancelled
            with scope:
                self._current_pass_scope = scope
                await rp(self.target)

            if budget_exhausted:
                # The pass was abandoned by its probation budget. It still
                # has untried candidates, so it must be re-run before the
                # reduction can finish.
                self.incomplete_passes[pass_name] = rp
                if made_progress:
                    self.pass_probation[pass_name] = False
            elif scope.cancelled_caught:
                # The pass was skipped externally; mark that passes were
                # skipped so the main loop runs it again later.
                self._passes_were_skipped = True
            else:
                self.incomplete_passes.pop(pass_name, None)
                self.pass_probation[pass_name] = not made_progress
                if made_progress:
                    self.pass_fingerprints.pop(pass_name, None)
                else:
                    self.pass_fingerprints[pass_name] = problem.current_test_case

        finally:
            self.current_reduction_pass = None
            problem.current_pass_stats = None
            problem.pass_call_monitor = None
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

    async def __minimize_single_byte(self, c: int) -> None:
        """Try to replace the current single-byte test case with a smaller
        interesting byte, scanning upwards from zero. Interesting bytes are
        adopted by is_interesting as a side effect only if they sort below
        the current test case, so the scan only stops once one is adopted."""
        for i in range(c):
            candidate = bytes([i])
            if (
                await self.target.is_interesting(candidate)
                and self.target.current_test_case == candidate
            ):
                return

    async def run(self) -> None:
        try:
            await self._run()
        finally:
            await self.close_external_reducers()

    async def _run(self) -> None:
        await self.target.setup()

        if await self.target.is_interesting(b""):
            return

        for c in [0, 1, ord(b"\n"), ord(b"0"), ord(b"z"), 255]:
            if await self.target.is_interesting(bytes([c])):
                await self.__minimize_single_byte(c)
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
            if self.target.current_test_case != prev:
                continue
            # Only once everything else has converged do the expensive,
            # rarely-productive polish passes run.
            for rp in self.polish_passes:
                await self.run_pass(rp)
            if self.target.current_test_case != prev:
                continue
            # Passes abandoned by a probation budget may have missed
            # reductions. Re-run them without a budget before concluding,
            # so the final result is a fixpoint of every pass.
            if self.incomplete_passes:
                for name in sorted(self.incomplete_passes):
                    await self.run_pass(self.incomplete_passes[name], budgeted=False)
                if self.target.current_test_case != prev:
                    continue
            # Only terminate if no passes were skipped
            # If passes were skipped, we need another full run to be sure
            if not self._passes_were_skipped:
                # Give the problem a chance to change something (e.g.
                # raise an adaptive timeout) that makes another round
                # worth trying before we give up for good.
                if not await self.target.attempt_unstick():
                    break
                # Something changed, so a pass that previously ran to
                # completion without progress may now succeed on the same
                # test case: no-progress fingerprints are no longer valid.
                self.pass_fingerprints.clear()


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
        # Use the appropriate sort key for this value (natural for text, shortlex for binary)
        self._sort_key_fn = sort_key_for_initial(self.current_test_case)

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
        return self._sort_key_fn(test_case)

    def display(self, value: bytes) -> str:
        return repr(value)


@define
class DirectoryShrinkRay(Reducer[dict[str, bytes]]):
    # Forwarded to each per-file ShrinkRay. See ShrinkRay for details.
    external_reducers: list[list[str]] = attrs.Factory(list)
    python_reducer: bool = True
    reducer_log_dir: str | None = None

    async def run(self):
        while True:
            prev = self.target.current_test_case
            await self.delete_keys()
            await self.shrink_values()
            if self.target.current_test_case == prev:
                if not await self.target.attempt_unstick():
                    break

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
                if self.reducer_log_dir is not None:
                    key_log_dir: str | None = os.path.join(
                        self.reducer_log_dir, sanitize_for_filename(k)
                    )
                else:
                    key_log_dir = None
                key_shrinkray = ShrinkRay(
                    enable_cpp_passes=any(k.endswith(s) for s in C_FILE_EXTENSIONS),
                    treesitter_language=language_for_filename(k),
                    target=key_problem,
                    external_reducers=self.external_reducers,
                    python_reducer=self.python_reducer,
                    reducer_log_dir=key_log_dir,
                )
                nursery.start_soon(key_shrinkray.run)
