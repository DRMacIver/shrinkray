from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from time import time
from typing import Generic, Iterable, Optional, TypeVar

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
    lower_bytes,
    lower_individual_bytes,
    remove_indents,
    remove_whitespace,
    replace_space_with_newlines,
    short_deletions,
    standard_substitutions,
)
from shrinkray.passes.clangdelta import ClangDelta, clang_delta_pumps
from shrinkray.passes.definitions import Format, ReductionPass, ReductionPump, compose
from shrinkray.passes.genericlanguages import (
    combine_expressions,
    merge_adjacent_strings,
    normalize_identifiers,
    reduce_integer_literals,
    replace_falsey_with_zero,
    simplify_brackets,
)
from shrinkray.passes.json import JSON, JSON_PASSES
from shrinkray.passes.python import PYTHON_PASSES, is_python
from shrinkray.passes.sat import SAT_PASSES, DimacsCNF
from shrinkray.passes.sequences import block_deletion, delete_duplicates
from shrinkray.problem import ReductionProblem

S = TypeVar("S")
T = TypeVar("T")


@define
class Reducer(Generic[T], ABC):
    target: ReductionProblem[T]

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


@define
class BasicReducer(Reducer[T]):
    reduction_passes: Iterable[ReductionPass[T]]
    pumps: Iterable[ReductionPump[T]] = ()
    status: str = "Starting up"

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

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
class ShrinkRay(Reducer[bytes]):
    clang_delta: Optional[ClangDelta] = None

    current_reduction_pass: Optional[ReductionPass[bytes]] = None
    current_pump: Optional[ReductionPump[bytes]] = None

    unlocked_ok_passes: bool = False

    initial_cuts: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            hollow,
            compose(Split(b"\n"), delete_duplicates),
            compose(Split(b"\n"), block_deletion(10, 100)),
            lift_braces,
            remove_indents,
            remove_whitespace,
            replace_space_with_newlines,
        ]
    )

    great_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            debracket,
            compose(Split(b"\n"), delete_duplicates),
            compose(Split(b"\n"), block_deletion(1, 10)),
            compose(Split(b";"), block_deletion(1, 10)),
            remove_indents,
            hollow,
            lift_braces,
            delete_byte_spans,
        ]
    )

    ok_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
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
            # short_replacements,
            # sort_whitespace,
        ]
    )

    def __attrs_post_init__(self) -> None:
        if is_python(self.target.current_test_case):
            self.great_passes.extend(PYTHON_PASSES)
        self.register_format_specific_pass(JSON, JSON_PASSES)
        self.register_format_specific_pass(
            DimacsCNF,
            SAT_PASSES,
        )

    def register_format_specific_pass(
        self, format: Format[bytes, T], passes: Iterable[ReductionPass[T]]
    ):
        if format.is_valid(self.target.current_test_case):
            self.great_passes.extend(compose(format, p) for p in passes)

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
        try:
            assert self.current_reduction_pass is None
            start = time()
            self.current_reduction_pass = rp
            await rp(self.target)
        finally:
            self.current_reduction_pass = None

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
        for rp in self.great_passes:
            await self.run_pass(rp)

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
            prev = self.target.current_test_case
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
            if prev == self.target.current_test_case:
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
            prev = self.target.current_test_case
            await self.run_some_passes()
            if self.target.current_test_case != prev:
                continue
            for pump in self.pumps:
                await self.pump(pump)
            if self.target.current_test_case == prev:
                break
