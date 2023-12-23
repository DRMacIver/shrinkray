import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from typing import Any, Generic, Iterable, Optional, TypeVar

import attrs
import trio
from attrs import define

from shrinkray.passes.bytes import (
    Split,
    Tokenize,
    debrace,
    delete_byte_spans,
    hollow,
    lexeme_based_deletions,
    lift_braces,
    remove_indents,
    remove_whitespace,
    replace_space_with_newlines,
    short_deletions,
)
from shrinkray.passes.clangdelta import ClangDelta, clang_delta_pumps
from shrinkray.passes.definitions import ReductionPass, ReductionPump
from shrinkray.passes.genericlanguages import (
    combine_expressions,
    merge_adjacent_strings,
    reduce_integer_literals,
)
from shrinkray.passes.python import is_python, lift_indented_constructs
from shrinkray.passes.sequences import block_deletion, delete_duplicates
from shrinkray.problem import Format, ParseError, ReductionProblem

S = TypeVar("S")
T = TypeVar("T")


def compose(format: Format[S, T], reduction_pass: ReductionPass[T]) -> ReductionPass[S]:
    @wraps(reduction_pass)
    async def wrapped_pass(problem: ReductionProblem[S]) -> None:
        view = problem.view(format)

        try:
            view.current_test_case
        except ParseError:
            return

        await reduction_pass(view)

    wrapped_pass.__name__ = f"{format.name}/{reduction_pass.__name__}"

    return wrapped_pass


@define
class Reducer(Generic[T], ABC):
    target: ReductionProblem[T]

    @contextmanager
    def backtrack(self, restart: T):
        current = self.target
        try:
            self.target = self.target.backtrack(restart)
            yield
        finally:
            self.target = current

    @abstractmethod
    async def run(self) -> None:
        ...


@define
class BasicReducer(Reducer[T]):
    reduction_passes: Iterable[ReductionPass[T]]
    pumps: Iterable[ReductionPump[T]] = ()
    status: str = "Starting up"

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

    async def run_pass(self, rp):
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


class ReductionLimitedProblem(ReductionProblem[T]):
    def __init__(self, base_problem, halt_at: float = 0.5):
        super().__init__(work=base_problem.work)
        self.base_problem = base_problem
        n = self.base_problem.size(self.base_problem.current_test_case)
        self.threshold = min(n - 1, math.ceil(halt_at * n))
        self.stats = base_problem.stats

    async def is_interesting(self, test_case: T) -> bool:
        result = await self.base_problem.is_interesting(test_case)
        if self.current_size <= self.threshold:
            raise RestartPass()
        return result

    def sort_key(self, test_case: T) -> Any:
        return self.base_problem.sort_key(test_case)

    def size(self, test_case: T) -> int:
        return self.base_problem.size(test_case)

    @property
    def current_test_case(self) -> T:
        return self.base_problem.current_test_case


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
            debrace,
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
            combine_expressions,
            merge_adjacent_strings,
            lexeme_based_deletions,
        ]
    )

    last_ditch_passes: list[ReductionPass[bytes]] = attrs.Factory(
        lambda: [
            compose(Split(b"\n"), block_deletion(21, 100)),
            short_deletions,
            replace_space_with_newlines,
        ]
    )

    def __attrs_post_init__(self):
        if is_python(self.target.current_test_case):
            self.great_passes.append(lift_indented_constructs)

    @property
    def pumps(self):
        if self.clang_delta is None:
            return ()
        else:
            return clang_delta_pumps(self.clang_delta)

    @property
    def status(self):
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

    async def run_pass(self, rp: ReductionPass[bytes]):
        try:
            assert self.current_reduction_pass is None
            self.current_reduction_pass = rp
            await rp(self.target)
        finally:
            self.current_reduction_pass = None

    async def pump(self, rp: ReductionPump[bytes]):
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

    async def run_great_passes(self):
        for rp in self.great_passes:
            await self.run_pass(rp)

    async def run_ok_passes(self):
        for rp in self.ok_passes:
            await self.run_pass(rp)

    async def run_last_ditch_passes(self):
        for rp in self.last_ditch_passes:
            await self.run(rp)

    async def run_some_passes(self):
        prev = self.target.current_test_case
        await self.run_great_passes()
        if prev == self.target.current_test_case and not self.unlocked_ok_passes:
            return
        self.unlocked_ok_passes = True
        await self.run_ok_passes()
        if prev == self.target.current_test_case:
            return
        await self.run_last_ditch_passes()

    async def initial_cut(self):
        while True:
            prev = self.target.current_test_case
            for rp in self.initial_cuts:
                async with trio.open_nursery() as nursery:

                    @nursery.start_soon
                    async def _():
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
                        best_reduction_rate = None

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

                            if (
                                rate < 0.5 * best_reduction_rate
                                or current == self.target.current_test_case
                            ):
                                nursery.cancel_scope.cancel()
                                break

                    await self.run_pass(rp)
            if prev == self.target.current_test_case:
                return

    async def run(self):
        await self.target.setup()

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
