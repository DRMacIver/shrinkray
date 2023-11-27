from functools import wraps
import math
from pdb import Restart
from typing import Any, Awaitable, Callable, Generic, Iterable, Optional, TypeVar

import trio
from attrs import define

from shrinkray.problem import Format, ParseError, ReductionProblem
from shrinkray.work import Volume, WorkContext

S = TypeVar("S")
T = TypeVar("T")


ReductionPass = Callable[[ReductionProblem[T]], Awaitable[None]]


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
class Reducer(Generic[T]):
    target: ReductionProblem[T]
    reduction_passes: Iterable[ReductionPass[T]]
    dumb_mode: bool = False

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)
        if len(self.reduction_passes) <= 1:
            self.dumb_mode = True

    async def run_pass(self, rp):
        self.target.work.note(rp.__name__)

        halt = 0.9
        while True:
            try:
                await rp(ReductionLimitedProblem(self.target, halt_at=halt))
                break
            except* RestartPass:
                halt *= 0.95

    async def run(self) -> None:
        await self.target.setup()

        if self.dumb_mode:
            while True:
                prev = self.target.current_test_case
                for rp in self.reduction_passes:
                    await self.run_pass(rp)
                if prev == self.target.current_test_case:
                    return

        budget = 100
        while True:
            reduction_pass = await self.select_pass()
            if reduction_pass is None:
                break

            self.target.work.note(reduction_pass.__name__)
            budget_exceeded = False
            try:
                await reduction_pass(
                    BudgetedProblem(base_problem=self.target, budget=budget)
                )
            except* BudgetExceeded:
                budget_exceeded = True

            if budget_exceeded:
                budget = int(budget * 1.05)

    async def select_pass(self) -> Optional[ReductionPass[T]]:
        """Select a reduction pass to run, or None if no passes can make progress."""

        # Idea: Run all reduction passes in parallel. If one of them succeeds,
        # we probably want to run that one. But we might get unlucky and succeed
        # with some pass that mostly makes tiny changes, so we run for a little
        # longer after we've found the first successful reduction and then return
        # whichever pass found the best example in that window.
        async with trio.open_nursery() as nursery:
            passes = self.reduction_passes

            work_queue_send, work_queue_receive = trio.open_memory_channel(
                self.target.work.parallelism * 2
            )

            completed_passes = 0

            async def pass_worker(i):
                nonlocal completed_passes
                problem = ChannelBackedProblem(
                    base_problem=self.target,
                    work_queue=work_queue_send,
                    label=i,
                )
                try:
                    await passes[i](problem)
                except trio.BrokenResourceError:
                    pass
                finally:
                    completed_passes += 1

            for i in range(len(self.reduction_passes)):
                nursery.start_soon(pass_worker, i)

            results_queue_send, results_queue_receive = trio.open_memory_channel(
                float("inf")
            )

            async def results_worker():
                try:
                    while True:
                        chunk = await work_queue_receive.receive()
                        result = await self.target.is_interesting(chunk.value)
                        await chunk.response.send(result)
                        await results_queue_send.send(
                            CompletedWork(
                                calling_pass=chunk.label,
                                value=chunk.value,
                                result=result,
                            )
                        )
                except trio.BrokenResourceError:
                    pass

            for _ in range(self.target.work.parallelism):
                nursery.start_soon(results_worker)

            count = 0
            succeeded_at = None
            best: Any = None
            best_pass = None

            while completed_passes < len(passes):
                if succeeded_at is not None and count > max(10, 2 * succeeded_at):
                    break
                with trio.move_on_after(1) as cancel_scope:
                    completed: CompletedWork[T] = await results_queue_receive.receive()
                if cancel_scope.cancel_called:
                    continue
                count += 1
                if completed.result:
                    if succeeded_at is None:
                        succeeded_at = count
                    key = self.target.sort_key(completed.value)
                    if best is None or key < best:
                        best = key
                        best_pass = completed.calling_pass

            await results_queue_receive.aclose()

            nursery.cancel_scope.cancel()

        if best_pass is not None:
            return passes[best_pass]


@define(frozen=True, slots=True)
class WorkChunk(Generic[T]):
    label: int
    value: T
    response: trio.MemorySendChannel[bool]


@define(frozen=True, slots=True)
class CompletedWork(Generic[T]):
    calling_pass: int
    value: T
    result: bool


class ChannelBackedProblem(ReductionProblem[T]):
    def __init__(
        self,
        base_problem,
        work_queue: trio.MemorySendChannel[WorkChunk],
        label: int,
    ):
        super().__init__(
            work=WorkContext(
                parallelism=1,
                volume=Volume.quiet,
                random=base_problem.work.random,
            )
        )

        self.base_problem = base_problem
        self.send_channel, self.receive_channel = trio.open_memory_channel(0)
        self.work_queue = work_queue
        self.__current = self.base_problem.current_test_case
        self.label = label

    def cached_is_interesting(self, test_case: T) -> bool | None:
        return self.base_problem.cached_is_interesting(test_case)

    async def is_interesting(self, test_case: T) -> bool:
        cached = self.cached_is_interesting(test_case)
        if cached is not None:
            result = cached
        else:
            await self.work_queue.send(
                WorkChunk(label=self.label, value=test_case, response=self.send_channel)
            )
            result = await self.receive_channel.receive()

        if result and self.sort_key(test_case) < self.sort_key(self.current_test_case):
            self.__current = test_case

        return result

    def sort_key(self, test_case: T) -> Any:
        return self.base_problem.sort_key(test_case)

    @property
    def current_test_case(self) -> T:
        return self.__current


class BudgetExceeded(Exception):
    pass


class BudgetedProblem(ReductionProblem[T]):
    def __init__(self, base_problem, budget: int):
        super().__init__(work=base_problem.work)
        self.base_problem = base_problem
        self.budget = budget
        self.__call_count = 0

    def cached_is_interesting(self, test_case: T) -> bool | None:
        return self.base_problem.cached_is_interesting(test_case)

    async def is_interesting(self, test_case: T) -> bool:
        cached = self.cached_is_interesting(test_case)
        if cached is not None:
            return cached
        self.__call_count += 1
        if self.__call_count >= self.budget:
            raise BudgetExceeded()

        return await self.base_problem.is_interesting(test_case)

    def sort_key(self, test_case: T) -> Any:
        return self.base_problem.sort_key(test_case)

    @property
    def current_test_case(self) -> T:
        return self.base_problem.current_test_case


class RestartPass(Exception):
    pass


class ReductionLimitedProblem(ReductionProblem[T]):
    def __init__(self, base_problem, halt_at: float = 0.5):
        super().__init__(work=base_problem.work)
        self.base_problem = base_problem
        n = self.base_problem.size(self.base_problem.current_test_case)
        self.threshold = min(n - 1, math.ceil(halt_at * n))

    def cached_is_interesting(self, test_case: T) -> bool | None:
        return self.base_problem.cached_is_interesting(test_case)

    async def is_interesting(self, test_case: T) -> bool:
        cached = self.cached_is_interesting(test_case)
        if cached is not None:
            return cached
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
