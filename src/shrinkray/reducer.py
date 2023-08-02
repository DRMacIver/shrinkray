from typing import Awaitable, Callable, Generic, Sequence, TypeVar
from shrink_ray.problem import ReductionProblem, Format
from attrs import define
from abc import ABC, abstractmethod

S = TypeVar("S")
T = TypeVar("T")


class ReductionPass(Generic[S]):
    @abstractmethod
    async def __call__(self, problem: ReductionProblem[S]) -> None:
        ...


@define
class FunctionBasedPass(ReductionPass[S]):
    function: Callable[[ReductionProblem[S]], Awaitable[None]]

    async def __call__(self, problem: ReductionProblem[S]) -> None:
        return await self.function(problem)


@define
class ViewBasedPass(Generic[S, T], ReductionPass[S]):
    format: Format[S, T]
    base_pass: ReductionPass[T]

    async def __call__(self, problem: ReductionProblem[S]) -> None:
        await self.base_pass(problem.view(self.format))


@define
class Reducer(Generic[T]):
    target: ReductionProblem[T]
    reduction_passes: Sequence[ReductionPass[T]]

    async def run(self) -> None:
        # TODO: Better algorithms go here

        while True:
            prev = self.target.current_test_case

            for rp in self.reduction_passes:
                await rp(self.target)

            if self.target.current_test_case == prev:
                break
