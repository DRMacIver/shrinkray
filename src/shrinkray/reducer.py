from abc import ABC, abstractmethod
from functools import wraps
from typing import Awaitable, Callable, Generic, Iterable, Sequence, TypeVar

from attrs import define

from shrinkray.problem import Format, ParseError, ReductionProblem


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

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

    async def run(self) -> None:
        # TODO: Better algorithms go here

        while True:
            prev = self.target.current_test_case

            for rp in self.reduction_passes:
                self.target.work.note(rp.__name__)
                await rp(self.target)

            if self.target.current_test_case == prev:
                break
