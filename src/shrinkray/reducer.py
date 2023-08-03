from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import Awaitable
from typing import Callable
from typing import Generic
from typing import Sequence
from typing import TypeVar

from attrs import define

from shrinkray.problem import Format
from shrinkray.problem import ParseError
from shrinkray.problem import ReductionProblem


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

        reduction_pass(view)

    wrapped_pass.__name__ = f"compose({format}, {reduction_pass.__name__})"

    return wrapped_pass


@define
class Reducer(Generic[T]):
    target: ReductionProblem[T]
    reduction_passes: Sequence[ReductionPass[T]]

    async def run(self) -> None:
        # TODO: Better algorithms go here

        while True:
            prev = self.target.current_test_case

            for rp in self.reduction_passes:
                self.target.work.note(rp.__name__)
                await rp(self.target)

            if self.target.current_test_case == prev:
                break
