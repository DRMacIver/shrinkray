from abc import ABC, abstractmethod
from functools import wraps
from typing import Awaitable, Callable, Generic, TypeVar

from shrinkray.problem import ReductionProblem

S = TypeVar("S")
T = TypeVar("T")


ReductionPass = Callable[[ReductionProblem[T]], Awaitable[None]]
ReductionPump = Callable[[ReductionProblem[T]], Awaitable[T]]


class ParseError(Exception):
    pass


class DumpError(Exception):
    pass


class Format(Generic[S, T], ABC):
    @property
    def name(self) -> str:
        return repr(self)

    @abstractmethod
    def parse(self, input: S) -> T: ...

    def is_valid(self, input: S) -> bool:
        try:
            self.parse(input)
            return True
        except ParseError:
            return False

    @abstractmethod
    def dumps(self, input: T) -> S: ...


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
