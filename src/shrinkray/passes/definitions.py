from typing import Awaitable, Callable, TypeVar

from shrinkray.problem import ReductionProblem

T = TypeVar("T")


ReductionPass = Callable[[ReductionProblem[T]], Awaitable[None]]
ReductionPump = Callable[[ReductionProblem[T]], Awaitable[T]]
