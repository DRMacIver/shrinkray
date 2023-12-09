import math
from functools import wraps
from typing import Any, Awaitable, Callable, Generic, Iterable, TypeVar

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

    status: str = "Starting up"

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

    async def run_pass(self, rp):
        self.status = f"Running reduction pass {rp.__name__}"
        await rp(self.target)

    async def run(self) -> None:
        await self.target.setup()

        while True:
            prev = self.target.current_test_case
            for rp in self.reduction_passes:
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
