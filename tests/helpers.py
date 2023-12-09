from typing import Callable, Iterable, TypeVar

import trio

from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import Reducer, ReductionPass
from shrinkray.work import WorkContext

T = TypeVar("T")


def reduce_with(
    rp: Iterable[ReductionPass[T]],
    initial: T,
    is_interesting: Callable[[T], bool],
    parallelism=1,
) -> T:
    async def acondition(x):
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> T:
        problem: BasicReductionProblem[T] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
        )

        reducer = Reducer(
            target=problem,
            reduction_passes=rp,
        )

        await reducer.run()

        return problem.current_test_case  # type: ignore

    return trio.run(calc_result)
