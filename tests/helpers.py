from typing import Callable, Iterable, TypeVar

import trio

from shrinkray.passes.definitions import ReductionPass
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import BasicReducer
from shrinkray.work import WorkContext

T = TypeVar("T")


def reduce_with(
    rp: Iterable[ReductionPass[T]],
    initial: T,
    is_interesting: Callable[[T], bool],
    parallelism: int = 1,
) -> T:
    async def acondition(x: T) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> T:
        problem: BasicReductionProblem[T] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
        )

        reducer = BasicReducer(
            target=problem,
            reduction_passes=rp,
        )

        await reducer.run()

        return problem.current_test_case

    return trio.run(calc_result)
