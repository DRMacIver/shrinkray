import random
from typing import Callable, Iterable, TypeVar

import trio

from shrinkray.passes.definitions import ReductionPass
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import BasicReducer, ShrinkRay
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


def reduce(
    initial: bytes,
    is_interesting: Callable[[bytes], bool],
    parallelism: int = 1,
) -> T:
    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> bytes:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
        )

        reducer = ShrinkRay(
            target=problem,
        )

        await reducer.run()

        return problem.current_test_case

    return trio.run(calc_result)


def assert_no_blockers(
    is_interesting: Callable[[bytes], bool],
    potential_blockers: list[bytes],
    lower_bounds=random.sample(range(1000), 12),
):
    potential_blockers = sorted(set(potential_blockers), key=lambda s: (len(s), s))

    for lower_bound in lower_bounds:

        async def acondition(x: bytes) -> bool:
            await trio.lowlevel.checkpoint()
            if len(x) < lower_bound:
                return False
            return is_interesting(x)

        current_best = None

        max_blockers = 1000
        candidates = [
            candidate
            for candidate in potential_blockers
            if len(candidate) >= lower_bound
        ]

        if len(candidates) > max_blockers:
            candidates = random.sample(candidates, max_blockers)
            candidates.sort(key=lambda s: (len(s), s))

        for initial in candidates:
            if len(initial) < lower_bound or not is_interesting(initial):
                continue
            problem: BasicReductionProblem[T] = BasicReductionProblem(
                initial=initial,
                is_interesting=acondition,
                work=WorkContext(parallelism=1),
            )

            async def calc_result() -> bytes:
                reducer = ShrinkRay(
                    target=problem,
                )

                await reducer.run()

                return problem.current_test_case

            result = trio.run(calc_result)

            if current_best is None:
                current_best = result
            elif result != current_best:
                current_best, result = sorted(
                    (result, current_best), key=problem.sort_key
                )

                raise AssertionError(
                    f"With lower bound {lower_bound}, {result} does not reduce to {current_best}"
                )
