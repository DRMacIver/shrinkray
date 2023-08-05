from typing import Any, Iterable, Sequence, TypeVar

import trio

from shrinkray.problem import ReductionProblem
from shrinkray.reducer import ReductionPass
from shrinkray.work import NotFound

Seq = TypeVar("Seq", bound=Sequence[Any])


async def single_forward_delete(problem: ReductionProblem[Seq]) -> None:
    test_case = problem.current_test_case

    if not test_case:
        await trio.lowlevel.checkpoint()
        return

    def deleted(j: int, k: int) -> Seq:
        return test_case[:j] + test_case[k:]  # type: ignore

    async def can_delete(j: int, k: int) -> bool:
        return await problem.is_interesting(deleted(j, k))

    initial_length = len(test_case)
    i = 0

    async with problem.work.pb(
        total=lambda: initial_length,
        current=lambda: i + initial_length - len(test_case),
        desc="Deletion steps",
    ):
        while i < len(test_case):
            try:
                i = await problem.work.find_first_value(
                    range(i, len(test_case)), lambda j: can_delete(j, j + 1)
                )
            except NotFound:
                break

            test_case = deleted(i, i + 1)

            async def delete_k(k: int) -> bool:
                if i + k > len(test_case):
                    await trio.lowlevel.checkpoint()
                    return False
                return await can_delete(i, i + k)

            k = await problem.work.find_large_integer(delete_k)
            test_case = deleted(i, i + k)

            i += 1


async def single_backward_delete(problem: ReductionProblem[Seq]) -> None:
    test_case = problem.current_test_case
    if not test_case:
        await trio.lowlevel.checkpoint()
        return

    def deleted(j: int, k: int) -> Seq:
        return test_case[:j] + test_case[k:]  # type: ignore

    async def can_delete(j: int, k: int) -> bool:
        return await problem.is_interesting(deleted(j, k))

    i = len(test_case) - 1
    initial_length = i

    async with problem.work.pb(
        total=lambda: initial_length,
        current=lambda: initial_length - i,
        desc="Deletion steps",
    ):
        while i >= 0:
            try:
                i = await problem.work.find_first_value(
                    range(i, -1, -1), lambda j: can_delete(j, j + 1)
                )
            except NotFound:
                break

            test_case = deleted(i, i + 1)

            async def delete_k(k: int) -> bool:
                if k > i:
                    await trio.lowlevel.checkpoint()
                    return False
                return await can_delete(i - k, i)

            k = await problem.work.find_large_integer(delete_k)
            test_case = deleted(i - k, i)
            i -= k + 1


def sequence_passes(
    problem: ReductionProblem[Seq],
) -> Iterable[ReductionPass[Seq]]:
    yield single_backward_delete
    yield single_forward_delete
