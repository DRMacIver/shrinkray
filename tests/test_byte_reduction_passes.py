import ast
from typing import Callable, Iterable, TypeVar

import trio
from shrinkray.passes.bytes import short_deletions
from shrinkray.passes.sequences import single_backward_delete
from shrinkray.problem import BasicReductionProblem

from shrinkray.reducer import Reducer, ReductionPass
from shrinkray.work import WorkContext


T = TypeVar("T")


def reduce_with(
    rp: Iterable[ReductionPass[T]],
    initial: T,
    is_interesting: Callable[[T], bool],
    dumb=True,
    parallelism=1,
) -> T:
    async def acondition(x):
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> T:
        problem: BasicReductionProblem[T] = await BasicReductionProblem(  # type: ignore
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
        )

        reducer = Reducer(
            target=problem,
            reduction_passes=rp,
            dumb_mode=dumb,
        )

        await reducer.run()

        return problem.current_test_case  # type: ignore

    return trio.run(calc_result)


def test_basic_delete():
    assert (
        reduce_with([single_backward_delete], b"abracadabra", lambda s: b"a" in s)
        == b"a"
    )


def is_hello(data: bytes) -> bool:
    try:
        tree = ast.parse(data)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "Hello world!":
            return True

    return False


def test_short_deletions_can_delete_brackets():
    assert (
        reduce_with([short_deletions], b'"Hello world!"()', is_hello)
        == b'"Hello world!"'
    )
