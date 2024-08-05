import json

from shrinkray.passes.json import delete_identifiers
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import WorkContext


def json_problem(initial, is_interesting):
    return BasicReductionProblem(
        initial,
        is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=lambda s: len(json.dumps(s)),
    )


async def test_can_remove_some_identifiers():
    initial = [{"a": 0, "b": 0, "c": i} for i in range(10)]

    async def is_interesting(ls):
        return len(ls) == 10 and all(x.get("c", -1) == i for i, x in enumerate(ls))

    assert await is_interesting(initial)

    problem = json_problem(initial, is_interesting)

    await delete_identifiers(problem)

    assert problem.current_test_case == [{"c": i} for i in range(10)]
