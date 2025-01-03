from contextlib import aclosing

import pytest

from shrinkray.work import WorkContext, parallel_map


async def identity(x: int) -> int:
    return x


@pytest.mark.parametrize("p", [1, 2, 3, 4])
async def test_parallel_map(p: int) -> None:
    input = [1, 2, 3]
    async with parallel_map(input, identity, parallelism=p) as mapped:
        values = [x async for x in mapped]
        print(values)
    assert values == input


@pytest.mark.parametrize("p", [1, 2, 3, 4])
async def test_worker_map(p: int) -> None:
    work = WorkContext(parallelism=p)

    input = range(1000)

    i = 0
    async with work.map(input, identity) as mapped:
        async for x in mapped:
            assert input[i] == x
            i += 1
