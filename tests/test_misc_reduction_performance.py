from tests.helpers import reduce
import pytest
from time import time


@pytest.mark.parametrize(
    "initial",
    [
        b"." * 378,
    ],
)
@pytest.mark.parametrize("parallelism", [1, 2])
def test_failure_performance(initial, parallelism):
    start = time()
    final = reduce(initial, lambda s: s == initial)
    assert final == initial
    runtime = time() - start
    assert runtime <= 2
