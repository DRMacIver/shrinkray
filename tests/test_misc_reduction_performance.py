from time import time

import pytest

from tests.helpers import reduce


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


ASSIGNMENT_CHAIN = b"""
agmas = 42
benighted = agmas
squabashing = benighted
paradisaically = squabashing
simar = paradisaically
output.append(simar)
"""


def test_can_normalize_identifiers():
    def is_interesting(test_case):
        output = []
        data = {"output": output}
        try:
            exec(test_case, data, data)
        except BaseException:
            return False
        return output == [42]

    # Ideally we would reduce further than this, but it's tricky for now.
    assert reduce(ASSIGNMENT_CHAIN, is_interesting) == b"A=42\noutput.append(A)"
