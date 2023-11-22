from shrinkray.passes.bytes import byte_passes
from shrinkray.passes.genericlanguages import language_passes
from shrinkray.problem import ReductionProblem


def common_passes(problem: ReductionProblem[bytes]):
    yield from byte_passes(problem)
    yield from language_passes(problem)
