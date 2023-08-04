from collections import Counter
from typing import Iterable

from attr import define

from shrinkray.passes.sequences import single_backward_delete
from shrinkray.problem import Format, ReductionProblem
from shrinkray.reducer import ReductionPass, compose


@define
class Split(Format[str, list[str]]):
    splitter: str

    def parse(self, value: str) -> list[str]:
        return value.split(self.splitter)

    def dumps(self, value: list[str]) -> str:
        return self.splitter.join(value)


def string_reduction_passes(
    problem: ReductionProblem[str],
) -> Iterable[ReductionPass[str]]:
    for splitter in [";", "\n", "n"]:
        yield compose(Split(splitter), single_backward_delete)

    yield single_backward_delete
