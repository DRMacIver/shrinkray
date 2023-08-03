from typing import Iterator
from attr import define

from shrinkray.problem import Format, ReductionProblem

from shrinkray.reducer import ReductionPass, compose

from shrinkray.passes.sequences import single_backward_delete


@define(frozen=True)
class Split(Format[str, list[str]]):
    splitter: str

    def parse(self, value: str) -> list[str]:
        return value.split(self.splitter)

    def dumps(self, value: list[str]) -> str:
        return self.splitter.join(value)


def string_passes(problem: ReductionProblem[str]) -> Iterator[ReductionPass[str]]:
    for split in [";", "\n", " "]:
        if split in problem.current_test_case:
            yield compose(Split(split), single_backward_delete)

    yield single_backward_delete
