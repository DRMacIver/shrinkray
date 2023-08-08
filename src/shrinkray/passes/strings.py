from typing import Iterator

from attr import define

from shrinkray.passes.sequences import single_backward_delete
from shrinkray.problem import Format, ReductionProblem
from shrinkray.reducer import ReductionPass, compose


@define(frozen=True)
class Split(Format[str, list[str]]):
    splitter: str

    def __repr__(self) -> str:
        return f"Split({repr(self.splitter)})"

    @property
    def name(self) -> str:
        return f"split({repr(self.splitter)})"

    def parse(self, value: str) -> list[str]:
        return value.split(self.splitter)

    def dumps(self, value: list[str]) -> str:
        return self.splitter.join(value)


def string_passes(problem: ReductionProblem[str]) -> Iterator[ReductionPass[str]]:
    for split in ['"', "'", ";", "\n", " "]:
        if split in problem.current_test_case:
            yield compose(Split(split), single_backward_delete)

    yield single_backward_delete
