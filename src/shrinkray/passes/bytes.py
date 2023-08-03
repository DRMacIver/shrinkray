from typing import Iterator

import chardet
from attrs import define

from shrinkray.passes.sequences import sequence_passes
from shrinkray.problem import Format
from shrinkray.problem import ReductionProblem
from shrinkray.reducer import ReductionPass
from shrinkray.reducer import compose


@define
class Encoding(Format[bytes, str]):
    encoding: str

    def parse(self, value: bytes) -> str:
        return value.decode(self.encoding)

    def dumps(self, value: str) -> bytes:
        return value.encode(self.encoding)


def byte_passes(problem: ReductionProblem[bytes]) -> Iterator[ReductionPass[bytes]]:
    yield from sequence_passes(problem)

    value = problem.current_test_case

    for info in chardet.detect_all(problem.current_test_case):
        encoding = info["encoding"]
        if encoding is None:
            continue

        try:
            value.decode(encoding)
        except UnicodeDecodeError:
            continue

        format = Encoding(encoding)
        view = problem.view(format)
        for reduction_pass in sequence_passes(view):
            yield compose(format, reduction_pass)
