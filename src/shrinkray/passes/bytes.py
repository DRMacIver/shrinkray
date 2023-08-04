from typing import Iterator

import chardet
from attrs import define

from shrinkray.passes.sequences import sequence_passes
from shrinkray.passes.strings import string_passes
from shrinkray.problem import Format, ReductionProblem
from shrinkray.reducer import ReductionPass, compose


@define(frozen=True)
class Encoding(Format[bytes, str]):
    encoding: str

    def __repr__(self) -> str:
        return f"Encoding({repr(self.encoding)})"

    @property
    def name(self) -> str:
        return self.encoding

    def parse(self, value: bytes) -> str:
        return value.decode(self.encoding)

    def dumps(self, value: str) -> bytes:
        return value.encode(self.encoding)


def byte_passes(problem: ReductionProblem[bytes]) -> Iterator[ReductionPass[bytes]]:
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
        for reduction_pass in string_passes(view):
            yield compose(format, reduction_pass)

    yield from sequence_passes(problem)
