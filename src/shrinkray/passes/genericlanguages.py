"""
Module of reduction passes designed for "things that look like programming languages".
"""

from functools import wraps
import re
from typing import Callable, AnyStr

from attr import define
from shrinkray.problem import Format, ParseError, ReductionProblem
from shrinkray.reducer import ReductionPass


@define(frozen=True)
class Substring(Format[AnyStr, AnyStr]):
    prefix: AnyStr
    suffix: AnyStr

    @property
    def name(self) -> str:
        return f"Substring({len(self.prefix)}, {len(self.suffix)})"

    def parse(self, input: AnyStr) -> AnyStr:
        if input.startswith(self.prefix) and input.endswith(self.suffix):
            return input[len(self.prefix) : len(input) - len(self.suffix)]
        else:
            raise ParseError()

    def dumps(self, input: AnyStr) -> AnyStr:
        return self.prefix + input + self.suffix


def regex_pass(
    pattern: AnyStr | re.Pattern[AnyStr],
) -> Callable[[ReductionPass[AnyStr]], ReductionPass[AnyStr]]:
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)  # type: ignore

    def inner(fn):
        @wraps(fn)
        async def reduction_pass(problem: ReductionProblem[AnyStr]) -> None:
            i = 0
            while i < len(problem.current_test_case):
                search = pattern.search(problem.current_test_case, i)
                if search is None:
                    break

                u, v = search.span()

                i = v

                subformat = Substring(
                    problem.current_test_case[:u], problem.current_test_case[v:]
                )

                await fn(problem.view(subformat))

        return reduction_pass

    return inner


async def reduce_integer(problem: ReductionProblem[int]) -> None:
    assert problem.current_test_case >= 0

    if await problem.is_interesting(0):
        return

    lo = 0
    hi = problem.current_test_case

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if await problem.is_interesting(mid):
            hi = mid
        else:
            lo = mid

        if await problem.is_interesting(hi - 1):
            hi -= 1

        if await problem.is_interesting(lo + 1):
            return
        else:
            lo += 1


class IntegerFormat(Format[bytes, int]):
    def parse(self, s: bytes) -> int:
        try:
            return int(s.decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            raise ParseError()

    def dumps(self, i: int) -> bytes:
        return str(i).encode("ascii")


@regex_pass(b"[0-9]+")
async def reduce_integer_literals(problem):
    await reduce_integer(problem.view(IntegerFormat()))


@regex_pass(rb"[0-9]+ [*+-/] [0-9]+")
async def combine_expressions(problem):
    try:
        await problem.is_interesting(
            str(eval(problem.current_test_case)).encode("ascii")
        )
    except ArithmeticError:
        pass


def language_passes(problem: ReductionProblem[bytes]):
    yield reduce_integer_literals
    yield combine_expressions
