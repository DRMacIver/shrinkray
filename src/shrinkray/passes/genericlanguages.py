"""
Module of reduction passes designed for "things that look like programming languages".
"""

import re
from collections.abc import Callable, Sized
from functools import wraps
from string import ascii_lowercase, ascii_uppercase
from typing import AnyStr, TypeVar

import trio
from attr import define

from shrinkray.passes.bytes import ByteReplacement, delete_intervals
from shrinkray.passes.definitions import Format, ParseError, ReductionPass
from shrinkray.passes.patching import PatchApplier, Patches, apply_patches
from shrinkray.problem import BasicReductionProblem, ReductionProblem
from shrinkray.work import NotFound


T = TypeVar("T", bound=Sized)


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


class RegionReplacingPatches(Patches[dict[int, AnyStr], AnyStr]):
    def __init__(self, regions: list[tuple[int, int]]):
        assert regions
        for (_, v), (u, _) in zip(regions, regions[1:], strict=False):
            assert v <= u
        self.regions = regions

    @property
    def empty(self):
        return {}

    def combine(self, *patches):
        result = {}
        for p in patches:
            result.update(p)
        return result

    def apply(self, patch, target):
        empty = target[:0]
        parts = []
        prev = 0
        for j, (u, v) in enumerate(self.regions):
            assert v <= len(target)
            parts.append(target[prev:u])
            try:
                parts.append(patch[j])
            except KeyError:
                parts.append(target[u:v])
            prev = v
        parts.append(target[prev:])
        return empty.join(parts)

    def size(self, patch):
        for i, s in patch.items():
            u, v = self.regions[i]
            return v - u - len(s)
        raise AssertionError(f"expected nonempty {patch=}")


def regex_pass(
    pattern: AnyStr | re.Pattern[AnyStr],
    flags: re.RegexFlag = re.RegexFlag.NOFLAG,
) -> Callable[[ReductionPass[AnyStr]], ReductionPass[AnyStr]]:
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern, flags=flags)

    def inner(fn: ReductionPass[AnyStr]) -> ReductionPass[AnyStr]:
        @wraps(fn)
        async def reduction_pass(problem: ReductionProblem[AnyStr]) -> None:
            matching_regions = []
            initial_values_for_regions = []

            i = 0
            while i < len(problem.current_test_case):
                search = pattern.search(problem.current_test_case, i)
                if search is None:
                    break

                u, v = search.span()
                matching_regions.append((u, v))
                initial_values_for_regions.append(problem.current_test_case[u:v])

                i = v

            if not matching_regions:
                return

            patches = RegionReplacingPatches(matching_regions)

            patch_applier = PatchApplier(patches, problem)

            async with trio.open_nursery() as nursery:

                async def reduce_region(i: int) -> None:
                    async def is_interesting(s):
                        return await patch_applier.try_apply_patch({i: s})

                    subproblem = BasicReductionProblem(
                        initial_values_for_regions[i],
                        is_interesting,
                        work=problem.work,
                    )
                    nursery.start_soon(fn, subproblem)

                for i in range(len(matching_regions)):
                    await reduce_region(i)

        return reduction_pass

    return inner


async def reduce_integer(problem: ReductionProblem[int]) -> None:
    """Reduce an integer to its smallest interesting value.

    Uses binary search to find the smallest integer that maintains
    interestingness. Tries 0 first, then narrows down the range.
    """
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
    def parse(self, input: bytes) -> int:
        try:
            return int(input.decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            raise ParseError()

    def dumps(self, input: int) -> bytes:
        return str(input).encode("ascii")


@regex_pass(b"[0-9]+")
async def reduce_integer_literals(problem: ReductionProblem[bytes]) -> None:
    """Reduce integer literals in source code to smaller values.

    Finds numeric literals and tries to reduce each one independently
    using binary search.
    """
    await reduce_integer(problem.view(IntegerFormat()))


@regex_pass(rb"[0-9]+ [*+-/] [0-9]+")
async def combine_expressions(problem: ReductionProblem[bytes]) -> None:
    """Evaluate and simplify simple arithmetic expressions.

    Finds expressions like "2 + 3" and replaces them with their result "5".
    Only handles basic integer arithmetic to avoid changing program semantics.
    """
    try:
        # NB: Use of eval is safe, as everything passed to this is a simple
        # arithmetic expression. Would ideally replace with a guaranteed
        # safe version though.
        await problem.is_interesting(
            str(eval(problem.current_test_case)).encode("ascii")
        )
    except ArithmeticError:
        pass


@regex_pass(rb'([\'"])\s*\1')
async def merge_adjacent_strings(problem: ReductionProblem[bytes]) -> None:
    """Remove empty string concatenations like '' '' or "" "".

    These patterns (quote, whitespace, same quote) often result from
    other reductions and can be eliminated entirely.
    """
    await problem.is_interesting(b"")


@regex_pass(rb"''|\"\"|false|\(\)|\[\]", re.IGNORECASE)
async def replace_falsey_with_zero(problem: ReductionProblem[bytes]) -> None:
    """Replace falsey values with 0.

    Tries to replace empty strings, 'false', empty parentheses, and empty
    brackets with the single character '0', which is shorter and often
    equivalent in boolean contexts.
    """
    await problem.is_interesting(b"0")


async def simplify_brackets(problem: ReductionProblem[bytes]) -> None:
    """Try to replace bracket types with simpler ones.

    Attempts to replace {} with [] or (), and [] with (). This can
    help normalize syntax when the specific bracket type doesn't matter.
    """
    bracket_types = [b"[]", b"{}", b"()"]

    patches = [
        dict(zip(u, v, strict=True))
        for u in bracket_types
        for v in bracket_types
        if u > v
    ]

    await apply_patches(problem, ByteReplacement(), patches)


IDENTIFIER = re.compile(rb"(\b[A-Za-z][A-Za-z0-9_]*\b)|([0-9]+)")


def shortlex(s: T) -> tuple[int, T]:
    return (len(s), s)


async def normalize_identifiers(problem: ReductionProblem[bytes]) -> None:
    """Replace identifiers with shorter alternatives.

    Finds all identifiers in the source and tries to replace longer ones
    with shorter alternatives (single letters like 'a', 'b', etc.). This
    normalizes variable/function names to minimal forms.
    """
    identifiers = {m.group(0) for m in IDENTIFIER.finditer(problem.current_test_case)}
    replacements = set(identifiers)

    for char_type in [ascii_lowercase, ascii_uppercase]:
        for cc in char_type.encode("ascii"):
            c = bytes([cc])
            if c not in replacements:
                replacements.add(c)
                break

    replacements = sorted(replacements, key=shortlex)
    targets = sorted(identifiers, key=shortlex, reverse=True)

    # TODO: This could use better parallelisation.
    for t in targets:
        pattern = re.compile(rb"\b" + t + rb"\b")
        source = problem.current_test_case
        if not pattern.search(source):
            continue

        async def can_replace(r):
            if shortlex(r) >= shortlex(t):
                return False
            attempt = pattern.sub(r, source)
            assert attempt != source
            return await problem.is_interesting(attempt)

        try:
            await problem.work.find_first_value(replacements, can_replace)
        except NotFound:
            pass


def iter_indices(s, substring):
    try:
        i = s.index(substring)
        yield i
        while True:
            i = s.index(substring, i + 1)
            yield i
    except ValueError:
        return


async def cut_comments(problem: ReductionProblem[bytes], start, end, include_end=True):
    """Remove comment-like regions bounded by start and end markers.

    Finds all regions starting with 'start' and ending with 'end', then
    tries to delete them. Used to remove comments from various languages.
    If include_end is False, the end marker itself is not deleted.
    """
    cuts = []
    target = problem.current_test_case
    # python comments
    for i in iter_indices(target, start):
        try:
            j = target.index(end, i + 1)
        except ValueError:
            if include_end:
                continue
            j = len(target)
        if include_end:
            cuts.append((i, j + len(end)))
        else:
            cuts.append((i, j))
    await delete_intervals(problem, cuts)


async def cut_comment_like_things(problem: ReductionProblem[bytes]):
    """Remove common comment syntaxes from source code.

    Tries to delete Python-style (#), C++-style (//), Python docstrings
    (triple quotes), and C-style block comments (/* ... */).
    """
    await cut_comments(problem, b"#", b"\n", include_end=False)
    await cut_comments(problem, b"//", b"\n", include_end=False)
    await cut_comments(problem, b'"""', b'"""')
    await cut_comments(problem, b"/*", b"*/")
