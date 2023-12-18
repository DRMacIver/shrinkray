import math
from collections import Counter, defaultdict, deque
from typing import Iterator

import trio
from attrs import define

from shrinkray.passes.sequences import delete_elements
from shrinkray.problem import Format, ReductionProblem
from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.patching import Patches, apply_patches, Cuts


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


@define(frozen=True)
class Split(Format[bytes, list[bytes]]):
    splitter: bytes

    def __repr__(self) -> bytes:
        return f"Split({repr(self.splitter)})"

    @property
    def name(self) -> bytes:
        return f"split({repr(self.splitter)})"

    def parse(self, value: bytes) -> list[bytes]:
        return value.split(self.splitter)

    def dumps(self, value: list[bytes]) -> bytes:
        return self.splitter.join(value)


def find_ngram_endpoints(value: bytes) -> list[list[int]]:
    queue = deque([(0, range(len(value)))])
    results = []

    while queue and len(results) < 10000:
        k, indices = queue.popleft()

        if k > 1:
            normalized = []
            for i in indices:
                if not normalized or i >= normalized[-1] + k:
                    normalized.append(i)
            indices = normalized

        while (
            indices[-1] + k < len(value) and len({value[i + k] for i in indices}) == 1
        ):
            k += 1

        if k > 0 and (indices[0] == 0 or len({value[i - 1] for i in indices}) > 1):
            assert isinstance(indices, list)
            results.append((k, indices))

        split = defaultdict(list)
        for i in indices:
            try:
                split[value[i + k]].append(i)
            except IndexError:
                pass
        queue.extend([(k + 1, v) for v in split.values() if len(v) > 1])

    return results


def tokenize(text: bytes) -> list[bytes]:
    result = []
    i = 0
    while i < len(text):
        c = bytes([text[i]])
        j = i + 1
        if b"A" <= c <= b"z":
            while j < len(text) and (
                b"A"[0] <= text[j] <= b"z"[0]
                or text[j] == b"_"[0]
                or b"0"[0] <= text[j] <= b"9"[0]
            ):
                j += 1
        elif b"0" <= c <= b"9":
            while j < len(text) and (
                text[j] == b"."[0] or b"0"[0] <= text[j] <= b"9"[0]
            ):
                j += 1
        elif c == b" ":
            while j < len(text) and (text[j] == b" "[0]):
                j += 1
        result.append(text[i:j])
        i = j
    assert b"".join(result) == text
    return result


MAX_DELETE_INTERVAL = 8


async def lexeme_based_deletions(problem: ReductionProblem[bytes], min_size=8) -> None:
    intervals_by_k = defaultdict(set)

    for k, endpoints in find_ngram_endpoints(problem.current_test_case):
        intervals_by_k[k].update(zip(endpoints, endpoints[1:]))

    intervals_to_delete = [
        t
        for _, intervals in sorted(intervals_by_k.items(), reverse=True)
        for t in sorted(intervals, key=lambda t: (t[1] - t[0], t[0]), reverse=True)
        if t[1] - t[0] >= min_size
    ]

    await delete_intervals(problem, intervals_to_delete, shuffle=True)


async def delete_intervals(
    problem: ReductionProblem[bytes],
    intervals_to_delete: list[tuple[int, int]],
    shuffle=False,
) -> None:
    await apply_patches(problem, Cuts(), [[t] for t in intervals_to_delete])


def brace_intervals(target: bytes, brace: bytes) -> list[tuple[int, int]]:
    open, close = brace
    intervals = []
    stack = []
    for i, c in enumerate(target):
        if c == open:
            stack.append(i)
        elif c == close and stack:
            start = stack.pop() + 1
            end = i
            if end > start:
                intervals.append((start, end))
    return intervals


async def debrace(problem: ReductionProblem[bytes]):
    await apply_patches(
        problem,
        Cuts(),
        [
            [(u - 1, u), (v, v + 1)]
            for u, v in brace_intervals(problem.current_test_case, b"{}")
        ],
    )


def quote_intervals(target: bytes) -> list[tuple[int, int]]:
    indices = defaultdict(list)
    for i, c in enumerate(target):
        indices[c].append(i)

    intervals = []
    for quote in b"\"'":
        xs = indices[quote]
        for u, v in zip(xs, xs[1:], strict=False):
            if u + 1 < v:
                intervals.append((u + 1, v))
    return intervals


async def hollow(problem: ReductionProblem[bytes]):
    target = problem.current_test_case
    intervals = []
    for b in [
        quote_intervals(target),
        brace_intervals(target, b"[]"),
        brace_intervals(target, b"{}"),
    ]:
        b.sort(key=lambda t: (t[1] - t[0], t[0]))
        intervals.extend(b)
    await delete_intervals(
        problem,
        intervals,
    )


async def short_deletions(problem: ReductionProblem[bytes]) -> None:
    target = problem.current_test_case
    await delete_intervals(
        problem,
        [
            (i, j)
            for i in range(len(target))
            for j in range(i + 1, min(i + 11, len(target) + 1))
        ],
    )


async def lift_braces(problem: ReductionProblem[bytes]):
    target = problem.current_test_case

    open_brace, close_brace = b"{}"
    start_stack = []
    child_stack = []

    results = []

    for i, c in enumerate(target):
        if c == open_brace:
            start_stack.append(i)
            child_stack.append([])
        elif c == close_brace and start_stack:
            start = start_stack.pop() + 1
            end = i
            children = child_stack.pop()
            if child_stack:
                child_stack[-1].append((start, end))
            if end > start:
                results.append((start, end, children))

    cuts = []
    for start, end, children in results:
        for child_start, child_end in children:
            cuts.append([(start, child_start), (child_end, end)])

    await apply_patches(problem, Cuts(), cuts)


@define(frozen=True)
class Tokenize(Format[bytes, list[bytes]]):
    def __repr__(self) -> bytes:
        return "tokenize"

    @property
    def name(self) -> bytes:
        return "tokenize"

    def parse(self, value: bytes) -> list[bytes]:
        return tokenize(value)

    def dumps(self, value: list[bytes]) -> bytes:
        return b"".join(value)


async def delete_byte_spans(problem: ReductionProblem[bytes]):
    indices = defaultdict(list)
    target = problem.current_test_case
    for i, c in enumerate(target):
        indices[c].append(i)

    spans = []

    for c, ix in sorted(indices.items()):
        if len(ix) > 1:
            spans.append((0, ix[0] + 1))
            spans.extend(zip(ix, ix[1:]))
            spans.append((ix[-1], len(target)))

    await apply_patches(problem, Cuts(), [[s] for s in spans])


async def remove_indents(problem: ReductionProblem[bytes]):
    target = problem.current_test_case
    spans = []

    newline = ord(b"\n")
    space = ord(b" ")

    for i, c in enumerate(target):
        if c == newline:
            j = i + 1
            while j < len(target) and target[j] == space:
                j += 1

            if j > i + 1:
                spans.append([(i + 1, j)])

    await apply_patches(problem, Cuts(), spans)


async def remove_whitespace(problem: ReductionProblem[bytes]):
    target = problem.current_test_case
    spans = []

    for i, c in enumerate(target):
        c = bytes([c])
        if c.isspace():
            j = i + 1
            while j < len(target) and target[j : j + 1].isspace():
                j += 1

            if j > i + 1:
                spans.append([(i, j)])
            if j > i + 2:
                spans.append([(i + 1, j)])

    await apply_patches(problem, Cuts(), spans)


class NewlineReplacer(Patches[frozenset[int], bytes]):
    def empty(self):
        return frozenset()

    def combine(self, *patches: frozenset[int]) -> frozenset[int]:
        result = set()
        for p in patches:
            result.update(p)
        return frozenset(result)

    def apply(self, patch: frozenset[int], target: bytes) -> bytes:
        result = bytearray()

        for i, c in enumerate(target):
            if i in patch:
                result.extend(b"\n")
            else:
                result.append(c)
        return bytes(result)

    def size(self, patch: frozenset[int]) -> int:
        return len(patch)


async def replace_space_with_newlines(problem: ReductionProblem[bytes]):
    await apply_patches(
        problem,
        NewlineReplacer(),
        [
            frozenset({i})
            for i, c in enumerate(problem.current_test_case)
            if c in b" \t"
        ],
    )
