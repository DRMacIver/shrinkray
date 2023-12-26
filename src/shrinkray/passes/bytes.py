from collections import defaultdict, deque
from typing import Sequence

from attrs import define

from shrinkray.passes.patching import Cuts, Patches, apply_patches
from shrinkray.problem import Format, ReductionProblem


@define(frozen=True)
class Encoding(Format[bytes, str]):
    encoding: str

    def __repr__(self) -> str:
        return f"Encoding({repr(self.encoding)})"

    @property
    def name(self) -> str:
        return self.encoding

    def parse(self, input: bytes) -> str:
        return input.decode(self.encoding)

    def dumps(self, input: str) -> bytes:
        return input.encode(self.encoding)


@define(frozen=True)
class Split(Format[bytes, list[bytes]]):
    splitter: bytes

    def __repr__(self) -> str:
        return f"Split({repr(self.splitter)})"

    @property
    def name(self) -> str:
        return f"split({repr(self.splitter)})"

    def parse(self, input: bytes) -> list[bytes]:
        return input.split(self.splitter)

    def dumps(self, input: list[bytes]) -> bytes:
        return self.splitter.join(input)


def find_ngram_endpoints(value: bytes) -> list[tuple[int, list[int]]]:
    if len(set(value)) <= 1:
        return []
    queue: deque[tuple[int, Sequence[int]]] = deque([(0, range(len(value)))])
    results: list[tuple[int, list[int]]] = []

    while queue and len(results) < 10000:
        k, indices = queue.popleft()

        if k > 1:
            normalized: list[int] = []
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

        split: dict[int, list[int]] = defaultdict(list)
        for i in indices:
            try:
                split[value[i + k]].append(i)
            except IndexError:
                pass
        queue.extend([(k + 1, v) for v in split.values() if len(v) > 1])

    return results


def tokenize(text: bytes) -> list[bytes]:
    result: list[bytes] = []
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


async def lexeme_based_deletions(
    problem: ReductionProblem[bytes], min_size: int = 8
) -> None:
    intervals_by_k: dict[int, set[tuple[int, int]]] = defaultdict(set)

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
    shuffle: bool = False,
) -> None:
    await apply_patches(problem, Cuts(), [[t] for t in intervals_to_delete])


def brace_intervals(target: bytes, brace: bytes) -> list[tuple[int, int]]:
    open, close = brace
    intervals: list[tuple[int, int]] = []
    stack: list[int] = []
    for i, c in enumerate(target):
        if c == open:
            stack.append(i)
        elif c == close and stack:
            start = stack.pop() + 1
            end = i
            if end > start:
                intervals.append((start, end))
    return intervals


async def debrace(problem: ReductionProblem[bytes]) -> None:
    await apply_patches(
        problem,
        Cuts(),
        [
            [(u - 1, u), (v, v + 1)]
            for u, v in brace_intervals(problem.current_test_case, b"{}")
        ],
    )


def quote_intervals(target: bytes) -> list[tuple[int, int]]:
    indices: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(target):
        indices[c].append(i)

    intervals: list[tuple[int, int]] = []
    for quote in b"\"'":
        xs = indices[quote]
        for u, v in zip(xs, xs[1:], strict=False):
            if u + 1 < v:
                intervals.append((u + 1, v))
    return intervals


async def hollow(problem: ReductionProblem[bytes]) -> None:
    target = problem.current_test_case
    intervals: list[tuple[int, int]] = []
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


async def lift_braces(problem: ReductionProblem[bytes]) -> None:
    target = problem.current_test_case

    open_brace, close_brace = b"{}"
    start_stack: list[int] = []
    child_stack: list[list[tuple[int, int]]] = []

    results: list[tuple[int, int, list[tuple[int, int]]]] = []

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

    cuts: list[list[tuple[int, int]]] = []
    for start, end, children in results:
        for child_start, child_end in children:
            cuts.append([(start, child_start), (child_end, end)])

    await apply_patches(problem, Cuts(), cuts)


@define(frozen=True)
class Tokenize(Format[bytes, list[bytes]]):
    def __repr__(self) -> str:
        return "tokenize"

    @property
    def name(self) -> str:
        return "tokenize"

    def parse(self, input: bytes) -> list[bytes]:
        return tokenize(input)

    def dumps(self, input: list[bytes]) -> bytes:
        return b"".join(input)


async def delete_byte_spans(problem: ReductionProblem[bytes]) -> None:
    indices: dict[int, list[int]] = defaultdict(list)
    target = problem.current_test_case
    for i, c in enumerate(target):
        indices[c].append(i)

    spans: list[tuple[int, int]] = []

    for c, ix in sorted(indices.items()):
        if len(ix) > 1:
            spans.append((0, ix[0] + 1))
            spans.extend(zip(ix, ix[1:]))
            spans.append((ix[-1], len(target)))

    await apply_patches(problem, Cuts(), [[s] for s in spans])


async def remove_indents(problem: ReductionProblem[bytes]) -> None:
    target = problem.current_test_case
    spans: list[list[tuple[int, int]]] = []

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


async def remove_whitespace(problem: ReductionProblem[bytes]) -> None:
    target = problem.current_test_case
    spans: list[list[tuple[int, int]]] = []

    for i, c in enumerate(target):
        char = bytes([c])
        if char.isspace():
            j = i + 1
            while j < len(target) and target[j : j + 1].isspace():
                j += 1

            if j > i + 1:
                spans.append([(i, j)])
            if j > i + 2:
                spans.append([(i + 1, j)])

    await apply_patches(problem, Cuts(), spans)


class NewlineReplacer(Patches[frozenset[int], bytes]):
    @property
    def empty(self) -> frozenset[int]:
        return frozenset()

    def combine(self, *patches: frozenset[int]) -> frozenset[int]:
        result: set[int] = set()
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


async def replace_space_with_newlines(problem: ReductionProblem[bytes]) -> None:
    await apply_patches(
        problem,
        NewlineReplacer(),
        [
            frozenset({i})
            for i, c in enumerate(problem.current_test_case)
            if c in b" \t"
        ],
    )
