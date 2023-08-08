from collections import defaultdict, deque
from typing import Iterator

import chardet
from attrs import define

from shrinkray.passes.sequences import sequence_passes, single_backward_delete
from shrinkray.passes.strings import string_passes
from shrinkray.problem import Format, ReductionProblem
from shrinkray.reducer import ReductionPass, compose

from shrinkray.work import NotFound


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

        if k > 0 and (indices[0] == 0 or len(set(value[i - 1] for i in indices)) > 1):
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


MAX_DELETE_INTERVAL = 8


async def lexeme_based_deletions(problem: ReductionProblem[bytes]) -> None:
    intervals_by_k = defaultdict(set)

    for k, endpoints in find_ngram_endpoints(problem.current_test_case):
        intervals_by_k[k].update(zip(endpoints, endpoints[1:]))

    intervals_to_delete = [
        t
        for _, intervals in sorted(intervals_by_k.items(), reverse=True)
        for t in sorted(intervals, key=lambda t: (t[1] - t[0], t[0]), reverse=True)
        if t[1] - t[0] > MAX_DELETE_INTERVAL
    ]

    await delete_intervals(problem, intervals_to_delete)


async def delete_intervals(
    problem: ReductionProblem[bytes], intervals_to_delete: list[tuple[int, int]]
) -> None:
    target = problem.current_test_case
    applied_deletions = []

    def with_extra_deletions(deletions):
        deletions = applied_deletions + list(deletions)
        normalized = []
        for start, end in sorted(deletions):
            if normalized and normalized[-1][-1] >= start:
                normalized[-1][-1] = max(normalized[-1][-1], end)
            else:
                normalized.append([start, end])
        result = bytearray()
        prev = 0
        for start, end in normalized:
            result.extend(target[prev:start])
            prev = end
        result.extend(target[prev:])
        return bytes(result)

    initial_size = len(intervals_to_delete)

    offset = 0
    async with problem.work.pb(
        total=lambda: initial_size,
        current=lambda: initial_size - len(intervals_to_delete) + offset,
        desc="Intervals considered",
    ):
        while True:

            async def delete_initial(k):
                if k > len(intervals_to_delete):
                    return False
                return await problem.is_interesting(
                    with_extra_deletions(intervals_to_delete[:k])
                )

            k = await problem.work.find_large_integer(delete_initial)
            problem.work.debug(f"k={k}")
            if k > 0:
                applied_deletions.extend(intervals_to_delete[:k])
                intervals_to_delete = intervals_to_delete[k + 1 :]

            offset = 0
            try:

                def check_interesting(i):
                    nonlocal offset
                    offset = max(offset, i)
                    return problem.is_interesting(
                        with_extra_deletions([intervals_to_delete[i]])
                    )

                i = await problem.work.find_first_value(
                    range(len(intervals_to_delete)),
                    check_interesting,
                )
            except NotFound:
                return

            deleted = intervals_to_delete[i]

            applied_deletions.append(deleted)

            intervals_to_delete = [
                t
                for t in intervals_to_delete[i:]
                if t[0] >= deleted[1] or t[1] <= deleted[0]
            ]

            intervals_to_delete.sort(
                key=lambda t: t[1] == deleted[0] or t[0] == deleted[1], reverse=True
            )


async def hollow_braces(problem: ReductionProblem[bytes]):
    target = problem.current_test_case
    open, close = b"{}"

    intervals = []
    stack = []
    for i, c in enumerate(target):
        if c == open:
            stack.append(i)
        elif c == close and stack:
            start = stack.pop() + 1
            end = i
            intervals.append((start, end))

    await delete_intervals(problem, intervals)


def byte_passes(problem: ReductionProblem[bytes]) -> Iterator[ReductionPass[bytes]]:
    yield hollow_braces
    yield compose(Split(b"\n"), single_backward_delete)
    # yield lexeme_based_deletions
    return
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
