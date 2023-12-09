import math
from collections import Counter, defaultdict, deque
from typing import Iterator

import trio
from attrs import define

from shrinkray.passes.sequences import delete_elements
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


def merged_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized = []
    for start, end in sorted(map(tuple, intervals)):
        if normalized and normalized[-1][-1] >= start:
            normalized[-1][-1] = max(normalized[-1][-1], end)
        else:
            normalized.append([start, end])
    return list(map(tuple, normalized))


def with_deletions(target: bytes, deletions: list[tuple[int, int]]) -> bytes:
    result = bytearray()
    prev = 0
    total_deleted = 0
    for start, end in deletions:
        total_deleted += end - start
        result.extend(target[prev:start])
        prev = end
    result.extend(target[prev:])
    assert len(result) + total_deleted == len(target)
    return bytes(result)


class CutTarget:
    def __init__(
        self, problem: ReductionProblem[bytes], intervals: list[tuple[int, int]]
    ):
        self.problem = problem
        self.target = problem.current_test_case
        self.intervals = list(map(tuple, intervals))
        self.intervals.sort()

        self.intervals_by_content = defaultdict(list)
        for u, v in intervals:
            self.intervals_by_content[self.target[u:v]].append([u, v])

        self.applied = []
        self.generation = 0
        self.attempted = {}

    def is_redundant(self, interval: tuple[int, int]) -> bool:
        if not self.applied:
            return False
        interval = tuple(interval)
        if interval in self.attempted:
            return True
        start, end = interval
        assert start < end

        if start >= self.applied[-1][0]:
            return end <= self.applied[-1][-1]

        if start < self.applied[0][0]:
            return False

        lo = 0
        hi = len(self.applied) - 1

        # Invariant: start is >= start of lo and < start of hi
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if start >= self.applied[mid][0]:
                lo = mid
            else:
                hi = mid

        return end <= self.intervals[lo][1]

    def similar_cuts(self, cut):
        return [
            s
            for s in self.intervals_by_content[self.target[cut[0] : cut[1]]]
            if s != cut
        ]

    async def try_apply(self, interval: tuple[int, int]) -> bool:
        interval = tuple(interval)
        while True:
            if self.is_redundant(interval):
                return False

            generation_at_start = self.generation

            merged = merged_intervals(self.applied + [interval])
            attempt = with_deletions(self.target, merged)

            succeeded = await self.problem.is_interesting(attempt)

            if not succeeded:
                self.attempted[interval] = False
                return False

            if self.generation == generation_at_start:
                self.generation += 1
                self.applied = merged
                self.attempted[interval] = True
                return True

    async def try_merge(self, intervals: list[tuple[int, int]]) -> bool:
        while True:
            generation_at_start = self.generation

            merged = merged_intervals(self.applied + intervals)
            if merged == self.applied:
                return True
            attempt = with_deletions(self.target, merged)

            succeeded = await self.problem.is_interesting(attempt)

            if not succeeded:
                return False

            if self.generation == generation_at_start:
                self.generation += 1
                self.applied = merged
                return True

            # Sleep for a bit to give whichever other task is making progress
            # that's stomping on ours time to do its thing.
            await trio.sleep(self.problem.work.random.expovariate(1))

    def intervals_touching(self, interval: tuple[int, int]) -> list[tuple[int, int]]:
        start, end = interval
        # TODO: Use binary search to cut this down to size.
        return [t for t in self.intervals if not (t[1] < start or t[0] > end)]


async def delete_intervals(
    problem: ReductionProblem[bytes],
    intervals_to_delete: list[tuple[int, int]],
    shuffle=False,
) -> None:
    intervals_considered = 0

    async with trio.open_nursery() as nursery:
        cuts = CutTarget(problem, intervals_to_delete)
        send_intervals, receive_intervals = trio.open_memory_channel(1000)

        async def fill_queue():
            nonlocal intervals_considered

            intervals = list(cuts.intervals)
            if shuffle:
                problem.work.random.shuffle(intervals)
            else:
                intervals.sort(key=lambda t: (t[1] - t[0], t[0]), reverse=True)

            for interval in intervals:
                if cuts.is_redundant(interval):
                    intervals_considered += 1
                else:
                    await send_intervals.send(interval)

            send_intervals.close()

        nursery.start_soon(fill_queue)

        async def apply_intervals():
            nonlocal intervals_considered
            while True:
                try:
                    interval = await receive_intervals.receive()
                except trio.EndOfChannel:
                    return
                intervals_considered += 1
                if await cuts.try_apply(interval):
                    similar = cuts.similar_cuts(interval)
                    if similar:
                        await cuts.try_merge(similar)

        async with trio.open_nursery() as sub_nursery:
            for _ in range(problem.work.parallelism * 2):
                sub_nursery.start_soon(apply_intervals)

        nursery.cancel_scope.cancel()


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


def quote_intervals(target: bytes) -> list[tuple[int, int]]:
    indices = defaultdict(list)
    for i, c in enumerate(target):
        indices[c].append(i)

    intervals = []
    for quote in b"\"'":
        xs = indices[quote]
        for u, v in zip(xs, xs[1:], strict=False):
            intervals.append((u + 1, v))
    return intervals


async def hollow(problem: ReductionProblem[bytes]):
    target = problem.current_test_case
    await delete_intervals(
        problem,
        brace_intervals(target, b"{}")
        + brace_intervals(target, b"[]")
        + quote_intervals(target),
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


class SimpleCutTarget:
    def __init__(self, problem: ReductionProblem[bytes]):
        self.problem = problem
        self.target = problem.current_test_case

        self.applied = []
        self.generation = 0

    async def try_merge(self, intervals: list[tuple[int, int]]) -> bool:
        while True:
            generation_at_start = self.generation

            merged = merged_intervals(self.applied + intervals)
            if merged == self.applied:
                return True
            attempt = with_deletions(self.target, merged)

            succeeded = await self.problem.is_interesting(attempt)

            if not succeeded:
                return False

            if self.generation == generation_at_start:
                self.generation += 1
                self.applied = merged
                return True

            # Sleep for a bit to give whichever other task is making progress
            # that's stomping on ours time to do its thing.
            await trio.sleep(self.problem.work.random.expovariate(1))


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

    target = SimpleCutTarget(problem)

    async with trio.open_nursery() as nursery:
        async with trio.open_nursery() as nursery:
            send_cuts, receive_cuts = trio.open_memory_channel(max_buffer_size=math.inf)
            sent_any = False
            for start, end, children in results:
                for child_start, child_end in children:
                    sent_any = True
                    await send_cuts.send([(start, child_start), (child_end, end)])

            send_cuts.close()

            if sent_any:

                async def consumer():
                    try:
                        while True:
                            cuts = await receive_cuts.receive()
                            await target.try_merge(cuts)
                    except trio.EndOfChannel:
                        pass

                for _ in range(problem.work.parallelism):
                    nursery.start_soon(consumer)


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


def byte_passes(problem: ReductionProblem[bytes]) -> Iterator[ReductionPass[bytes]]:
    yield hollow

    high_prio_splitters = [b"\n", b";"]
    for split in high_prio_splitters:
        yield compose(Split(split), delete_elements)

    yield lift_braces

    yield compose(Tokenize(), delete_elements)

    all_bytes = Counter(problem.current_test_case)

    for s in sorted(all_bytes, key=all_bytes.__getitem__):
        split = bytes([s])
        if split not in high_prio_splitters:
            yield compose(Split(split), delete_elements)

    yield lexeme_based_deletions
    yield short_deletions
