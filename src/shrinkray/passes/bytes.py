from collections import defaultdict, deque
import math
from typing import Iterator

import chardet
from attrs import define

from shrinkray.passes.sequences import sequence_passes, single_backward_delete
from shrinkray.passes.strings import string_passes
from shrinkray.problem import Format, ReductionProblem
from shrinkray.reducer import ReductionPass, compose
from shrinkray.work import parallel_map

from shrinkray.work import NotFound
import numpy as np
from sklearn import tree


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


class IntervalApplier:
    def __init__(self, problem: ReductionProblem, intervals: list[tuple[int, int]]):
        self.problem = problem
        self.intervals = list(map(list, intervals))
        self.applied_deletions = []
        self.target = problem.current_test_case
        self.current = self.target
        self.attempted = []
        self.attempted_results = []
        self.offset = 0
        self.batch_size = 100

    def with_extra_deletions(self, deletions):
        deletions = self.applied_deletions + list(deletions)
        normalized = []
        for start, end in sorted(deletions):
            if normalized and normalized[-1][-1] >= start:
                normalized[-1][-1] = max(normalized[-1][-1], end)
            else:
                normalized.append([start, end])
        result = bytearray()
        prev = 0
        for start, end in normalized:
            result.extend(self.target[prev:start])
            prev = end
        result.extend(self.target[prev:])
        return bytes(result)

    def apply_deletions(self, deletions):
        self.current = self.with_extra_deletions(deletions)
        self.applied_deletions.extend(deletions)

    async def can_apply(self, deletions):
        attempt = self.with_extra_deletions(deletions)
        if attempt == self.current:
            return False
        return await self.problem.is_interesting(attempt)

    async def can_apply_single(self, deletion):
        return await self.can_apply([deletion])

    async def consider_batch(self, batch):
        batch = list(batch)

        self.offset = 0
        async with parallel_map(
            batch, self.can_apply_single, parallelism=self.problem.work.parallelism
        ) as fetch_results:
            results = []
            async for r in fetch_results:
                results.append(r)
                self.offset = len(results)

        self.attempted.extend(batch)
        self.attempted_results.extend(results)

        good_intervals = [b for b, r in zip(batch, results) if r]

        if not good_intervals:
            print("Dead batch")
            self.batch_size *= 2
            return

        to_merge = len(good_intervals)

        if await self.can_apply(good_intervals):
            merged = len(good_intervals)
            self.apply_deletions(good_intervals)
        else:
            good_intervals.sort(key=lambda t: t[1] - t[0], reverse=True)
            merged = 0
            while good_intervals:

                async def can_apply_prefix(k):
                    if k > len(good_intervals):
                        return False
                    return await self.can_apply(good_intervals[:k])

                k = await self.problem.work.find_large_integer(can_apply_prefix)
                merged += k
                self.apply_deletions(good_intervals[:k])
                good_intervals = good_intervals[k + 1 :]
        print(f"Successfully merged {merged} / {to_merge} good intervals")
        self.batch_size = math.ceil(len(batch) * merged / to_merge)

    def normalize_intervals(self):
        self.intervals = [
            d
            for d in self.intervals
            if all(
                start >= d[1] or end <= d[0] for start, end in self.applied_deletions
            )
        ]

    async def run(self):
        initial_size = len(self.intervals)
        async with self.problem.work.pb(
            total=lambda: initial_size,
            current=lambda: initial_size - len(self.intervals) + self.offset,
            desc="Intervals considered",
        ):
            rnd = self.problem.work.random

            while self.intervals and not self.applied_deletions:
                rnd.shuffle(self.intervals)
                await self.consider_batch(self.intervals[-self.batch_size :])
                del self.intervals[-self.batch_size :]

            self.normalize_intervals()

            self.intervals.sort(key=lambda t: t[1] - t[0])
            await self.consider_batch(self.intervals[-self.batch_size :])
            del self.intervals[-self.batch_size :]

            while self.intervals:
                self.normalize_intervals()

                if self.batch_size >= len(self.intervals):
                    await self.consider_batch(self.intervals)
                    self.intervals.clear()
                    return

                classifier = tree.DecisionTreeClassifier(
                    min_samples_leaf=10,
                )
                classifier.fit(self.attempted, self.attempted_results)

                predictions = classifier.predict_proba(self.intervals)[:, 1]

                if not (predictions > 0).any():
                    rnd.shuffle(self.intervals)
                    await self.consider_batch(self.intervals[-self.batch_size :])
                    del self.intervals[-self.batch_size :]
                    continue

                p = predictions.mean()

                print(predictions)
                self.problem.work.note(
                    f"Estimate {p * 100:.2f}% of remaining intervals are deletable."
                )

                assert len(predictions) == len(self.intervals)

                scores = {
                    (u, v): (v - u) * p
                    for (u, v), p in zip(self.intervals, predictions)
                }

                self.intervals.sort(key=lambda t: scores[tuple(t)])

                await self.consider_batch(self.intervals[-self.batch_size :])
                del self.intervals[-self.batch_size :]


async def delete_intervals(
    problem: ReductionProblem[bytes], intervals_to_delete: list[tuple[int, int]]
) -> None:
    await IntervalApplier(problem, intervals_to_delete).run()


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
            if end > start:
                intervals.append((start, end))

    await delete_intervals(problem, intervals)


def byte_passes(problem: ReductionProblem[bytes]) -> Iterator[ReductionPass[bytes]]:
    yield lexeme_based_deletions
    yield hollow_braces
    yield compose(Split(b"\n"), single_backward_delete)
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
