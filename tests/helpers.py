import random
import subprocess
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import TypeVar

import trio
from attrs import define

from shrinkray.passes.clangdelta import find_clang_delta
from shrinkray.passes.definitions import ReductionPass, ReductionPump
from shrinkray.passes.python import is_python
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import Reducer, ShrinkRay
from shrinkray.state import sort_key_for_initial
from shrinkray.work import WorkContext


@define
class BasicReducer[T](Reducer[T]):
    """Simple reducer for testing - runs passes in sequence until no progress."""

    reduction_passes: Iterable[ReductionPass[T]]
    pumps: Iterable[ReductionPump[T]] = ()
    _status: str = "Starting up"

    def __attrs_post_init__(self) -> None:
        self.reduction_passes = list(self.reduction_passes)

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value

    async def run_pass(self, rp: ReductionPass[T]) -> None:
        await rp(self.target)

    async def run(self) -> None:
        await self.target.setup()

        while True:
            prev = self.target.current_test_case
            for rp in self.reduction_passes:
                self.status = f"Running reduction pass {rp.__name__}"
                await self.run_pass(rp)
            for pump in self.pumps:
                self.status = f"Pumping with {pump.__name__}"
                pumped = await pump(self.target)
                if pumped != self.target.current_test_case:
                    with self.backtrack(pumped):
                        for rp in self.reduction_passes:
                            self.status = f"Running reduction pass {rp.__name__} under pump {pump.__name__}"
                            await self.run_pass(rp)
            if prev == self.target.current_test_case:
                return


@lru_cache(maxsize=1)
def clang_delta_works() -> bool:
    """Check if clang_delta can actually execute.

    This verifies not just that the binary exists, but that it can run.
    On some systems (e.g., Ubuntu 24.04), creduce is installed but
    clang_delta fails at runtime due to shared library issues.
    """
    clang_delta = find_clang_delta()
    if not clang_delta:
        return False
    try:
        # Run a simple test to verify clang_delta works
        result = subprocess.run(
            [clang_delta, "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


T = TypeVar("T")


def reduce_with(
    rp: Iterable[ReductionPass[T]],
    initial: T,
    is_interesting: Callable[[T], bool],
    parallelism: int = 1,
    sort_key=None,
) -> T:
    if sort_key is None:
        sort_key = sort_key_for_initial(initial)

    async def acondition(x: T) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> T:
        problem: BasicReductionProblem[T] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
            sort_key=sort_key,
        )

        reducer = BasicReducer(
            target=problem,
            reduction_passes=rp,
        )

        await reducer.run()

        return problem.current_test_case

    return trio.run(calc_result)


def reduce(
    initial: bytes,
    is_interesting: Callable[[bytes], bool],
    parallelism: int = 1,
    sort_key=None,
) -> bytes:
    if sort_key is None:
        sort_key = sort_key_for_initial(initial)

    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> bytes:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
            sort_key=sort_key,
        )

        reducer = ShrinkRay(
            target=problem,
        )

        await reducer.run()

        return problem.current_test_case

    return trio.run(calc_result)


def assert_no_blockers(
    is_interesting: Callable[[bytes], bool],
    potential_blockers: list[bytes],
    lower_bounds=random.sample(range(1000), 12),  # noqa: B008
):
    potential_blockers = sorted(set(potential_blockers), key=lambda s: (len(s), s))

    for lower_bound in lower_bounds:

        async def acondition(x: bytes) -> bool:
            await trio.lowlevel.checkpoint()
            if len(x) < lower_bound:
                return False
            return is_interesting(x)

        current_best = None

        max_blockers = 1000
        candidates = [
            candidate
            for candidate in potential_blockers
            if len(candidate) >= lower_bound
        ]

        if len(candidates) > max_blockers:
            candidates = random.sample(candidates, max_blockers)
            candidates.sort(key=lambda s: (len(s), s))

        for initial in candidates:
            if len(initial) < lower_bound or not is_interesting(initial):
                continue
            problem: BasicReductionProblem[bytes] = BasicReductionProblem(
                initial=initial,
                is_interesting=acondition,
                work=WorkContext(parallelism=1),
            )

            async def calc_result() -> bytes:
                reducer = ShrinkRay(
                    target=problem,
                )

                await reducer.run()

                return problem.current_test_case

            result = trio.run(calc_result)

            if current_best is None:
                current_best = result
            elif result != current_best:
                current_best, result = sorted(
                    (result, current_best), key=problem.sort_key
                )

                raise AssertionError(
                    f"With lower bound {lower_bound}, {result} does not reduce to {current_best}"
                )


def direct_reductions(
    origin: bytes, *, parallelism=1, sort_key=None, passes=None
) -> set[bytes]:
    children = set()
    if sort_key is None:
        sort_key = sort_key_for_initial(origin)

    def is_interesting(b: bytes) -> bool:
        if sort_key(b) < sort_key(origin):
            children.add(b)
        return b == origin

    if passes is not None:
        reduce_with(
            passes,
            origin,
            is_interesting,
            parallelism=parallelism,
            sort_key=sort_key,
        )
    else:
        reduce(origin, is_interesting, parallelism=parallelism, sort_key=sort_key)

    return children


def assert_reduces_to(
    *,
    origin: bytes,
    target: bytes,
    parallelism=1,
    language_restrictions=True,
    passes=None,
    sort_key=None,
):
    if origin == target:
        raise AssertionError("A value cannot reduce to itself")
    if sort_key is None:
        sort_key = sort_key_for_initial(origin)
    if sort_key(origin) < sort_key(target):
        raise AssertionError(
            f"It is impossible for {origin} to reduce to {target} as it is more reduced."
        )

    if language_restrictions and is_python(origin) and is_python(target):
        require_python = True
    else:
        require_python = False

    def is_interesting(value: bytes) -> bool:
        if require_python and not is_python(value):
            return False
        return sort_key(value) >= sort_key(target)

    if passes is None:
        best = reduce(
            origin,
            is_interesting,
            parallelism=parallelism,
            sort_key=sort_key,
        )
    else:
        best = reduce_with(
            passes,
            origin,
            is_interesting,
            parallelism=parallelism,
            sort_key=sort_key,
        )

    if best == target:
        return

    if best == origin:
        raise AssertionError(f"Unable to make any progress from {origin}")
    raise AssertionError(
        f"Unable to reduce {origin} to {target}. Best achieve was {best}"
    )
