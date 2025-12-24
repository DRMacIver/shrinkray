"""UI abstractions for shrink ray."""

from abc import ABC
from typing import TYPE_CHECKING, Generic, TypeVar

import humanize
import trio
from attrs import define


if TYPE_CHECKING:
    from shrinkray.problem import BasicReductionProblem
    from shrinkray.state import ShrinkRayState

TestCase = TypeVar("TestCase")


@define(slots=False)
class ShrinkRayUI(Generic[TestCase], ABC):
    """Base class for shrink ray UI implementations."""

    state: "ShrinkRayState[TestCase]"

    @property
    def reducer(self):
        return self.state.reducer

    @property
    def problem(self) -> "BasicReductionProblem":
        return self.reducer.target  # type: ignore

    def install_into_nursery(self, nursery: trio.Nursery): ...

    async def run(self, nursery: trio.Nursery): ...


class BasicUI(ShrinkRayUI[TestCase]):
    """Simple text-based UI for non-interactive use."""

    async def run(self, nursery: trio.Nursery):
        initial = self.state.initial
        size = self.state.problem.size
        print(
            f"Starting reduction. Initial test case size: "
            f"{humanize.naturalsize(size(initial))}",
            flush=True,
        )
        prev_reduction = 0
        while True:
            initial = self.state.initial
            current = self.state.problem.current_test_case
            size = self.state.problem.size
            reduction = size(initial) - size(current)
            if reduction > prev_reduction:
                print(
                    f"Reduced test case to {humanize.naturalsize(size(current))} "
                    f"(deleted {humanize.naturalsize(reduction)}, "
                    f"{humanize.naturalsize(reduction - prev_reduction)} since last time)",
                    flush=True,
                )
                prev_reduction = reduction
                await trio.sleep(5)
            else:
                await trio.sleep(0.1)
