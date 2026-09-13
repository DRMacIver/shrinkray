"""UI abstractions for shrink ray."""

from abc import ABC
from typing import TYPE_CHECKING

import humanize
import trio
from attrs import define


if TYPE_CHECKING:
    from shrinkray.problem import BasicReductionProblem
    from shrinkray.state import ShrinkRayState


@define(slots=False)
class ShrinkRayUI[TestCase](ABC):
    """Base class for shrink ray UI implementations."""

    state: "ShrinkRayState[TestCase]"

    @property
    def reducer(self):
        return self.state.reducer

    @property
    def problem(self) -> "BasicReductionProblem":
        return self.reducer.target  # type: ignore

    def install_into_nursery(self, nursery: trio.Nursery):  # noqa: B027
        """Install any background tasks into the nursery.

        Subclasses may override this to add tasks that run alongside reduction.
        The default implementation does nothing.
        """

    async def run(self, nursery: trio.Nursery):  # noqa: B027
        """Run the UI update loop.

        Subclasses may override this to display progress or status updates.
        The default implementation does nothing.
        """


class BasicUI[TestCase](ShrinkRayUI[TestCase]):
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
        announced_nondeterminism = False
        while True:
            initial = self.state.initial
            problem = self.state.problem
            current = problem.current_test_case
            size = problem.size
            reduction = size(initial) - size(current)
            if problem.nondeterministic and not announced_nondeterminism:
                announced_nondeterminism = True
                print(
                    "Nondeterministic interestingness test detected: candidates "
                    "are now confirmed by repeated runs before being adopted.",
                    flush=True,
                )
            if reduction < prev_reduction:
                print(
                    f"Backtracked to a test case of {humanize.naturalsize(size(current))} "
                    "that reproduces the bug more reliably",
                    flush=True,
                )
                prev_reduction = reduction
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
