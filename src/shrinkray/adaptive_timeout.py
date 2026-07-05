"""Adaptive timeouts for interestingness tests.

Test runtime varies a lot over the course of a reduction: candidates
usually get faster as the test case shrinks, but some reductions make the
test dramatically slower (e.g. a loop that now takes many more iterations),
and parallel load adds variance on top. A fixed timeout is either too tight
(losing reductions to spurious timeouts) or too generous (wasting most of
the run waiting for candidates that will never finish).

AdaptiveTimeoutPolicy tracks recent test runtimes and continuously chooses
a timeout from them:

- The base timeout is a generous multiple of a high quantile of recent
  completed runtimes, with an extra floor based on the slowest recent
  *interesting* runs (interesting runs are the ones we can't afford to
  lose, and uninteresting candidates often fail much faster than
  interesting ones succeed).

- When reduction stalls while a significant fraction of runs are timing
  out, the policy explores upwards: it doubles the timeout (up to the
  user-specified maximum, or a hardcoded cap if none was given) to see
  whether the timeouts were hiding progress. Each doubling gets a budget
  of runs; if raising the timeout stops producing timeouts without
  producing reductions, or we run out of headroom, the timeout drops back
  down and exploration is paused until the next successful reduction.

- When the reducer runs out of things to try (`attempt_unstick`), any
  timeouts seen since the last reduction trigger the same upward
  exploration, so a reduction never ends while a raised timeout might
  still unlock progress.

Timeout results interact with result caching: a candidate that timed out
at 5s might succeed at 10s, so cached timeout-failures are only valid
while the current timeout is no larger than the timeout they ran under.
`cached_timeout_valid` implements that check for the caching layer.
"""

import math
import time
from collections import deque
from collections.abc import Callable


# Hard cap on the timeout when the user didn't specify one.
DEFAULT_TIMEOUT_CAP = 300.0
# Never set the timeout below this, to avoid pathological behaviour on
# very fast tests.
MIN_TIMEOUT = 1.0

# The base timeout is the larger of these two estimates, so it adapts to
# the common case while staying safe for slow interesting runs.
RUNTIME_MULTIPLIER = 10.0  # x the p90 of recent completed runtimes
INTERESTING_MULTIPLIER = 5.0  # x the slowest recent interesting runtime
RUNTIME_QUANTILE = 0.9

# Window sizes for recent measurements. Interesting runs are much rarer
# than ordinary completions, so their window is smaller to keep it recent.
RUNTIME_WINDOW = 200
INTERESTING_WINDOW = 20
OUTCOME_WINDOW = 100

# Stall-based exploration: don't explore until we have a meaningful amount
# of data showing a meaningful rate of timeouts, and enough time has
# passed since the last reduction (scaled up when the timeout itself is
# long, since everything happens more slowly then).
MIN_OUTCOMES_FOR_EXPLORATION = 20
TIMEOUT_RATE_THRESHOLD = 0.05
STALL_MIN_SECONDS = 20.0
STALL_TIMEOUT_MULTIPLIER = 3.0

# How many runs to allow at each raised timeout before deciding whether to
# raise it further or give up.
RUNG_RUN_BUDGET = 50


class AdaptiveTimeoutPolicy:
    """Chooses and adapts the timeout for interestingness test runs.

    All decisions are made from data reported through `record_completion`,
    `record_timeout` and `note_reduction`; the current choice is read with
    `current_timeout`. The clock is injectable for testing.
    """

    def __init__(
        self,
        *,
        user_timeout: float | None,
        clock: Callable[[], float] = time.monotonic,
        min_timeout: float = MIN_TIMEOUT,
        default_cap: float = DEFAULT_TIMEOUT_CAP,
        rung_budget: int = RUNG_RUN_BUDGET,
    ) -> None:
        # A user timeout of infinity means timeouts are disabled entirely,
        # in which case there is nothing to adapt.
        self.__enabled = user_timeout != math.inf
        if user_timeout is None:
            self.__cap = default_cap
        else:
            self.__cap = user_timeout
        self.__clock = clock
        self.__min_timeout = min_timeout
        self.__rung_budget = rung_budget

        self.__runtimes: deque[float] = deque(maxlen=RUNTIME_WINDOW)
        self.__interesting_runtimes: deque[float] = deque(maxlen=INTERESTING_WINDOW)
        # True for each recent run that timed out, False for completions.
        self.__timed_out_flags: deque[bool] = deque(maxlen=OUTCOME_WINDOW)

        # Exploration state: the current timeout is the base timeout
        # doubled once per level.
        self.__level = 0
        self.__runs_at_level = 0
        self.__timeouts_at_level = 0
        # Set when exploration has been tried and didn't help; cleared by
        # the next successful reduction.
        self.__exploration_exhausted = False
        # Timeouts seen since the last reduction that happened below the
        # cap (timeouts at the cap can't be helped by raising it).
        self.__timeouts_below_cap = 0
        self.__last_progress = clock()

    @property
    def enabled(self) -> bool:
        return self.__enabled

    @property
    def cap(self) -> float:
        return self.__cap

    @property
    def recent_timeout_rate(self) -> float:
        """Fraction of recent runs that timed out."""
        if not self.__timed_out_flags:
            return 0.0
        return sum(self.__timed_out_flags) / len(self.__timed_out_flags)

    def current_timeout(self) -> float:
        """The timeout to use for the next test run."""
        if not self.__enabled:
            return math.inf
        self.__evaluate_exploration()
        return self.__rung_timeout(self.__level)

    def record_completion(self, runtime: float, *, interesting: bool) -> None:
        """Record a test run that finished (however it exited)."""
        if not self.__enabled:
            return
        self.__runtimes.append(runtime)
        if interesting:
            self.__interesting_runtimes.append(runtime)
        self.__timed_out_flags.append(False)
        if self.__level > 0:
            self.__runs_at_level += 1

    def record_timeout(self, timeout_used: float) -> None:
        """Record a test run that was killed at `timeout_used` seconds."""
        if not self.__enabled:
            return
        self.__timed_out_flags.append(True)
        if timeout_used < self.__cap:
            self.__timeouts_below_cap += 1
        if self.__level > 0:
            self.__runs_at_level += 1
            self.__timeouts_at_level += 1

    def note_reduction(self) -> None:
        """Record that a successful reduction happened."""
        if not self.__enabled:
            return
        self.__last_progress = self.__clock()
        self.__timeouts_below_cap = 0
        self.__exploration_exhausted = False
        self.__reset_exploration()

    def attempt_unstick(self) -> bool:
        """Called when the reducer has run out of things to try.

        Returns True if the timeout was raised, in which case another
        round of reduction may now make progress (previously cached
        timeout-failures become invalid and will be retried).
        """
        if not self.__enabled or self.__exploration_exhausted:
            return False
        if self.__timeouts_below_cap > 0 and self.__rung_timeout(
            self.__level + 1
        ) > self.__rung_timeout(self.__level):
            self.__level += 1
            self.__runs_at_level = 0
            self.__timeouts_at_level = 0
            return True
        if self.__level > 0:
            self.__reset_exploration()
            self.__exploration_exhausted = True
        return False

    def cached_timeout_valid(self, timeout_used: float) -> bool:
        """Whether a cached timeout-failure is still valid.

        A run that timed out at `timeout_used` would still time out at any
        timeout no larger than that, but might succeed at a larger one.
        """
        if not self.__enabled:
            return True
        return self.current_timeout() <= timeout_used

    def reset(self) -> None:
        """Discard all learned state (used when restarting reduction)."""
        self.__runtimes.clear()
        self.__interesting_runtimes.clear()
        self.__timed_out_flags.clear()
        self.__timeouts_below_cap = 0
        self.__exploration_exhausted = False
        self.__reset_exploration()
        self.__last_progress = self.__clock()

    def __reset_exploration(self) -> None:
        self.__level = 0
        self.__runs_at_level = 0
        self.__timeouts_at_level = 0

    def __clamp(self, value: float) -> float:
        # The cap wins over the minimum: a user-specified maximum below
        # the normal minimum timeout must still be respected.
        return min(self.__cap, max(self.__min_timeout, value))

    def __base_timeout(self) -> float:
        """The timeout suggested by recent runtime measurements alone."""
        if not self.__runtimes:
            # No data: be maximally generous.
            return self.__cap
        sorted_runtimes = sorted(self.__runtimes)
        index = min(
            len(sorted_runtimes) - 1,
            math.ceil(RUNTIME_QUANTILE * len(sorted_runtimes)),
        )
        candidate = RUNTIME_MULTIPLIER * sorted_runtimes[index]
        if self.__interesting_runtimes:
            candidate = max(
                candidate, INTERESTING_MULTIPLIER * max(self.__interesting_runtimes)
            )
        return self.__clamp(candidate)

    def __rung_timeout(self, level: int) -> float:
        return self.__clamp(self.__base_timeout() * 2.0**level)

    def __stall_threshold(self) -> float:
        return max(STALL_MIN_SECONDS, STALL_TIMEOUT_MULTIPLIER * self.__rung_timeout(0))

    def __evaluate_exploration(self) -> None:
        """Advance the exploration state machine.

        Called whenever the current timeout is read, so decisions are made
        promptly (every test run reads the timeout before starting).
        """
        if self.__exploration_exhausted:
            return
        if self.__level == 0:
            stalled = self.__clock() - self.__last_progress > self.__stall_threshold()
            if (
                stalled
                and len(self.__timed_out_flags) >= MIN_OUTCOMES_FOR_EXPLORATION
                and self.recent_timeout_rate >= TIMEOUT_RATE_THRESHOLD
                and self.__rung_timeout(1) > self.__rung_timeout(0)
            ):
                self.__level = 1
                self.__runs_at_level = 0
                self.__timeouts_at_level = 0
        elif self.__runs_at_level >= self.__rung_budget:
            if self.__timeouts_at_level > 0 and self.__rung_timeout(
                self.__level + 1
            ) > self.__rung_timeout(self.__level):
                # Still timing out with headroom left: raise further.
                self.__level += 1
                self.__runs_at_level = 0
                self.__timeouts_at_level = 0
            else:
                # Raising the timeout didn't help: go back down and don't
                # try again until something else changes.
                self.__reset_exploration()
                self.__exploration_exhausted = True
