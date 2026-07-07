"""Tests for the adaptive timeout policy."""

import math

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from shrinkray.adaptive_timeout import (
    DEFAULT_TIMEOUT_CAP,
    MIN_OUTCOMES_FOR_EXPLORATION,
    MIN_TIMEOUT,
    RUNG_RUN_BUDGET,
    STALL_MIN_SECONDS,
    AdaptiveTimeoutPolicy,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_policy(user_timeout=None, **kwargs) -> tuple[AdaptiveTimeoutPolicy, FakeClock]:
    clock = FakeClock()
    policy = AdaptiveTimeoutPolicy(user_timeout=user_timeout, clock=clock, **kwargs)
    return policy, clock


def record_fast_run_with_timeouts(policy, n_completions=50, n_timeouts=10):
    """Record a mix of fast completions and timeouts (rate well over threshold)."""
    for _ in range(n_completions):
        policy.record_completion(0.1, interesting=False)
    for _ in range(n_timeouts):
        policy.record_timeout(policy.current_timeout())


def trigger_exploration(policy, clock):
    """Put the policy into a state where stall-based exploration has begun."""
    policy.record_completion(0.1, interesting=True)
    record_fast_run_with_timeouts(policy)
    base = policy.current_timeout()
    clock.advance(STALL_MIN_SECONDS + 1)
    raised = policy.current_timeout()
    assert raised == 2 * base
    return base, raised


# === basic configuration ===


def test_infinite_user_timeout_still_adapts():
    policy, _ = make_policy(user_timeout=math.inf)
    assert policy.cap == math.inf
    # No data yet: no timeout at all.
    assert policy.current_timeout() == math.inf
    # But once we have runtime measurements, the timeout adapts as usual.
    policy.record_completion(1.0, interesting=True)
    assert policy.current_timeout() == 10.0


def test_unstick_stops_raising_after_a_fruitless_raise_when_uncapped():
    """Regression test: with --timeout 0 (no cap), a candidate whose test
    never terminates re-armed the timeout counter every round, so
    attempt_unstick doubled the timeout and replayed the round forever.
    A raise that leads to no reduction must not be followed by another."""
    policy, _ = make_policy(user_timeout=math.inf)
    policy.record_completion(0.5, interesting=True)  # base 5.0
    raises = 0
    for _ in range(20):
        # Each round replays the hanging candidate, which times out below
        # the (infinite) cap and re-arms the counter.
        policy.record_timeout(policy.current_timeout())
        if not policy.attempt_unstick():
            break
        raises += 1
    assert raises == 1


def test_default_cap_when_no_user_timeout():
    policy, _ = make_policy(user_timeout=None)
    assert policy.cap == DEFAULT_TIMEOUT_CAP
    # No data yet: be maximally generous
    assert policy.current_timeout() == DEFAULT_TIMEOUT_CAP


def test_user_timeout_is_cap():
    policy, _ = make_policy(user_timeout=7.0)
    assert policy.cap == 7.0
    assert policy.current_timeout() == 7.0


# === base timeout computation ===


def test_base_timeout_tracks_runtime():
    policy, _ = make_policy()
    policy.record_completion(2.0, interesting=True)
    # 10x the observed runtime, same as the old dynamic timeout heuristic.
    assert policy.current_timeout() == 20.0


def test_base_timeout_clamped_to_minimum():
    policy, _ = make_policy()
    policy.record_completion(0.001, interesting=True)
    assert policy.current_timeout() == MIN_TIMEOUT


def test_base_timeout_clamped_to_cap():
    policy, _ = make_policy()
    policy.record_completion(100.0, interesting=True)
    assert policy.current_timeout() == DEFAULT_TIMEOUT_CAP


def test_base_timeout_uses_high_quantile_of_runtimes():
    policy, _ = make_policy()
    # Mostly fast runs with some slow ones: the slow tail should dominate.
    for _ in range(90):
        policy.record_completion(0.1, interesting=False)
    for _ in range(10):
        policy.record_completion(1.0, interesting=False)
    # p90 is 1.0, so timeout should be 10.0, not 1.0.
    assert policy.current_timeout() == 10.0


def test_interesting_runtimes_keep_timeout_generous():
    policy, _ = make_policy()
    # Uninteresting candidates fail fast, but interesting ones are slow.
    # The timeout must stay high enough for interesting runs.
    policy.record_completion(2.0, interesting=True)
    for _ in range(150):
        policy.record_completion(0.01, interesting=False)
    # 5x the slowest recent interesting runtime.
    assert policy.current_timeout() == 10.0


def test_timeout_adapts_down_as_tests_get_faster():
    policy, _ = make_policy()
    policy.record_completion(5.0, interesting=True)
    slow = policy.current_timeout()
    # As reduction progresses the test gets much faster and old
    # measurements age out of the windows.
    for _ in range(500):
        policy.record_completion(0.05, interesting=True)
    fast = policy.current_timeout()
    assert fast < slow
    assert fast == MIN_TIMEOUT


# === stall-based exploration ===


def test_raises_timeout_when_stalled_with_timeouts():
    policy, clock = make_policy()
    trigger_exploration(policy, clock)


def test_no_exploration_without_timeouts():
    policy, clock = make_policy()
    policy.record_completion(0.1, interesting=True)
    record_fast_run_with_timeouts(policy, n_completions=100, n_timeouts=0)
    base = policy.current_timeout()
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == base


def test_no_exploration_when_not_stalled():
    policy, clock = make_policy()
    policy.record_completion(0.1, interesting=True)
    record_fast_run_with_timeouts(policy)
    base = policy.current_timeout()
    clock.advance(1.0)
    assert policy.current_timeout() == base


def test_no_exploration_with_too_few_outcomes():
    policy, clock = make_policy()
    policy.record_completion(0.1, interesting=True)
    # Plenty of timeouts by rate, but not enough data overall.
    for _ in range(MIN_OUTCOMES_FOR_EXPLORATION - 2):
        policy.record_timeout(policy.current_timeout())
    base = policy.current_timeout()
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == base


def test_stall_threshold_scales_with_timeout():
    policy, clock = make_policy(user_timeout=10000.0)
    policy.record_completion(60.0, interesting=True)
    record_fast_run_with_timeouts(policy, n_completions=50, n_timeouts=10)
    base = policy.current_timeout()
    # Well past the fixed minimum stall time, but when the timeout itself
    # is long we should wait proportionally longer before concluding that
    # reduction has stalled.
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == base


def test_no_exploration_when_already_at_cap():
    policy, clock = make_policy(user_timeout=1.5)
    policy.record_completion(1.0, interesting=True)
    record_fast_run_with_timeouts(policy)
    assert policy.current_timeout() == 1.5
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == 1.5


def test_exploration_capped_at_cap():
    policy, clock = make_policy(user_timeout=2.5)
    policy.record_completion(0.4, interesting=True)  # base 2.0
    record_fast_run_with_timeouts(policy)
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == 2.5


def test_exploration_advances_through_rungs_while_timeouts_continue():
    policy, clock = make_policy()
    base, _ = trigger_exploration(policy, clock)
    # Timeouts continue at the raised timeout with no reduction: after the
    # budget of runs at this rung we escalate further.
    for _ in range(RUNG_RUN_BUDGET):
        policy.record_timeout(policy.current_timeout())
    assert policy.current_timeout() == 4 * base


def test_exploration_reverts_when_raising_stops_timeouts_but_no_progress():
    policy, clock = make_policy()
    base, _ = trigger_exploration(policy, clock)
    # At the raised timeout nothing times out any more, but nothing reduces
    # either: raising didn't help, so go back down.
    for _ in range(RUNG_RUN_BUDGET):
        policy.record_completion(0.1, interesting=False)
    assert policy.current_timeout() == base


def test_failed_exploration_is_not_retried_until_reduction():
    policy, clock = make_policy()
    base, _ = trigger_exploration(policy, clock)
    for _ in range(RUNG_RUN_BUDGET):
        policy.record_completion(0.1, interesting=False)
    assert policy.current_timeout() == base
    # Conditions for exploration still hold, but we already tried and failed.
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == base
    # A reduction resets this: exploration may trigger again later.
    policy.note_reduction()
    record_fast_run_with_timeouts(policy)
    clock.advance(STALL_MIN_SECONDS + 1)
    assert policy.current_timeout() == 2 * base


def test_exploration_reverts_after_exhausting_all_rungs():
    policy, clock = make_policy(user_timeout=4.0)
    policy.record_completion(0.1, interesting=True)
    record_fast_run_with_timeouts(policy)
    base = policy.current_timeout()
    assert base == MIN_TIMEOUT
    clock.advance(STALL_MIN_SECONDS + 1)
    # Climb: 2.0, 4.0 (cap), then nowhere left to go.
    for expected in [2.0, 4.0]:
        assert policy.current_timeout() == expected
        for _ in range(RUNG_RUN_BUDGET):
            policy.record_timeout(policy.current_timeout())
    assert policy.current_timeout() == base


def test_reduction_during_exploration_resets_and_keeps_timeout_generous():
    policy, clock = make_policy()
    base, _ = trigger_exploration(policy, clock)
    # A slow interesting run completes at the raised timeout and a
    # reduction follows.
    slow_runtime = base * 1.5
    policy.record_completion(slow_runtime, interesting=True)
    policy.note_reduction()
    # Exploration ends, but the slow interesting runtime keeps the base
    # comfortably above what the successful run needed.
    assert policy.current_timeout() >= slow_runtime
    assert policy.current_timeout() == 5 * slow_runtime


# === attempt_unstick (reducer ran out of things to try) ===


def test_unstick_raises_timeout_when_timeouts_were_seen():
    policy, _ = make_policy()
    policy.record_completion(0.5, interesting=True)  # base 5.0
    policy.record_timeout(5.0)
    assert policy.attempt_unstick()
    assert policy.current_timeout() == 10.0


def test_unstick_does_nothing_without_timeouts():
    policy, _ = make_policy()
    policy.record_completion(0.5, interesting=True)
    assert not policy.attempt_unstick()
    assert policy.current_timeout() == 5.0


def test_unstick_gives_up_after_fruitless_raise():
    policy, _ = make_policy(user_timeout=18.0)
    policy.record_completion(0.5, interesting=True)  # base 5.0
    policy.record_timeout(5.0)
    assert policy.attempt_unstick()
    assert policy.current_timeout() == 10.0
    # The raise produced no reduction, only more timeouts: refuse to
    # raise again rather than climbing pointlessly.
    policy.record_timeout(10.0)
    assert not policy.attempt_unstick()
    # After giving up we return to the base timeout.
    assert policy.current_timeout() == 5.0


def test_unstick_raise_that_finds_reduction_allows_later_raises():
    policy, _ = make_policy(user_timeout=math.inf)
    policy.record_completion(0.5, interesting=True)  # base 5.0
    policy.record_timeout(policy.current_timeout())
    assert policy.attempt_unstick()
    # The raised timeout let a slow candidate complete and reduce.
    policy.record_completion(policy.current_timeout() / 2, interesting=True)
    policy.note_reduction()
    # A later round runs out of things to try with fresh timeouts: the
    # timeout may be raised again.
    policy.record_timeout(policy.current_timeout())
    assert policy.attempt_unstick()


def test_unstick_ignores_timeouts_at_cap():
    policy, _ = make_policy(user_timeout=5.0)
    policy.record_completion(0.5, interesting=True)  # base 5.0 == cap
    policy.record_timeout(5.0)
    assert not policy.attempt_unstick()


def test_unstick_not_retried_after_exhaustion_until_reduction():
    policy, _ = make_policy(user_timeout=8.0)
    policy.record_completion(0.5, interesting=True)  # base 5.0
    policy.record_timeout(5.0)
    assert policy.attempt_unstick()  # 8.0 (cap)
    assert not policy.attempt_unstick()
    assert not policy.attempt_unstick()
    policy.note_reduction()
    policy.record_timeout(policy.current_timeout())
    assert policy.attempt_unstick()


def test_unstick_timeout_counter_resets_on_reduction():
    policy, _ = make_policy()
    policy.record_completion(0.5, interesting=True)
    policy.record_timeout(5.0)
    policy.note_reduction()
    assert not policy.attempt_unstick()


# === cache validity ===


def test_cached_timeout_results_valid_at_or_below_recorded_timeout():
    policy, _ = make_policy()
    policy.record_completion(0.5, interesting=True)  # base 5.0
    assert policy.cached_timeout_valid(5.0)
    assert policy.cached_timeout_valid(10.0)


def test_cached_timeout_results_invalidated_by_raise():
    policy, _ = make_policy()
    policy.record_completion(0.5, interesting=True)  # base 5.0
    policy.record_timeout(5.0)
    assert policy.attempt_unstick()
    assert not policy.cached_timeout_valid(5.0)
    assert policy.cached_timeout_valid(10.0)


# === reset (restart from history) ===


def test_reset_clears_learned_state():
    policy, clock = make_policy()
    trigger_exploration(policy, clock)
    policy.reset()
    # Back to knowing nothing: maximally generous.
    assert policy.current_timeout() == DEFAULT_TIMEOUT_CAP
    assert policy.recent_timeout_rate == 0.0


def test_reset_forgets_fruitless_unstick_raise():
    policy, _ = make_policy(user_timeout=math.inf)
    policy.record_completion(0.5, interesting=True)
    policy.record_timeout(policy.current_timeout())
    assert policy.attempt_unstick()
    policy.reset()
    # A restarted reduction starts with a clean slate: it may raise again.
    policy.record_completion(0.5, interesting=True)
    policy.record_timeout(policy.current_timeout())
    assert policy.attempt_unstick()


# === stats ===


def test_recent_timeout_rate():
    policy, _ = make_policy()
    assert policy.recent_timeout_rate == 0.0
    for _ in range(8):
        policy.record_completion(0.1, interesting=False)
    for _ in range(2):
        policy.record_timeout(policy.current_timeout())
    assert policy.recent_timeout_rate == pytest.approx(0.2)


# === property-based invariants ===


@st.composite
def policy_events(draw):
    return draw(
        st.lists(
            st.one_of(
                st.tuples(
                    st.just("completion"),
                    st.floats(min_value=0.0, max_value=1000.0),
                    st.booleans(),
                ),
                st.just(("timeout",)),
                st.just(("reduction",)),
                st.just(("unstick",)),
                st.tuples(
                    st.just("advance"), st.floats(min_value=0.0, max_value=10000.0)
                ),
            ),
            max_size=100,
        )
    )


@given(
    user_timeout=st.one_of(
        st.none(),
        st.just(math.inf),
        st.floats(min_value=0.5, max_value=1000.0),
    ),
    events=policy_events(),
)
@example(
    user_timeout=None,
    events=[
        ("completion", 1.0, True),
        ("timeout",),
        ("reduction",),
        ("unstick",),
        ("advance", 30.0),
    ],
)
def test_current_timeout_always_within_bounds(user_timeout, events):
    policy, clock = make_policy(user_timeout=user_timeout)
    for event in events:
        kind = event[0]
        if kind == "completion":
            policy.record_completion(event[1], interesting=event[2])
        elif kind == "timeout":
            policy.record_timeout(policy.current_timeout())
        elif kind == "reduction":
            policy.note_reduction()
        elif kind == "unstick":
            policy.attempt_unstick()
        else:
            assert kind == "advance"
            clock.advance(event[1])
        current = policy.current_timeout()
        # The cap always wins, even when the user sets it below the
        # normal minimum timeout.
        assert min(MIN_TIMEOUT, policy.cap) <= current <= policy.cap
