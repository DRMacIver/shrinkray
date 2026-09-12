"""Tests for the nondeterminism statistics module."""

import math
from statistics import NormalDist

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.nondeterminism import (
    ANCHOR_ALPHA,
    ANCHOR_SEED_RUNS,
    CHARGE_FLUKE_RATE,
    CONFIRM_CAP,
    CONFIRM_MIN_HITS,
    GATE_RUNS,
    GAUNTLET_ALPHA_BUDGET,
    GAUNTLET_CAP,
    GAUNTLET_FLOOR,
    GAUNTLET_GAMMA,
    GAUNTLET_MIN_HITS,
    MIN_HITS_CEILING,
    RETENTION_HIGH_WATER,
    AlphaBudget,
    Evidence,
    NondeterminismPolicy,
    Verdict,
    anchor_z,
    confirmation_bar,
    gauntlet,
    gauntlet_alpha,
    gauntlet_threshold,
    wilson_bound,
)


evidences = st.builds(
    lambda runs, frac: Evidence(interesting=int(runs * frac), runs=runs),
    st.integers(0, 200),
    st.floats(0.0, 1.0),
)


# === Wilson bounds ===


@pytest.mark.parametrize(
    "interesting, runs, lower, upper",
    [
        pytest.param(4, 30, 0.0531, 0.2968, id="min-hits-at-cap"),
        pytest.param(1, 1, 0.2065, 1.0, id="single-hit"),
        pytest.param(20, 20, 0.8389, 1.0, id="all-hit-seed"),
        pytest.param(10, 20, 0.2993, 0.7007, id="half"),
        pytest.param(40, 40, 0.9124, 1.0, id="forty"),
        pytest.param(30, 30, 0.8865, 1.0, id="cap-reachable"),
        pytest.param(0, 10, 0.0, 0.2775, id="no-hits"),
    ],
)
def test_wilson_reference_values(interesting, runs, lower, upper):
    ev = Evidence(interesting=interesting, runs=runs)
    assert ev.lower_bound() == pytest.approx(lower, abs=1e-4)
    assert ev.upper_bound() == pytest.approx(upper, abs=1e-4)


def test_wilson_with_no_runs_is_vacuous():
    assert wilson_bound(0, 0, upper=False) == 0.0
    assert wilson_bound(0, 0, upper=True) == 1.0


@given(evidences)
def test_wilson_bounds_bracket_the_rate(ev):
    if ev.runs == 0:
        return
    rate = ev.interesting / ev.runs
    assert 0.0 <= ev.lower_bound() <= rate <= ev.upper_bound() <= 1.0


@given(st.integers(1, 200), st.integers(0, 200))
def test_more_hits_never_lower_the_bounds(runs, k):
    k = min(k, runs - 1)
    before = Evidence(interesting=k, runs=runs)
    after = Evidence(interesting=k + 1, runs=runs)
    assert after.lower_bound() >= before.lower_bound()
    assert after.upper_bound() >= before.upper_bound()


def test_evidence_record_counts_hits_and_runs():
    ev = Evidence()
    ev.record(True)
    ev.record(False)
    ev.record(True)
    assert (ev.interesting, ev.runs) == (2, 3)
    assert ev.rate == pytest.approx(2 / 3)


def test_evidence_rate_with_no_runs_is_zero():
    assert Evidence().rate == 0.0


# === Gauntlet ===


def test_threshold_is_gamma_scaled_above_the_floor():
    assert gauntlet_threshold(0.5) == pytest.approx(GAUNTLET_GAMMA * 0.5)


def test_threshold_never_drops_below_the_floor():
    assert gauntlet_threshold(0.0) == GAUNTLET_FLOOR
    assert gauntlet_threshold(0.03) == GAUNTLET_FLOOR


def test_threshold_is_full_anchor_at_the_high_water():
    assert gauntlet_threshold(RETENTION_HIGH_WATER) == RETENTION_HIGH_WATER
    assert gauntlet_threshold(0.9) == 0.9


def test_floor_costs_no_power_at_min_hits():
    # Four hits within the cap always clear the floor, so the floor never
    # rejects a candidate the hit minimum would have accepted.
    assert GAUNTLET_FLOOR < Evidence(GAUNTLET_MIN_HITS, GAUNTLET_CAP).lower_bound()


def test_gauntlet_continues_short_of_min_hits():
    for hits in range(GAUNTLET_MIN_HITS):
        assert (
            gauntlet(Evidence(hits, hits), 0.5, GAUNTLET_MIN_HITS) == Verdict.CONTINUE
        )


def test_gauntlet_accepts_at_min_hits_when_bound_clears():
    ev = Evidence(GAUNTLET_MIN_HITS, GAUNTLET_MIN_HITS)
    assert ev.lower_bound() >= gauntlet_threshold(0.5)
    assert gauntlet(ev, 0.5, GAUNTLET_MIN_HITS) == Verdict.ACCEPT


def test_gauntlet_rejects_when_upper_bound_is_below_threshold():
    assert gauntlet(Evidence(0, 8), 0.5, GAUNTLET_MIN_HITS) == Verdict.REJECT


def test_gauntlet_rejects_at_the_cap():
    ev = Evidence(GAUNTLET_MIN_HITS, GAUNTLET_CAP)
    assert gauntlet(ev, 0.5, GAUNTLET_MIN_HITS) == Verdict.REJECT


def test_gauntlet_rejects_once_the_threshold_is_unreachable():
    # At the high water a single miss makes the threshold unreachable by
    # the cap: LCB(29/30) < LCB(20/20). Rejecting immediately is what stops
    # near-deterministic candidates from burning the whole cap.
    anchor = Evidence(ANCHOR_SEED_RUNS, ANCHOR_SEED_RUNS).lower_bound()
    assert gauntlet(Evidence(1, 2), anchor, GAUNTLET_MIN_HITS) == Verdict.REJECT
    # With no miss it keeps going.
    assert gauntlet(Evidence(2, 2), anchor, GAUNTLET_MIN_HITS) == Verdict.CONTINUE


def test_gauntlet_can_accept_at_the_cap():
    ev = Evidence(GAUNTLET_CAP, GAUNTLET_CAP)
    assert gauntlet(ev, 0.7, GAUNTLET_MIN_HITS) == Verdict.ACCEPT


def test_gauntlet_respects_pinned_min_hits():
    ev = Evidence(5, 5)
    assert gauntlet(ev, 0.5, 4) == Verdict.ACCEPT
    assert gauntlet(ev, 0.5, 6) == Verdict.CONTINUE


@given(evidences, st.floats(0.0, 1.0), st.integers(1, MIN_HITS_CEILING))
def test_gauntlet_always_decides_at_the_cap(ev, anchor, min_hits):
    if ev.runs >= GAUNTLET_CAP:
        assert gauntlet(ev, anchor, min_hits) != Verdict.CONTINUE


@given(evidences, st.floats(0.0, 1.0), st.integers(1, MIN_HITS_CEILING))
def test_unreachability_reject_never_changes_an_accept(ev, anchor, min_hits):
    # If every remaining run were a hit and the result would still not
    # accept, rejecting now loses nothing.
    threshold = gauntlet_threshold(anchor)
    verdict = gauntlet(ev, anchor, min_hits)
    if verdict == Verdict.REJECT and ev.runs < GAUNTLET_CAP:
        best = Evidence(ev.interesting + GAUNTLET_CAP - ev.runs, GAUNTLET_CAP)
        assert ev.upper_bound() < threshold or best.lower_bound() < threshold


def exact_gauntlet_outcome(
    p: float, anchor: float, min_hits: int, *, recruited: bool
) -> tuple[float, float]:
    """Independent DP: P(accept) and E[runs] for a candidate at true rate p.

    `recruited` starts from a single hit (a candidate whose first run was
    interesting), matching the fast-sweep entry into the gauntlet.
    """
    start = (1, 1) if recruited else (0, 0)
    mass = {start: 1.0}
    accept = 0.0
    expected_runs = 0.0
    while mass:
        following: dict[tuple[int, int], float] = {}
        for (hits, runs), m in mass.items():
            verdict = gauntlet(Evidence(hits, runs), anchor, min_hits)
            if verdict == Verdict.CONTINUE:
                following[(hits + 1, runs + 1)] = (
                    following.get((hits + 1, runs + 1), 0.0) + m * p
                )
                following[(hits, runs + 1)] = following.get(
                    (hits, runs + 1), 0.0
                ) + m * (1 - p)
            else:
                expected_runs += m * runs
                if verdict == Verdict.ACCEPT:
                    accept += m
        mass = following
    return accept, expected_runs


@pytest.mark.parametrize(
    "anchor, p, accept, runs",
    [
        # The operating points experiment 008 derived for these rules, all
        # unchanged by the unreachability reject except the run counts at
        # the high water.
        pytest.param(0.05, 0.02, 0.0198, 28.4, id="floor-fluke"),
        pytest.param(0.05, 0.10, 0.5650, 23.9, id="floor-target"),
        pytest.param(0.05, 0.90, 1.0000, 4.3, id="floor-reliable"),
        pytest.param(0.30, 0.02, 0.0002, 20.4, id="mid-fluke"),
        pytest.param(0.30, 0.10, 0.0182, 21.8, id="mid-target"),
        pytest.param(0.30, 0.90, 1.0000, 4.3, id="mid-reliable"),
        pytest.param(0.839, 0.02, 0.0000, 2.0, id="high-fluke"),
        pytest.param(0.839, 0.10, 0.0000, 2.1, id="high-target"),
        pytest.param(0.839, 0.90, 0.1216, 9.8, id="high-reliable"),
    ],
)
def test_gauntlet_operating_points(anchor, p, accept, runs):
    got_accept, got_runs = exact_gauntlet_outcome(
        p, anchor, GAUNTLET_MIN_HITS, recruited=True
    )
    assert got_accept == pytest.approx(accept, abs=5e-4)
    assert got_runs == pytest.approx(runs, abs=0.1)


# === Confirmation bar ===


def test_bar_continues_at_first():
    assert confirmation_bar(Evidence(0, 0)) == Verdict.CONTINUE
    assert confirmation_bar(Evidence(1, 3)) == Verdict.CONTINUE


def test_bar_accepts_on_the_min_hits():
    assert confirmation_bar(Evidence(CONFIRM_MIN_HITS, 5)) == Verdict.ACCEPT


def test_bar_rejects_no_hits_at_the_gate():
    assert confirmation_bar(Evidence(0, GATE_RUNS - 1)) == Verdict.CONTINUE
    assert confirmation_bar(Evidence(0, GATE_RUNS)) == Verdict.REJECT


def test_bar_rejects_when_quota_unreachable():
    remaining_needed = CONFIRM_MIN_HITS - 1
    assert (
        confirmation_bar(Evidence(1, CONFIRM_CAP - remaining_needed))
        == Verdict.CONTINUE
    )
    assert (
        confirmation_bar(Evidence(1, CONFIRM_CAP - remaining_needed + 1))
        == Verdict.REJECT
    )


@given(evidences)
def test_bar_always_decides_at_the_cap(ev):
    if ev.runs >= CONFIRM_CAP:
        assert confirmation_bar(ev) != Verdict.CONTINUE


def exact_bar_outcome(p: float) -> tuple[float, float]:
    mass = {(0, 0): 1.0}
    accept = 0.0
    expected_runs = 0.0
    while mass:
        following: dict[tuple[int, int], float] = {}
        for (hits, runs), m in mass.items():
            verdict = confirmation_bar(Evidence(hits, runs))
            if verdict == Verdict.CONTINUE:
                following[(hits + 1, runs + 1)] = (
                    following.get((hits + 1, runs + 1), 0.0) + m * p
                )
                following[(hits, runs + 1)] = following.get(
                    (hits, runs + 1), 0.0
                ) + m * (1 - p)
            else:
                expected_runs += m * runs
                if verdict == Verdict.ACCEPT:
                    accept += m
        mass = following
    return accept, expected_runs


def test_bar_operating_points():
    # Experiment 005A's derivation: 0.6% false accepts on a 2% fluke at
    # about 15 replays, 45% power at the 10% target.
    accept, runs = exact_bar_outcome(0.02)
    assert accept == pytest.approx(0.0059, abs=2e-4)
    assert runs == pytest.approx(15.2, abs=0.1)
    accept, _ = exact_bar_outcome(0.1)
    assert accept == pytest.approx(0.454, abs=1e-3)


# === Alpha budget ===


def test_gauntlet_alpha_matches_the_independent_dp():
    for threshold in (GAUNTLET_FLOOR, 0.24, 0.839):
        for seed in (Evidence(0, 0), Evidence(1, 1), Evidence(2, 3)):
            for min_hits in (4, 6, 8):
                expected, _ = exact_gauntlet_outcome_from(
                    CHARGE_FLUKE_RATE, threshold, min_hits, seed
                )
                assert gauntlet_alpha(seed, threshold, min_hits) == pytest.approx(
                    expected, abs=1e-9
                )


def exact_gauntlet_outcome_from(
    p: float, threshold: float, min_hits: int, seed: Evidence
) -> tuple[float, float]:
    """Like exact_gauntlet_outcome but from an arbitrary seed and a raw
    threshold (an anchor whose gauntlet threshold is exactly `threshold`)."""
    anchor = (
        threshold if threshold >= RETENTION_HIGH_WATER else threshold / GAUNTLET_GAMMA
    )
    assert gauntlet_threshold(anchor) == pytest.approx(threshold)
    mass = {(seed.interesting, seed.runs): 1.0}
    accept = 0.0
    expected_runs = 0.0
    while mass:
        following: dict[tuple[int, int], float] = {}
        for (hits, runs), m in mass.items():
            verdict = gauntlet(Evidence(hits, runs), anchor, min_hits)
            if verdict == Verdict.CONTINUE:
                following[(hits + 1, runs + 1)] = (
                    following.get((hits + 1, runs + 1), 0.0) + m * p
                )
                following[(hits, runs + 1)] = following.get(
                    (hits, runs + 1), 0.0
                ) + m * (1 - p)
            else:
                expected_runs += m * runs
                if verdict == Verdict.ACCEPT:
                    accept += m
        mass = following
    return accept, expected_runs


def test_alpha_at_the_floor_is_the_008_figure():
    fast = CHARGE_FLUKE_RATE * gauntlet_alpha(Evidence(1, 1), GAUNTLET_FLOOR, 4)
    assert fast == pytest.approx(4.0e-4, abs=2e-5)
    drive = gauntlet_alpha(Evidence(0, 0), GAUNTLET_FLOOR, 4)
    assert drive == pytest.approx(2.9e-3, abs=1e-4)


def test_unreachable_threshold_charges_nothing():
    assert gauntlet_alpha(Evidence(0, 0), 0.9, 4) == 0.0


def test_budget_affords_about_fifty_fast_floor_proposals():
    budget = AlphaBudget()
    charged = 0
    while budget.charge(Evidence(0, 0), anchor=0.0, drive=False, pinned=None) == 4:
        charged += 1
    assert charged == 50


def test_budget_escalates_to_the_ceiling_and_no_further():
    budget = AlphaBudget()
    seen = set()
    for _ in range(200):
        seen.add(budget.charge(Evidence(0, 0), anchor=0.0, drive=True, pinned=None))
    assert max(seen) == MIN_HITS_CEILING
    assert seen == set(range(GAUNTLET_MIN_HITS, MIN_HITS_CEILING + 1))


def test_pinned_candidates_keep_their_minimum_past_the_budget():
    budget = AlphaBudget()
    for _ in range(200):
        budget.charge(Evidence(0, 0), anchor=0.0, drive=True, pinned=None)
    before = budget.remaining
    assert budget.charge(Evidence(2, 5), anchor=0.0, drive=True, pinned=4) == 4
    # The overdraft is exactly that candidate's own charge, nothing more.
    charge = gauntlet_alpha(Evidence(2, 5), GAUNTLET_FLOOR, 4)
    assert charge > 0.0
    assert budget.remaining == pytest.approx(before - charge)
    assert budget.remaining < 0.0


def test_mid_anchor_proposals_are_effectively_free():
    budget = AlphaBudget()
    for _ in range(1000):
        assert budget.charge(Evidence(0, 0), anchor=0.5, drive=False, pinned=None) == 4
    assert budget.remaining > GAUNTLET_ALPHA_BUDGET * 0.9


# === Policy ===


def test_policy_starts_deterministic():
    policy = NondeterminismPolicy()
    assert not policy.active
    assert policy.anchor == 0.0
    assert not policy.confirming


def test_flip_is_sticky_and_idempotent():
    policy = NondeterminismPolicy()
    assert policy.flip()
    assert policy.active
    assert not policy.flip()
    assert policy.active


def test_anchor_is_monotone():
    policy = NondeterminismPolicy()
    policy.flip()
    policy.raise_anchor(Evidence(10, 20))
    first = policy.anchor
    assert first == pytest.approx(Evidence(10, 20).lower_bound())
    policy.raise_anchor(Evidence(2, 20))
    assert policy.anchor == first
    policy.raise_anchor(Evidence(20, 20))
    assert policy.anchor > first


def test_evidence_since_an_earlier_snapshot():
    ev = Evidence(3, 5)
    later = Evidence(7, 12)
    assert later.since(ev) == Evidence(4, 7)


def test_wilson_bound_tightens_with_z():
    assert Evidence(10, 20).lower_bound(z=1.0) > Evidence(10, 20).lower_bound()
    assert Evidence(10, 20).lower_bound(z=3.0) < Evidence(10, 20).lower_bound()


def test_anchor_z_starts_at_z_and_grows_with_attempts():
    assert anchor_z(0) == anchor_z(1) == 1.96
    assert anchor_z(1) < anchor_z(10) < anchor_z(1000)
    # The Bonferroni level: 1000 attempts share the single-test tail.
    assert anchor_z(1000) == pytest.approx(
        NormalDist().inv_cdf(1 - ANCHOR_ALPHA / 1000)
    )


def test_repeated_raises_are_held_to_a_stricter_level():
    # The same evidence raises the anchor less the more attempts there
    # have been, so a run of hundreds of adoptions cannot ratchet the
    # anchor above the rate they all share by picking the luckiest batch.
    first = NondeterminismPolicy()
    first.flip()
    first.raise_anchor(Evidence(16, 20))
    later = NondeterminismPolicy()
    later.flip()
    later.anchor_attempts = 200
    later.raise_anchor(Evidence(16, 20))
    assert later.anchor < first.anchor
    assert later.anchor_attempts == 201


def test_threshold_follows_the_anchor():
    policy = NondeterminismPolicy()
    policy.flip()
    policy.raise_anchor(Evidence(20, 20))
    assert policy.threshold == gauntlet_threshold(policy.anchor)


def test_policy_counts_replays_by_site():
    policy = NondeterminismPolicy()
    policy.record_replay(True, "detection")
    policy.record_replay(False, "gauntlet")
    policy.record_replay(False, "gauntlet")
    assert policy.replay_calls == 3
    assert policy.replay_interesting == 1
    assert policy.replay_sites == {"detection": 1, "gauntlet": 2}


def test_charge_delegates_to_the_budget():
    policy = NondeterminismPolicy()
    policy.flip()
    assert policy.charge(Evidence(0, 0), pinned=None) == GAUNTLET_MIN_HITS
    assert policy.budget.remaining < GAUNTLET_ALPHA_BUDGET


def test_charge_uses_drive_mode_when_confirming():
    fast = NondeterminismPolicy()
    fast.flip()
    fast.charge(Evidence(0, 0), pinned=None)
    confirm = NondeterminismPolicy()
    confirm.flip()
    confirm.confirming = True
    confirm.charge(Evidence(0, 0), pinned=None)
    assert confirm.budget.remaining < fast.budget.remaining


def test_constants_are_consistent():
    # The seed batch's all-hit lower bound must be matchable within the cap,
    # or a deterministic incumbent could never be displaced.
    seed = Evidence(ANCHOR_SEED_RUNS, ANCHOR_SEED_RUNS).lower_bound()
    assert seed <= Evidence(GAUNTLET_CAP, GAUNTLET_CAP).lower_bound()
    assert seed >= RETENTION_HIGH_WATER
    assert math.isclose(GAUNTLET_GAMMA, 0.8)
