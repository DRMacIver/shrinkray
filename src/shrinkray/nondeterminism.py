"""Statistics for reducing under a nondeterministic interestingness test.

A deterministic reducer treats one run of the interestingness test as a
verdict. When the test is nondeterministic (a race, a timing dependence,
hidden state) one run is a sample, and every decision that used to be a
lookup becomes a decision about an estimated reproduction rate:

- A candidate is adopted only when its estimated rate clears a threshold
  priced off the incumbent's rate (the *gauntlet*), so that reduction
  cannot walk the test case down to something that reproduces the bug
  far less often than the original, or not at all.
- The incumbent's rate is tracked as a monotone *anchor*: a lower
  confidence bound raised only at validated events (a confirmation batch
  or an adopted candidate's own evidence), never lowered, and never fed
  by re-measurements of the standing incumbent, which would price fresh
  candidates out.
- Every candidate keeps a *ledger* of accumulated evidence across
  retries, so a candidate rejected on one unlucky run gains power when a
  later pass proposes it again, and rejections stay cheap (charge
  accepts, not rejects).
- Because a reduction proposes an unbounded number of candidates, the
  false-accept mass each proposal may spend is charged against a
  per-reduction *alpha budget*; once it runs out, the number of
  interesting runs an accept needs escalates.

Everything here is pure arithmetic with no I/O, so each rule is tested
directly against exact dynamic programs over its stopping rule. The
design and its constants follow Hegel's nondeterminism work, with one
change: the gauntlet also rejects as soon as the threshold has become
unreachable within the run cap, which changes no accept decision and
stops near-deterministic candidates from burning the whole cap.
"""

import math
from enum import Enum, auto

from attrs import define, field


# Wilson score interval z for a two-sided 95% interval. The stopping rules
# below peek after every run, so this is a tuning constant: the exact
# operating points (in tests/test_nondeterminism.py) are the specification.
Z = 1.96

# Replays of the initial test case at startup, and of the final result at
# the end, used to detect nondeterminism. A deterministic test pays this
# many extra calls in total; detection misses a test that reproduces with
# probability p with probability p ** DETECTION_REPLAYS.
DETECTION_REPLAYS = 4

# While the run is still believed deterministic, the current test case is
# re-run once every this many interestingness calls, so a test that is
# nondeterministic but reproduces reliably is still caught early rather
# than only at the end.
VERIFY_INTERVAL = 100

# The confirmation bar: the batch that decides whether a test case that
# was adopted on a single run really reproduces. Reject with no hits once
# the gate is reached, accept on the fourth hit, give up at the cap.
GATE_RUNS = 10
CONFIRM_CAP = 40
CONFIRM_MIN_HITS = 4

# The gauntlet: a candidate whose first run was interesting keeps running
# until its lower bound clears max(gamma * anchor, floor) with at least
# the hit minimum, its upper bound drops below the threshold, the
# threshold becomes unreachable, or the cap is spent.
GAUNTLET_CAP = 30
GAUNTLET_GAMMA = 0.8
# Derived: below LCB(4/30), so at the hit minimum the floor never rejects
# a candidate the minimum would have accepted.
GAUNTLET_FLOOR = 0.05
# A single hit has lower bound 0.2065, so without a minimum every
# threshold below that would accept a candidate on its first run.
GAUNTLET_MIN_HITS = 4
# Once the anchor is this high the incumbent is indistinguishable from
# deterministic and gamma becomes 1.0: a deterministic region, once
# reached, is not traded away for size.
RETENTION_HIGH_WATER = 0.8
# Batches that may seed or raise the anchor are extended to this many runs
# past their stopping point, so the bound estimates the rate rather than
# the stopping rule. It is the largest batch whose all-hit bound (0.839) a
# candidate can still match within GAUNTLET_CAP.
ANCHOR_SEED_RUNS = 20

# Per-reduction false-accept budget, charged per proposal with the exact
# accept probability of a fluke that reproduces at CHARGE_FLUKE_RATE.
GAUNTLET_ALPHA_BUDGET = 0.02
MIN_HITS_CEILING = 8
CHARGE_FLUKE_RATE = 0.02


def wilson_bound(interesting: int, runs: int, *, upper: bool) -> float:
    """One side of the Wilson score interval for interesting / runs."""
    if runs <= 0:
        return 1.0 if upper else 0.0
    # The closed form drifts a few ulps past the ends; the ends are exact.
    if interesting == 0 and not upper:
        return 0.0
    if interesting == runs and upper:
        return 1.0
    p = interesting / runs
    z2 = Z * Z
    denominator = 1.0 + z2 / runs
    centre = p + z2 / (2.0 * runs)
    margin = Z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * runs)) / runs)
    bound = (centre + margin if upper else centre - margin) / denominator
    return min(1.0, max(0.0, bound))


@define
class Evidence:
    """How many of a test case's runs were interesting."""

    interesting: int = 0
    runs: int = 0

    def record(self, interesting: bool) -> None:
        self.runs += 1
        if interesting:
            self.interesting += 1

    @property
    def rate(self) -> float:
        if self.runs == 0:
            return 0.0
        return self.interesting / self.runs

    def lower_bound(self) -> float:
        return wilson_bound(self.interesting, self.runs, upper=False)

    def upper_bound(self) -> float:
        return wilson_bound(self.interesting, self.runs, upper=True)


class Verdict(Enum):
    ACCEPT = auto()
    REJECT = auto()
    CONTINUE = auto()


def gauntlet_threshold(anchor: float) -> float:
    gamma = 1.0 if anchor >= RETENTION_HIGH_WATER else GAUNTLET_GAMMA
    return max(gamma * anchor, GAUNTLET_FLOOR)


def gauntlet(evidence: Evidence, anchor: float, min_hits: int) -> Verdict:
    """Decide whether a candidate's evidence clears the incumbent's anchor."""
    return _gauntlet_at(evidence, gauntlet_threshold(anchor), min_hits)


def _gauntlet_at(evidence: Evidence, threshold: float, min_hits: int) -> Verdict:
    if evidence.interesting >= min_hits and evidence.lower_bound() >= threshold:
        return Verdict.ACCEPT
    if evidence.upper_bound() < threshold or evidence.runs >= GAUNTLET_CAP:
        return Verdict.REJECT
    # Even if every remaining run were interesting the bound would not
    # clear the threshold, so nothing is left to learn.
    remaining = GAUNTLET_CAP - evidence.runs
    best = Evidence(evidence.interesting + remaining, GAUNTLET_CAP)
    if best.lower_bound() < threshold:
        return Verdict.REJECT
    return Verdict.CONTINUE


def confirmation_bar(evidence: Evidence) -> Verdict:
    """Decide whether an incumbent adopted on one run really reproduces."""
    if evidence.interesting >= CONFIRM_MIN_HITS:
        return Verdict.ACCEPT
    if evidence.interesting == 0 and evidence.runs >= GATE_RUNS:
        return Verdict.REJECT
    if evidence.interesting + (CONFIRM_CAP - evidence.runs) < CONFIRM_MIN_HITS:
        return Verdict.REJECT
    return Verdict.CONTINUE


def gauntlet_alpha(seed: Evidence, threshold: float, min_hits: int) -> float:
    """P(the gauntlet accepts | the candidate is a CHARGE_FLUKE_RATE fluke),
    starting from `seed`: the false-accept mass one driven proposal
    contributes. An exact dynamic program over the (interesting, runs)
    probability mass under the per-run verdict checks."""
    mass = {(seed.interesting, seed.runs): 1.0}
    accept = 0.0
    while mass:
        following: dict[tuple[int, int], float] = {}
        for (hits, runs), m in mass.items():
            verdict = _gauntlet_at(Evidence(hits, runs), threshold, min_hits)
            if verdict == Verdict.ACCEPT:
                accept += m
            elif verdict == Verdict.CONTINUE:
                hit = (hits + 1, runs + 1)
                miss = (hits, runs + 1)
                following[hit] = following.get(hit, 0.0) + m * CHARGE_FLUKE_RATE
                following[miss] = following.get(miss, 0.0) + m * (
                    1.0 - CHARGE_FLUKE_RATE
                )
        mass = following
    return accept


@define
class AlphaBudget:
    """One reduction's false-accept spending state.

    Every proposal is charged, before its outcome is known, its exact
    false-accept mass against a fluke. Charging per proposal makes the
    total bound the expected number of false accepts by linearity. A
    candidate already `pinned` at a hit minimum is charged at that
    minimum even past the budget (a stopping rule never changes mid-test,
    and the overdraft per candidate is bounded by one charge); a new
    candidate is pinned at the current minimum, escalated up to
    MIN_HITS_CEILING first when the remainder cannot afford it.
    """

    remaining: float = GAUNTLET_ALPHA_BUDGET
    min_hits: int = GAUNTLET_MIN_HITS

    def charge(
        self, seed: Evidence, *, anchor: float, drive: bool, pinned: int | None
    ) -> int:
        threshold = gauntlet_threshold(anchor)

        def alpha(min_hits: int) -> float:
            if drive:
                return gauntlet_alpha(seed, threshold, min_hits)
            recruited = Evidence(seed.interesting + 1, seed.runs + 1)
            return CHARGE_FLUKE_RATE * gauntlet_alpha(recruited, threshold, min_hits)

        if pinned is not None:
            self.remaining -= alpha(pinned)
            return pinned
        while True:
            cost = alpha(self.min_hits)
            if cost <= self.remaining or self.min_hits >= MIN_HITS_CEILING:
                self.remaining -= cost
                return self.min_hits
            self.min_hits += 1


@define
class NondeterminismPolicy:
    """The run-level state of nondeterminism handling.

    `active` is sticky: a run is deterministic until proven otherwise and
    never goes back. `anchor` is the monotone lower bound on the
    incumbent's reproduction rate. `confirming` selects the sweep mode:
    in a fast sweep a candidate whose first run misses is rejected at the
    cost of that one run (its evidence retained for later retries); in a
    confirmation sweep every candidate's ledger is driven to a bound
    verdict, so a fixpoint reached that way carries a certificate.
    """

    active: bool = False
    anchor: float = 0.0
    confirming: bool = False
    budget: AlphaBudget = field(factory=AlphaBudget)
    # Runs spent on replays (every run of a candidate beyond its first,
    # plus detection and confirmation runs), and how many were interesting.
    replay_calls: int = 0
    replay_interesting: int = 0

    def flip(self) -> bool:
        """Switch into nondeterministic handling. Returns whether this call
        was the one that switched."""
        if self.active:
            return False
        self.active = True
        return True

    @property
    def threshold(self) -> float:
        return gauntlet_threshold(self.anchor)

    def raise_anchor(self, evidence: Evidence) -> None:
        self.anchor = max(self.anchor, evidence.lower_bound())

    def record_replay(self, interesting: bool) -> None:
        self.replay_calls += 1
        if interesting:
            self.replay_interesting += 1

    def charge(self, seed: Evidence, *, pinned: int | None) -> int:
        return self.budget.charge(
            seed, anchor=self.anchor, drive=self.confirming, pinned=pinned
        )
