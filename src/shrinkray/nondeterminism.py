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
- A reduction proposes an unbounded number of candidates, so any fixed
  per-proposal false-accept rate compounds. Instead of assuming a
  background rate of spurious hits and budgeting against it, the run
  *monitors its incumbent* with fresh replays: an incumbent that turns
  out not to reproduce at the rate the gauntlet required is a detected
  false accept, recovered from by backtracking through the adopted
  history, and answered by raising the hit minimum for every later
  candidate. Multiplicity is controlled by feedback from what actually
  happened rather than by an assumed model of the noise.

Everything here is pure arithmetic with no I/O, so each rule is tested
directly against exact dynamic programs over its stopping rule. The
design and its constants follow Hegel's nondeterminism work, with one
change: the gauntlet also rejects as soon as the threshold has become
unreachable within the run cap, which changes no accept decision and
stops near-deterministic candidates from burning the whole cap.
"""

import math
from enum import Enum, auto
from statistics import NormalDist

from attrs import define, field


# Wilson score interval z for a two-sided 95% interval. The stopping rules
# below peek after every run, so this is a tuning constant: the exact
# operating points (in tests/test_nondeterminism.py) are the specification.
Z = 1.96
# The one-sided level anchor raises are held to, over all of a run's
# attempts together (see anchor_z). 0.025 is Z's one-sided tail, so the
# first attempt uses Z itself.
ANCHOR_ALPHA = 0.025

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
# threshold below that would accept a candidate on its first run. This is
# the minimum the operating points in the tests are derived at; the
# minimum in force starts at INITIAL_MIN_HITS and escalates with observed
# false accepts.
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

# The hit minimum a run starts with, and how far observed false accepts
# may escalate it. Rather than charging every proposal against an assumed
# background rate of spurious hits, the run watches its incumbent: a
# falsely accepted candidate (or one that reproduces far less often than
# the anchor promised) is detected by the incumbent monitor, recovered
# from by backtracking, and answered by demanding one more hit of every
# later candidate. A one-sided test, where nothing without the bug is
# ever interesting, therefore pays only the small starting minimum.
INITIAL_MIN_HITS = 2
MIN_HITS_CEILING = 12
# The incumbent monitor judges the incumbent only once it has this many
# fresh replays: against a near-deterministic anchor a single missed
# replay would otherwise condemn an incumbent that reproduces nine times
# in ten.
MONITOR_MIN_RUNS = 5


def wilson_bound(interesting: int, runs: int, *, upper: bool, z: float = Z) -> float:
    """One side of the Wilson score interval for interesting / runs."""
    if runs <= 0:
        return 1.0 if upper else 0.0
    # The closed form drifts a few ulps past the ends; the ends are exact.
    if interesting == 0 and not upper:
        return 0.0
    if interesting == runs and upper:
        return 1.0
    p = interesting / runs
    z2 = z * z
    denominator = 1.0 + z2 / runs
    centre = p + z2 / (2.0 * runs)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * runs)) / runs)
    bound = (centre + margin if upper else centre - margin) / denominator
    return min(1.0, max(0.0, bound))


def anchor_z(attempts: int) -> float:
    """The z for the `attempts`-th attempt to raise the anchor.

    The anchor is a running maximum over every adopted candidate's bound,
    and the maximum of many noisy bounds overshoots the rate they all
    estimate (a reduction adopts hundreds of candidates at the same true
    rate, and the luckiest of their twenty-run batches looks far better
    than the rate). Each attempt therefore uses a bound at the Bonferroni
    level for the number of attempts so far, so the chance that any raise
    ever overshoots the true rate stays at the single-test level.
    """
    return max(Z, NormalDist().inv_cdf(1.0 - ANCHOR_ALPHA / max(attempts, 1)))


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

    def lower_bound(self, z: float = Z) -> float:
        return wilson_bound(self.interesting, self.runs, upper=False, z=z)

    def since(self, earlier: "Evidence") -> "Evidence":
        """The runs recorded after `earlier`, an earlier snapshot of this
        evidence."""
        return Evidence(
            self.interesting - earlier.interesting, self.runs - earlier.runs
        )

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
    # The hit minimum new candidates are pinned at, and the false accepts
    # observed so far (each raises the minimum by one, up to the ceiling).
    min_hits: int = INITIAL_MIN_HITS
    false_accepts: int = 0
    # Fresh, unselected evidence about the current incumbent: seeded from
    # the runs after its accept decision and fed by the incumbent monitor.
    # When its upper bound falls below the threshold the incumbent is not
    # what the anchor promised, and the run backtracks.
    incumbent: Evidence = field(factory=Evidence)
    # The same evidence pooled over every incumbent so far: the rate the
    # reduction's incumbents have actually reproduced at. A raise must
    # beat it, so that in a landscape where every candidate reproduces at
    # the same rate the anchor stays where the confirmation batch put it
    # instead of creeping up on the luckiest of hundreds of batches, and
    # rises only where candidates genuinely reproduce more reliably.
    pool: Evidence = field(factory=Evidence)
    # Attempts made to raise the anchor, which set the level each is held to.
    anchor_attempts: int = 0
    # Runs spent on replays (every run of a candidate beyond its first,
    # plus detection and confirmation runs), how many were interesting,
    # and where they were spent, by site:
    #   detection     replays of the incumbent looking for nondeterminism
    #   confirmation  confirmation-bar batches (incumbent and backtracking)
    #   gauntlet      reruns of a candidate until the gauntlet decides
    #   seed          topping an accepted candidate's ledger up to the seed
    #   monitor       periodic replays of the incumbent under handling
    #   report        the final measurement for the report
    replay_calls: int = 0
    replay_interesting: int = 0
    replay_sites: dict[str, int] = field(factory=dict)

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

    @property
    def raise_bar(self) -> float:
        """What a raise must beat: the anchor, and the rate incumbents have
        reproduced at so far."""
        return max(self.anchor, self.pool.rate)

    def raise_reachable(self, unselected: Evidence, remaining: int) -> bool:
        """Whether `remaining` more runs after `unselected` could raise the
        anchor on the next attempt, even if every one of them hit. When
        they cannot, the runs are not worth spending."""
        best = Evidence(unselected.interesting + remaining, unselected.runs + remaining)
        return best.lower_bound(anchor_z(self.anchor_attempts + 1)) > self.raise_bar

    def raise_anchor(self, evidence: Evidence) -> None:
        """Raise the anchor to `evidence`'s lower bound if that beats the
        raise bar. `evidence` must not have been selected on: runs that
        decided an accept are biased upwards by the stopping rule, so
        callers pass the runs recorded after the decision (see
        Evidence.since)."""
        self.anchor_attempts += 1
        z = anchor_z(self.anchor_attempts)
        bound = evidence.lower_bound(z)
        if bound > self.raise_bar:
            self.anchor = bound

    def record_incumbent_run(self, interesting: bool) -> None:
        """A fresh replay of the incumbent, from the monitor."""
        self.incumbent.record(interesting)
        self.pool.record(interesting)

    def adopt(self, unselected: Evidence) -> None:
        """A new incumbent: start its monitor from the runs that did not
        take part in accepting it, and add them to the pool."""
        self.incumbent = Evidence(unselected.interesting, unselected.runs)
        self.pool.interesting += unselected.interesting
        self.pool.runs += unselected.runs

    def incumbent_failing(self) -> bool:
        """Whether the incumbent's fresh evidence rules out the rate the
        gauntlet required of it."""
        return (
            self.incumbent.runs >= MONITOR_MIN_RUNS
            and self.incumbent.upper_bound() < self.threshold
        )

    def record_false_accept(self) -> None:
        self.false_accepts += 1
        self.min_hits = min(MIN_HITS_CEILING, self.min_hits + 1)

    def record_replay(self, interesting: bool, site: str) -> None:
        self.replay_calls += 1
        if interesting:
            self.replay_interesting += 1
        self.replay_sites[site] = self.replay_sites.get(site, 0) + 1
