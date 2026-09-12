# Nondeterministic interestingness tests

A reducer normally treats one run of the interestingness test as a verdict.
Some tests do not admit that: a race, a timing dependence, or hidden state
makes the same test case exit 0 on some runs and not on others. Reducing
such a test with single-run verdicts goes wrong in two ways:

- **Drift.** Every adopted candidate was seen interesting once. A candidate
  that reproduces the bug a third as often as the incumbent is adopted with
  probability one third, and the reducer adopts thousands of candidates, so
  the result reproduces the bug far more rarely than the input did.
- **Loss.** If test cases without the bug are sometimes spuriously
  interesting (an occasional crash in the harness itself, say), one lucky
  run adopts a candidate that no longer has the bug at all, and nothing
  ever recovers from it.

The design follows Hegel's nondeterminism work (the `DRMacIver/nondeterminism`
branch of hegel-rust; its notes are the primary source for the statistics).
Only the outcome-nondeterminism part applies here: Shrink Ray has no
generator, so there are no timelines, pools, or origins.

## Mechanism

Everything lives in `BasicReductionProblem` (`problem.py`) over the pure
arithmetic in `nondeterminism.py`; passes are untouched.

**Ledger.** The interestingness cache stores per-candidate `Evidence`
(interesting runs, total runs) and a latched verdict, instead of a boolean.
Under a deterministic test one run latches the verdict, exactly as before.

**Detection.** A run is deterministic until proven otherwise. The initial
test case is replayed `DETECTION_REPLAYS` times at setup (concurrently, so
wall-clock cost is one run), the current test case is replayed once every
`VERIFY_INTERVAL` calls, and the final result is replayed through
`attempt_unstick` before the reducer is allowed to finish. Any completed
replay that is not interesting flips the run into nondeterministic handling,
which is sticky. Timed-out replays are ignored: a timeout says nothing
about determinism. `--assume-deterministic` skips all of this.

**Confirmation and backtracking.** At the flip the current test case was
adopted on single runs and may not reproduce. It faces the confirmation
bar (accept on the fourth interesting run, reject on none in ten or when
four are unreachable in forty); if it fails, the run's adopted history
(`HistoryBacktrackSource`, reading the `.shrinkray` history directory, or
just the original input without history) is scanned newest-first at
geometrically growing distances, then bisected, for the newest entry that
clears the bar, and the reduction reverts to it.

**Anchor.** A monotone Wilson lower bound on the incumbent's reproduction
rate, seeded from the confirmation batch extended to `ANCHOR_SEED_RUNS`
runs and raised only when an accepted candidate is adopted. It never falls
and is never fed by re-measurements of the standing incumbent.

**Gauntlet.** Under nondeterministic handling a candidate's first run
recruits it: in a fast sweep a miss rejects it at the cost of that one run,
with the evidence kept so a later retry has more power; a hit starts a
sequential test against `max(gamma * anchor, floor)` (gamma 0.8 below the
retention high-water, 1.0 above it) that needs at least `GAUNTLET_MIN_HITS`
interesting runs, rejects when the upper bound falls below the threshold or
the threshold has become unreachable within `GAUNTLET_CAP` runs, and on an
accept tops the ledger up to the seed size before latching. Adoption
requires an accept plus a smaller sort key, and raises the anchor.

**Alpha budget.** Every proposal is charged, before it runs, its exact
false-accept probability against a 2% fluke (an exact DP over the stopping
rule). When the per-reduction budget is spent, new candidates need more
interesting runs, up to `MIN_HITS_CEILING`.

**Confirmed-dry stopping.** A fixpoint under fast sweeps is not a
certificate, since candidates were rejected on single misses. At a fixpoint
`attempt_unstick` switches the policy into a confirmation sweep, in which
every candidate is driven to a bound verdict, and clears the pass
fingerprints so every pass runs again; an adoption during the sweep drops
back to fast sweeps. Only a confirmation sweep that adopts nothing ends the
reduction. Pass fingerprints are never recorded under nondeterministic
handling, since a fruitless pass may succeed on a retry.

## Differences from Hegel

- The gauntlet rejects as soon as the threshold is unreachable within the
  cap (Hegel's discovery bar has this rule, its gauntlet did not). This
  changes no accept decision and removes the cost lottery above the
  retention high-water, where a p = 0.9 candidate otherwise spent nearly
  the whole cap being rejected.
- Detection also runs periodically and at the end, and backtracks through
  the whole adopted history, because Shrink Ray adopts thousands of
  candidates before any check and Hegel's evidence was that the losses come
  from what happens before detection.
- No pools, timelines, boost, or persistence.

## Measuring it

`evaluation/benchmark.py` has `flaky_*` problems whose oracle flips a
seeded coin: one-sided at p = 0.9 / 0.5 / 0.2 and a two-sided variant with
a 2% background rate. Judge by calls, final size, the `rate` column (the
result's true reproduction probability) and the bug-kept check.
