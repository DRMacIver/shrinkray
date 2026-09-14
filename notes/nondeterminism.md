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

Under handling, a candidate run that times out is not a sample either: the
candidate is rejected for as long as the timeout it ran under stands (the
rejection is retried after a raise, as under a deterministic test) without
the timeout entering its ledger, and a timed-out seed run just ends the
top-up. The batches that judge fallback test cases (confirmation and
recovery) do count a timeout as a miss, so that they stay bounded and err
towards the older entry.

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
and is never fed by re-measurements of the standing incumbent. Two things
keep the running maximum honest, because a reduction adopts hundreds of
candidates and the luckiest of their batches looks far better than the
rate they all share (the first benchmark showed the anchor climbing to
0.58 on a p = 0.5 bug, after which most valid reductions were rejected at
nearly the full cap): a raise uses only the runs recorded *after* the
accept decision (the stopping rule selected on the earlier ones), and the
n-th raise attempt uses a bound at the Bonferroni level for n attempts
(`anchor_z`).

**Gauntlet.** Under nondeterministic handling a candidate's first run
recruits it: in a fast sweep a miss rejects it at the cost of that one run,
with the evidence kept so a later retry has more power; a hit starts a
sequential test against `max(gamma * anchor, floor)` (gamma 0.8 below the
retention high-water, 1.0 above it) that needs at least `GAUNTLET_MIN_HITS`
interesting runs, rejects when the upper bound falls below the threshold or
the threshold has become unreachable within `GAUNTLET_CAP` runs, and on an
accept tops the ledger up to the seed size before latching. Adoption
requires an accept plus a smaller sort key, and raises the anchor.

**Multiplicity by feedback, not by an assumed noise model.** Hegel charges
every proposal its exact false-accept probability against a fluke assumed
to reproduce at 2%, and escalates the hit minimum when the budget runs
out. That bound is only as good as the 2%: at a real background rate of
10% the charges are fiction, and Shrink Ray proposes tens of thousands of
candidates. Instead the run watches its incumbent. Adoption seeds the
*incumbent monitor* with the candidate's post-decision runs, and the
periodic verify keeps feeding it fresh replays; the final measurement
before the reducer may finish adds twenty more. When the incumbent's upper
bound falls below the gauntlet threshold (with at least `MONITOR_MIN_RUNS`
replays) it did not reproduce at the rate the gauntlet required: a false
accept, or drift the anchor never covered. The run then backtracks
through the adopted history to the newest entry that clears the gauntlet
at the current anchor, and raises the hit minimum by one (from
`INITIAL_MIN_HITS` up to `MIN_HITS_CEILING`) for every candidate proposed
from then on. A one-sided test, where nothing without the bug is ever
interesting, never trips the monitor and pays only the small starting
minimum; a two-sided one escalates until false accepts become rare
relative to the run, at a cost of one detected-and-recovered event per
escalation step.

**Confirmed-dry stopping.** A fixpoint under fast sweeps is not a
certificate, since candidates were rejected on single misses. At a fixpoint
`attempt_unstick` switches the policy into a confirmation sweep, in which
every candidate is driven to a bound verdict, and clears the pass
fingerprints so every pass runs again; an adoption during the sweep drops
back to fast sweeps. Only a confirmation sweep that adopts nothing ends the
reduction. Pass fingerprints are never recorded under nondeterministic
handling, since a fruitless pass may succeed on a retry.

## Differences from Hegel

- Anchor raises are multiplicity-corrected and use post-decision evidence
  only (above). Hegel takes the max over every adopted candidate's full
  twenty-run bound, which is the selection effect behind its replay cost
  lottery.
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

## Alternatives tried

An alternative "retry budget" search, which treats every verdict during
search as speculative and only accepts a result once fresh replays
certify its expected cost to the next reproduction, was implemented and
evaluated against this design in September 2026. It won on a few
one-sided synthetic landscapes and lost on two-sided noise and most real
entries, so the gauntlet above stays. See
`notes/performance-programme-2026-09.md`.

## Measuring it

`evaluation/benchmark.py` has `flaky_*` problems whose oracle flips a
seeded coin: one-sided at p = 0.9 / 0.5 / 0.2 and a two-sided variant with
a 2% background rate. Judge by calls, final size, the `rate` column (the
result's true reproduction probability) and the bug-kept check.
