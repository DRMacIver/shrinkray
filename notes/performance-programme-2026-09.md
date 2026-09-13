# The September 2026 performance programme

Between 2026-09-12 and 2026-09-13 an autonomous agent (Codex) ran a bounded
performance investigation on the nondeterminism branch. This note records
what it tried, what was kept, what was rejected and why, so that the work
is not repeated. The raw material (about a gigabyte of result JSON, source
snapshots, logs, figures and a draft paper) was deliberately not committed;
the maintainer has it on disk under `notes/nondeterministic-shrinking/`,
`notes/performance-program/` and `notes/performance-diagnostics/`.

## What was kept

Four overhead reductions in the production reducer, each supported by an
equivalence property test and an operation-level microbenchmark against
the previous implementation. All are exact: they change how fast the
reducer does something, never what it does.

| Change | Operation-level speedup | Whole-run effect |
| --- | --- | --- |
| Lazy layout tie-breakers in `reflow_sort_key` (`problem.py`): the raw layout distance is only computed on a canonical-content tie | Sorting candidate sets 1.3–10x faster (largest on Z3's 122 KB input) | Neutral on real one-worker runs |
| Batched scanners, header prefilter and a bounded inline-fragment cache in `reformat.py` | Formatting 1.2–1.5x faster on corpus inputs | Neutral to slightly better on large synthetic inputs |
| Tuple normalisation in `Cuts.combine` and slice-joining in `Cuts.apply` (`patching.py`) | Combining 1.2–2.6x faster; applying to bytes 8x–1000x faster (allocation-free slicing instead of an int-at-a-time rebuild) | Neutral; copying rarely dominates oracle cost |
| Patch workers pull from a shared iterator with a scheduler checkpoint every 64 patches instead of a channel receive per patch | Scheduler iterations on a large synthetic nondeterministic problem fell from ~565,000 to ~32,000; wall time 14.5 s to 6.7 s (parallelism 1, cheap oracle) | The one change that is visible end to end on cheap oracles |

Measured together on `evaluation/benchmark.py` (cheap in-process oracles,
parallelism 1, idle machine, 2026-09-13), the tip of the branch before this
cleanup versus after it:

| Problems | Wall time, after / before |
| --- | --- |
| All 19 problems | 0.42 (444 s to 187 s) |
| Python-syntax-constrained (`python_syntax`, `corpus_mypy`, `corpus_pylint`, `coupled_arity_python`) | 0.09–0.16 |
| `flaky_*` nondeterministic problems | 0.34–0.54 (partly fewer calls, from the reject bar) |
| `keep_markers`, `deep_parens`, `coupled_total_text` | 0.31, 0.76, 0.74 |

Interestingness calls on the deterministic problems were within a few of
the previous tip, so this is overhead, not search. With a real oracle
costing hundreds of milliseconds per call these gains are diluted by
the oracle, which is what the programme's neutral whole-run results on
real corpus entries show.

Also kept: three real races in the external-reducer protocol, found when
the experiment's driver stalled (see the commit `fix: close three races
in the external reducer protocol`), and a `reject_bar` in the gauntlet
that drops a candidate early once it is clearly worse than what the
reduction has been adopting (the maintainer's own change, merged from the
same working tree).

## What was rejected

**Retry-budget search.** The bulk of the programme was an alternative
search algorithm for nondeterministic tests: keep the production passes,
but treat every verdict during search as speculative, and only accept a
result once a *fresh* ledger of replays certifies that its mean cost to
the next reproduction is at most `max(10, 1.25 / p_original)` executions
(an e-value / anytime-valid test, tuned through some twenty named policy
variants). Against the production gauntlet at equal call caps it was:

- better on a few synthetic landscapes (one-sided p = 0.2, a
  "nonmonotone valley"), where it finished in a third of the calls;
- much worse on two-sided noise (a 2% background hit rate turned a
  51-byte result into 929 bytes) and on several real corpus entries
  (Rust 30 → 50 bytes, Z3 32 KB → 69 KB, both lark entries larger);
- prone to returning the original input unchanged when the wall-clock
  budget expired before it had certified anything.

Its final full-matrix evaluation (690 runs over 16 real bug families,
four of them reserved and unseen during tuning) did not meet its own
promotion criterion, and the write-up recommended keeping the production
policy. Nothing of it is in the tree.

**Evidence compaction** (storing rejected candidates by digest rather than
full bytes in the experimental engine): saved memory, but a clean Z3
ablation was 10% slower with worse outputs. Reverted before the final
evaluation.

**Disabling the initial-cut inactivity watchdog** (the five-second timer
in `ShrinkRay.initial_cut`): only the experimental engine needed it. The
parameter was reverted along with the engine.

**A separate benchmark harness** (`evaluation/performance/`): a manifest of
16 real bugs with an A/B runner that compares two source trees in-process
on wall time, CPU, memory and time-to-target, plus a protocol-freezing
mechanism for reserved evaluations. It duplicated `run.py` and
`nd_corpus.py` with a second oracle format and a second set of venvs, and
the freeze machinery is research process rather than something the
project needs. It was dropped; the four new real bugs it introduced were
moved into the standard corpora instead (below).

## Unexplained observation

With four workers, Ruff's time to reach a tenth of its original size was
1.3–1.5x slower on the optimised tree than on the baseline in two separate
batches of three pairs, while one-worker Ruff and every other case were
neutral. One-file ablations could not attribute it to any single change:
each individual rollback moved the median in both directions with pair
ranges far wider than the effect. Traces show the four-worker histories
diverge in the order the initial comment-cut patches are dispatched and
adopted, after which the two trees propose different candidates; trio's
scheduler randomness is independent of the reducer seed, so paired runs
are not paired trajectories. If this is real it is a scheduling
interaction, not a defect in any primitive. Worth a look if parallel
reductions ever seem slower than expected.

## Benchmarks that exist now

- `evaluation/benchmark.py`: synthetic problems with cheap in-process
  oracles, including the `flaky_*` nondeterministic ones. Judge by
  interestingness calls; because the oracles are nearly free, its
  `seconds` column is also a direct measure of reducer overhead, which is
  how the retained optimisations above should be checked in future.
- `evaluation/run.py` and `evaluation/corpus/`: real deterministic bugs,
  now including `toml-0.10.2-mixed-array`, `sqlparse-0.4.4-nested-list`
  and `sympy-1.8-piecewise-subs` from the programme's reserved set.
- `evaluation/nd_corpus.py` and `evaluation/nd-corpus/`: real
  nondeterministic bugs, now including `jinja-3.1.4-unpack-hashseed`,
  which has a deterministic variant for ground-truth comparison.
