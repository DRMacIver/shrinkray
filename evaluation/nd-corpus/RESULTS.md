# Results

Reductions run with `evaluation/nd_corpus.py` (basic UI, `--seed 0`,
`--no-history`, default parallelism) on a macOS arm64 machine on
2026-09-12, at commit 43ca627 (incumbent monitoring, hit minimum starting
at 2). Rates are `test.sh` passes out of 20 runs, measured independently
by the driver; "reported" is Shrink Ray's own final replay summary.

| entry | original rate | size | final size | seconds | reported | reduced rate |
|---|---|---|---|---|---|---|
| pygments-2.0.2-txt-lexer-guess-hashseed | 15/20 | 18 | 1 | 15 | 13/20 | 16/20 |
| jq-1.5-string-repeat-uninitialised | 20/20 | 12 | 7 | 50 | 20/20 | 20/20 |
| duckdb-1.1.2-read-csv-empty-names | 5/20 | 16 | 3 | 59 | 3/20 | 2/20 |
| lark-1.1.9-earley-ambiguous-quantifier | 11/20 | 135 | 31 | 647 | 7/20 | 12/20 |
| rustc-nightly-2025-06-05-zthreads-diagnostics | 3/20 | 54 | 31 | 51 | 20/20 | 20/20 |

Observations:

- Every result still reproduces at about the original's rate or better;
  none lost the bug.
- rustc went from a 3-in-20 race to a 31-byte variant that hits the E0391
  diagnostic on every run: the "raise the reproduction rate where
  possible" contract working on a real bug.
- jq's original reproduced 20/20 in this session (18/20 when the entry
  was built), so nondeterminism handling never engaged; the reduction was
  an ordinary deterministic one.
- The two rates below 1 in 4 (DuckDB, and rustc before reduction) are the
  expensive regime: wide intervals, many replays per decision.
- The lark entry is the slow one (each test run starts a Python
  interpreter and parses a grammar), not a reducer problem: 647 s for 135
  bytes is dominated by test runtime.

The remaining entries (lark 0.8.5, Z3) were not reduced in this session:
Z3's rate had dropped to 4/20 and lark 0.8.5 takes over ten minutes.

## 2026-09-13: the Jinja entry

Same driver and settings, at commit 899850c (after the reject bar and the
overhead work), on the entry added that day:

| entry | original rate | size | final size | seconds | reported | reduced rate |
|---|---|---|---|---|---|---|
| jinja-3.1.4-unpack-hashseed | 12/20 | 139 | 18 | 103 | 9/20 | 12/20 |

The result is a single two-name unpacking `set`, the smallest template that
can still compile two ways, and reproduces at the same rate as the original.
