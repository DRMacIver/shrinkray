# Nondeterministic-bug corpus

Real bugs whose interestingness test does not give the same answer on every
run: hash-seed-dependent output, uninitialised memory, thread scheduling.
They exist to evaluate Shrink Ray's nondeterminism handling against real
oracles rather than the synthetic coin-flip problems in `benchmark.py`.
Candidates were researched in `notes/nondeterminism-corpus-candidates.md`;
every entry here was set up and measured on a macOS arm64 machine, and only
bugs whose reproduction rate on the original input was strictly between 0
and 20 out of 20 runs were kept.

## Layout

One directory per entry, `evaluation/nd-corpus/<slug>/`:

| file | purpose |
|---|---|
| `README.md` | source issue, pinned tool, the bug and its nondeterminism source, how the test detects it, measured reproduction rate, setup notes |
| `setup.sh` | idempotent; installs the pinned tool into `<slug>/.tool/` (git-ignored) and nothing anywhere else |
| `original.<ext>` | the triggering input, as close to the issue's attachment as possible (the README says when it was reconstructed from inline text) |
| `test.sh` | the interestingness test in Shrink Ray's convention: called with the candidate path as `$1`, exits 0 iff the bug reproduced on *this* run; self-contained (finds `.tool` relative to itself), cwd-independent, runtime bounded with `timeout` |
| `test-deterministic.sh` | optional: the same bug pinned to a fixed seed/environment, for comparing a reduction under nondeterminism against deterministic ground truth |
| other files | helper scripts the test needs (`parse.py`, `query.py`, ...) and fixed secondary inputs |

## Entries

| entry | tool | symptom | source of nondeterminism | rate on original | deterministic variant |
|---|---|---|---|---|---|
| `duckdb-1.1.2-read-csv-empty-names` | duckdb 1.1.2 (PyPI) | NUL garbage / crash in `read_csv` result | uninitialised memory | 4/20 | no |
| `jinja-3.1.4-unpack-hashseed` | Jinja2 3.1.4 (PyPI) | generated Python source differs between runs | hash seed (sets) | 15/20 | yes |
| `jq-1.5-string-repeat-uninitialised` | jq 1.5 release binary | NUL/invalid-UTF-8 garbage in output / assertion abort | reads freed memory | 18/20 | no |
| `lark-0.8.5-earley-hashseed-tree` | lark-parser 0.8.5 (PyPI) | parse tree differs between runs | hash seed (sets) | 10/20 | yes |
| `lark-1.1.9-earley-ambiguous-quantifier` | lark 1.1.9 (PyPI) | parse tree differs between runs | hash seed (sets) | 7/20 | yes |
| `pygments-2.0.2-txt-lexer-guess-hashseed` | Pygments 2.0.2 on CPython 3.5.10 (source build) | wrong lexer guessed | hash seed (dict order, pre-3.6) | 15/20 | yes |
| `rustc-nightly-2025-06-05-zthreads-diagnostics` | rustc nightly-2025-06-05 | different diagnostics between runs | thread scheduling (`-Zthreads=8`) | 6/20 | no |
| `z3-4.12.1-dminor-intermittent-segfault` | Z3 4.12.1 release binary | SIGSEGV | memory layout | 54/60 | no |

Each entry's README has the exact numbers and how they were measured.

## Running the driver

`evaluation/nd_corpus.py` sets entries up, measures the reproduction rate of
the original input, reduces a copy with the installed `shrinkray` CLI in
basic UI mode, and independently re-measures the rate on the reduced file:

```bash
uv run python evaluation/nd_corpus.py --list
uv run python evaluation/nd_corpus.py                         # all entries
uv run python evaluation/nd_corpus.py jq-1.5-string-repeat-uninitialised
uv run python evaluation/nd_corpus.py --no-reduce             # rates only
uv run python evaluation/nd_corpus.py --deterministic         # test-deterministic.sh + --assume-deterministic
uv run python evaluation/nd_corpus.py --runs 50 --json out.json --keep-results /tmp/nd
```

Useful options: `--timeout` (per test run, passed to `shrinkray --timeout`),
`--max-seconds` (kill a reduction that runs too long), `--parallelism`,
`--keep-results DIR` (keeps each reduced file and the shrinkray log).
Reductions run with `--no-llm --no-history --seed 0 --trivial-is-not-error`;
the last because a real bug can have a one-byte reproducer (Pygments does),
and Shrink Ray's trivial-result guard would otherwise exit 1 and skip its
final replay summary.

Setup notes worth knowing before a first run: the Pygments entry builds
CPython 3.5 from source (about four minutes); the rustc entry downloads a
~500 MB toolchain into its `.tool/`; the jq entry's 1.5 binary is x86_64 and
needs Rosetta 2 on Apple silicon.

## Rejected candidates

Measured on this machine and not included, with the reason:

- **Z3 5.0.0 MBQI segfault (Z3Prover/z3#10385).** The attached
  `repro_z3_500_mbqi_segv.smt2` (from the reporter's gist) crashes on
  **20 / 20** runs with the official arm64 5.0.0 binary: deterministic in
  *whether* it crashes (only the fault site varies), so it does not exercise
  nondeterminism handling. Each run also takes ~22 s. Reductions of it might
  well become flaky, but that is speculation, not a measurement.
- **QuickJS-ng cycle-GC use-after-free (quickjs-ng/quickjs#1570).** Built
  commit d950d55 four ways (Release; Release + ASan; Debug + GC stress;
  Debug + ASan + GC stress). The thread's actual reproducer `min1570b.js`
  ran clean on every build (0 / 20 on the stock build, 0 / 20 with ASan,
  0 / 3 on each GC-stress build), and the schematic PoC in the report just
  spins (as the maintainer also found). The reporter's deterministic
  reproduction was on Linux/glibc; it does not transfer to macOS arm64.
- **Go 1.23.0 range-over-func miscompile (golang/go#69507).** The playground
  program (with `golang.org/x/exp` pinned to a September 2024 commit and
  `GOTOOLCHAIN=go1.23.0`) crashes with `fatal error: fault` on **20 / 20**
  runs, twice. Only the faulting address varies, so on this machine it is a
  deterministic crash and belongs in the ordinary corpus, not here.
- **CPython 3.14.6 specialised-CALL corruption (python/cpython#155145).**
  Built 3.14.6 from source (uv 0.9.18 has no download for it). One run of the
  reproducer completed without incident and took **550 s**; the maintainer
  who tried could not reproduce it either. Even if it does fail sometimes,
  nine minutes per interestingness-test run makes it unusable for reduction
  experiments.
- **Pygments on modern Python.** Pygments 2.0.2-2.19.2 on CPython 3.12 (and
  2.2.0+ on 3.5) answer "Text only" for every hash seed; the bug needs
  pre-3.6 dict ordering *and* the 2.0.x/2.1.x tie-break, which is why the
  kept entry builds Python 3.5.
- **jq 1.6 with the issue's exact command.** `jq-1.6` on `.*1024*1024 |
  length` reproduces at about 4% here (2 / 40, 0 / 20, 1 / 30), too rare for
  Shrink Ray to get past its startup gate; the kept entry uses jq 1.5 (the
  reporter's version) and the raw `.*1024*1024` program, at 18 / 20. See the
  entry's README for the full version/input table and for the differential
  oracle (jq 1.5 vs 1.7.1) that was tried and rejected because Shrink Ray
  reduced it to a deterministic non-bug in three seconds.
- **DuckDB CLI.** The `duckdb_cli-osx-universal` 1.1.2 binary gives the same
  clean wrong answer on 20 / 20 runs; only the Python wheel shows the
  nondeterministic garbage, so the entry uses the wheel.
- **Z3 4.12.1 ddsmt-minimised files (from the same #6615 zip).** 0 / 20
  crashes each; the reporter warned they only crash on the platform they
  were minimised on. The unminimised `original.smt2` is used instead.
- **lark #1386, Z3 #3877, mruby #3423, Lua GC bugs, SymPy, dask, NetworkX,
  and the rest of the candidate list** were not attempted: the recommended
  order already yielded enough entries, and the remaining ones need source
  builds at old commits or have weaker oracles.
