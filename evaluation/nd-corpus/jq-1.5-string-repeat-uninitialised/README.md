# jq 1.5: repeated string multiplication reads freed memory

- **Source issue:** https://github.com/jqlang/jq/issues/2192
- **Tool:** jq 1.5, official release binary (`jq-osx-amd64`, run under
  Rosetta 2 on Apple silicon; `setup.sh` picks the asset for the host). The
  reporter was on jq 1.5.1 (Ubuntu) and also reproduced on jqplay's 1.6.
- **Input under reduction:** `original.jq`, the jq program from the issue's
  title and reproduction, `.*1024*1024` (reconstructed from the inline shell
  loop; the issue has no attachment; the issue's `| length` was only there to
  print a summary, see below). The JSON input is fixed to `"ab"` inside
  `test.sh` (the issue says "any input"; why `"ab"` rather than the issue's
  `"a"` is explained under the measured rates).

## The bug

`jq ".*1024*1024"` on a string sometimes produces a corrupted result (NUL
bytes and other garbage inside the repeated string, so `| length` reports
e.g. 1046536 instead of 1048576) and sometimes aborts with
`jq: jv_print.c:78: jvp_dump_string: Assertion 'c != -1' failed.`. The
reporter saw one wrong answer in ten runs. Fixed on master by commit
9163e09605383a88f6e953d6cb5cc2aebe18c84f (string multiplication keeps a
pointer into a buffer that `jv_string_append_buf` reallocates).

## Nondeterminism source

Reading freed memory: the bytes depend on what the allocator has done with
the reused block, so the output varies between runs. There is no switch that
makes it deterministic, so **no `test-deterministic.sh` is provided**.

## Detection

`test.sh` runs the candidate program on the input `"ab"` with jq 1.5 under
a 60 s timeout and is interesting iff jq either dies with SIGABRT (exit 134,
the assertion) or exits 0 with output containing the signature of
uninitialised memory inside a string: an escaped NUL (`\u0000`) or a
replacement character (`\ufffd`, jq's rendering of invalid UTF-8). A
well-behaved program derived from `.*1024*1024` by deletion cannot print
either escape, so the oracle does not depend on what the candidate
computes: `.*1024` (one multiplication), `.`, `.0` and an invalid program
are all correctly not interesting.

Two oracles were tried and rejected first:

- The issue's own `| length` check (wrong length vs 1048576) is tied to the
  program's meaning; any reduction changes the expected value.
- A differential oracle against the fixed jq 1.7.1 ("jq 1.5's output
  differs from 1.7.1's") is semantics-independent but exploitable: Shrink
  Ray reduced `.*1024*1024` to `.0` in 3 s, a program on which the two
  versions differ deterministically (20 / 20) for reasons unrelated to the
  bug. That is exactly the failure mode a corpus entry must not have.

## Measured reproduction rate (this machine, macOS arm64, x86_64 jq under Rosetta)

- `test.sh` on `original.jq`: **18 / 20** runs interesting (the driver's
  own measurement in the same session: 15 / 20); a 60-run direct measurement
  of the same signature gave 55 / 60.
- The rate is a step function of the allocation sizes, not a smooth one, so
  "any input" is true but the rate is not. All rows below are the corruption
  signature (or a differential check, which agreed with it where both were
  measured), 20-60 runs each:

| binary | program | input | rate |
|---|---|---|---|
| jq 1.5 | `.*1024*1024` | `"a"` | 29 / 30, 57 / 60 |
| jq 1.5 | `.*1024*1024` | `"ab"` | 55 / 60 (**used**) |
| jq 1.5 | `.*1024*1024` | `"abc"` | 0 / 60 |
| jq 1.5 | `.*1024` | `"ab"` | 0 / 20 |
| jq 1.6 | `.*1024*1024` | `"a"` | 30 / 30 (NUL bytes on every run) |
| jq 1.6 | `.*1024*1024` | `"ab"`, `"abc"`, longer strings, non-strings | 0-2 / 30 |
| jq 1.6 | `.*1024*1024 \| length` | `"a"` | 2 / 40, 0 / 20, 1 / 30 (the issue's own command) |
| jq 1.6 native arm64 build | `.*1024*1024` | `"a"`, `"ab"` / `"abc"` | 40 / 40 / 0 / 40 |

The issue's exact command (jq 1.6, `| length`) reproduces at roughly 4%
here, which is faithful but unusable: Shrink Ray gives an initial test case
ten tries before refusing to start, and at 4% it refused on the first
attempt. The `| length` symptom is rarer than the corruption itself because
`length` counts code points, so NUL bytes do not change it; only invalid
UTF-8 does.

- The assertion abort was never observed on this machine (~700 runs across
  versions).

## Setup notes

`setup.sh` downloads the release binary into `.tool/`. On Apple silicon
the jq 1.5 binary needs Rosetta 2 (`softwareupdate --install-rosetta` if it
is not present). Nothing is installed globally.
