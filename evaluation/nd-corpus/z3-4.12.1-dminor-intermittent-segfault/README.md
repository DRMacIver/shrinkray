# Z3 4.12.1: intermittent segfault on a Dminor-generated SMT-LIB file

- **Source issue:** https://github.com/Z3Prover/z3/issues/6615
- **Tool:** Z3 4.12.1, official release binary
  (`z3-4.12.1-arm64-osx-11.0.zip` on this machine; `setup.sh` picks the
  asset for the host).
- **Input under reduction:** `original.smt2` (122 KB), the `original.smt2`
  from the issue's `examples.zip` attachment, byte for byte. The zip also
  contains two ddsmt-minimised files (`minimized-macos.smt2`,
  `minimized-ubuntu.smt2`); neither crashes on this machine (0 / 20 each),
  matching the reporter's note that they only crash on the platform they
  were produced on, so they are not included.

## The bug

Z3 4.12.1 (and commit 25d45a350) segfaults "maybe 1/10 times" on a file
produced by the reporter's Dminor refinement type checker. No root cause is
recorded on the issue (still open at research time). Plain `z3 file.smt2`
with default options triggers it.

## Nondeterminism source

Memory-layout dependent: same binary, same file, same options, different
outcome run to run. There is no known switch that makes it deterministic, so
**no `test-deterministic.sh` is provided**.

## Detection

`test.sh` runs `z3 "$1"` under a 60 s timeout and is interesting iff the
process exits with status 139 (killed by SIGSEGV). Any answer, error or
timeout is not interesting.

## Measured reproduction rate (this machine, macOS arm64)

The rate here is much higher than the reporter's ~10%:

- `test.sh` on `original.smt2`: **54 / 60** runs interesting (batches of 20
  gave 20, 17 and 17).
- Running the bare binary without the `timeout` wrapper: 18 / 20 and 17 / 20.
- Each run takes about 0.25 s.
- The rate varies a lot between sessions: a later session on the same
  machine measured 0 / 5 and then 4 / 20 with the same binary and input.
  Treat it as a low-and-variable-rate entry, not the ~90% above.

## Setup notes

`setup.sh` downloads the release zip into `.tool/` and keeps only `bin/`.
Nothing is installed globally.
