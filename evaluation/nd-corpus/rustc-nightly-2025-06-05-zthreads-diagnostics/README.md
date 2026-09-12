# rustc parallel front-end: diagnostics flip-flop between runs

- **Source issue:** https://github.com/rust-lang/rust/issues/142063
- **Tool:** `rustc 1.89.0-nightly (4b27a04cc 2025-06-04)`, installed as the
  rustup toolchain `nightly-2025-06-05` into a private `RUSTUP_HOME` under
  `.tool/` (about 500 MB).
- **Input under reduction:** `original.rs`, the two-line file from the issue
  (reconstructed verbatim from the inline snippet).

## The bug

Compiling

```rust
trait A { fn foo() -> A; }
trait B { fn foo() -> A; }
```

with `--edition=2024 -Zthreads=8 --crate-type=lib` sometimes reports a single
`E0391` "cycle detected when computing function signature of `A::foo`" and
sometimes three `E0782` "expected a type, found a trait" errors. Which set
appears depends on how the parallel well-formedness checks are scheduled
across threads.

On this machine there is a third outcome: the E0391 diagnostic followed by
an internal compiler error (`entered unreachable code` at
`compiler/rustc_middle/src/values.rs:62`, exit 101), which is
rust-lang/rust#142064 on the same input (reported there with `-Zthreads=2`
as deterministic; here it is not).

## Nondeterminism source

Thread scheduling in the parallel query system (`-Zthreads=8`). With
`-Zthreads=1` the E0391 outcome does not occur, so there is no deterministic
variant of the *interesting* outcome and **no `test-deterministic.sh` is
provided**.

## Detection

`test.sh` compiles the candidate with the flags above under a 120 s timeout
and is interesting iff rustc exits 1 and its stderr contains
`error[E0391]`. The E0782 outcome, the ICE (exit 101, even though it also
prints E0391), successful compilation, and any other error are not
interesting. `RUSTC_ICE=0` is set so the ICE outcome does not write
`rustc-ice-*.txt` dumps into the working directory.

## Measured reproduction rate (this machine, macOS arm64, 8+ cores)

The mix of outcomes shifts with machine load, which changes thread timing;
the interesting (clean E0391) outcome stayed in the 15-30% range across
conditions, the other two traded places:

- `test.sh` on `original.rs`: **6 / 20** runs interesting; the other 14
  were ICEs (other builds were running at the time).
- First batch, machine otherwise idle: 4 / 20 E0391, 16 / 20 E0782, 0 ICEs.
- A later batch with several reductions and a benchmark running: 3 / 20
  clean E0391, 17 / 20 ICE, 0 E0782.

Expect the rate to move if reductions are run in parallel with other work.

## Setup notes

`setup.sh` uses the `rustup` binary already on `PATH` but sets
`RUSTUP_HOME=.tool/rustup`, so the toolchain is downloaded into the entry
directory and the user's own `~/.rustup` is untouched. `test.sh` sets the
same `RUSTUP_HOME` and invokes `rustc +nightly-2025-06-05`.
