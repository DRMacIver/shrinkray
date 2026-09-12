# lark 0.8.5: nondeterministic Earley parse of an ambiguous grammar

- **Source issue:** https://github.com/lark-parser/lark/issues/595
- **Tool:** `lark-parser==0.8.5` (PyPI), run under Python 3.12 from a uv venv.
- **Input under reduction:** `original.lark`, the ~50-line grammar from the
  issue (reconstructed verbatim from the inline `GRAMMAR` string; the issue
  has no attachment). The parsed text is the issue's 8-line `x: 1 ...`
  snippet, stored as `input.txt` and read by `parse.py`.

## The bug

The grammar (a cut-down CUE-like language) is ambiguous. With
`parser='earley'`, lark 0.8.5 returns a different parse tree depending on
`PYTHONHASHSEED`: the reporter's diff shows the `c` in `cd : 30` being
attached under an extra `embedding` / `aliasexpr` chain for one seed and
directly under `letter` for another. The maintainers closed it as resolved by
later work on ambiguity handling.

## Nondeterminism source

Python hash randomisation (set iteration order inside the Earley parser's
ambiguity resolution). Fixing `PYTHONHASHSEED` makes any run reproducible, so
a deterministic variant exists.

## Detection

`test.sh` runs `parse.py` twice on the candidate grammar: once with the
default (randomised) hash seed and once with `PYTHONHASHSEED=0`, and is
interesting iff both parse successfully and the pretty-printed trees differ.
A grammar that parses the input unambiguously can never be interesting, so
reduction cannot "succeed" by removing the ambiguity.

`test-deterministic.sh` pins the candidate run to `PYTHONHASHSEED=2` (which
on the original grammar yields the other tree than seed 0) and is interesting
iff the two fixed-seed trees differ.

## Measured reproduction rate (this machine, macOS arm64)

- `test.sh` on `original.lark`: **10 / 20** runs interesting.
- Direct measurement of the underlying split over 20 hash-randomised runs:
  16 runs gave one tree, 4 the other.
- `test-deterministic.sh`: 3 / 3 (deterministic, as expected).

## Setup notes

`setup.sh` creates `.tool/` as a uv venv with `lark-parser==0.8.5` (the
package name used before lark 1.0). Nothing is installed globally.
