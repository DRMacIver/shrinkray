# lark 1.1.9: Earley picks a different derivation depending on the hash seed

- **Source issue:** https://github.com/lark-parser/lark/issues/1434
- **Tool:** `lark==1.1.9` (PyPI), run under Python 3.12 from a uv venv.
- **Input under reduction:** `original.lark`, the grammar from the issue
  (reconstructed verbatim from the inline Python snippet; the issue has no
  attachment). The parsed text is fixed to the issue's `a.?` inside
  `parse.py`.

## The bug

The grammar is ambiguous: a regex quantifier such as `?` can also be lexed as
a `printable_char`. lark 1.1.9's Earley parser resolves the ambiguity by
iterating over Python sets, so which derivation wins depends on
`PYTHONHASHSEED`. Two different trees come out of repeated runs of the same
program:

```
start                      start
  printable_char  a          printable_char  a
  .                          factor
  printable_char  ?            .
                               quantifier
```

Fixed on lark master after the issue (the reporter confirmed); pinned 1.1.9
still shows it.

## Nondeterminism source

Python hash randomisation (set iteration order inside the Earley forest
resolution). Fixing `PYTHONHASHSEED` makes any given run reproducible, so a
deterministic variant exists.

## Detection

`test.sh` runs `parse.py` twice on the candidate grammar: once with the
default (randomised) hash seed and once with `PYTHONHASHSEED=0`, and is
interesting iff both parse successfully and the pretty-printed trees differ.
This oracle cannot be satisfied by a grammar that parses `a.?` in only one
way, so reduction cannot "succeed" by deleting the ambiguity.

`test-deterministic.sh` pins the candidate run to `PYTHONHASHSEED=1`, which
on the original grammar selects the other derivation than seed 0, and is
interesting iff the seed-1 and seed-0 trees differ. This is the deterministic
ground truth for comparison.

## Measured reproduction rate (this machine, macOS arm64)

- `test.sh` on `original.lark`: **7 / 20** runs interesting.
- Direct measurement of the underlying split over 20 hash-randomised runs:
  11 runs gave one tree, 9 the other.
- `test-deterministic.sh`: 3 / 3 (deterministic, as expected).

## Setup notes

`setup.sh` creates `.tool/` as a uv venv with `lark==1.1.9`. Nothing is
installed globally.
