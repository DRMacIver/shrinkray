# Pygments 2.0.2: lexer guessed for a .txt file depends on PYTHONHASHSEED

- **Source issue:** https://github.com/pygments/pygments/issues/852
  (migrated from Bitbucket issue 1145, filed 2015-08-28; still open).
- **Tool:** Pygments 2.0.2 (the current release when the issue was filed;
  the issue names no version) on **CPython 3.5.10**, both built/unpacked by
  `setup.sh` into `.tool/`.
- **Input under reduction:** `original.txt`, the one-line file from the
  issue, `foo bar@email.com` (reconstructed from the inline text; the issue
  has no attachment). The bug needs the file to be named `*.txt`, so
  `guess.py` always passes the name `test.txt` to Pygments together with the
  candidate's content.

## The bug

`get_lexer_for_filename("test.txt", code)`, which is what `pygmentize
test.txt` uses, finds several lexers registered for `*.txt` (Text,
RobotFramework, ResourceBundle in 2.0.2). All of them score the content 0,
so the sort by score leaves them in the order they were pulled out of the
`LEXERS` dict, and the last one wins. The issue reports RobotFramework
(seed 1), Resource (seed 2) or Text (seed 8) for the same file.

## Nondeterminism source

Python hash randomisation applied to **dict** iteration order. That is why
the entry pins CPython 3.5: from 3.6 on dicts iterate in insertion order and
this tie-break is stable (verified here: Pygments 2.0.2 through 2.19.2 on
Python 3.12 always answer "Text only" for every seed, and 2.2.0+ answer
"Text only" even on 3.5 because the tie-break changed). On 3.5 with
2.0.2 or 2.1.3 the three answers all appear. Fixing `PYTHONHASHSEED`
makes any run reproducible, so a deterministic variant exists.

## Detection

`test.sh` runs `guess.py` on the candidate with hash randomisation on and is
interesting iff Pygments returns a lexer other than `Text only`, the correct
answer for plain text. A candidate Pygments rejects (`ClassNotFound`, a
decoding failure) exits non-zero and is not interesting.

The tie-break ignores the content entirely, so an empty file named
`test.txt` is misdetected just as often (a first reduction attempt went
straight to 0 bytes, still interesting on 11 / 20 runs). The test therefore
also requires the candidate to contain at least one non-whitespace byte, so
that a reduction has a one-character minimum to find rather than tripping
Shrink Ray's trivial-result guard.

`test-deterministic.sh` is the same with `PYTHONHASHSEED=1`, a seed on which
the original is misdetected as RobotFramework on every run.

## Measured reproduction rate (this machine, macOS arm64)

- `test.sh` on `original.txt`: **14 / 20** and **15 / 20** runs interesting
  (two batches; the second after the non-blank requirement was added).
- A one-character file `x`: 10 / 20. Empty and whitespace-only files: 0 / 10
  each (rejected by the non-blank requirement, not by Pygments).
- Direct measurement over 20 hash-randomised runs: 9 RobotFramework,
  4 ResourceBundle, 7 Text only. Seeds 1-12: 8 RobotFramework,
  2 ResourceBundle, 2 Text only.
- `test-deterministic.sh`: 3 / 3.

## Setup notes

`setup.sh` downloads the CPython 3.5.10 tarball and builds it into
`.tool/python` (about four minutes; the only non-trivial setup in this
corpus). Two adjustments are needed on a modern toolchain and are made by
the script: 3.5's `configure` rejects `arch` output other than i386/ppc on
macOS, so `arm64` is accepted alongside `i386`; and implicit function
declarations are downgraded from errors to warnings for modern clang. It
then unpacks the Pygments 2.0.2 sdist into `.tool/pygments`, which the test
puts on `PYTHONPATH` (uv cannot manage a 3.5 environment). Nothing is
installed globally.
