# DuckDB 1.1.2: uninitialised memory in `read_csv` with empty column names

- **Source issue:** https://github.com/duckdb/duckdb/issues/14428
- **Tool:** `duckdb==1.1.2` (PyPI wheel), run under Python 3.12 from a uv venv.
- **Input under reduction:** `original.csv`, the four-row CSV from the issue
  (reconstructed verbatim from the inline text; the issue has no attachment).
  The query is fixed inside `query.py`:
  `from read_csv('<file>', header=false, names=['', ''])`.

## The bug

Passing two empty strings as column names to `read_csv` makes DuckDB 1.1.2
return wrong answers, garbage bytes, or segfault, varying from run to run.
Closed; fixed by duckdb/duckdb#14466.

## Nondeterminism source

Uninitialised memory / memory corruption in the CSV reader, so the observed
output depends on allocator state. No switch is known that makes the
manifestation deterministic, so **no `test-deterministic.sh` is provided**.

## Detection

`test.sh` runs `query.py` and is interesting iff the process is killed by a
signal (segfault or abort in the library) or the printed rows contain a NUL
byte (`\x00`, uninitialised memory leaking into a cell).

On this machine the query never returns the correct answer: every run
yields the wrong rows `[('b', ''), ('d', 'd'), ('f', 'f'), ('h', 'h')]` or a
variant with NUL-filled cells. A "wrong but clean" answer is therefore
deliberately *not* counted as interesting, because it would make the oracle
deterministic and hide the nondeterministic memory-corruption symptom the
issue is about.

## Measured reproduction rate (this machine, macOS arm64)

- `test.sh` on `original.csv`: **4 / 20** runs interesting.
- Direct measurement of the NUL symptom through the Python API: 6 / 20 and
  21 / 100. No segfault was observed in 140 runs.
- The DuckDB 1.1.2 CLI (`duckdb_cli-osx-universal`) printed the same wrong,
  NUL-free answer on 20 / 20 runs, so the CLI is not used.

## Setup notes

`setup.sh` creates `.tool/` as a uv venv with `duckdb==1.1.2`. Nothing is
installed globally.
