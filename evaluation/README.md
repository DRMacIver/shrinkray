# Evaluation corpus

A corpus of **real bugs in real tools** — inputs that make a specific
pinned version of a compiler, formatter, linter, JSON parser, or SAT
solver crash — used to evaluate shrink ray's reduction quality on
realistic material across the formats it supports (C/C++, Python, JSON,
DIMACS CNF), and to compare it against other reducers.

Each entry is a plausible, pre-reduction-sized input with a genuine bug
trigger buried inside realistic scaffolding (application-shaped code,
config-shaped JSON, a problem-shaped CNF). Reducing it should strip the
scaffolding back down to the essential trigger, exercising the passes
shrink ray dispatches for that format.

## Layout

```
evaluation/
├── run.py        # reduce entries with shrink ray, write result.json
├── report.py     # regenerate the tables in RESULTS.md
├── RESULTS.md    # generated tables + hand-written analysis
├── corpus/<id>/  # one directory per bug
└── creduce/      # c-reduce comparison driver (C/C++ entries)
```

Each `corpus/<id>/` holds:

- `original.<ext>` — the realistic pre-reduction input.
- `meta.json` — how to run the buggy tool and recognise the bug (schema below).
- `setup.sh` — only for `command` oracles: idempotent pinned build into `evaluation/tools/` (gitignored).
- `shrinkray_reduced.<ext>` — shrink ray's output (committed for comparison).
- `creduce_reduced.<ext>` — c-reduce's output, where a comparison exists.
- `result.json` — metrics from the last `run.py` reduction (sizes, time).
- `work/` — gitignored scratch: the in-progress reduction (resumable), the
  generated check script, per-entry venv.

## meta.json schema

```json
{
  "id": "mypy-0971-crash-example",
  "format": "python",
  "tool": "mypy 0.971",
  "extension": ".py",
  "bug_url": "https://github.com/python/mypy/issues/NNNN",
  "symptom": "What goes wrong and when.",
  "oracle": { ... },
  "interesting": {
    "output_contains": ["INTERNAL ERROR", "in some_function"],
    "exit_code": 134
  },
  "parallelism": 1,
  "timeout": 10
}
```

- `interesting` — how the generated check script recognises the bug in
  the tool's combined stdout+stderr and exit status. `output_contains`
  (string or list; all must match) should pin the *specific* bug (an
  assertion message, a distinctive traceback frame), not just any
  failure. `exit_code` is exact (signals appear as 128+N).
- `parallelism` — passed to shrinkray if present. Docker entries under
  emulation set 1 (emulation penalises concurrency); native-tool entries
  omit it and use shrink ray's default.
- `timeout` — per-call timeout passed to shrinkray, avoiding its
  dynamic-timeout calibration.
- `shrinkray_args` — optional extra flags for the shrinkray invocation.
  For example the black entry needs `--formatter=none`: shrink ray's
  default formatting step runs a *current* black on each Python
  candidate, which rewrites away the very construct that crashes the
  old black under test.

`oracle` declares how to run the tool, one of three types:

```json
{"type": "docker", "image": "gcc:4.9", "platform": "linux/amd64",
 "command": ["g++", "-std=c++1y", "-x", "c++", "-c", "{file}", "-o", "/dev/null"]}

{"type": "venv", "python": "3.10", "requirements": ["black==21.12b0", "click==8.0.4"],
 "command": ["black", "--check", "{file}"]}

{"type": "command", "setup": "setup.sh",
 "command": ["{tools}/cadical-1.5.0/build/cadical", "{file}"]}
```

- **docker** — a persistent container per entry (started once, reused
  across the thousands of oracle calls a reduction makes), bind-mounting
  the work dir at `/w`. `{file}` becomes the in-container candidate path.
- **venv** — `uv venv --python <python>` + `uv pip install <requirements>`
  into `work/venv`, whose `bin` is prepended to PATH.
- **command** — arbitrary command; `setup.sh` (run from the entry dir
  with `$TOOLS_DIR` = `evaluation/tools/`) builds the pinned tool once.

Placeholders in `command` and `env` values: `{file}` (candidate copy),
`{tools}`, `{entry}`, `{workdir}`. An optional `oracle.env` object
exports environment variables in the check script.

## Running

```bash
# Verify every entry still reproduces its bug
python3 evaluation/run.py --check

# Reduce every entry (writes work/reduced.<ext>, resumable; on success
# updates shrinkray_reduced.<ext> and result.json)
python3 evaluation/run.py

# Reduce / check specific entries
python3 evaluation/run.py mypy-0971-crash-example

# Regenerate the tables in RESULTS.md
python3 evaluation/report.py
```

The C/C++ entries use amd64-only compiler images which run under
emulation on Apple Silicon: slow but functional, and single-threaded
(`parallelism: 1`) because emulation penalises concurrency. The other
entries run natively at full parallelism and reduce in seconds to
minutes.

## Provenance

Every entry corresponds to a real bug: `bug_url` links the upstream
issue/PR/changelog, and `symptom` describes the failure. The C/C++
triggers were found by screening gcc's own `g++.dg/cpp1y` testsuite and
C++11/14 dark-corner snippets against old compiler Docker images; the
Python/JSON/CNF entries reproduce reported crashes in pinned tool
versions (formatters, type checkers, linters, JSON parsers, SAT
solvers). Each trigger is then embedded in a larger realistic input and
the failure re-verified.

## Comparing against c-reduce

`creduce/` holds a driver that reduces the C/C++ entries with c-reduce
2.11.0 (and its bundled `clang_delta`) against the identical
compiler-in-Docker oracle, so the two tools are compared on equal
footing. Build the c-reduce host image once, then run the driver:

```bash
docker build --platform linux/amd64 -t creduce-host \
    -f evaluation/creduce/Dockerfile.creduce evaluation/creduce

# one entry (writes <entry>/creduce_reduced.cpp)
evaluation/creduce/run_creduce.sh gcc49-udlit-char-pack-template
```

c-reduce runs inside `creduce-host` and reaches the compiler container
via the mounted Docker socket. Because c-reduce's long tail is
prohibitively slow under emulation, each run is capped by
`CREDUCE_BUDGET` seconds (default 900) and the in-place result is taken.

## Results

See `RESULTS.md` for the generated size/timing tables, the c-reduce
comparison, and the structural analysis of what each tool could and
couldn't remove (which has already driven new passes: type replacement
and namespace-qualifier rewriting in `src/shrinkray/passes/cpp.py`).
