# Evaluation corpus

A corpus of **real bugs in real tools** — inputs that make a specific
pinned version of a compiler, formatter, linter, JSON parser, or SAT
solver crash — used to evaluate shrink ray's reduction quality on
realistic material across the formats it supports (C/C++, Python, JSON,
DIMACS CNF, and tree-sitter-grammar languages like Go, Rust,
JavaScript, and TypeScript), and to compare it against other reducers.

Each entry is a plausible, pre-reduction-sized input with a genuine bug
trigger buried inside realistic scaffolding (application-shaped code,
config-shaped JSON, a problem-shaped CNF). Reducing it should strip the
scaffolding back down to the essential trigger, exercising the passes
shrink ray dispatches for that format.

## Layout

```
evaluation/
├── run.py        # reduce entries with shrink ray, write result.json
├── watch.py      # run one entry interactively in the TUI (records nothing)
├── report.py     # regenerate the tables in RESULTS.md
├── benchmark.py  # measure reducer efficiency (interestingness calls) against cheap in-process oracles
├── benchmark_baseline.json  # committed benchmark metrics for main (compare with benchmark.py --baseline)
├── llm_benchmark.py         # compare classical vs LLM-assisted reduction quality on the benchmark problems
├── llm_prompt_experiment.py # measure LLM prompt variants (validity rate, size gain per generation)
├── RESULTS.md    # generated tables + hand-written analysis
├── corpus/<id>/  # one directory per bug
├── sortkey/      # sort-key tuning corpus (see sortkey/README.md)
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

Every entry is reduced in shrink ray's **normal mode**: shrink ray runs
each interestingness test in a fresh temp dir it creates and removes, so
anything the tool writes beside the candidate is cleaned up (point a
tool's own scratch there via the `{sandbox}` placeholder, e.g. pylint's
`PYLINTHOME`).

- **docker** — a persistent container per entry (started once, reused
  across the thousands of oracle calls a reduction makes). The candidate
  is piped into the container on stdin (`docker exec -i`), so nothing is
  shared into it; `{file}` is `-`, and the compiler reads stdin (`-x c++
  -c -`).
- **venv** — `uv venv --python <python>` + `uv pip install <requirements>`
  into `work/venv`, whose `bin` is prepended to PATH.
- **command** — arbitrary command; `setup.sh` (run from the entry dir
  with `$TOOLS_DIR` = `evaluation/tools/`) builds the pinned tool once.

Placeholders in `command` and `env` values: `{file}` (the candidate, or
`-` for docker stdin), `{sandbox}` (the per-test temp dir shrink ray
cleans up), `{tools}`, `{entry}`, `{workdir}`. An optional `oracle.env`
object exports environment variables in the check script. Progress is
streamed to `<entry>/work/reduce.log`.

## Running

```bash
# Verify every entry still reproduces its bug
python3 evaluation/run.py --check

# Reduce every entry (writes work/reduced.<ext>, resumable; on success
# updates shrinkray_reduced.<ext> and result.json)
python3 evaluation/run.py

# Reduce / check specific entries
python3 evaluation/run.py mypy-0.942-match-union-tuple-crash

# Regenerate the tables in RESULTS.md
python3 evaluation/report.py
```

To *watch* a reduction in shrink ray's normal interactive TUI, use
`watch.py`. It sets up the entry's oracle exactly as run.py would, but
always restarts from the original input, runs attached to your terminal,
and records nothing (it reduces `<entry>/work/watch<ext>`, leaving
run.py's resumable state and the committed results untouched). Unknown
arguments are forwarded to shrinkray:

```bash
python3 evaluation/watch.py ruff-0.0.277-isort-skip-block-panic
python3 evaluation/watch.py pylint-2.17.4-duplicate-bases-mro-crash --parallelism 4
```

The C/C++ entries use amd64-only compiler images which run under
emulation on Apple Silicon: slow but functional, and single-threaded
(`parallelism: 1`) because emulation penalises concurrency. The other
entries run natively at full parallelism and reduce in seconds to
minutes.

## Provenance

Most entries correspond to a real bug in an external tool: `bug_url`
links the upstream issue/PR/changelog, and `symptom` describes the
failure. The C/C++ triggers were found by screening gcc's own
`g++.dg/cpp1y` testsuite and C++11/14 dark-corner snippets against old
compiler Docker images; the Python/JSON/CNF entries reproduce reported
crashes in pinned tool versions (formatters, type checkers, linters,
JSON parsers, SAT solvers). Each trigger is then embedded in a larger
realistic input and the failure re-verified.

A few entries (`shrinkray-*`) are regression evaluations for shrink ray's
own robustness rather than an external tool. They reduce deeply nested
inputs that used to crash shrink ray itself — deeply bracket-nested code
overflowed libcst's C stack in `is_python`, and deeply nested JSON
overflowed the recursion in the JSON passes. Their interestingness test
runs just the relevant bit of code (a bare `libcst.parse_module` /
`json.loads`) and requires the input to stay parseable and nested past a
threshold, so the reduced result is the minimal deeply nested trigger.
Running shrink ray on them at all is the regression check; the reduction
also exercises the generic passes on deep nesting.

## Comparing against c-reduce

`creduce/` holds a driver that reduces the C/C++ entries with the
Debian-packaged c-reduce (2.11.0 on bookworm, with its bundled
`clang_delta`) against the identical compiler-in-Docker oracle, so the
two tools are compared on equal footing. Build the c-reduce host image once, then run the driver:

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

## LLM evaluation

`benchmark.py` includes three `coupled_*` problems where no single deletion
can succeed because two distant parts of the file must change together (a
count that must match a list length, a checksum line, call-site arity).
The classical passes get stuck far above the minimum on these by design;
`llm_benchmark.py` runs each benchmark problem with and without the LLM
passes and compares final sizes, which is how the LLM mode's ability to
escape such fixpoints is measured (e.g. coupled_count_json: 143 bytes
classical vs 46 with the LLM, which is the global minimum).
`llm_prompt_experiment.py` measures prompt variants per generation and is
what the shipped prompt's design decisions were based on.
