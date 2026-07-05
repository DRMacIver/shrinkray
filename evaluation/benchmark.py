#!/usr/bin/env python3
"""Measure the reducer's *efficiency*, not tool speed.

Each benchmark problem reduces a realistic input against a cheap in-process
predicate, so the reported number of interestingness calls reflects the
reducer's own behaviour (pass ordering, stopping, tail churn) rather than
how slow a real oracle is. Wall-clock cost of a real reduction is roughly
`calls * oracle_time`, so fewer calls helps every entry in the corpus.

Problems come in two flavours:

- synthetic: constructed inputs isolating one reducer behaviour
  (bulk deletion, rigid-structure tail churn, stopping overhead).
- corpus-derived: `original.*` files from `evaluation/corpus/` with a cheap
  predicate approximating the real bug's requirements (required tokens,
  syntactic validity, structural properties). These give realistic
  reduction *trajectories* without a slow oracle.

Metrics per problem:

- calls: total interestingness-test calls (the primary cost metric).
- final size: quality of the result (guards against "fast but worse").
- c90 / c99: calls needed to achieve 90% / 99% of the total size
  reduction the run eventually achieved (how front-loaded progress is).
- tail: calls after the last successful reduction (pure stopping cost).

Runs at parallelism 1 with a fixed random seed (WorkContext seeds its own
Random(0)), so call counts are reproducible to within a couple of calls
run to run; judge changes by call counts, not the informational seconds
column, which varies with machine load.

    python3 evaluation/benchmark.py                 # run all, print a table
    python3 evaluation/benchmark.py NAME ...        # run selected problems
    python3 evaluation/benchmark.py --json out.json # also dump metrics
    python3 evaluation/benchmark.py --baseline B    # compare against a dump
    python3 evaluation/benchmark.py --passes        # per-pass stats tables
"""

import argparse
import ast
import json as json_module
import sys
import time
import warnings
from pathlib import Path

import trio

from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.state import sort_key_for_initial
from shrinkray.work import WorkContext


CORPUS = Path(__file__).resolve().parent / "corpus"


# --- predicate helpers ------------------------------------------------------


def bracket_depth(data: bytes) -> int:
    """Max nesting depth if brackets are balanced, else -1."""
    opens, closes = b"([{", b")]}"
    depth = worst = 0
    for b in data:
        if b in opens:
            depth += 1
            worst = max(worst, depth)
        elif b in closes:
            if depth == 0:
                return -1
            depth -= 1
    return worst if depth == 0 else -1


def compiles_as_python(data: bytes) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(data, "<benchmark>", "exec")
    except Exception:
        return False
    return True


def json_depth(data: bytes) -> int:
    """Max nesting depth of a valid JSON document, else -1."""
    try:
        doc = json_module.loads(data)
    except Exception:
        return -1
    depth = 0
    stack = [(doc, 1)]
    while stack:
        value, d = stack.pop()
        depth = max(depth, d)
        if isinstance(value, dict):
            stack.extend((child, d + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, d + 1) for child in value)
    return depth


def has_dup_bases_and_metaclass(data: bytes) -> bool:
    """Some class has syntactically duplicate bases, and some class uses a
    metaclass keyword. Approximates the pylint/astroid DuplicateBasesError
    corpus entry's requirements."""
    try:
        tree = ast.parse(data)
    except Exception:
        return False
    dup = meta = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            dumped = [ast.dump(base) for base in node.bases]
            if len(dumped) != len(set(dumped)):
                dup = True
            if any(kw.arg == "metaclass" for kw in node.keywords):
                meta = True
    return dup and meta


def coupled_count_json_ok(data: bytes) -> bool:
    try:
        doc = json_module.loads(data)
    except Exception:
        return False
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("events"), list)
        and doc.get("event_count") == len(doc["events"])
        and any(
            isinstance(e, dict) and e.get("type") == "crash" for e in doc["events"]
        )
    )


def coupled_total_text_ok(data: bytes) -> bool:
    total = None
    seen = 0
    saw_corrupted = False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["ENTRY"] and len(parts) == 3:
            try:
                seen += int(parts[2])
            except ValueError:
                return False
            if parts[1] == "job-corrupted":
                saw_corrupted = True
        elif parts[:1] == ["TOTAL"] and len(parts) == 2:
            try:
                total = int(parts[1])
            except ValueError:
                return False
    return saw_corrupted and total is not None and total == seen


def coupled_arity_python_ok(data: bytes) -> bool:
    """Sentinel in a called function; every call matches its def's arity.

    ast.parse alone (not compile) is deliberate: it accepts duplicate
    parameter names, so identifier normalisation stays available and the
    only thing the reducer can't do by deletion is drop a parameter,
    which requires editing the def and every call site together.
    """
    if b"MAGIC_SENTINEL" not in data:
        return False
    try:
        tree = ast.parse(data.decode("utf-8"))
    except Exception:
        return False
    arities: dict[str, int] = {}
    sentinel_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
                return False
            arities[node.name] = len(args.args)
            if "MAGIC_SENTINEL" in ast.dump(node):
                sentinel_fn = node.name
    if sentinel_fn is None:
        return False
    sentinel_called = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in arities:
                if node.keywords or len(node.args) != arities[name]:
                    return False
                if name == sentinel_fn:
                    sentinel_called = True
    return sentinel_called


def contains_all(*tokens: bytes):
    def predicate(data: bytes) -> bool:
        return all(t in data for t in tokens)

    return predicate


# --- synthetic problem inputs -----------------------------------------------


def _scaffolded_deep_parens() -> bytes:
    body = b"wrap(" * 600 + b"0" + b")" * 600
    return (
        b"# generated pipeline module\n"
        b"import sys\n\n"
        b"def wrap(stage):\n    return stage\n\n"
        b"DEFAULTS = [1, 2, 3]\n\n"
        b"def build():\n    composed = " + body + b"\n    return composed\n\n"
        b"if __name__ == '__main__':\n    print(build())\n"
    )


def _big_file_with_markers() -> bytes:
    lines = []
    for i in range(2000):
        lines.append(f"def function_{i}(a, b, c):".encode())
        lines.append(f"    result = a * {i} + b - c  # step {i}".encode())
        if i == 400:
            lines.append(b"    RARE_MARKER_ALPHA = 1")
        if i == 1200:
            lines.append(b"    RARE_MARKER_BETA = 2")
        if i == 1800:
            lines.append(b"    RARE_MARKER_GAMMA = 3")
        lines.append(b"    return result")
    return b"\n".join(lines) + b"\n"


def _already_minimal() -> bytes:
    # A small input the predicate keeps almost entirely: measures how many
    # calls the reducer wastes confirming it cannot shrink further.
    return b"x=RARE_MARKER_ALPHA+RARE_MARKER_BETA\n"


def _python_module_with_sentinel() -> bytes:
    """A plausible generated module where candidates must stay valid Python.

    Exercises the high-rejection regime typical of real reductions of
    strict-syntax languages: most byte-level candidates fail to compile.
    """
    parts = [
        b'"""Utilities for the frobnication service."""\n',
        b"import collections\n",
        b"import functools\n\n",
    ]
    for i in range(60):
        parts.append(
            (
                f"class Handler{i}:\n"
                f"    priority = {i}\n"
                f"    def process(self, item):\n"
                f"        queue = collections.deque(maxlen={i + 1})\n"
                f"        queue.append(item)\n"
                f"        return sorted(queue)\n\n"
            ).encode()
        )
        if i == 30:
            parts.append(
                b"def sentinel():\n    SENTINEL_KEEP = 1\n    return SENTINEL_KEEP\n\n"
            )
    return b"".join(parts)


def _coupled_count_json() -> bytes:
    events = []
    for i in range(25):
        events.append(
            {
                "id": f"evt-{i:04d}",
                "type": "request" if i != 13 else "crash",
                "duration_ms": 17 * (i + 3) % 211,
                "path": f"/api/v2/resource/{i}",
                "status": 200 if i % 7 else 503,
            }
        )
    doc = {
        "service": "billing-gateway",
        "region": "eu-west-1",
        "schema_version": 4,
        "event_count": len(events),
        "events": events,
        "sampled": False,
        "window": {"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z"},
    }
    return json_module.dumps(doc, indent=2).encode()


def _coupled_total_text() -> bytes:
    entries = []
    values = []
    for i in range(30):
        name = f"job-{i:03d}"
        value = (i * 37) % 101 + 3
        if i == 17:
            name = "job-corrupted"
        values.append(value)
        entries.append(f"ENTRY {name} {value}")
    lines = [
        "# batch accounting ledger",
        "# generated by billing-gateway 4.11.2",
        "FORMAT 2",
        *entries,
        f"TOTAL {sum(values)}",
        "END",
    ]
    return ("\n".join(lines) + "\n").encode()


def _coupled_arity_python() -> bytes:
    return b"""\
import logging

logger = logging.getLogger("ingest")


def normalise_headers(headers, casing, strip_empty, max_length):
    result = {}
    for key, value in headers.items():
        key = key.lower() if casing == "lower" else key
        if strip_empty and not value:
            continue
        result[key[:max_length]] = value
    return result


def process_record(record, options, retries, timeout, dry_run):
    payload = dict(record)
    payload["MAGIC_SENTINEL"] = True
    if dry_run:
        logger.info("dry run, skipping submit")
        return payload
    for attempt in range(retries):
        logger.info("submitting attempt %d timeout %s", attempt, timeout)
    return payload


def main():
    record = {"user": "test"}
    options = {"validate": True}
    cleaned = normalise_headers({"X-Test": "1"}, "lower", True, 64)
    logger.info("headers: %s", cleaned)
    return process_record(record, options, 3, 30.0, False)


main()
"""


# --- problem table ----------------------------------------------------------


class Problem:
    def __init__(
        self,
        initial: bytes,
        predicate,
        *,
        cpp: bool = False,
        treesitter_language: str | None = None,
    ):
        self.initial = initial
        self.predicate = predicate
        self.cpp = cpp
        # When set, the reducer runs its grammar-aware tree-sitter passes
        # for this language (as it does when reducing a file with the
        # matching extension), so their efficiency is measured too.
        self.treesitter_language = treesitter_language


def _corpus_file(entry: str, filename: str) -> bytes:
    return (CORPUS / entry / filename).read_bytes()


def build_problems() -> dict[str, Problem]:
    problems = {
        # Rigid deep structure: reduces to minimal parens at depth >= 400,
        # then the reducer keeps trying (and failing) to go smaller.
        "deep_parens": Problem(
            _scaffolded_deep_parens(),
            lambda x: bracket_depth(x) >= 400,
        ),
        # Bulk deletion of a large file down to three scattered markers.
        "keep_markers": Problem(
            _big_file_with_markers(),
            contains_all(
                b"RARE_MARKER_ALPHA", b"RARE_MARKER_BETA", b"RARE_MARKER_GAMMA"
            ),
        ),
        # Already minimal: pure measure of stopping / no-progress overhead.
        "already_minimal": Problem(
            _already_minimal(),
            contains_all(b"RARE_MARKER_ALPHA", b"RARE_MARKER_BETA"),
        ),
        # Valid-Python-required marker hunt: most candidates get rejected.
        "python_syntax": Problem(
            _python_module_with_sentinel(),
            lambda x: b"SENTINEL_KEEP" in x and compiles_as_python(x),
        ),
        # Corpus-derived problems. Predicates approximate each entry's real
        # bug requirements; see evaluation/corpus/<id>/meta.json.
        "corpus_mypy": Problem(
            _corpus_file("mypy-0.942-match-union-tuple-crash", "original.py"),
            lambda x: (
                compiles_as_python(x)
                and contains_all(b"match ", b"case ", b"Union[", b"tuple[")(x)
            ),
        ),
        "corpus_pylint": Problem(
            _corpus_file("pylint-2.17.4-duplicate-bases-mro-crash", "original.py"),
            has_dup_bases_and_metaclass,
        ),
        "corpus_ujson": Problem(
            _corpus_file("ujson-510-indent-buffer-overflow", "original.json"),
            lambda x: json_depth(x) >= 20,
        ),
        "corpus_udlit_cpp": Problem(
            _corpus_file("gcc49-udlit-char-pack-template", "original.cpp"),
            contains_all(b'operator""', b"decltype(", b"..."),
            cpp=True,
        ),
        # Coupled-edit problems: no single deletion can succeed because
        # two distant parts of the file must change together (a count that
        # must match a list length, a checksum, call-site arity), so the
        # deletion-based passes reach a fixpoint far above the minimum.
        # Added to measure how well the LLM passes escape such fixpoints;
        # classical runs also document the size of the "stuck tax".
        "coupled_count_json": Problem(
            _coupled_count_json(),
            coupled_count_json_ok,
        ),
        "coupled_total_text": Problem(
            _coupled_total_text(),
            coupled_total_text_ok,
        ),
        "coupled_arity_python": Problem(
            _coupled_arity_python(),
            coupled_arity_python_ok,
        ),
        "corpus_minisat": Problem(
            _corpus_file("minisat-dimacs-int-overflow", "original.cnf"),
            contains_all(b"2147483648"),
        ),
        # Tree-sitter (Go): keeps the generic ~bool constraint and a
        # comparison (the untyped-bool ICE trigger) while letting the rest
        # of the module — and the imports coupled to it — be deleted, so the
        # grammar-aware passes (delete_orphaned_declarations in particular)
        # are exercised and measured.
        "corpus_go": Problem(
            _corpus_file("go11810-generic-untyped-bool-ice", "original.go"),
            contains_all(b"~bool", b"=="),
            treesitter_language="go",
        ),
    }
    return problems


# --- running and metrics ----------------------------------------------------


def run_problem(name: str, problem: Problem) -> dict:
    events: list[dict] = []

    async def acond(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return problem.predicate(x)

    async def go() -> tuple[bytes, object, object]:
        reduction_problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=problem.initial,
            is_interesting=acond,
            work=WorkContext(parallelism=1),
            sort_key=sort_key_for_initial(problem.initial),
        )

        async def record(test_case: bytes) -> None:
            stats = reduction_problem.current_pass_stats
            events.append(
                {
                    "calls": reduction_problem.stats.calls,
                    "size": len(test_case),
                    "pass": stats.pass_name if stats is not None else None,
                }
            )

        reduction_problem.on_reduce(record)
        reducer = ShrinkRay(
            target=reduction_problem,
            enable_cpp_passes=problem.cpp,
            treesitter_language=problem.treesitter_language,
        )
        await reducer.run()
        return reduction_problem.current_test_case, reduction_problem, reducer

    start = time.monotonic()
    result, reduction_problem, reducer = trio.run(go)
    seconds = time.monotonic() - start

    initial_size = len(problem.initial)
    final_size = len(result)
    calls = reduction_problem.stats.calls

    def calls_to_fraction(fraction: float) -> int:
        if initial_size == final_size:
            return 0
        threshold = initial_size - fraction * (initial_size - final_size)
        for event in events:
            if event["size"] <= threshold:
                return event["calls"]
        return calls

    tail_calls = calls - events[-1]["calls"] if events else calls

    pass_stats = [
        {
            "pass": s.pass_name,
            "runs": s.run_count,
            "calls": s.test_evaluations,
            "reductions": s.successful_reductions,
            "bytes_deleted": s.bytes_deleted,
        }
        for s in reducer.pass_stats.get_stats_in_order()
    ]

    return {
        "name": name,
        "initial_size": initial_size,
        "final_size": final_size,
        "calls": calls,
        "c90": calls_to_fraction(0.90),
        "c99": calls_to_fraction(0.99),
        "tail": tail_calls,
        "reductions": len(events),
        "seconds": round(seconds, 2),
        "pass_stats": pass_stats,
        "events": events,
    }


# --- output -----------------------------------------------------------------

TABLE_COLUMNS = [
    ("problem", "name", "<24"),
    ("calls", "calls", ">8"),
    ("final", "final_size", ">7"),
    ("c90", "c90", ">8"),
    ("c99", "c99", ">8"),
    ("tail", "tail", ">7"),
    ("secs", "seconds", ">7"),
]


def print_table(results: list[dict]) -> None:
    header = " ".join(f"{title:{fmt}}" for title, _, fmt in TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for r in results:
        print(" ".join(f"{r[key]:{fmt}}" for _, key, fmt in TABLE_COLUMNS))
    total_calls = sum(r["calls"] for r in results)
    print("-" * len(header))
    print(f"{'TOTAL':<24} {total_calls:>8}")


def print_pass_stats(results: list[dict]) -> None:
    for r in results:
        print()
        print(f"=== {r['name']} (calls={r['calls']}) ===")
        print(f"{'pass':<40} {'runs':>5} {'calls':>8} {'reds':>6} {'deleted':>9}")
        for s in sorted(r["pass_stats"], key=lambda s: -s["calls"]):
            if s["calls"] == 0 and s["reductions"] == 0:
                continue
            print(
                f"{s['pass']:<40} {s['runs']:>5} {s['calls']:>8} "
                f"{s['reductions']:>6} {s['bytes_deleted']:>9}"
            )


def print_comparison(baseline: dict, results: list[dict]) -> None:
    by_name = {r["name"]: r for r in baseline["results"]}
    print(f"{'problem':<24} {'calls':>16} {'Δ%':>7} {'final':>12} {'c99':>16}")
    print("-" * 80)
    total_old = total_new = 0
    for r in results:
        old = by_name.get(r["name"])
        if old is None:
            print(f"{r['name']:<24} {'(new)':>16}")
            continue
        total_old += old["calls"]
        total_new += r["calls"]
        delta = (
            (r["calls"] - old["calls"]) / old["calls"] * 100 if old["calls"] else 0.0
        )
        print(
            f"{r['name']:<24} "
            f"{old['calls']:>7} → {r['calls']:>6} {delta:>+6.1f}% "
            f"{old['final_size']:>5} → {r['final_size']:>4} "
            f"{old['c99']:>7} → {r['c99']:>6}"
        )
    if total_old:
        overall = (total_new - total_old) / total_old * 100
        print("-" * 80)
        print(f"{'TOTAL':<24} {total_old:>7} → {total_new:>6} {overall:>+6.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="problems to run (default: all)")
    parser.add_argument("--json", help="write metrics to this file")
    parser.add_argument("--baseline", help="compare against a previous --json dump")
    parser.add_argument(
        "--passes", action="store_true", help="print per-pass stats tables"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="include per-reduction event lists in the --json dump",
    )
    args = parser.parse_args()

    problems = build_problems()
    names = args.names or list(problems)
    unknown = [n for n in names if n not in problems]
    if unknown:
        parser.error(
            f"unknown problems: {', '.join(unknown)}. Available: {', '.join(problems)}"
        )

    for name, problem in problems.items():
        if not problem.predicate(problem.initial):
            raise AssertionError(
                f"problem {name}: predicate is false on its initial input"
            )

    results = []
    for name in names:
        results.append(run_problem(name, problems[name]))

    print_table(results)
    if args.passes:
        print_pass_stats(results)
    if args.baseline:
        print()
        with open(args.baseline) as f:
            baseline = json_module.load(f)
        print_comparison(baseline, results)
    if args.json:
        dumped = results
        if not args.full:
            dumped = [{k: v for k, v in r.items() if k != "events"} for r in results]
        with open(args.json, "w") as f:
            json_module.dump({"results": dumped}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
