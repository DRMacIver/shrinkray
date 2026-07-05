#!/usr/bin/env python3
"""Measure the reducer's *efficiency*, not tool speed.

Each benchmark problem reduces a realistic input against a cheap in-process
predicate, so the reported number of interestingness calls reflects the
reducer's own behaviour (pass ordering, stopping, tail churn) rather than
how slow a real oracle is. Wall-clock cost of a real reduction is roughly
`calls * oracle_time`, so fewer calls helps every entry in the corpus.

Runs at parallelism 1 for reproducibility.

    python3 evaluation/benchmark.py            # run all, print a table
    python3 evaluation/benchmark.py <name> ... # run selected problems
"""

import sys
import trio

from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.state import sort_key_for_initial
from shrinkray.work import WorkContext


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


def run_problem(initial: bytes, is_interesting) -> tuple[bytes, int]:
    async def acond(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def go() -> tuple[bytes, int]:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=initial,
            is_interesting=acond,
            work=WorkContext(parallelism=1),
            sort_key=sort_key_for_initial(initial),
        )
        reducer = ShrinkRay(target=problem)
        await reducer.run()
        return problem.current_test_case, problem.stats.calls

    return trio.run(go)


# --- problem definitions: (name, initial, predicate) -----------------------


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


PROBLEMS: dict[str, tuple[bytes, object]] = {
    # Rigid deep structure: reduces to minimal parens at depth >= 400, then
    # the reducer keeps trying (and failing) to go smaller -> tail churn.
    "deep_parens": (
        _scaffolded_deep_parens(),
        lambda x: bracket_depth(x) >= 400,
    ),
    # Bulk deletion of a large file down to three scattered required markers.
    "keep_markers": (
        _big_file_with_markers(),
        lambda x: b"RARE_MARKER_ALPHA" in x
        and b"RARE_MARKER_BETA" in x
        and b"RARE_MARKER_GAMMA" in x,
    ),
    # Already minimal: pure measure of the stopping / no-progress overhead.
    "already_minimal": (
        _already_minimal(),
        lambda x: b"RARE_MARKER_ALPHA" in x and b"RARE_MARKER_BETA" in x,
    ),
}


def main() -> int:
    names = sys.argv[1:] or list(PROBLEMS)
    print(f"{'problem':<18} {'calls':>8} {'final size':>11}")
    print("-" * 40)
    total = 0
    for name in names:
        initial, pred = PROBLEMS[name]
        result, calls = run_problem(initial, pred)
        total += calls
        print(f"{name:<18} {calls:>8} {len(result):>11}")
    print("-" * 40)
    print(f"{'TOTAL':<18} {total:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
