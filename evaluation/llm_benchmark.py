#!/usr/bin/env python3
"""Compare classical reduction with LLM-assisted reduction on the
benchmark problems.

Runs each benchmark problem twice against its cheap in-process oracle:
once with the classical passes only, and once with the LLM passes enabled
(as in a default `shrinkray` run), using the natural-language oracle
descriptions from llm_prompt_experiment.py as the prompt's interestingness
context. Reports final sizes side by side; per-problem outputs are written
to work/llm_benchmark/.

This measures reduction *quality*, not efficiency: the LLM passes' value
is escaping fixpoints the deletion-based passes can't (see the coupled_*
problems), at a wall-clock cost benchmark.py deliberately doesn't model.

Needs the model (downloaded on first use) and benefits from GPU
acceleration; a full run takes tens of minutes.

    uv run python evaluation/llm_benchmark.py                # all problems
    uv run python evaluation/llm_benchmark.py coupled_count_json
    uv run python evaluation/llm_benchmark.py --modes llm    # skip classical
"""

import argparse
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import trio
from benchmark import Problem, build_problems
from llm_prompt_experiment import FILENAMES, ORACLES

from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.llm import (
    DEFAULT_MODEL_SPEC,
    LLMConfig,
    parse_model_spec,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.state import sort_key_for_initial
from shrinkray.work import Volume, WorkContext

RESULTS_DIR = EVAL_DIR / "work" / "llm_benchmark"
PER_PROBLEM_TIMEOUT = 900.0


def run_problem(
    name: str,
    problem: Problem,
    mode: str,
    client: LlamaCppClient | None,
) -> tuple[int, float, bool]:
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        try:
            return bool(problem.predicate(x))
        except Exception:
            return False

    reduction_problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=problem.initial,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1, volume=Volume.quiet),
        sort_key=sort_key_for_initial(problem.initial),
    )
    kwargs = {}
    if mode == "llm":
        assert client is not None
        kwargs = {
            "llm_client": client,
            "llm_config": LLMConfig(
                filename=FILENAMES.get(name),
                oracle=ORACLES.get(name),
            ),
        }
    reducer = ShrinkRay(
        target=reduction_problem,
        enable_cpp_passes=problem.cpp,
        **kwargs,
    )

    timed_out = False

    async def run() -> None:
        nonlocal timed_out
        with trio.move_on_after(PER_PROBLEM_TIMEOUT) as scope:
            await reducer.run()
        timed_out = scope.cancelled_caught

    start = time.time()
    trio.run(run)
    final = reduction_problem.current_test_case
    out = RESULTS_DIR / f"{name}.{mode}"
    out.write_bytes(final)
    return len(final), time.time() - start, timed_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="*")
    parser.add_argument(
        "--modes", nargs="*", default=["classical", "llm"], help="modes to run"
    )
    args = parser.parse_args()

    problems = build_problems()
    names = args.problems or list(problems)
    unknown = set(names) - set(problems)
    if unknown:
        print(f"Unknown problems: {sorted(unknown)}", file=sys.stderr)
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = None
    if "llm" in args.modes:
        client = LlamaCppClient(model=parse_model_spec(DEFAULT_MODEL_SPEC))

    print(f"{'problem':<22} {'initial':>8} {'classical':>9} {'llm':>8}  notes")
    for name in names:
        problem = problems[name]
        sizes: dict[str, int] = {}
        notes = []
        for mode in args.modes:
            size, elapsed, timed_out = run_problem(name, problem, mode, client)
            sizes[mode] = size
            if timed_out:
                notes.append(f"{mode} timed out")
        print(
            f"{name:<22} {len(problem.initial):>8} "
            f"{sizes.get('classical', ''):>9} {sizes.get('llm', ''):>8}  "
            f"{'; '.join(notes)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
