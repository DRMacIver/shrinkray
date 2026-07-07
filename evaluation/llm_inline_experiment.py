#!/usr/bin/env python3
"""Measure the grammar-guided LLM inlining pump against a real model.

Each problem is a small program with a function whose definition cannot
be deleted while a call to it remains, judged by a real semantic oracle
(run the program, require exact output). Classical passes can shrink
names and whitespace but cannot rewrite the call into its inlined form,
so the function definition survives; the inlining pump should remove it.

Runs each problem in three modes and reports final sizes side by side:

- classical: the ordinary passes only (no LLM at all).
- rewrite: the ordinary passes plus the whole-file llm_rewrite pass,
  with the pumps disabled.
- pump: the ordinary passes plus llm_inline_calls, with the whole-file
  llm_rewrite pass stripped out so the pump's contribution is isolated.

restart_at_fixpoint is disabled in every mode: the restarted
sub-reduction builds its own pass list, which would let llm_rewrite
leak into pump mode and muddy the attribution.

Model prompts and responses, and each mode's final test case, are
written to work/llm_inline_experiment/.

Needs the model (downloaded on first use) and the python3/node/ruby/cc
binaries for the oracles.

    uv run python evaluation/llm_inline_experiment.py             # all problems
    uv run python evaluation/llm_inline_experiment.py python_simple
    uv run python evaluation/llm_inline_experiment.py --modes pump
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import trio
from attrs import define, field

from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.llm import (
    DEFAULT_MODEL_SPEC,
    LLMClient,
    LLMConfig,
    parse_model_spec,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.state import sort_key_for_initial
from shrinkray.work import Volume, WorkContext


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "work" / "llm_inline_experiment"
PER_MODE_TIMEOUT = 600.0
ORACLE_TIMEOUT = 10.0


@define
class InlineProblem:
    language: str
    extension: str
    initial: bytes
    expected_output: str
    # The function the problem wants inlined away. Success is judged by
    # eye from the final test case: passes rename identifiers, so the
    # name itself cannot be checked for.
    function_name: bytes
    # argv to run a source file, with {file} (and for C {binary})
    # placeholders. A two-step compile-and-run is expressed by compile_argv.
    run_argv: list[str]
    compile_argv: list[str] | None = None
    # A line the oracle requires verbatim in the candidate, mimicking a
    # bug that needs a specific construct. Whole-file rewrites tend to
    # drop it; targeted splicing leaves it untouched.
    required_line: bytes | None = None
    # Natural-language description of the oracle, shown to llm_rewrite
    # the way a real run shows the user's test script.
    oracle_description: str | None = None


def _python_large() -> bytes:
    """A bigger file: one function to inline, plus deletable filler and
    a class the oracle (silently) requires kept intact."""
    lines = [
        "class Config:",
        "    def __init__(self):",
        "        self.value = 7",
        "        self.scale = 3",
        "",
        "def combine(a, b):",
        "    return a * b + 1",
        "",
    ]
    for i in range(15):
        lines += [f"def helper_{i}(x):", f"    return x + {i}", ""]
    lines += ["cfg = Config()", "print(combine(cfg.value, cfg.scale))", ""]
    return "\n".join(lines).encode()


PROBLEMS: dict[str, InlineProblem] = {
    "python_simple": InlineProblem(
        language="python",
        extension=".py",
        initial=(b"def add_one(x):\n    return x + 1\n\nprint(add_one(3))\n"),
        expected_output="4\n",
        function_name=b"add_one",
        run_argv=["python3", "{file}"],
        oracle_description='Running the file must print exactly "4".',
    ),
    # A multi-statement body: the hand-written C++ inliner refuses
    # these, so this is territory only the model can reach.
    "python_multi": InlineProblem(
        language="python",
        extension=".py",
        initial=(
            b"def scale(x):\n"
            b"    y = x * 2\n"
            b"    y = y + 1\n"
            b"    return y\n"
            b"\n"
            b"print(scale(10))\n"
        ),
        expected_output="21\n",
        function_name=b"scale",
        run_argv=["python3", "{file}"],
        oracle_description='Running the file must print exactly "21".',
    ),
    "javascript_simple": InlineProblem(
        language="javascript",
        extension=".js",
        initial=(b"function addOne(x) { return x + 1; }\nconsole.log(addOne(3));\n"),
        expected_output="4\n",
        function_name=b"addOne",
        run_argv=["node", "{file}"],
        oracle_description='Running the file must print exactly "4".',
    ),
    "ruby_simple": InlineProblem(
        language="ruby",
        extension=".rb",
        initial=(b"def add_one(x)\n  x + 1\nend\nputs add_one(3)\n"),
        expected_output="4\n",
        function_name=b"add_one",
        run_argv=["ruby", "{file}"],
        oracle_description='Running the file must print exactly "4".',
    ),
    "c_simple": InlineProblem(
        language="c",
        extension=".c",
        initial=(
            b"#include <stdio.h>\n"
            b"static int add_one(int x) { return x + 1; }\n"
            b'int main(void) { printf("%d\\n", add_one(3)); return 0; }\n'
        ),
        expected_output="4\n",
        function_name=b"add_one",
        run_argv=["{binary}"],
        compile_argv=["cc", "-x", "c", "{file}", "-o", "{binary}"],
        oracle_description='Running the file must print exactly "4".',
    ),
    "python_large": InlineProblem(
        language="python",
        extension=".py",
        initial=_python_large(),
        expected_output="22\n",
        function_name=b"combine",
        run_argv=["python3", "{file}"],
        required_line=b"self.value = 7",
        oracle_description=(
            'Running the file must print exactly "22", and the file must '
            'still contain the line "self.value = 7".'
        ),
    ),
}


@define
class LoggingClient(LLMClient):
    """Wraps a client, recording every prompt/response exchange."""

    inner: LLMClient
    exchanges: list[dict[str, str]] = field(factory=list)

    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        start = time.monotonic()
        response = await self.inner.complete(
            prompt, max_tokens=max_tokens, seed=seed, temperature=temperature
        )
        self.exchanges.append(
            {
                "prompt": prompt,
                "response": response,
                "seconds": round(time.monotonic() - start, 1),
            }
        )
        return response

    def start_loading(self) -> None:
        self.inner.start_loading()

    async def wait_until_ready(self) -> None:
        await self.inner.wait_until_ready()

    def is_disabled(self) -> bool:
        return self.inner.is_disabled()


class NoPumpShrinkRay(ShrinkRay):
    """ShrinkRay with pumps disabled, isolating the whole-file rewrite."""

    @property
    def pumps(self):
        return ()


def make_oracle(problem: InlineProblem):
    """An async interestingness test running the candidate for real."""

    async def is_interesting(candidate: bytes) -> bool:
        if problem.required_line is not None and problem.required_line not in candidate:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / f"candidate{problem.extension}"
            source.write_bytes(candidate)
            binary = Path(tmp) / "candidate.bin"
            substitutions = {"{file}": str(source), "{binary}": str(binary)}

            async def run(argv: list[str]) -> subprocess.CompletedProcess | None:
                argv = [substitutions.get(a, a) for a in argv]
                with trio.move_on_after(ORACLE_TIMEOUT):
                    return await trio.run_process(
                        argv,
                        capture_stdout=True,
                        capture_stderr=True,
                        check=False,
                    )
                return None

            if problem.compile_argv is not None:
                compiled = await run(problem.compile_argv)
                if compiled is None or compiled.returncode != 0:
                    return False
            result = await run(problem.run_argv)
            if result is None or result.returncode != 0:
                return False
            return result.stdout.decode("utf-8", errors="replace") == (
                problem.expected_output
            )

    return is_interesting


async def reduce_problem(
    name: str, problem: InlineProblem, mode: str, client: LlamaCppClient | None
) -> dict:
    logging_client = None
    reduction_problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=problem.initial,
        is_interesting=make_oracle(problem),
        work=WorkContext(parallelism=2, volume=Volume.quiet),
        sort_key=sort_key_for_initial(problem.initial),
    )
    kwargs = {}
    reducer_class = ShrinkRay
    if mode in ("rewrite", "pump"):
        assert client is not None
        logging_client = LoggingClient(inner=client)
        kwargs = {
            "llm_client": logging_client,
            "llm_config": LLMConfig(
                filename=f"candidate{problem.extension}",
                oracle=problem.oracle_description,
            ),
        }
        if mode == "rewrite":
            reducer_class = NoPumpShrinkRay
    reducer = reducer_class(
        target=reduction_problem,
        treesitter_language=problem.language,
        python_reducer=False,
        restart_at_fixpoint=False,
        **kwargs,
    )
    if mode == "pump":
        # Isolate the inlining pump: drop the whole-file rewrite pass.
        reducer.last_ditch_passes = [
            p for p in reducer.last_ditch_passes if p.__name__ != "llm_rewrite"
        ]
    start = time.monotonic()
    with trio.move_on_after(PER_MODE_TIMEOUT) as scope:
        await reducer.run()
    elapsed = time.monotonic() - start

    final = reduction_problem.current_test_case
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{mode}-final{problem.extension}").write_bytes(final)
    exchanges = logging_client.exchanges if logging_client is not None else []
    if exchanges:
        (out_dir / f"{mode}-exchanges.json").write_text(
            json.dumps(exchanges, indent=2)
        )
    return {
        "mode": mode,
        "initial_size": len(problem.initial),
        "final_size": len(final),
        "final": final.decode("utf-8", errors="replace"),
        "calls": reduction_problem.stats.calls,
        "model_calls": len(exchanges),
        "seconds": round(elapsed, 1),
        "timed_out": scope.cancelled_caught,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problems", nargs="*", default=None)
    parser.add_argument("--modes", default="classical,rewrite,pump")
    args = parser.parse_args()
    names = args.problems or list(PROBLEMS)
    modes = args.modes.split(",")

    client = None
    if "pump" in modes or "rewrite" in modes:
        client = LlamaCppClient(model=parse_model_spec(DEFAULT_MODEL_SPEC))
        client.start_loading()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[dict]] = {}
    for name in names:
        problem = PROBLEMS[name]
        all_results[name] = []
        for mode in modes:
            print(f"=== {name} [{mode}] ===", flush=True)
            result = await reduce_problem(name, problem, mode, client)
            all_results[name].append(result)
            print(
                f"  {result['initial_size']}B -> {result['final_size']}B "
                f"in {result['seconds']}s "
                f"({result['calls']} oracle calls, "
                f"{result['model_calls']} model calls"
                f"{', TIMED OUT' if result['timed_out'] else ''})",
                flush=True,
            )
            print("  final: " + result["final"].replace("\n", "\\n"), flush=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results in {RESULTS_DIR}")


if __name__ == "__main__":
    trio.run(main)
