"""Prompt-variant experiments for shrink ray's LLM mode.

Runs prompt variants against benchmark.py problems (cheap in-process
oracles over real corpus files) and reports validity rate, size
reduction, and cost per generation. Results are appended to
work/llm_experiments.jsonl for later analysis.

Needs the llm extra (llama-cpp-python) and downloads the model on first
use. Runs GPU-accelerated where available (pass --cpu to measure
CPU-only speed); prompt-quality conclusions transfer, generation speed
does not.

    uv run python evaluation/llm_prompt_experiment.py --samples 2
    uv run python evaluation/llm_prompt_experiment.py --variants shipped_oracle --problems corpus_mypy
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SHRINKRAY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SHRINKRAY / "evaluation"))

import benchmark  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from llama_cpp import Llama  # noqa: E402
from shrinkray.passes.llm import LLMConfig as ShippedConfig  # noqa: E402
from shrinkray.passes.llm import extract_candidates as shipped_extract  # noqa: E402
from shrinkray.passes.llm import reduction_prompt as shipped_prompt  # noqa: E402
from shrinkray.state import sort_key_for_initial  # noqa: E402

CORPUS = SHRINKRAY / "evaluation" / "corpus"
RESULTS = SHRINKRAY / "evaluation" / "work" / "llm_experiments.jsonl"

# Inputs bigger than this are skipped: they don't fit the context window.
MAX_INPUT_BYTES = 40_000

# Natural-language description of each predicate, used by the *_oracle
# variants (stands in for the user's interestingness script).
ORACLES = {
    "python_syntax": "The file must be valid Python (compile() succeeds) and must still contain the string SENTINEL_KEEP.",
    "corpus_mypy": "The file must be valid Python (compile() succeeds) and must still contain all of these substrings: 'match ', 'case ', 'Union[', 'tuple['.",
    "corpus_pylint": "The file must be valid Python and must still define a class with duplicate bases and a metaclass (the pattern that crashes pylint 2.17.4).",
    "corpus_ujson": "The file must still be JSON nested at least 20 levels deep.",
    "corpus_udlit_cpp": "The file must still contain all of: 'operator\"\"', 'decltype(', '...'.",
    "corpus_minisat": "The file must still contain the number 2147483648.",
}

# Map benchmark problem -> corpus entry whose shrinkray_reduced.* we can
# also use as a "polished" input (can the LLM beat shrinkray's fixpoint?).
REDUCED_INPUTS = {
    "corpus_mypy": "mypy-0.942-match-union-tuple-crash/shrinkray_reduced.py",
    "corpus_pylint": "pylint-2.17.4-duplicate-bases-mro-crash/shrinkray_reduced.py",
    "corpus_ujson": "ujson-510-indent-buffer-overflow/shrinkray_reduced.json",
    "corpus_udlit_cpp": "gcc49-udlit-char-pack-template/shrinkray_reduced.cpp",
    "corpus_minisat": "minisat-dimacs-int-overflow/shrinkray_reduced.cnf",
}

PROBLEM_NAMES = list(ORACLES)

FENCE_RE = re.compile(rb"```[^\n]*\n(.*?)```", re.DOTALL)
THINK_RE = re.compile(rb"<think>.*?</think>", re.DOTALL)


@dataclass
class Variant:
    name: str
    build_prompt: Callable[[bytes, str], str]
    # Turn the raw response into candidate byte strings.
    extract: Callable[[bytes, bytes], list[bytes]]
    max_tokens: int = 4096


def strip_thinking(response: bytes) -> bytes:
    return THINK_RE.sub(b"", response)


def extract_blocks(response: bytes, current: bytes) -> list[bytes]:
    return [m.group(1) for m in FENCE_RE.finditer(strip_thinking(response))]


def base_prompt(test_case: bytes, oracle: str) -> str:
    return (
        "You are helping minimize a test case that triggers a bug in a tool. "
        "Your job is to produce a smaller version of the file that still "
        "triggers the same bug. Remove or simplify as much as possible: "
        "delete unused code, shorten names, inline things, drop anything "
        "not needed to trigger the bug."
        f"{oracle}\n\n"
        "The current test case is:\n\n"
        "```\n" + test_case.decode("utf-8", errors="replace") + "\n```\n\n"
    )


def full_rewrite_prompt(test_case: bytes, oracle: str) -> str:
    return base_prompt(test_case, oracle) + (
        "Output ONLY the reduced file contents inside a single fenced code "
        "block, with no explanation."
    )


def multi_candidate_prompt(test_case: bytes, oracle: str) -> str:
    return base_prompt(test_case, oracle) + (
        "Output THREE different reduced versions, each in its own fenced "
        "code block, ordered from most aggressive (smallest, riskiest) to "
        "most conservative (largest, safest). No explanations."
    )


def line_deletion_prompt(test_case: bytes, oracle: str) -> str:
    lines = test_case.split(b"\n")
    numbered = "\n".join(
        f"{i + 1}: {line.decode('utf-8', errors='replace')}"
        for i, line in enumerate(lines)
    )
    return (
        "You are helping minimize a test case that triggers a bug in a tool. "
        "Identify lines that can be DELETED while still triggering the bug."
        f"{oracle}\n\n"
        "The current test case, with line numbers, is:\n\n"
        f"```\n{numbered}\n```\n\n"
        "Output ONLY a single line of the form DELETE: followed by a "
        "comma-separated list of line numbers or ranges, e.g. "
        "'DELETE: 3,7-12,20'. Prefer deleting a lot. No explanations."
    )


DELETE_RE = re.compile(rb"DELETE:\s*([0-9,\-\s]+)")


def extract_line_deletion(response: bytes, current: bytes) -> list[bytes]:
    m = DELETE_RE.search(strip_thinking(response))
    if m is None:
        return []
    to_delete: set[int] = set()
    for part in m.group(1).split(b","):
        part = part.strip()
        if not part:
            continue
        if b"-" in part:
            lo, _, hi = part.partition(b"-")
            try:
                to_delete.update(range(int(lo), int(hi) + 1))
            except ValueError:
                return []
        else:
            try:
                to_delete.add(int(part))
            except ValueError:
                return []
    lines = current.split(b"\n")
    kept = [line for i, line in enumerate(lines, 1) if i not in to_delete]
    if len(kept) == len(lines):
        return []
    return [b"\n".join(kept)]


def oracle_text(name: str, include: bool) -> str:
    if not include:
        return ""
    return (
        "\n\nThe reduced file is only accepted if this check still passes: "
        + ORACLES[name]
    )


VARIANTS = {
    "shipped_oracle": Variant(
        "shipped_oracle",
        lambda tc, o: "",  # replaced in run_one
        lambda resp, cur: shipped_extract(resp.decode("utf-8", errors="replace")),
    ),
    "full_rewrite": Variant(
        "full_rewrite",
        lambda tc, o: full_rewrite_prompt(tc, o),
        extract_blocks,
    ),
    "full_rewrite_oracle": Variant(
        "full_rewrite_oracle",
        lambda tc, o: full_rewrite_prompt(tc, o),
        extract_blocks,
    ),
    "multi_candidate": Variant(
        "multi_candidate",
        lambda tc, o: multi_candidate_prompt(tc, o),
        extract_blocks,
    ),
    "line_deletion": Variant(
        "line_deletion",
        lambda tc, o: line_deletion_prompt(tc, o),
        extract_line_deletion,
        max_tokens=1024,
    ),
    "multi_candidate_oracle": Variant(
        "multi_candidate_oracle",
        lambda tc, o: multi_candidate_prompt(tc, o),
        extract_blocks,
    ),
}

FILENAMES = {
    "python_syntax": "input.py",
    "corpus_mypy": "original.py",
    "corpus_pylint": "original.py",
    "corpus_ujson": "original.json",
    "corpus_udlit_cpp": "original.cpp",
    "corpus_minisat": "original.cnf",
}


@dataclass
class RunResult:
    problem: str
    input_kind: str  # original | reduced
    variant: str
    sample: int
    input_size: int
    time: float
    prompt_tokens: int
    completion_tokens: int
    n_candidates: int
    n_valid: int
    n_improving: int  # valid and sorts below input
    best_size: int | None
    candidates: list[dict] = field(default_factory=list)


def run_one(
    llm: Llama,
    name: str,
    input_kind: str,
    test_case: bytes,
    predicate,
    variant: Variant,
    sample: int,
    temperature: float,
    no_think: bool,
) -> RunResult:
    include_oracle = variant.name.endswith("_oracle") or variant.name == "line_deletion"
    if variant.name == "shipped_oracle":
        shipped = shipped_prompt(
            test_case,
            config=ShippedConfig(
                oracle=ORACLES[name],
                filename=FILENAMES.get(name),
                n_candidates=3,
            ),
        )
        assert shipped is not None
        prompt = shipped
    else:
        prompt = variant.build_prompt(test_case, oracle_text(name, include_oracle))
    if no_think:
        prompt += "\n/no_think"
    t0 = time.time()
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=variant.max_tokens,
        temperature=temperature,
    )
    elapsed = time.time() - t0
    usage = result["usage"]
    text = (result["choices"][0]["message"]["content"] or "").encode()
    candidates = variant.extract(text, test_case)

    sk = sort_key_for_initial(test_case)
    input_key = sk(test_case)
    cand_info = []
    n_valid = n_improving = 0
    best_size = None
    for c in candidates:
        c = c.rstrip(b"\n") + b"\n" if c.endswith(b"\n") else c
        try:
            valid = bool(predicate(c))
        except Exception:
            valid = False
        improving = valid and sk(c) < input_key
        n_valid += valid
        n_improving += improving
        if improving and (best_size is None or len(c) < best_size):
            best_size = len(c)
        cand_info.append({"size": len(c), "valid": valid, "improving": improving})
    return RunResult(
        problem=name,
        input_kind=input_kind,
        variant=variant.name,
        sample=sample,
        input_size=len(test_case),
        time=elapsed,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        n_candidates=len(candidates),
        n_valid=n_valid,
        n_improving=n_improving,
        best_size=best_size,
        candidates=cand_info,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="*", default=PROBLEM_NAMES)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cpu", action="store_true", help="disable Metal")
    parser.add_argument("--think", action="store_true", help="allow thinking mode")
    parser.add_argument("--inputs", nargs="*", default=["original", "reduced"])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    path = hf_hub_download("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf")
    llm = Llama(
        model_path=path,
        n_ctx=16384,
        n_gpu_layers=0 if args.cpu else -1,
        verbose=False,
    )

    problems = benchmark.build_problems()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        for name in args.problems:
            problem = problems[name]
            inputs = [("original", problem.initial)]
            if name in REDUCED_INPUTS:
                inputs.append(
                    ("reduced", (CORPUS / REDUCED_INPUTS[name]).read_bytes())
                )
            for input_kind, data in inputs:
                if input_kind not in args.inputs:
                    continue
                if len(data) > MAX_INPUT_BYTES:
                    print(f"{name}/{input_kind}: skipped ({len(data)} bytes)")
                    continue
                for vname in args.variants:
                    variant = VARIANTS[vname]
                    for sample in range(args.samples):
                        try:
                            r = run_one(
                                llm,
                                name,
                                input_kind,
                                data,
                                problem.predicate,
                                variant,
                                sample,
                                args.temperature,
                                no_think=not args.think,
                            )
                        except Exception as e:
                            print(
                                f"{name}/{input_kind}/{vname}#{sample}: ERROR {e!r}",
                                flush=True,
                            )
                            continue
                        row = {"tag": args.tag, "think": args.think, **r.__dict__}
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(
                            f"{name}/{input_kind}/{vname}#{sample}: "
                            f"{r.n_candidates} cand, {r.n_valid} valid, "
                            f"{r.n_improving} improving, best {r.best_size} "
                            f"(from {r.input_size}), {r.time:.1f}s "
                            f"{r.completion_tokens}tok",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
