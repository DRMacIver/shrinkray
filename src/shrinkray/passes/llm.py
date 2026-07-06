"""Reduction passes that ask a language model to propose smaller test cases.

The pass feeds the whole current test case to a model with instructions to
produce reduced versions, extracts fenced code blocks from the response,
and offers each as a candidate to the reduction problem. Candidates that
don't pass the interestingness test, or that don't sort below the current
test case, are simply discarded, so a wrong or hallucinating model costs
time but never correctness.

Model inference runs in-process through llama-cpp-python (see
:mod:`shrinkray.llm_client`); the pass itself only depends on the abstract
:class:`LLMClient`, so any other completion source can be plugged in.
"""

import os
import re
from abc import ABC, abstractmethod

from attrs import define, frozen

from shrinkray.passes.definitions import ReductionPass
from shrinkray.problem import ReductionProblem


DEFAULT_MODEL_SPEC = "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf"


@frozen
class LocalModel:
    """A GGUF model file on the local filesystem."""

    path: str


@frozen
class HuggingFaceModel:
    """A GGUF file within a Hugging Face repository."""

    repo_id: str
    filename: str


def parse_model_spec(spec: str) -> LocalModel | HuggingFaceModel:
    """Parse a --llm-model argument.

    Accepts either a path to a local .gguf file or a Hugging Face
    ``repo_id:filename`` pair naming a GGUF file to download.
    """
    if os.path.exists(spec):
        return LocalModel(path=spec)
    repo_id, _, filename = spec.partition(":")
    if "/" in repo_id and repo_id and filename:
        return HuggingFaceModel(repo_id=repo_id, filename=filename)
    raise ValueError(
        f"Invalid model spec {spec!r}: expected a path to a local .gguf "
        "file or a Hugging Face 'repo/name:filename.gguf' reference."
    )


class LLMClient(ABC):
    """A source of text completions for the LLM passes."""

    @abstractmethod
    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        """Return the model's response to a single-message chat prompt."""
        ...


@define
class LLMConfig:
    """Tuning knobs for the LLM passes."""

    # Test cases larger than this are skipped: they don't fit comfortably
    # in a small model's context window, and generating a full rewrite of
    # a large file is too slow to be worth attempting.
    max_input_bytes: int = 32_768

    # How many reduced versions a single completion is asked to produce.
    n_candidates: int = 3

    # The pass ends after this many consecutive completions that produce
    # no adopted reduction.
    patience: int = 2

    # Sampling temperature, passed through to the model backend.
    temperature: float = 0.7

    # The name of the file being reduced, if known. Giving the model the
    # filename tells it the language without wasting prompt space.
    filename: str | None = None

    # A description of the interestingness condition, if known (for
    # example the text of the user's test script), included in the prompt.
    oracle: str | None = None


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_candidates(response: str) -> list[bytes]:
    """Extract candidate test cases from a model response.

    Candidates are the contents of fenced code blocks, in order, with
    empty blocks and duplicates dropped. Anything inside <think> tags is
    ignored: models with visible reasoning quote fragments there that are
    not intended as output.
    """
    seen: set[bytes] = set()
    result: list[bytes] = []
    for match in _FENCED_BLOCK.finditer(_THINK_BLOCK.sub("", response)):
        candidate = match.group(1).encode()
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def completion_max_tokens(input_size: int) -> int:
    """Token budget for a completion rewriting an input of this size.

    Enough to rewrite the whole input (roughly a token per three bytes)
    several times over, capped so a looping model can't stall the
    reduction indefinitely.
    """
    return min(8192, 512 + input_size)


def reduction_prompt(test_case: bytes, *, config: LLMConfig) -> str | None:
    """The prompt asking for reduced versions of ``test_case``.

    Returns None if the test case isn't valid UTF-8: the LLM passes only
    operate on text.
    """
    try:
        text = test_case.decode("utf-8")
    except UnicodeDecodeError:
        return None
    filename_part = (
        f" The file is called `{config.filename}`." if config.filename else ""
    )
    oracle_part = (
        "A reduced version is only accepted if the following interestingness "
        f"test still passes on it:\n\n{config.oracle}\n\n"
        if config.oracle
        else ""
    )
    return (
        "You are helping to minimize a test case that triggers a bug in a "
        "tool. Your job is to produce smaller versions of the file that "
        "still trigger the same bug: delete anything unnecessary, simplify "
        "what remains, and shorten names, while preserving whatever makes "
        f"the bug fire.{filename_part}\n\n"
        f"{oracle_part}"
        "The current test case is:\n\n"
        f"```\n{text}```\n\n"
        f"Output {config.n_candidates} different reduced versions, each in "
        "its own fenced code block, ordered from most aggressive (smallest) "
        "to most conservative (largest). Output nothing else: no "
        "explanations, no commentary."
    )


def llm_rewrite(client: LLMClient, config: LLMConfig) -> ReductionPass[bytes]:
    """A pass that asks the model for whole-file rewrites of the test case.

    Each round feeds the current test case to the model and tries every
    candidate it proposes; the pass ends after ``config.patience``
    consecutive rounds in which nothing was adopted.
    """

    async def apply(problem: ReductionProblem[bytes]) -> None:
        seen: set[bytes] = set()
        fruitless = 0
        while fruitless < config.patience:
            current = problem.current_test_case
            if len(current) > config.max_input_bytes:
                return
            prompt = reduction_prompt(current, config=config)
            if prompt is None:
                return
            response = await client.complete(
                prompt,
                max_tokens=completion_max_tokens(len(current)),
                seed=problem.work.random.getrandbits(32),
                temperature=config.temperature,
            )
            for candidate in extract_candidates(response):
                if candidate in seen:
                    continue
                seen.add(candidate)
                # Candidates that don't sort below the current test case
                # could never be adopted; don't waste a test run on them.
                if problem.sort_key(candidate) >= problem.sort_key(
                    problem.current_test_case
                ):
                    continue
                await problem.is_interesting(candidate)
            if problem.current_test_case == current:
                fruitless += 1
            else:
                fruitless = 0

    apply.__name__ = "llm_rewrite"
    return apply
