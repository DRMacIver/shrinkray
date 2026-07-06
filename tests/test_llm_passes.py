"""Tests for the LLM-based reduction passes.

These test the prompt construction, response parsing, and pass behaviour
against a fake in-memory client. No model is loaded and no network is
used; the llama-cpp integration is tested separately in
test_llm_client.py.
"""

from typing import Any

import pytest
import trio
from attrs import define, field

from shrinkray.cli import InputType
from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.llm import (
    DEFAULT_MODEL_SPEC,
    HuggingFaceModel,
    LLMClient,
    LLMConfig,
    LocalModel,
    completion_max_tokens,
    extract_candidates,
    llm_rewrite,
    parse_model_spec,
    read_oracle_script,
    reduction_prompt,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.state import ShrinkRayStateSingleFile
from shrinkray.work import Volume, WorkContext
from tests.helpers import reduce_with


# === Model spec parsing ===


def test_parse_model_spec_local_path(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"gguf")
    spec = parse_model_spec(str(path))
    assert spec == LocalModel(path=str(path))


def test_parse_model_spec_hf_repo():
    spec = parse_model_spec("unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf")
    assert spec == HuggingFaceModel(
        repo_id="unsloth/Qwen3.5-4B-GGUF", filename="Qwen3.5-4B-Q4_K_M.gguf"
    )


def test_default_model_spec_is_valid():
    spec = parse_model_spec(DEFAULT_MODEL_SPEC)
    assert isinstance(spec, HuggingFaceModel)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "no-such-file.gguf",
        "repo-with-no-filename",
        "unsloth/Qwen3.5-4B-GGUF:",
        ":file.gguf",
    ],
)
def test_parse_model_spec_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_model_spec(bad)


# === Response parsing ===


def test_extracts_single_fenced_block():
    assert extract_candidates("```\nhello\n```") == [b"hello\n"]


def test_extracts_block_with_language_tag():
    assert extract_candidates("```python\nx = 1\n```") == [b"x = 1\n"]


def test_extracts_multiple_blocks_in_order():
    response = "First:\n```\naa\n```\nSecond:\n```\nbb\n```\n"
    assert extract_candidates(response) == [b"aa\n", b"bb\n"]


def test_strips_thinking_blocks():
    response = "<think>\n```\nnot a candidate\n```\n</think>\n```\nreal\n```"
    assert extract_candidates(response) == [b"real\n"]


def test_drops_empty_and_duplicate_blocks():
    response = "```\n```\n```\naa\n```\n```\naa\n```"
    assert extract_candidates(response) == [b"aa\n"]


def test_no_blocks_means_no_candidates():
    assert extract_candidates("I cannot help with that.") == []


# === Prompt construction ===


def test_prompt_includes_test_case_and_instructions():
    prompt = reduction_prompt(b"int main() {}", config=LLMConfig())
    assert prompt is not None
    assert "int main() {}" in prompt
    assert "smaller" in prompt


def test_prompt_includes_filename_when_known():
    prompt = reduction_prompt(b"x", config=LLMConfig(filename="crash.cpp"))
    assert prompt is not None
    assert "crash.cpp" in prompt


def test_prompt_omits_filename_when_unknown():
    prompt = reduction_prompt(b"x", config=LLMConfig())
    assert prompt is not None
    assert "crash.cpp" not in prompt


def test_prompt_includes_oracle_when_known():
    prompt = reduction_prompt(
        b"x", config=LLMConfig(oracle="output must contain 'boom'")
    )
    assert prompt is not None
    assert "output must contain 'boom'" in prompt


def test_prompt_is_none_for_undecodable_input():
    assert reduction_prompt(b"\xc3\x28", config=LLMConfig()) is None


def test_completion_max_tokens_scales_with_input_but_is_capped():
    assert completion_max_tokens(100) < completion_max_tokens(10_000)
    assert completion_max_tokens(10_000_000) <= 8192


# === The pass itself ===


@define
class FakeLLMClient(LLMClient):
    """Returns scripted responses in order, then empty strings."""

    responses: list[str] = field(factory=list)
    prompts: list[str] = field(factory=list)
    seeds: list[int] = field(factory=list)

    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        await trio.lowlevel.checkpoint()
        self.prompts.append(prompt)
        self.seeds.append(seed)
        if len(self.prompts) <= len(self.responses):
            return self.responses[len(self.prompts) - 1]
        return ""


def test_adopts_a_valid_reduction():
    client = FakeLLMClient(responses=["```\nboom\n```"])
    result = reduce_with(
        [llm_rewrite(client, LLMConfig())],
        b"say boom please\n",
        lambda x: b"boom" in x,
    )
    assert result == b"boom\n"


def test_ignores_candidates_that_do_not_pass_the_test():
    client = FakeLLMClient(responses=["```\nquiet\n```"])
    result = reduce_with(
        [llm_rewrite(client, LLMConfig())],
        b"say boom please\n",
        lambda x: b"boom" in x,
    )
    assert result == b"say boom please\n"


def test_tries_multiple_candidates_from_one_response():
    client = FakeLLMClient(responses=["```\nquiet\n```\n```\nboom\n```"])
    result = reduce_with(
        [llm_rewrite(client, LLMConfig())],
        b"say boom please\n",
        lambda x: b"boom" in x,
    )
    assert result == b"boom\n"


def test_keeps_going_while_making_progress():
    client = FakeLLMClient(
        responses=[
            "```\nboom boom boom\n```",
            "```\nboom\n```",
        ]
    )
    result = reduce_with(
        [llm_rewrite(client, LLMConfig())],
        b"say boom please and boom again\n",
        lambda x: b"boom" in x,
    )
    assert result == b"boom\n"


def test_stops_after_patience_fruitless_calls():
    client = FakeLLMClient()
    reduce_with(
        [llm_rewrite(client, LLMConfig(patience=3))],
        b"say boom please\n",
        lambda x: b"boom" in x,
    )
    assert len(client.prompts) == 3


def test_does_not_retest_candidates_across_calls():
    # The same useless candidate twice: the second response is a repeat, so
    # the pass makes no second is_interesting call for it and gives up.
    calls: list[bytes] = []

    def is_interesting(x: bytes) -> bool:
        calls.append(x)
        return b"boom" in x

    client = FakeLLMClient(responses=["```\nquiet\n```", "```\nquiet\n```"])
    reduce_with(
        [llm_rewrite(client, LLMConfig(patience=2))],
        b"say boom please\n",
        is_interesting,
    )
    assert calls.count(b"quiet\n") == 1


def test_skips_oversized_input_without_calling_the_model():
    client = FakeLLMClient(responses=["```\nboom\n```"])
    result = reduce_with(
        [llm_rewrite(client, LLMConfig(max_input_bytes=5))],
        b"say boom please\n",
        lambda x: b"boom" in x,
    )
    assert result == b"say boom please\n"
    assert client.prompts == []


def test_skips_undecodable_input_without_calling_the_model():
    client = FakeLLMClient(responses=["```\nboom\n```"])
    result = reduce_with(
        [llm_rewrite(client, LLMConfig())],
        b"\xc3\x28 boom \xc3\x28",
        lambda x: b"boom" in x,
    )
    assert result == b"\xc3\x28 boom \xc3\x28"
    assert client.prompts == []


def test_does_not_test_candidates_that_sort_above_current():
    # A "reduction" that's larger than the current test case must not be
    # run against the interestingness test at all.
    calls: list[bytes] = []

    def is_interesting(x: bytes) -> bool:
        calls.append(x)
        return b"boom" in x

    client = FakeLLMClient(responses=["```\nboom boom boom boom boom boom boom\n```"])
    reduce_with(
        [llm_rewrite(client, LLMConfig(patience=1))],
        b"say boom\n",
        is_interesting,
    )
    assert b"boom boom boom boom boom boom boom\n" not in calls


def test_seeds_are_drawn_from_the_work_context():
    async def is_boom(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"boom" in x

    async def run_once() -> list[int]:
        client = FakeLLMClient()
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=b"say boom\n",
            is_interesting=is_boom,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        await llm_rewrite(client, LLMConfig(patience=2))(problem)
        return client.seeds

    first = trio.run(run_once)
    second = trio.run(run_once)
    assert first == second
    assert len(set(first)) == len(first)


# === Oracle script reading ===


def test_read_oracle_script_returns_text(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep boom "$1"\n')
    assert read_oracle_script(str(script)) == '#!/bin/sh\ngrep boom "$1"\n'


def test_read_oracle_script_missing_file(tmp_path):
    assert read_oracle_script(str(tmp_path / "nope.sh")) is None


def test_read_oracle_script_binary(tmp_path):
    script = tmp_path / "test"
    script.write_bytes(b"\x7fELF\xc3\x28\x00\x01")
    assert read_oracle_script(str(script)) is None


def test_read_oracle_script_too_large(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("x" * 100_000)
    assert read_oracle_script(str(script)) is None


# === Reducer wiring ===


def make_shrinkray(initial: bytes = b"say boom\n", **kwargs) -> ShrinkRay:
    async def is_boom(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"boom" in x

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=initial,
        is_interesting=is_boom,
        work=WorkContext(parallelism=1),
    )
    return ShrinkRay(target=problem, **kwargs)


def test_llm_pass_runs_late_when_a_client_is_configured():
    reducer = make_shrinkray(llm_client=FakeLLMClient())
    assert [p.__name__ for p in reducer.last_ditch_passes].count("llm_rewrite") == 1
    assert "llm_rewrite" not in [p.__name__ for p in reducer.great_passes]


def test_no_llm_pass_without_a_client():
    reducer = make_shrinkray()
    all_passes = (
        reducer.initial_cuts
        + reducer.great_passes
        + reducer.ok_passes
        + reducer.last_ditch_passes
        + reducer.polish_passes
    )
    assert "llm_rewrite" not in [p.__name__ for p in all_passes]


def test_llm_only_disables_every_other_pass():
    reducer = make_shrinkray(
        llm_client=FakeLLMClient(), llm_only=True, enable_cpp_passes=True
    )
    assert [p.__name__ for p in reducer.great_passes] == ["llm_rewrite"]
    assert reducer.initial_cuts == []
    assert reducer.ok_passes == []
    assert reducer.last_ditch_passes == []
    assert reducer.polish_passes == []
    assert list(reducer.pumps) == []


def test_llm_only_requires_a_client():
    with pytest.raises(ValueError, match="llm_only"):
        make_shrinkray(llm_only=True)


def test_llm_only_reduction_end_to_end():
    client = FakeLLMClient(responses=["```\nboom\n```"])
    reducer = make_shrinkray(
        initial=b"say boom please\n", llm_client=client, llm_only=True
    )

    trio.run(reducer.run)
    assert reducer.target.current_test_case == b"boom\n"


# === State wiring ===


def make_llm_state(tmp_path, **overrides) -> ShrinkRayStateSingleFile:
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep boom "$1"\n')
    script.chmod(0o755)
    target = tmp_path / "target.txt"
    target.write_text("say boom\n")
    kwargs: dict[str, Any] = {
        "input_type": InputType.all,
        "in_place": False,
        "test": [str(script)],
        "filename": str(target),
        "timeout": 30.0,
        "base": "target.txt",
        "parallelism": 1,
        "initial": b"say boom\n",
        "formatter": "none",
        "trivial_is_error": True,
        "seed": 0,
        "volume": Volume.quiet,
        "history_enabled": False,
    }
    kwargs.update(overrides)
    return ShrinkRayStateSingleFile(**kwargs)


def make_state_problem(state: ShrinkRayStateSingleFile) -> BasicReductionProblem[bytes]:
    async def is_boom(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"boom" in x

    return BasicReductionProblem(
        initial=b"say boom\n",
        is_interesting=is_boom,
        work=WorkContext(parallelism=1),
    )


def test_state_does_not_configure_llm_by_default(tmp_path):
    state = make_llm_state(tmp_path)
    reducer = state.new_reducer(make_state_problem(state))
    assert isinstance(reducer, ShrinkRay)
    assert reducer.llm_client is None


def test_state_wires_llm_into_the_reducer(tmp_path):
    state = make_llm_state(tmp_path, llm_enabled=True)
    reducer = state.new_reducer(make_state_problem(state))
    assert isinstance(reducer, ShrinkRay)
    client = reducer.llm_client
    assert isinstance(client, LlamaCppClient)
    assert client.model == parse_model_spec(DEFAULT_MODEL_SPEC)
    assert reducer.llm_config.filename == "target.txt"
    assert reducer.llm_config.oracle is not None
    assert "grep boom" in reducer.llm_config.oracle
    assert not reducer.llm_only

    # The client (and so the loaded model) is shared across reducers.
    second = state.new_reducer(make_state_problem(state))
    assert isinstance(second, ShrinkRay)
    assert second.llm_client is client


def test_state_passes_llm_only_through(tmp_path):
    state = make_llm_state(tmp_path, llm_enabled=True, llm_only=True)
    reducer = state.new_reducer(make_state_problem(state))
    assert isinstance(reducer, ShrinkRay)
    assert reducer.llm_only
