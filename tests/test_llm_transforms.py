"""Tests for the grammar-guided LLM transformation pumps.

These use real tree-sitter grammars to find transformation targets, and
a fake in-memory client for the model side; no model is loaded and no
network is used.
"""

import contextlib
import io
import re
import warnings

import pytest
import trio
from attrs import define

from shrinkray.passes.cpp import CPP_PUMPS
from shrinkray.passes.llm import LLMConfig
from shrinkray.passes.llmtransforms import (
    MAX_SNIPPET_BYTES,
    TransformTarget,
    inline_call_targets,
    llm_transform_pump,
    llm_transform_pumps,
    splice,
    transform_prompt,
)
from shrinkray.passes.treesitter import parse_tree
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.work import WorkContext
from tests.helpers import FakeLLMClient, RecordingClient


def targets_for(language: str, source: bytes) -> list[TransformTarget]:
    return inline_call_targets(parse_tree(language, source), source)


def target_texts(language: str, source: bytes) -> list[str]:
    return [t.text for t in targets_for(language, source)]


# === Finding inlinable calls ===


PY_SOURCE = b"def f(x):\n    return x + 1\n\nprint(f(3))\n"


def test_finds_a_python_call_with_its_definition():
    (target,) = targets_for("python", PY_SOURCE)
    assert PY_SOURCE[target.span[0] : target.span[1]] == b"f(3)"
    assert target.text == "f(3)"
    assert "def f(x):" in target.context
    assert "`f`" in target.instruction


def test_finds_a_c_call_through_the_declarator_chain():
    source = (
        b"int add(int x, int y) { return x + y; }\n"
        b"int main(void) { return add(1, 2); }\n"
    )
    assert target_texts("c", source) == ["add(1, 2)"]


def test_finds_calls_in_other_grammars():
    go = b"package main\nfunc f(x int) int { return x + 1 }\nfunc main() { println(f(3)) }\n"
    assert target_texts("go", go) == ["f(3)"]
    js = b"function f(x) { return x + 1; }\nconsole.log(f(3));\n"
    assert target_texts("javascript", js) == ["f(3)"]
    ruby = b"def f(x)\n  x + 1\nend\nputs f(3)\n"
    assert target_texts("ruby", ruby) == ["f(3)"]


def test_skips_calls_through_an_attribute():
    source = b"def f(x):\n    return x\n\nobj.f(3)\n"
    assert targets_for("python", source) == []


def test_skips_calls_with_a_receiver():
    source = b"def f(x)\n  x + 1\nend\nobj.f(3)\n"
    assert targets_for("ruby", source) == []


def test_skips_java_calls_on_an_explicit_object():
    source = (
        b"class A { static int f(int x) { return x + 1; } "
        b"int g() { return f(3) + this.f(4); } }\n"
    )
    assert target_texts("java", source) == ["f(3)"]


def test_skips_calls_with_no_recognisable_callee():
    # Rust macro invocations are call-like but name no function.
    source = b'fn f(x: i32) -> i32 { x + 1 }\nfn main() { let y = f(3); println!("hi"); }\n'
    assert target_texts("rust", source) == ["f(3)"]


def test_skips_recursive_calls_inside_the_definition():
    source = b"def f(x):\n    return f(x - 1)\n\nprint(f(3))\n"
    assert target_texts("python", source) == ["f(3)"]


def test_skips_ambiguously_overloaded_names():
    source = (
        b"int f(int x) { return x; }\n"
        b"int f(int x, int y) { return x + y; }\n"
        b"int main() { return f(1); }\n"
    )
    assert targets_for("cpp", source) == []


def test_skips_calls_to_unknown_functions():
    assert targets_for("python", b"print(g(3))\n") == []


def test_skips_definitions_without_bodies():
    source = b"interface I { int h(int x); }\nclass A { int g() { return h(3); } }\n"
    assert targets_for("java", source) == []


def test_skips_definitions_without_names():
    source = (
        b"struct A{}; int operator+(A a, A b) { return 0; }\n"
        b"int main() { A x; A y; x + y; return 0; }\n"
    )
    assert targets_for("cpp", source) == []


def test_skips_oversized_definitions():
    body = b" + ".join([b"x"] * MAX_SNIPPET_BYTES)
    source = b"def f(x):\n    return " + body + b"\n\nprint(f(3))\n"
    assert targets_for("python", source) == []


def test_skips_oversized_calls():
    argument = b"'" + b"a" * MAX_SNIPPET_BYTES + b"'"
    source = b"def f(x):\n    return x\n\nprint(f(" + argument + b"))\n"
    assert targets_for("python", source) == []


def test_skips_undecodable_definitions():
    source = b'def f(x):\n    return "\xff"\n\nprint(f(3))\n'
    assert targets_for("python", source) == []


def test_skips_undecodable_calls():
    source = b'def f(x):\n    return x\n\nprint(f("\xff"))\n'
    assert targets_for("python", source) == []


# === Prompt construction and splicing ===


def test_splice_replaces_the_span():
    assert splice(b"abcdef", (2, 4), b"XY") == b"abXYef"
    assert splice(b"abcdef", (2, 4), b"") == b"abef"


def test_prompt_contains_instruction_context_and_target():
    (target,) = targets_for("python", PY_SOURCE)
    prompt = transform_prompt(target)
    assert target.instruction in prompt
    assert target.context in prompt
    assert target.text in prompt
    assert "fenced code block" in prompt


# === The pump ===


def run_pump(pump, initial, is_interesting):
    calls = []

    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        calls.append(x)
        return is_interesting(x)

    async def calc_result() -> bytes:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=initial,
            is_interesting=acondition,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await pump(problem)

    return trio.run(calc_result), calls


def inline_pump(client, **kwargs):
    return llm_transform_pump(
        client,
        LLMConfig(),
        "python",
        "llm_inline_calls(python)",
        inline_call_targets,
        **kwargs,
    )


def test_inlines_a_call_end_to_end():
    client = FakeLLMClient(responses=["```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), PY_SOURCE, lambda x: b"print" in x)
    assert result == b"def f(x):\n    return x + 1\n\nprint((3 + 1))\n"
    (prompt,) = client.prompts
    assert "def f(x):" in prompt
    assert "f(3)" in prompt


def test_does_not_adopt_uninteresting_candidates():
    client = FakeLLMClient(responses=["```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), PY_SOURCE, lambda x: b"f(3)" in x)
    assert result == PY_SOURCE


def test_a_model_echoing_the_call_is_not_retested():
    client = FakeLLMClient(responses=["```\nf(3)\n```"])
    result, calls = run_pump(inline_pump(client), PY_SOURCE, lambda x: b"print" in x)
    assert result == PY_SOURCE
    # Only the initial test-case check; the echoed candidate is skipped.
    assert calls == [PY_SOURCE]


def test_identical_calls_are_inlined_from_one_completion():
    # Both calls produce the same prompt; after the first adoption the
    # second call's splice is satisfied from the cached response.
    source = b"def f(x):\n    return x + 1\n\nprint(f(3))\nprint(f(3))\n"
    client = FakeLLMClient(responses=["```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), source, lambda x: b"print" in x)
    assert result == b"def f(x):\n    return x + 1\n\nprint((3 + 1))\nprint((3 + 1))\n"
    assert len(client.prompts) == 1


def test_retries_a_fruitless_prompt_with_a_fresh_seed():
    client = FakeLLMClient(responses=["no code block here", "```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), PY_SOURCE, lambda x: b"print" in x)
    assert result == b"def f(x):\n    return x + 1\n\nprint((3 + 1))\n"
    assert len(client.prompts) == 2
    assert client.prompts[0] == client.prompts[1]
    assert client.seeds[0] != client.seeds[1]


def test_retries_per_prompt_are_bounded():
    client = FakeLLMClient()  # always answers with no code block
    result, _ = run_pump(
        inline_pump(client, max_prompt_attempts=3), PY_SOURCE, lambda x: b"print" in x
    )
    assert result == PY_SOURCE
    assert len(client.prompts) == 3
    assert len(set(client.prompts)) == 1


def test_rederives_targets_after_each_adoption():
    source = b"def f(x):\n    return x + 1\n\nprint(f(3))\nprint(f(5))\n"
    client = FakeLLMClient(responses=["```\n(3 + 1)\n```", "```\n(5 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), source, lambda x: b"print" in x)
    assert result == b"def f(x):\n    return x + 1\n\nprint((3 + 1))\nprint((5 + 1))\n"
    assert len(client.prompts) == 2


def test_max_adoptions_bounds_a_single_invocation():
    source = b"def f(x):\n    return x + 1\n\nprint(f(3))\nprint(f(5))\n"
    client = FakeLLMClient(responses=["```\n(3 + 1)\n```", "```\n(5 + 1)\n```"])
    result, _ = run_pump(
        inline_pump(client, max_adoptions=1), source, lambda x: b"print" in x
    )
    assert result == b"def f(x):\n    return x + 1\n\nprint((3 + 1))\nprint(f(5))\n"


def test_max_completions_bounds_model_calls():
    source = b"def f(x):\n    return x + 1\n\nprint(f(3))\nprint(f(5))\n"
    client = FakeLLMClient(responses=["nope", "nope"])
    result, _ = run_pump(
        inline_pump(client, max_completions=1), source, lambda x: b"print" in x
    )
    assert result == source
    assert len(client.prompts) == 1


class DisabledClient(RecordingClient):
    def is_disabled(self) -> bool:
        return True


def test_disabled_client_is_never_asked():
    client = DisabledClient(responses=["```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), PY_SOURCE, lambda x: b"print" in x)
    assert result == PY_SOURCE
    assert "complete" not in client.events


def test_waits_for_readiness_before_completing():
    client = RecordingClient(responses=["```\n(3 + 1)\n```"])
    run_pump(inline_pump(client), PY_SOURCE, lambda x: b"print" in x)
    assert client.events.index("wait_until_ready") < client.events.index("complete")


def test_no_model_interaction_without_targets():
    client = RecordingClient(responses=["```\n(3 + 1)\n```"])
    result, _ = run_pump(inline_pump(client), b"hello world\n", lambda x: True)
    assert result == b"hello world\n"
    assert client.events == []


_REWRITE_BLOCK = re.compile(r"The code to rewrite is:\n\n```\n(.*?)\n```", re.DOTALL)


@define
class InlineOnlyClient(FakeLLMClient):
    """Inlines `name(number)` calls to an add-one function; nothing else.

    A scripted stand-in for the model: whatever the function and its
    argument have been renamed to by other passes, a call shaped like
    `name(number)` is answered with `(number+1)`.
    """

    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        await super().complete(
            prompt, max_tokens=max_tokens, seed=seed, temperature=temperature
        )
        block = _REWRITE_BLOCK.search(prompt)
        if "mechanical refactoring" in prompt and block is not None:
            call = re.fullmatch(r"\w+\((\d+)\)", block.group(1))
            if call is not None:
                return f"```\n({call.group(1)}+1)\n```"
        return ""


@pytest.mark.parametrize("parallelism", [1, 2])
def test_inlining_unlocks_deleting_the_definition(parallelism):
    # A semantic oracle: the program must still print 4. The definition
    # cannot be deleted while the call needs it, and no ordinary pass
    # rewrites the call into its inlined form; only the combination
    # reaches a result with no function definition left.
    def is_interesting(x: bytes) -> bool:
        out = io.StringIO()
        try:
            # Compiling mangled candidates raises SyntaxWarnings (e.g.
            # "'int' object is not callable"); they are the oracle
            # rejecting a candidate, not a problem to report.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code = compile(x.decode("utf-8"), "<candidate>", "exec")
            with contextlib.redirect_stdout(out):
                exec(code, {})
        except Exception:
            return False
        return out.getvalue() == "4\n"

    async def acondition(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return is_interesting(x)

    async def calc_result() -> bytes:
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=PY_SOURCE,
            is_interesting=acondition,
            work=WorkContext(parallelism=parallelism),
        )
        reducer = ShrinkRay(
            target=problem,
            llm_client=InlineOnlyClient(),
            treesitter_language="python",
            python_reducer=False,
        )
        await reducer.run()
        return problem.current_test_case

    result = trio.run(calc_result)
    assert is_interesting(result)
    assert b"def" not in result
    assert len(result) <= len(b"print((3 + 1))\n")


# === Reducer wiring ===


def make_shrinkray(**kwargs) -> ShrinkRay:
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"boom" in x

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=b"say boom\n",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    return ShrinkRay(target=problem, **kwargs)


def pump_names(reducer: ShrinkRay) -> list[str]:
    return [p.__name__ for p in reducer.pumps]


def test_llm_transform_pumps_are_named_for_their_language():
    pumps = llm_transform_pumps(FakeLLMClient(), LLMConfig(), "python")
    assert [p.__name__ for p in pumps] == ["llm_inline_calls(python)"]


def test_reducer_builds_llm_pumps_when_fully_configured():
    reducer = make_shrinkray(llm_client=FakeLLMClient(), treesitter_language="python")
    assert pump_names(reducer) == ["llm_inline_calls(python)"]


def test_reducer_keeps_cpp_pumps_ahead_of_llm_pumps():
    reducer = make_shrinkray(
        llm_client=FakeLLMClient(),
        treesitter_language="cpp",
        enable_cpp_passes=True,
    )
    assert pump_names(reducer) == [p.__name__ for p in CPP_PUMPS] + [
        "llm_inline_calls(cpp)"
    ]


def test_no_llm_pumps_without_a_client():
    reducer = make_shrinkray(treesitter_language="python")
    assert pump_names(reducer) == []


def test_no_llm_pumps_without_a_grammar():
    reducer = make_shrinkray(llm_client=FakeLLMClient())
    assert pump_names(reducer) == []
