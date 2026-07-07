"""Grammar-guided LLM transformation pumps.

Some transformations help reduction in nearly every language. Inlining
a function call is the canonical example: it usually makes the file
bigger, but afterwards the function definition may be deletable, for a
large net win. Shrink Ray implements this by hand for C and C++
(:mod:`shrinkray.passes.cpp`), but a hand-written inliner needs
per-language knowledge of syntax, so that approach doesn't scale across
languages.

These pumps combine two features that individually can't do the job:

- A tree-sitter grammar can *identify* the ingredients of such a
  transformation in any language it covers (here: a function
  definition and a call to it) but cannot perform the rewrite.
- A language model can *perform* the rewrite, but is unreliable at
  leaving the rest of the file alone when asked to rewrite whole files.

So tree-sitter picks the byte span to rewrite, the model is asked for
replacement text for that span only, and the replacement is spliced in
with the rest of the file untouched byte-for-byte. A wrong rewrite
produces a candidate that fails the interestingness test and is
discarded, so an imperfect model costs time but never correctness.
"""

from collections.abc import Callable

import tree_sitter
from attrs import frozen

from shrinkray.passes.definitions import ReductionPump
from shrinkray.passes.llm import (
    LLMClient,
    LLMConfig,
    completion_max_tokens,
    extract_candidates,
)
from shrinkray.passes.treesitter import iter_nodes, parse_tree
from shrinkray.problem import ReductionProblem


@frozen
class TransformTarget:
    """One place in the file a transformation could be applied."""

    # The byte range of the current test case to replace.
    span: tuple[int, int]

    # The text of that range.
    text: str

    # What to ask the model to do to it.
    instruction: str

    # Supporting code the model needs to see (e.g. the definition of
    # the function being inlined).
    context: str


TargetFinder = Callable[[tree_sitter.Tree, bytes], list[TransformTarget]]


# Definitions and calls bigger than this are not offered to the model:
# past this size the rewrite is unlikely to succeed and the prompt and
# completion get expensive.
MAX_SNIPPET_BYTES = 2048

# Bound on how many candidates a pump adopts in a single invocation.
# Pumps get rerun by the reducer as long as they lead to progress, so
# this only limits how much one invocation can balloon the test case.
MAX_ADOPTIONS = 20

# Bound on model calls in a single invocation, so a file with very many
# targets can't stall the reduction on one pump.
MAX_COMPLETIONS = 50

# How many completions to try per distinct prompt (with fresh seeds)
# before giving up on that target. Sampling means a single bad answer
# is common even when the model usually gets the rewrite right; a
# couple of retries recovers those cheaply.
MAX_PROMPT_ATTEMPTS = 3


# Node types whose underscore-separated words intersect these are taken
# to be function definitions / calls. This convention holds across the
# tree-sitter grammars: function_definition (Python, C, C++),
# function_declaration (Go, JavaScript), function_item (Rust),
# method_declaration (Java), method (Ruby), call, call_expression,
# method_invocation, ...
_DEFINITION_WORDS = frozenset({"function", "method"})
_CALL_WORDS = frozenset({"call", "invocation"})


def _matches(node_type: str, words: frozenset[str]) -> bool:
    return not words.isdisjoint(node_type.split("_"))


def _definition_name(node: tree_sitter.Node) -> tree_sitter.Node | None:
    """The identifier a function definition binds, or None.

    Most grammars expose a `name` field directly; C and C++ bury the
    name inside a chain of `declarator` fields (pointer declarators,
    function declarators) ending at an identifier.
    """
    name = node.child_by_field_name("name")
    if name is not None:
        return name
    declarator = node.child_by_field_name("declarator")
    while declarator is not None:
        if declarator.child_count == 0 and "identifier" in declarator.type:
            return declarator
        declarator = declarator.child_by_field_name("declarator")
    return None


def _callee_name(node: tree_sitter.Node) -> tree_sitter.Node | None:
    """The plain identifier a call names, or None.

    Calls with a receiver (`obj.f(x)`) are excluded: inlining them
    needs the receiver expression, which this transformation doesn't
    model. Grammars put the callee under `function` (most languages),
    `name` (Java) or `method` (Ruby); anything else call-shaped (e.g.
    Rust macro invocations) has no callee to match.
    """
    if (
        node.child_by_field_name("object") is not None
        or node.child_by_field_name("receiver") is not None
    ):
        return None
    for field in ("function", "name", "method"):
        callee = node.child_by_field_name(field)
        if callee is not None:
            if callee.child_count == 0 and "identifier" in callee.type:
                return callee
            return None
    return None


def inline_call_targets(
    tree: tree_sitter.Tree, source: bytes
) -> list[TransformTarget]:
    """Calls to functions defined in the file, paired with instructions
    to inline them.

    Only unambiguous targets are produced: the callee must be a plain
    identifier naming exactly one definition (overloaded names are
    skipped), the call must not be inside that definition (recursion),
    and both must be small enough to prompt with.
    """
    definitions: dict[bytes, tree_sitter.Node | None] = {}
    for node in iter_nodes(tree):
        if not node.is_named or not _matches(node.type, _DEFINITION_WORDS):
            continue
        # Bodiless function-like nodes (declarations, interface
        # members, C declarators) define nothing to inline.
        if node.child_by_field_name("body") is None:
            continue
        name_node = _definition_name(node)
        if name_node is None:
            continue
        name = source[name_node.start_byte : name_node.end_byte]
        definitions[name] = None if name in definitions else node

    inlinable: dict[bytes, tuple[tuple[int, int], str, str]] = {}
    for name, definition in definitions.items():
        if definition is None:
            continue
        if definition.end_byte - definition.start_byte > MAX_SNIPPET_BYTES:
            continue
        try:
            context = source[definition.start_byte : definition.end_byte].decode(
                "utf-8"
            )
            name_text = name.decode("utf-8")
        except UnicodeDecodeError:
            continue
        span = (definition.start_byte, definition.end_byte)
        inlinable[name] = (span, name_text, context)

    targets: list[TransformTarget] = []
    for node in iter_nodes(tree):
        if not node.is_named or not _matches(node.type, _CALL_WORDS):
            continue
        callee = _callee_name(node)
        if callee is None:
            continue
        entry = inlinable.get(source[callee.start_byte : callee.end_byte])
        if entry is None:
            continue
        definition_span, name_text, context = entry
        if (
            definition_span[0] <= node.start_byte
            and node.end_byte <= definition_span[1]
        ):
            continue
        if node.end_byte - node.start_byte > MAX_SNIPPET_BYTES:
            continue
        try:
            text = source[node.start_byte : node.end_byte].decode("utf-8")
        except UnicodeDecodeError:
            continue
        targets.append(
            TransformTarget(
                span=(node.start_byte, node.end_byte),
                text=text,
                instruction=(
                    f"Inline this call to the function `{name_text}`: "
                    "substitute the argument expressions for its "
                    "parameters and produce an equivalent inline "
                    "replacement for the call, parenthesised if needed. "
                    "The replacement must preserve the code's behaviour."
                ),
                context=context,
            )
        )
    return targets


def transform_prompt(target: TransformTarget) -> str:
    """The prompt asking the model to rewrite one target."""
    return (
        "You are helping to minimize a test case by applying a small "
        "mechanical refactoring to one piece of it.\n\n"
        f"{target.instruction}\n\n"
        "For reference:\n\n"
        f"```\n{target.context}\n```\n\n"
        "The code to rewrite is:\n\n"
        f"```\n{target.text}\n```\n\n"
        "Output only the replacement code, in a single fenced code "
        "block. Output nothing else: no explanations, no commentary."
    )


def splice(source: bytes, span: tuple[int, int], replacement: bytes) -> bytes:
    """source with the bytes in span replaced by replacement."""
    return source[: span[0]] + replacement + source[span[1] :]


def llm_transform_pump(
    client: LLMClient,
    config: LLMConfig,
    language: str,
    name: str,
    find_targets: TargetFinder,
    *,
    max_adoptions: int = MAX_ADOPTIONS,
    max_completions: int = MAX_COMPLETIONS,
    max_prompt_attempts: int = MAX_PROMPT_ATTEMPTS,
) -> ReductionPump[bytes]:
    """A pump applying one grammar-guided transformation with the model.

    Each round parses the current result, finds the transformation's
    targets, and asks the model to rewrite one span at a time; whenever
    a spliced candidate is interesting it is adopted and the targets
    are rederived (adoption shifts every later span). Responses are
    remembered per prompt: rederived targets replay earlier answers
    before asking again, and a prompt whose answers all failed is
    retried with a fresh seed up to max_prompt_attempts times, since a
    single bad sample is common even when the model usually gets the
    rewrite right.
    """

    async def pump(problem: ReductionProblem[bytes]) -> bytes:
        current = problem.current_test_case
        seen = {current}
        responses: dict[str, list[str]] = {}
        completions = 0
        adoptions = 0

        async def try_response(response: str, span: tuple[int, int]) -> bool:
            nonlocal current, adoptions
            for replacement in extract_candidates(response):
                candidate = splice(current, span, replacement.strip())
                if candidate in seen:
                    continue
                seen.add(candidate)
                if await problem.is_interesting(candidate):
                    current = candidate
                    adoptions += 1
                    return True
            return False

        while adoptions < max_adoptions:
            improved = False
            asked = False
            for target in find_targets(parse_tree(language, current), current):
                prompt = transform_prompt(target)
                cached = responses.setdefault(prompt, [])
                # Adoption shifts spans, so a previously useless answer
                # can produce a fresh candidate on the new state.
                for response in cached:
                    if await try_response(response, target.span):
                        improved = True
                        break
                if improved:
                    break
                if len(cached) >= max_prompt_attempts:
                    continue
                if completions >= max_completions:
                    return current
                # The model may still be downloading or loading in the
                # background; wait only now that there is a target that
                # needs it.
                await client.wait_until_ready()
                if client.is_disabled():
                    return current
                completions += 1
                asked = True
                response = await client.complete(
                    prompt,
                    max_tokens=completion_max_tokens(
                        len(target.context) + len(target.text)
                    ),
                    seed=problem.work.random.getrandbits(32),
                    temperature=config.temperature,
                )
                cached.append(response)
                if await try_response(response, target.span):
                    improved = True
                    break
            if not improved and not asked:
                break
        return current

    pump.__name__ = name
    return pump


def llm_transform_pumps(
    client: LLMClient, config: LLMConfig, language: str
) -> list[ReductionPump[bytes]]:
    """The grammar-guided LLM pumps for a language."""
    return [
        llm_transform_pump(
            client,
            config,
            language,
            f"llm_inline_calls({language})",
            inline_call_targets,
        ),
    ]
