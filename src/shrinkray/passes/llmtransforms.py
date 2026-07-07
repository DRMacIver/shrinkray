"""Grammar-guided LLM transformation pumps.

Some transformations help reduction in nearly every language. Inlining
a function call is the canonical example: it usually makes the file
bigger, but afterwards the function definition may be deletable, for a
large net win. Shrink Ray implements this by hand for C and C++
(:mod:`shrinkray.passes.cpp`), but a hand-written inliner needs
per-language knowledge of syntax, so that approach doesn't scale across
languages.

The transformations (see TRANSFORMATIONS):

- inline_calls: replace a call with the called function's body,
  arguments substituted (generalises the C/C++ simple inliner).
- inline_definitions: replace a use of a once-bound name (variable,
  constant, type alias, C macro) with its definition (generalises
  typedef inlining).
- stub_bodies: replace a function body with a minimal valid stub,
  returning a trivial constant where one is required (generalises
  function-def-to-decl and replace-body-with-ellipsis).
- unroll_loops: replace a loop with its first iteration as
  straight-line code.
- evaluate_constants: fold a constant expression to a literal.

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

from collections.abc import Callable, Iterable

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

# Node types that bind a name: assignment (Python, Ruby),
# init_declarator / variable_declarator (C, JavaScript, Java),
# let_declaration (Rust), var_spec / const_spec (Go), type_definition /
# alias_declaration / type_item (typedefs and type aliases), preproc_def
# (C macros). Deliberately absent: "parameter" (default values are not
# bindings of the call sites' values), "binary"/"unary" (expressions
# have left/right fields but bind nothing).
_BINDING_WORDS = frozenset(
    {
        "assignment",
        "declarator",
        "declaration",
        "spec",
        "item",
        "let",
        "definition",
        "typedef",
        "using",
        "alias",
        "def",
    }
)

# Binding nodes whose type words intersect these carry their definition
# in a `type` field rather than `value`/`right` (typedefs and aliases).
_TYPE_VALUE_WORDS = frozenset({"type", "alias", "typedef"})

_BINDING_NAME_FIELDS = ("name", "left", "declarator", "pattern")

# Loop node types: for_statement, while_statement, for_expression,
# loop_expression, repeat_statement, while, for, until (Ruby)...
# Requiring a `body` field excludes non-loop matches like Python's
# comprehension for_in_clause and Go's for_clause.
_LOOP_WORDS = frozenset({"for", "while", "loop", "repeat", "foreach", "until"})

# Expression node types eligible for constant folding: binary_operator,
# binary_expression, unary_expression, ...
_CONSTANT_EXPRESSION_WORDS = frozenset({"binary", "unary"})

# Constant expressions shorter than this cannot usefully fold: the
# replacement literal would be no smaller.
MIN_CONSTANT_BYTES = 3

# Function bodies at most this size are not worth stubbing: the stub
# would be no smaller, and mechanical passes empty them anyway.
MIN_STUB_BODY_BYTES = 16


def _matches(node_type: str, words: frozenset[str]) -> bool:
    return not words.isdisjoint(node_type.split("_"))


def _contains(span: tuple[int, int], node: tree_sitter.Node) -> bool:
    return span[0] <= node.start_byte and node.end_byte <= span[1]


def _unique_definitions(
    source: bytes, named_nodes: Iterable[tuple[bytes, tree_sitter.Node]]
) -> dict[bytes, tuple[tuple[int, int], str, str]]:
    """Map each name defined by exactly one of the given nodes to that
    node's (span, decoded name, decoded text).

    This is the shared notion of "the definition a name can be replaced
    by" for both inlining transformations: ambiguous names (defined by
    more than one node), oversized nodes, and undecodable text are all
    dropped.
    """
    nodes: dict[bytes, tree_sitter.Node | None] = {}
    for name, node in named_nodes:
        nodes[name] = None if name in nodes else node

    result: dict[bytes, tuple[tuple[int, int], str, str]] = {}
    for name, node in nodes.items():
        if node is None:
            continue
        if node.end_byte - node.start_byte > MAX_SNIPPET_BYTES:
            continue
        try:
            context = source[node.start_byte : node.end_byte].decode("utf-8")
            name_text = name.decode("utf-8")
        except UnicodeDecodeError:
            continue
        result[name] = ((node.start_byte, node.end_byte), name_text, context.strip())
    return result


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
    def named_definitions() -> Iterable[tuple[bytes, tree_sitter.Node]]:
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
            yield source[name_node.start_byte : name_node.end_byte], node

    inlinable = _unique_definitions(source, named_definitions())
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
        if _contains(definition_span, node):
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


def _binding_name(
    node: tree_sitter.Node, source: bytes
) -> tree_sitter.Node | None:
    """The name child of a node binding one plain identifier to a value
    or type, or None."""
    if not _matches(node.type, _BINDING_WORDS):
        return None
    # An augmented assignment (`x += 1`) reads the name's prior value
    # rather than defining it. Every grammar exposes the compound
    # operator through an `operator` field; plain bindings either have
    # no such field or a bare `=`.
    operator = node.child_by_field_name("operator")
    if operator is not None and source[operator.start_byte : operator.end_byte] != b"=":
        return None
    value = node.child_by_field_name("value")
    if value is None:
        value = node.child_by_field_name("right")
    if value is None and _matches(node.type, _TYPE_VALUE_WORDS):
        value = node.child_by_field_name("type")
    if value is None:
        return None
    for field in _BINDING_NAME_FIELDS:
        name = node.child_by_field_name(field)
        if name is None:
            continue
        if name.child_count == 0 and "identifier" in name.type:
            return name
        # A structured binding (tuple pattern, expression list): not
        # one name bound to one definition.
        return None
    return None


def inline_definition_targets(
    tree: tree_sitter.Tree, source: bytes
) -> list[TransformTarget]:
    """Uses of names bound once to a value or type, paired with
    instructions to inline the definition into the use.

    This generalises typedef inlining: variables, constants, type
    aliases and C macros all bind a name whose uses can be replaced by
    the definition, after which the binding itself is deletable. Names
    bound more than once (reassignment) are skipped: their value at a
    given use is not the textual definition. Uses that textually precede
    the binding are still offered: inside a function body they can
    legitimately refer to a binding made later, and a use the rewrite
    gets wrong just fails the interestingness test.
    """

    def named_bindings() -> Iterable[tuple[bytes, tree_sitter.Node]]:
        for node in iter_nodes(tree):
            if not node.is_named:
                continue
            name_node = _binding_name(node, source)
            if name_node is None:
                continue
            yield source[name_node.start_byte : name_node.end_byte], node

    inlinable = _unique_definitions(source, named_bindings())
    targets: list[TransformTarget] = []
    for node in iter_nodes(tree):
        if node.child_count != 0 or "identifier" not in node.type:
            continue
        entry = inlinable.get(source[node.start_byte : node.end_byte])
        if entry is None:
            continue
        binding_span, name_text, context = entry
        if _contains(binding_span, node):
            continue
        targets.append(
            TransformTarget(
                span=(node.start_byte, node.end_byte),
                text=name_text,
                instruction=(
                    f"Replace this use of `{name_text}` with the value or "
                    "type it is defined as, substituted in place. Add "
                    "parentheses only if they are needed to keep the "
                    "meaning unchanged."
                ),
                context=context,
            )
        )
    return targets


def _is_identifier_free(node: tree_sitter.Node) -> bool:
    """Whether an expression mentions no names and makes no calls, so
    its value is determined by the expression text alone."""
    stack = [node]
    while stack:
        n = stack.pop()
        if "identifier" in n.type or _matches(n.type, _CALL_WORDS):
            return False
        stack.extend(n.children)
    return True


def constant_expression_targets(
    tree: tree_sitter.Tree, source: bytes
) -> list[TransformTarget]:
    """Maximal constant expressions, paired with instructions to fold
    them to a literal.

    Mechanical folding (combine_expressions) only handles simple
    integer arithmetic; the model folds strings, floats, bit
    operations, and every language's own literal syntax.
    """
    targets: list[TransformTarget] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if (
            node.is_named
            and _matches(node.type, _CONSTANT_EXPRESSION_WORDS)
            and MIN_CONSTANT_BYTES
            <= node.end_byte - node.start_byte
            <= MAX_SNIPPET_BYTES
            and _is_identifier_free(node)
        ):
            try:
                text = source[node.start_byte : node.end_byte].decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                targets.append(
                    TransformTarget(
                        span=(node.start_byte, node.end_byte),
                        text=text,
                        instruction=(
                            "Evaluate this constant expression and replace "
                            "it with its simplest literal value."
                        ),
                        context="",
                    )
                )
                # Maximal expressions only: folding a subexpression of a
                # foldable expression is strictly worse.
                continue
        stack.extend(node.children)
    return targets


def loop_targets(tree: tree_sitter.Tree, source: bytes) -> list[TransformTarget]:
    """Loops, paired with instructions to unroll a single iteration.

    Replacing a loop with its first iteration's straight-line code
    deletes the loop scaffolding and often unblocks deleting whatever
    only the loop used. No mechanical pass can do the variable
    substitution this needs.
    """
    targets: list[TransformTarget] = []
    for node in iter_nodes(tree):
        if not node.is_named or not _matches(node.type, _LOOP_WORDS):
            continue
        if node.child_by_field_name("body") is None:
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
                    "Rewrite this loop as straight-line code that executes "
                    "the loop body exactly once, with the loop variable "
                    "(if any) bound to the first value it would take."
                ),
                context="",
            )
        )
    return targets


def stub_body_targets(tree: tree_sitter.Tree, source: bytes) -> list[TransformTarget]:
    """Function bodies, paired with instructions to replace them with a
    minimal stub.

    Mechanical hollowing empties brackets, but an empty body is invalid
    where a return value is required (Go, Rust, non-void C++ under
    -Werror); the model can synthesize the smallest body that still
    returns something of the right type.
    """
    targets: list[TransformTarget] = []
    for node in iter_nodes(tree):
        if not node.is_named or not _matches(node.type, _DEFINITION_WORDS):
            continue
        body = node.child_by_field_name("body")
        if body is None:
            continue
        if not (
            MIN_STUB_BODY_BYTES < body.end_byte - body.start_byte <= MAX_SNIPPET_BYTES
        ):
            continue
        try:
            signature = source[node.start_byte : body.start_byte].decode("utf-8")
            text = source[body.start_byte : body.end_byte].decode("utf-8")
        except UnicodeDecodeError:
            continue
        targets.append(
            TransformTarget(
                span=(body.start_byte, body.end_byte),
                text=text,
                instruction=(
                    "Replace this function body with the smallest body that "
                    "still parses and type-checks: return a trivial constant "
                    "if the function must return a value, otherwise do as "
                    "little as possible. Keep the original body's delimiters "
                    "and indentation style."
                ),
                context=signature.strip(),
            )
        )
    return targets


def transform_prompt(target: TransformTarget) -> str:
    """The prompt asking the model to rewrite one target."""
    reference = (
        f"For reference:\n\n```\n{target.context}\n```\n\n" if target.context else ""
    )
    return (
        "You are helping to minimize a test case by applying a small "
        "mechanical refactoring to one piece of it.\n\n"
        f"{target.instruction}\n\n"
        f"{reference}"
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

    Each round asks the model to rewrite one target span at a time;
    whenever a spliced candidate is interesting it is adopted and the
    targets are rederived by reparsing (adoption shifts every later
    span; fruitless rounds reuse the same targets). Responses are
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

        async def try_response(response: str, target: TransformTarget) -> bool:
            nonlocal current, adoptions
            context = target.context.encode()
            for block in extract_candidates(response):
                replacements = [block.strip()]
                # Small models often echo the reference context (e.g. a
                # function signature) ahead of the actual replacement;
                # salvage those answers by also trying the suffix, and
                # try it first since it is strictly smaller.
                if context and replacements[0].startswith(context):
                    replacements.insert(0, replacements[0][len(context) :].strip())
                for replacement in replacements:
                    candidate = splice(current, target.span, replacement)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    if await problem.is_interesting(candidate):
                        current = candidate
                        adoptions += 1
                        return True
            return False

        targets: list[TransformTarget] | None = None
        while adoptions < max_adoptions:
            if targets is None:
                targets = find_targets(parse_tree(language, current), current)
            improved = False
            asked = False
            for target in targets:
                prompt = transform_prompt(target)
                cached = responses.setdefault(prompt, [])
                # Adoption shifts spans, so a previously useless answer
                # can produce a fresh candidate on the new state.
                for response in cached:
                    if await try_response(response, target):
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
                if await try_response(response, target):
                    improved = True
                    break
            if improved:
                # Adoption changed the test case, shifting every later
                # span: the targets must be rederived.
                targets = None
            elif not asked:
                break
        return current

    pump.__name__ = name
    return pump


# The transformations, in rough order of expected value: the inlining
# transformations unlock deleting whole definitions, stubbing and
# unrolling unlock deleting what a body or loop used, and constant
# folding mostly polishes.
TRANSFORMATIONS: list[tuple[str, TargetFinder]] = [
    ("llm_inline_calls", inline_call_targets),
    ("llm_inline_definitions", inline_definition_targets),
    ("llm_stub_bodies", stub_body_targets),
    ("llm_unroll_loops", loop_targets),
    ("llm_evaluate_constants", constant_expression_targets),
]


def llm_transform_pumps(
    client: LLMClient, config: LLMConfig, language: str
) -> list[ReductionPump[bytes]]:
    """The grammar-guided LLM pumps for a language."""
    return [
        llm_transform_pump(client, config, language, f"{name}({language})", finder)
        for name, finder in TRANSFORMATIONS
    ]
