"""Reduction passes for C and C++.

These are pure Python replacements for the clang_delta transformations
that shrink ray previously shelled out to. They deliberately do not
attempt to parse C or C++ properly. Instead they use a sloppy lexer
plus bracket matching to find things that look like interesting
structures (function definitions, namespaces, class heads, template
parameter lists, call expressions, typedefs), and then generate lots of
candidate edits. Candidates that break the test case are simply
rejected by the interestingness test, so it's fine for the "parser" to
be wrong about what it's looking at as long as it's right often enough
to make progress.
"""

import re
from collections.abc import Callable

from attrs import define

from shrinkray.passes.definitions import ReductionPass, ReductionPump
from shrinkray.passes.patching import (
    CutPatch,
    Cuts,
    ReplacementPatch,
    Replacements,
    apply_patches,
)
from shrinkray.problem import ReductionProblem


NAME = "name"
NUMBER = "number"
STRING = "string"
CHARACTER = "char"
COMMENT = "comment"
PREPROC = "preproc"
PUNCT = "punct"


@define(frozen=True)
class Token:
    kind: str
    start: int
    end: int
    text: bytes


RAW_STRING_START = re.compile(rb'(?:u8|u|U|L)?R"([^()\\ \t\r\n]*)\(')

IDENTIFIER_START = re.compile(rb"[A-Za-z_$]")
IDENTIFIER_CONT = re.compile(rb"[A-Za-z0-9_$]")

# Multi-byte punctuation, longest first so maximal munch works by
# trying them in order.
MULTI_PUNCT = [
    b"<<=",
    b">>=",
    b"...",
    b"->*",
    b"<=>",
    b"::",
    b"->",
    b"<<",
    b">>",
    b"<=",
    b">=",
    b"==",
    b"!=",
    b"&&",
    b"||",
    b"+=",
    b"-=",
    b"*=",
    b"/=",
    b"%=",
    b"&=",
    b"|=",
    b"^=",
    b"++",
    b"--",
    b"##",
    b".*",
]

WHITESPACE = b" \t\r\n\f\v"


def _scan_string(source: bytes, i: int, quote: int) -> int:
    """Scan a string or character literal starting at the quote at
    position i, returning the position just after it ends.

    Backslash escapes are honoured. As a concession to malformed input
    (which mid-reduction test cases very often are), an unescaped
    newline also terminates the literal rather than letting it swallow
    the rest of the file."""
    n = len(source)
    j = i + 1
    while j < n:
        c = source[j]
        if c == ord("\\") and j + 1 < n:
            j += 2
            continue
        if c == quote:
            return j + 1
        if c == ord("\n"):
            return j
        j += 1
    return n


def _scan_preproc(source: bytes, i: int) -> int:
    """Scan a preprocessor directive starting at the # at position i,
    returning the position just after it ends. Backslash-continued
    lines are included."""
    n = len(source)
    j = i
    while j < n:
        if source[j] == ord("\n"):
            k = j - 1
            if k >= 0 and source[k] == ord("\r"):
                k -= 1
            if k >= 0 and source[k] == ord("\\"):
                j += 1
                continue
            return k + 1
        j += 1
    return n


def _scan_number(source: bytes, i: int) -> int:
    """Scan a pp-number starting at position i, returning the position
    just after it ends. This includes hex, exponents, suffixes, and
    C++14 digit separators."""
    n = len(source)
    j = i + 1
    while j < n:
        c = source[j]
        if (
            bytes([c])
            in b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_."
        ):
            j += 1
        elif c in b"+-" and source[j - 1] in b"eEpP":
            j += 1
        elif (
            c == ord("'") and j + 1 < n and IDENTIFIER_CONT.match(source, j + 1, j + 2)
        ):
            j += 2
        else:
            break
    return j


def lex(source: bytes) -> list[Token]:
    """Tokenize C/C++ source, sloppily.

    Whitespace is skipped (token positions preserve it in the source).
    Comments and preprocessor directives are emitted as single tokens.
    The lexer never fails: any byte it doesn't understand becomes a
    single-byte punctuation token.
    """
    tokens: list[Token] = []
    i = 0
    n = len(source)
    line_start = True

    def emit(kind: str, end: int) -> None:
        nonlocal i, line_start
        tokens.append(Token(kind=kind, start=i, end=end, text=source[i:end]))
        i = end
        line_start = False

    while i < n:
        c = source[i]
        if c in WHITESPACE:
            if c == ord("\n"):
                line_start = True
            i += 1
            continue
        if c == ord("#") and line_start:
            emit(PREPROC, _scan_preproc(source, i))
            continue
        if c == ord("/") and i + 1 < n:
            nxt = source[i + 1]
            if nxt == ord("/"):
                j = source.find(b"\n", i)
                emit(COMMENT, n if j < 0 else j)
                continue
            if nxt == ord("*"):
                j = source.find(b"*/", i + 2)
                emit(COMMENT, n if j < 0 else j + 2)
                continue
        if c in b"uULR":
            match = RAW_STRING_START.match(source, i)
            if match is not None:
                terminator = b")" + match.group(1) + b'"'
                j = source.find(terminator, match.end())
                emit(STRING, n if j < 0 else j + len(terminator))
                continue
        if c == ord('"'):
            emit(STRING, _scan_string(source, i, c))
            continue
        if c == ord("'"):
            emit(CHARACTER, _scan_string(source, i, c))
            continue
        if IDENTIFIER_START.match(source, i, i + 1):
            j = i + 1
            while j < n and IDENTIFIER_CONT.match(source, j, j + 1):
                j += 1
            emit(NAME, j)
            continue
        if bytes([c]) in b"0123456789" or (
            c == ord(".") and i + 1 < n and bytes([source[i + 1]]) in b"0123456789"
        ):
            emit(NUMBER, _scan_number(source, i))
            continue
        for punct in MULTI_PUNCT:
            if source.startswith(punct, i):
                emit(PUNCT, i + len(punct))
                break
        else:
            emit(PUNCT, i + 1)
    return tokens


OPEN_TO_CLOSE = {b"{": b"}", b"(": b")", b"[": b"]"}
CLOSE_TO_OPEN = {v: k for k, v in OPEN_TO_CLOSE.items()}


def match_brackets(tokens: list[Token]) -> dict[int, int]:
    """Match bracket tokens, returning a mapping between token indices
    that goes in both directions: each open bracket's index maps to its
    matching close bracket's index and vice versa.

    Unbalanced brackets are tolerated: unmatched brackets just don't
    appear in the result. Comments never take part in matching."""
    result: dict[int, int] = {}
    stacks: dict[bytes, list[int]] = {b"{": [], b"(": [], b"[": []}
    for i, token in enumerate(tokens):
        if token.kind != PUNCT:
            continue
        if token.text in OPEN_TO_CLOSE:
            stacks[token.text].append(i)
        elif token.text in CLOSE_TO_OPEN:
            stack = stacks[CLOSE_TO_OPEN[token.text]]
            if stack:
                j = stack.pop()
                result[i] = j
                result[j] = i
    return result


@define(frozen=True)
class TokenView:
    """A tokenized view of C/C++ source: the significant (non-comment)
    tokens plus bracket matching over them."""

    source: bytes
    tokens: list[Token]
    brackets: dict[int, int]


def token_view(source: bytes) -> TokenView:
    tokens = [t for t in lex(source) if t.kind != COMMENT]
    return TokenView(source=source, tokens=tokens, brackets=match_brackets(tokens))


# Names that can be followed by a parenthesised group without being a
# function name or callable: control flow, operators that take
# parenthesised operands, and attribute-ish decorations.
NOT_A_FUNCTION_NAME = frozenset(
    [
        b"if",
        b"while",
        b"for",
        b"switch",
        b"catch",
        b"return",
        b"sizeof",
        b"alignof",
        b"_Alignof",
        b"noexcept",
        b"decltype",
        b"throw",
        b"alignas",
        b"_Alignas",
        b"__attribute__",
        b"__declspec",
        b"defined",
        b"asm",
        b"__asm",
        b"__asm__",
        b"typeid",
        b"static_assert",
        b"_Static_assert",
        b"case",
        b"new",
        b"delete",
        b"co_await",
        b"co_return",
        b"co_yield",
    ]
)

ACCESS_SPECIFIERS = frozenset([b"public", b"private", b"protected"])

# How many candidate declaration starts to record per function. Each
# one produces a candidate cut, so this is purely a bound on wasted
# work when the statement boundary is ambiguous.
MAX_DECL_START_CANDIDATES = 3


@define(frozen=True)
class FunctionInfo:
    """A thing in the source that looks like a function definition.

    All fields other than name are token indices. decl_starts holds
    candidate indices for the first token of the declaration, best
    guess first: sloppy parsing can't always tell where a declaration
    starts, so we record several plausible options and let the
    interestingness test arbitrate."""

    decl_starts: tuple[int, ...]
    name: bytes
    params_open: int
    params_close: int
    init_colon: int | None
    body_open: int
    body_close: int


def _find_angle_close(view: TokenView, i: int) -> int | None:
    """Given the index of a '<' token, find the index of the '>' (or
    '>>') token that plausibly closes it as a template argument list.
    Returns None if this doesn't look like a template argument list."""
    tokens = view.tokens
    depth = 1
    j = i + 1
    while j < len(tokens):
        t = tokens[j]
        if t.kind in (STRING, PREPROC):
            return None
        if t.kind == PUNCT:
            if t.text == b"<":
                depth += 1
            elif t.text == b">":
                depth -= 1
                if depth == 0:
                    return j
            elif t.text == b">>":
                depth -= 2
                if depth <= 0:
                    return j
            elif t.text in (b"(", b"["):
                m = view.brackets.get(j)
                if m is None:
                    return None
                j = m
            elif t.text in (b";", b"{", b"}", b")", b"]", b"&&", b"||", b"?"):
                return None
        j += 1
    return None


def _split_on_top_level_commas(
    view: TokenView, lo: int, hi: int
) -> list[tuple[int, int]]:
    """Split the token range [lo, hi) into runs separated by top level
    commas, jumping over bracketed groups and template argument lists.
    Returns a list of (start, end) token index ranges."""
    tokens = view.tokens
    result: list[tuple[int, int]] = []
    start = lo
    j = lo
    while j < hi:
        t = tokens[j]
        if t.kind == PUNCT:
            if t.text in OPEN_TO_CLOSE:
                m = view.brackets.get(j)
                if m is not None and m < hi:
                    j = m
            elif t.text == b"<":
                m = _find_angle_close(view, j)
                if m is not None and m < hi:
                    j = m
            elif t.text == b",":
                result.append((start, j))
                start = j + 1
        j += 1
    result.append((start, hi))
    return result


def _statement_start_candidates(view: TokenView, body: int) -> list[int]:
    """Walk backwards from the '{' at token index body to find
    candidate token indices for the start of the statement containing
    it. The best guess comes first."""
    tokens = view.tokens
    candidates: list[int] = []
    k = body - 1
    while k >= 0 and len(candidates) < MAX_DECL_START_CANDIDATES:
        t = tokens[k]
        if t.kind == PREPROC or t.text in (b";", b"{"):
            candidates.append(k + 1)
            break
        if t.text == b"}":
            m = view.brackets.get(k)
            if m is None:
                candidates.append(k + 1)
                break
            # This could be the end of a previous definition, or part
            # of this declaration (e.g. a braced initializer in a
            # constructor's init list), so record a candidate here and
            # keep walking from before the group.
            candidates.append(k + 1)
            k = m - 1
            continue
        if t.text in (b")", b"]"):
            m = view.brackets.get(k)
            if m is None:
                candidates.append(k + 1)
                break
            k = m - 1
            continue
        if t.text == b":" and k >= 1 and tokens[k - 1].text in ACCESS_SPECIFIERS:
            candidates.append(k + 1)
            break
        k -= 1
    if k < 0:
        candidates.append(0)
    return candidates


def _analyze_function(view: TokenView, start: int, body: int) -> FunctionInfo | None:
    """Determine whether the token range [start, body) followed by the
    '{' at index body looks like a function definition, and if so
    describe it."""
    tokens = view.tokens
    paren_groups: list[tuple[int, int]] = []
    colon: int | None = None
    j = start
    while j < body:
        t = tokens[j]
        if t.kind == PUNCT:
            if t.text in OPEN_TO_CLOSE:
                m = view.brackets.get(j)
                if m is not None and m < body:
                    if t.text == b"(":
                        paren_groups.append((j, m))
                    j = m
            elif t.text == b":" and colon is None and paren_groups:
                colon = j
        j += 1

    if colon is not None:
        eligible = [g for g in paren_groups if g[1] < colon]
    else:
        eligible = paren_groups

    for open_idx, close_idx in reversed(eligible):
        prev = tokens[open_idx - 1] if open_idx > start else None
        if prev is not None and prev.kind == NAME:
            if prev.text in NOT_A_FUNCTION_NAME:
                continue
            name = prev.text
        elif any(
            tokens[q].kind == NAME and tokens[q].text == b"operator"
            for q in range(max(start, open_idx - 3), open_idx)
        ):
            name = b"operator"
        else:
            continue

        init_colon = None
        if colon is not None and colon > close_idx:
            init_colon = colon
        # This is never empty: `start` itself came from the same
        # candidate list, and a function match always has its parameter
        # list strictly after the region start.
        decl_starts = tuple(
            idx for idx in _statement_start_candidates(view, body) if idx < open_idx
        )
        return FunctionInfo(
            decl_starts=decl_starts,
            name=name,
            params_open=open_idx,
            params_close=close_idx,
            init_colon=init_colon,
            body_open=body,
            body_close=view.brackets[body],
        )
    return None


def find_function_definitions(view: TokenView) -> list[FunctionInfo]:
    """Find everything in the source that looks like a function
    definition."""
    tokens = view.tokens
    results: list[FunctionInfo] = []
    for body, token in enumerate(tokens):
        if token.kind != PUNCT or token.text != b"{" or body not in view.brackets:
            continue
        # The nearest statement boundary isn't always the right one
        # (e.g. a braced initializer in a constructor's init list looks
        # like one), so try successively wider regions until one looks
        # like a function.
        for start in _statement_start_candidates(view, body):
            info = _analyze_function(view, start, body)
            if info is not None:
                results.append(info)
                break
    return results


def _comma_element_cuts(view: TokenView, lo: int, hi: int) -> list[CutPatch]:
    """Cuts that delete individual elements of a comma-separated list
    spanning the token range [lo, hi), each together with an adjacent
    comma. Returns nothing for lists of fewer than two elements: there
    the whole-list cut is the only sensible candidate."""
    pieces = _split_on_top_level_commas(view, lo, hi)
    if len(pieces) <= 1:
        return []
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for i, (ts, te) in enumerate(pieces):
        if ts >= te:
            continue
        if i < len(pieces) - 1:
            # tokens[te] is the separating comma; delete through it.
            cuts.append([(tokens[ts].start, tokens[te].end)])
        else:
            # Last element: delete the preceding comma through the end.
            cuts.append([(tokens[ts - 1].start, tokens[te - 1].end)])
    return cuts


async def replace_function_bodies(problem: ReductionProblem[bytes]) -> None:
    """Replace function definitions with declarations, in the style of
    clang_delta's replace-function-def-with-decl.

    For each function we try both replacing just the braced body with a
    semicolon (which preserves trailing specifiers like const) and
    replacing everything after the parameter list (which is what's
    needed to kill constructor initializer lists)."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    patches: list[ReplacementPatch] = []
    for f in find_function_definitions(view):
        body_start = tokens[f.body_open].start
        body_end = tokens[f.body_close].end
        params_end = tokens[f.params_close].end
        patches.append(((params_end, body_end, b";"),))
        if problem.current_test_case[params_end:body_start].strip():
            patches.append(((body_start, body_end, b";"),))
    await apply_patches(problem, Replacements(), patches)


async def delete_function_definitions(problem: ReductionProblem[bytes]) -> None:
    """Delete entire function definitions, declaration and all.

    clang_delta's remove-unused-function needs to prove a function
    unused; we just try deleting every function and let the
    interestingness test tell us which ones matter."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for f in find_function_definitions(view):
        for start in f.decl_starts:
            cuts.append([(tokens[start].start, tokens[f.body_close].end)])
    await apply_patches(problem, Cuts(), cuts)


async def remove_constructor_initializers(problem: ReductionProblem[bytes]) -> None:
    """Remove constructor initializer lists (or individual elements of
    them), in the style of clang_delta's remove-ctor-initializer."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for f in find_function_definitions(view):
        if f.init_colon is None:
            continue
        cuts.append([(tokens[f.init_colon].start, tokens[f.body_open].start)])
        cuts.extend(_comma_element_cuts(view, f.init_colon + 1, f.body_open))
    await apply_patches(problem, Cuts(), cuts)


CLASS_KEYS = frozenset([b"class", b"struct", b"union", b"enum"])


def _find_class_base_lists(view: TokenView) -> list[tuple[int, int]]:
    """Find base class lists (and enum underlying types), returned as
    (colon, body_open) token index pairs."""
    tokens = view.tokens
    results: set[tuple[int, int]] = set()
    for i, t in enumerate(tokens):
        if t.kind != NAME or t.text not in CLASS_KEYS:
            continue
        colon: int | None = None
        j = i + 1
        while j < len(tokens):
            t2 = tokens[j]
            if t2.kind == NAME or (
                t2.kind == PUNCT and t2.text in (b"::", b",", b"...")
            ):
                if t2.text == b"," and colon is None:
                    break
                j += 1
                continue
            if t2.kind == PUNCT and t2.text in (b"(", b"["):
                m = view.brackets.get(j)
                if m is None:
                    break
                j = m + 1
                continue
            if t2.kind == PUNCT and t2.text == b"<":
                m = _find_angle_close(view, j)
                if m is None:
                    break
                j = m + 1
                continue
            if t2.kind == PUNCT and t2.text == b":" and colon is None:
                colon = j
                j += 1
                continue
            if t2.kind == PUNCT and t2.text == b"{":
                if colon is not None:
                    results.add((colon, j))
                break
            break
    return sorted(results)


async def remove_base_classes(problem: ReductionProblem[bytes]) -> None:
    """Remove base class lists (or individual bases), in the style of
    clang_delta's remove-base-class. Also removes enum underlying
    types, which look identical to a sloppy parser."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for colon, body in _find_class_base_lists(view):
        cuts.append([(tokens[colon].start, tokens[body].start)])
        cuts.extend(_comma_element_cuts(view, colon + 1, body))
    await apply_patches(problem, Cuts(), cuts)


def _template_prefixes(view: TokenView) -> dict[int, int]:
    """Map the token index that ends a `template < ... >` prefix (its
    closing `>`) to the index of the `template` keyword that starts it.
    Used to include a class template's `template<...>` prefix when
    deleting the class."""
    tokens = view.tokens
    result: dict[int, int] = {}
    for idx, tok in enumerate(tokens):
        if (
            tok.kind == NAME
            and tok.text == b"template"
            and idx + 1 < len(tokens)
            and tokens[idx + 1].text == b"<"
        ):
            close = _find_angle_close(view, idx + 1)
            if close is not None:
                result[close] = idx
    return result


def _declaration_end(view: TokenView, start: int) -> int | None:
    """From token index start, walk forward over any bracketed groups
    and template argument lists to the terminating top-level `;`,
    returning its index (or None if there isn't one)."""
    tokens = view.tokens
    j = start
    while j < len(tokens):
        t = tokens[j]
        if t.kind == PUNCT:
            if t.text in OPEN_TO_CLOSE:
                m = view.brackets.get(j)
                if m is None:
                    return None
                j = m
            elif t.text == b"<":
                m = _find_angle_close(view, j)
                if m is not None:
                    j = m
            elif t.text == b";":
                return j
        j += 1
    return None


async def replace_type_with_int(problem: ReductionProblem[bytes]) -> None:
    """Replace a class/struct/union type with the builtin `int`, in the
    style of clang_delta's empty-struct-to-int.

    For each class/struct/union definition or forward declaration
    (including template ones), delete it and rewrite every other use of
    its name — consuming any trailing `<...>` template arguments — to
    `int`. When the type only carried structure irrelevant to the bug
    this collapses it away entirely, which the deletion-only passes
    cannot do on their own."""
    source = problem.current_test_case
    view = token_view(source)
    tokens = view.tokens
    template_prefixes = _template_prefixes(view)
    patches: list[ReplacementPatch] = []
    for k, t in enumerate(tokens):
        if t.kind != NAME or t.text not in (b"struct", b"class", b"union"):
            continue
        if k + 1 >= len(tokens) or tokens[k + 1].kind != NAME:
            continue
        name = tokens[k + 1].text
        # Only a real definition/forward-declaration, not an elaborated
        # type used in a variable declaration (`struct S s;`): the token
        # after the name must open a body, a base list, or end the
        # declaration.
        after = tokens[k + 2] if k + 2 < len(tokens) else None
        if after is None or after.kind != PUNCT or after.text not in (b";", b"{", b":"):
            continue
        end = _declaration_end(view, k + 1)
        if end is None:
            continue
        decl_start = template_prefixes.get(k - 1, k)
        edits: list[tuple[int, int, bytes]] = [
            (tokens[decl_start].start, tokens[end].end, b"")
        ]
        p = 0
        while p < len(tokens):
            if decl_start <= p <= end:
                p += 1
                continue
            if tokens[p].kind == NAME and tokens[p].text == name:
                start_byte = tokens[p].start
                span_end = tokens[p].end
                if p + 1 < len(tokens) and tokens[p + 1].text == b"<":
                    m = _find_angle_close(view, p + 1)
                    if m is not None:
                        span_end = tokens[m].end
                        p = m
                edits.append((start_byte, span_end, b"int"))
            p += 1
        # By construction these edits never overlap: the definition span
        # is excluded from the use scan, and uses are distinct tokens.
        patches.append(tuple(sorted(edits)))
    await apply_patches(problem, Replacements(), patches)


async def remove_namespaces(problem: ReductionProblem[bytes]) -> None:
    """Remove namespaces, in the style of clang_delta's
    remove-namespace: either delete the whole namespace or splice its
    contents into the enclosing scope. extern "C" blocks are handled
    the same way."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for i, t in enumerate(tokens):
        if t.kind != NAME:
            continue
        name_path: tuple[int, int] | None = None
        if t.text == b"namespace":
            j = i + 1
            while j < len(tokens) and (
                tokens[j].kind == NAME
                or (tokens[j].kind == PUNCT and tokens[j].text == b"::")
            ):
                j += 1
            if j > i + 1:
                name_path = (i + 1, j)
        elif (
            t.text == b"extern" and i + 1 < len(tokens) and tokens[i + 1].kind == STRING
        ):
            j = i + 2
        else:
            continue
        if j >= len(tokens) or tokens[j].text != b"{" or j not in view.brackets:
            continue
        close = view.brackets[j]
        cuts.append([(t.start, tokens[close].end)])
        splice = [
            (t.start, tokens[j].end),
            (tokens[close].start, tokens[close].end),
        ]
        cuts.append(splice)
        # Splicing a named namespace leaves any `ns::name` reference
        # dangling, which fails to compile and gets the whole candidate
        # rejected. Offer a second splice that also strips the
        # namespace's qualifier from references, which is what actually
        # lets the namespace go.
        if name_path is not None:
            qualifier_cuts = _namespace_qualifier_cuts(view, name_path, i, close)
            if qualifier_cuts:
                cuts.append(splice + qualifier_cuts)
    await apply_patches(problem, Cuts(), cuts)


def _namespace_qualifier_cuts(
    view: TokenView, name_path: tuple[int, int], decl_start: int, decl_end: int
) -> list[tuple[int, int]]:
    """Find every `<path>::` qualifier that names the namespace declared
    by the tokens in [name_path[0], name_path[1]), outside the
    declaration itself, and return cuts that delete each one (the path
    tokens plus the trailing `::`). Deleting these turns `ns::name` into
    `name` so the namespace can be spliced away."""
    tokens = view.tokens
    lo, hi = name_path
    path_texts = [tokens[k].text for k in range(lo, hi)]
    n = len(path_texts)
    cuts: list[tuple[int, int]] = []
    p = 0
    limit = len(tokens) - n
    while p <= limit:
        if decl_start <= p <= decl_end:
            p += 1
            continue
        if (
            all(tokens[p + k].text == path_texts[k] for k in range(n))
            and p + n < len(tokens)
            and tokens[p + n].text == b"::"
        ):
            cuts.append((tokens[p].start, tokens[p + n].end))
            p += n + 1
        else:
            p += 1
    return cuts


async def remove_template_parts(problem: ReductionProblem[bytes]) -> None:
    """Remove template machinery: template<...> prefixes on
    declarations, template argument lists on uses, and individual
    template arguments or parameters. Together these cover clang_delta
    transformations like class-template-to-class and
    reduce-class-template-param."""
    view = token_view(problem.current_test_case)
    tokens = view.tokens
    cuts: list[CutPatch] = []
    for i, t in enumerate(tokens):
        if (
            t.kind != NAME
            or t.text in NOT_A_FUNCTION_NAME
            or t.text == b"operator"
            or i + 1 >= len(tokens)
            or tokens[i + 1].text != b"<"
        ):
            continue
        m = _find_angle_close(view, i + 1)
        if m is None:
            continue
        if t.text == b"template":
            cuts.append([(t.start, tokens[m].end)])
        else:
            cuts.append([(tokens[i + 1].start, tokens[m].end)])
        cuts.extend(_comma_element_cuts(view, i + 2, m))
    await apply_patches(problem, Cuts(), cuts)


async def simplify_call_expressions(problem: ReductionProblem[bytes]) -> None:
    """Simplify things that look like call expressions: replace the
    whole call with 0, with nothing, or with one of its arguments. This
    covers clang_delta's callexpr-to-value, replace-callexpr and
    simplify-callexpr."""
    source = problem.current_test_case
    view = token_view(source)
    tokens = view.tokens
    patches: list[ReplacementPatch] = []
    for i, t in enumerate(tokens):
        if (
            t.kind != NAME
            or t.text in NOT_A_FUNCTION_NAME
            or t.text in (b"operator", b"template", b"typename")
        ):
            continue
        open_idx = None
        if i + 1 < len(tokens) and tokens[i + 1].text == b"(":
            open_idx = i + 1
        elif i + 1 < len(tokens) and tokens[i + 1].text == b"<":
            m = _find_angle_close(view, i + 1)
            if m is not None and m + 1 < len(tokens) and tokens[m + 1].text == b"(":
                open_idx = m + 1
        if open_idx is None or open_idx not in view.brackets:
            continue
        close = view.brackets[open_idx]
        # Extend backwards over member access and scope resolution so
        # we replace the whole postfix chain: a.b->c(x), ns::f(x).
        k = i
        while (
            k >= 2
            and tokens[k - 1].text in (b"::", b".", b"->")
            and (tokens[k - 2].kind == NAME)
        ):
            k -= 2
        if k >= 1 and tokens[k - 1].text == b"~":
            k -= 1
        span_start = tokens[k].start
        span_end = tokens[close].end
        patches.append(((span_start, span_end, b"0"),))
        patches.append(((span_start, span_end, b""),))
        for ts, te in _split_on_top_level_commas(view, open_idx + 1, close):
            if ts < te:
                arg = source[tokens[ts].start : tokens[te - 1].end]
                patches.append(((span_start, span_end, arg),))
    await apply_patches(problem, Replacements(), patches)


@define(frozen=True)
class TypedefInfo:
    """A simple typedef (or using alias): one that defines a plain name
    for a type spelled entirely to its left. Typedefs where the name is
    embedded in the declarator (function pointers, arrays) are not
    reported since inlining them requires real understanding."""

    # Token indices of the whole declaration, inclusive of the ';'.
    decl_start: int
    decl_end: int
    name: bytes
    definition: bytes


def find_typedefs(view: TokenView) -> list[TypedefInfo]:
    tokens = view.tokens
    source = view.source
    results: list[TypedefInfo] = []
    for i, t in enumerate(tokens):
        if t.kind != NAME or t.text not in (b"typedef", b"using"):
            continue
        # Find the terminating semicolon at the top level.
        j = i + 1
        while j < len(tokens):
            t2 = tokens[j]
            if t2.kind == PREPROC:
                j = -1
                break
            if t2.kind == PUNCT and t2.text in OPEN_TO_CLOSE:
                m = view.brackets.get(j)
                if m is None:
                    j = -1
                    break
                j = m + 1
                continue
            if t2.kind == PUNCT and t2.text in (b";", b"}", b")"):
                break
            j += 1
        if j < 0 or j >= len(tokens) or tokens[j].text != b";":
            continue
        if t.text == b"using":
            # using name = definition;
            if i + 2 >= j or tokens[i + 1].kind != NAME or tokens[i + 2].text != b"=":
                continue
            name_idx = i + 1
            definition = source[tokens[i + 3].start : tokens[j - 1].end]
        else:
            # typedef definition name;
            name_idx = j - 1
            if name_idx <= i + 1 or tokens[name_idx].kind != NAME:
                continue
            definition = source[tokens[i + 1].start : tokens[name_idx - 1].end]
        if not definition:
            continue
        results.append(
            TypedefInfo(
                decl_start=i,
                decl_end=j,
                name=tokens[name_idx].text,
                definition=definition,
            )
        )
    return results


def typedef_inlining_candidates(source: bytes) -> list[bytes]:
    """For each simple typedef, produce a variant of the source with
    the typedef removed and every use of its name replaced by its
    definition."""
    view = token_view(source)
    tokens = view.tokens
    replacer = Replacements()
    results: list[bytes] = []
    for td in find_typedefs(view):
        edits: list[tuple[int, int, bytes]] = [
            (tokens[td.decl_start].start, tokens[td.decl_end].end, b"")
        ]
        for i, t in enumerate(tokens):
            if (
                t.kind == NAME
                and t.text == td.name
                and not td.decl_start <= i <= td.decl_end
            ):
                edits.append((t.start, t.end, td.definition))
        if len(edits) > 1:
            results.append(replacer.apply(tuple(sorted(edits)), source))
    return results


def _single_statement_body(view: TokenView, f: FunctionInfo) -> tuple[int, int] | None:
    """If the function's body consists of a single statement (with or
    without a leading return), return the token range of the expression
    in it."""
    tokens = view.tokens
    lo = f.body_open + 1
    hi = f.body_close
    if lo >= hi or tokens[hi - 1].text != b";":
        return None
    if tokens[lo].kind == NAME and tokens[lo].text == b"return":
        lo += 1
    j = lo
    while j < hi - 1:
        t = tokens[j]
        if t.kind == PREPROC:
            return None
        if t.kind == PUNCT and t.text in OPEN_TO_CLOSE:
            m = view.brackets.get(j)
            if m is None or m >= hi - 1:
                return None
            j = m + 1
            continue
        if t.kind == PUNCT and t.text in (b";", b"{", b"}"):
            return None
        if t.kind == NAME and t.text == b"return":
            return None
        j += 1
    if lo >= hi - 1:
        return None
    return (lo, hi - 1)


def _parameter_names(view: TokenView, f: FunctionInfo) -> list[bytes] | None:
    """Extract the parameter names of a function definition, sloppily:
    the name of each parameter is taken to be its last identifier
    token. Returns None if any parameter doesn't have one."""
    tokens = view.tokens
    pieces = _split_on_top_level_commas(view, f.params_open + 1, f.params_close)
    if pieces == [(f.params_open + 1, f.params_open + 1)]:
        return []
    names: list[bytes] = []
    for ts, te in pieces:
        name = None
        for q in range(ts, te):
            if tokens[q].kind == NAME:
                name = tokens[q].text
        if name is None:
            return None
        names.append(name)
    if names == [b"void"]:
        return []
    return names


def function_inlining_candidates(source: bytes) -> list[bytes]:
    """For each function whose body is a single statement, produce
    variants of the source where a call to it is replaced by the
    (parenthesised) body expression with arguments substituted for
    parameters. This is in the style of clang_delta's simple-inliner."""
    view = token_view(source)
    tokens = view.tokens
    replacer = Replacements()
    results: list[bytes] = []
    for f in find_function_definitions(view):
        expr_range = _single_statement_body(view, f)
        if expr_range is None:
            continue
        params = _parameter_names(view, f)
        if params is None:
            continue
        expr_lo, expr_hi = expr_range
        expr_start = tokens[expr_lo].start
        expr_end = tokens[expr_hi - 1].end
        for i, t in enumerate(tokens):
            if (
                t.kind != NAME
                or t.text != f.name
                or f.decl_starts[0] <= i <= f.body_close
                or i + 1 >= len(tokens)
                or tokens[i + 1].text != b"("
                or i + 1 not in view.brackets
            ):
                continue
            call_close = view.brackets[i + 1]
            args = [
                view.source[tokens[ts].start : tokens[te - 1].end]
                for ts, te in _split_on_top_level_commas(view, i + 2, call_close)
                if ts < te
            ]
            if len(args) != len(params):
                continue
            substitution = dict(zip(params, args, strict=True))
            expr_edits = tuple(
                (
                    tokens[q].start - expr_start,
                    tokens[q].end - expr_start,
                    b"(" + substitution[tokens[q].text] + b")",
                )
                for q in range(expr_lo, expr_hi)
                if tokens[q].kind == NAME and tokens[q].text in substitution
            )
            expr_source = source[expr_start:expr_end]
            inlined = b"(" + replacer.apply(expr_edits, expr_source) + b")"
            call_edit = ((tokens[i].start, tokens[call_close].end, inlined),)
            results.append(replacer.apply(call_edit, source))
    return results


# Bound on how many candidates a pump will adopt in a single
# invocation. Pumps get rerun by the reducer as long as they lead to
# progress, so this only limits how much a single invocation can
# balloon the test case.
MAX_PUMP_ADOPTIONS = 20


def _candidate_pump(
    name: str, derive: Callable[[bytes], list[bytes]]
) -> ReductionPump[bytes]:
    """Build a pump from a function that derives candidate variants
    (possibly larger than the input) from the current test case.

    Candidates are tried in order; whenever one is interesting we adopt
    it and rederive. The result is the last interesting variant, which
    the reducer will then try to reduce below the original."""

    async def pump(problem: ReductionProblem[bytes]) -> bytes:
        target = problem.current_test_case
        seen = {target}
        adoptions = 0
        improved = True
        while improved and adoptions < MAX_PUMP_ADOPTIONS:
            improved = False
            for candidate in derive(target):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if await problem.is_interesting(candidate):
                    target = candidate
                    adoptions += 1
                    improved = True
                    break
        return target

    pump.__name__ = name
    return pump


inline_typedefs: ReductionPump[bytes] = _candidate_pump(
    "inline_typedefs", typedef_inlining_candidates
)

inline_function_calls: ReductionPump[bytes] = _candidate_pump(
    "inline_function_calls", function_inlining_candidates
)


C_FILE_EXTENSIONS = (".c", ".cpp", ".h", ".hpp", ".cxx", ".cc")


CPP_PASSES: list[ReductionPass[bytes]] = [
    replace_function_bodies,
    delete_function_definitions,
    remove_namespaces,
    remove_base_classes,
    remove_constructor_initializers,
    remove_template_parts,
    simplify_call_expressions,
    replace_type_with_int,
]

CPP_PUMPS: list[ReductionPump[bytes]] = [
    inline_typedefs,
    inline_function_calls,
]
