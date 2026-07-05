"""Grammar-aware reduction passes built on tree-sitter.

tree-sitter is a parser, not a code generator, but that is enough for
reduction: every node carries the byte range of its source text, so
transformations can be expressed as byte-range patches (Cuts and
Replacements) against the current test case. These passes provide
structure-aware reductions for any language with a grammar in
tree-sitter-language-pack, without needing per-language code:

- delete_children: delete runs of named child nodes (statements,
  arguments, ...), eating adjacent separator tokens like `,`/`;`.
- lift_nodes: replace a node with one of its named descendants — a
  same-type descendant at any depth (grammar-aware lift_braces), or any
  named descendant within a couple of levels (grammar-aware debracket).
- delete_orphaned_declarations: delete a node together with any
  top-level declarations or import entries whose names no longer occur
  anywhere else. This finds coupled deletions that single cuts cannot:
  e.g. Go rejects programs with unused imports, so a dead function and
  the imports only it uses can only be deleted together.
- substitute_nodes: replace a node's text with the smallest text seen
  elsewhere in the file for a node of the same type (e.g. replacing a
  long call expression with a short one appearing later).

tree-sitter parses erroneous input tolerantly (inserting ERROR nodes),
so these passes keep working on the syntactically-broken intermediate
states other passes produce.
"""

import os
import re
from collections.abc import Iterator

import tree_sitter
import tree_sitter_language_pack

from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.patching import (
    CutPatch,
    Cuts,
    ReplacementPatch,
    Replacements,
    apply_patches,
)
from shrinkray.problem import ReductionProblem


# Extensions shrink ray maps to tree-sitter grammars. Formats with
# dedicated, complete pass support (JSON, DIMACS CNF) are deliberately
# absent; languages that have some dedicated passes but incomplete ones
# (Python, C/C++) are included because these passes still add
# capabilities (e.g. coupled import deletion) on top of them.
EXTENSION_LANGUAGES: dict[str, str] = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".cxx": "cpp",
    ".d": "d",
    ".dart": "dart",
    ".el": "elisp",
    ".erl": "erlang",
    ".ex": "elixir",
    ".exs": "elixir",
    ".go": "go",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hs": "haskell",
    ".html": "html",
    ".java": "java",
    ".jl": "julia",
    ".js": "javascript",
    # The javascript grammar includes JSX syntax.
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".lua": "lua",
    ".m": "objc",
    ".ml": "ocaml",
    ".mjs": "javascript",
    ".nim": "nim",
    ".php": "php",
    ".pl": "perl",
    ".py": "python",
    ".r": "r",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".swift": "swift",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vim": "vim",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zig": "zig",
}


def language_for_filename(filename: str) -> str | None:
    """The tree-sitter language for a file, judged by its extension."""
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_LANGUAGES.get(ext)


def parse_tree(language: str, source: bytes) -> tree_sitter.Tree:
    """Parse source with the named grammar.

    tree-sitter is error-tolerant: any input parses, with unparseable
    regions wrapped in ERROR nodes.
    """
    parser = tree_sitter.Parser(tree_sitter_language_pack.get_language(language))
    return parser.parse(source)


def iter_nodes(tree: tree_sitter.Tree) -> Iterator[tree_sitter.Node]:
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


SEPARATOR_TOKENS = (b",", b";")

# Length of the longest run of adjacent children deleted as one patch.
# Runs beyond this are reachable by combining patches (the patch
# applier merges compatible cuts), so longer runs add little.
MAX_CHILD_RUN = 3


def child_deletion_cuts(tree: tree_sitter.Tree, source: bytes) -> list[CutPatch]:
    """Cuts deleting runs of named children, eating adjacent separators."""
    cuts: list[CutPatch] = []
    for node in iter_nodes(tree):
        children = node.children
        named = [c for c in children if c.is_named]
        for i, child in enumerate(named):
            for run in range(1, MAX_CHILD_RUN + 1):
                if i + run > len(named):
                    break
                last = named[i + run - 1]
                start, end = child.start_byte, last.end_byte
                if end == start:
                    continue
                cuts.append([(start, end)])
                # A variant extending to the next named sibling's start,
                # eating the whitespace/newlines between them.
                if i + run < len(named):
                    cuts.append([(start, named[i + run].start_byte)])
                # Variants that also eat a separator token next to the
                # run, so deleting `bbb` from `f(aaa, bbb)` can remove
                # the now-dangling comma too.
                idx_last = children.index(last)
                if idx_last + 1 < len(children):
                    after = children[idx_last + 1]
                    if source[after.start_byte : after.end_byte] in SEPARATOR_TOKENS:
                        cuts.append([(start, after.end_byte)])
                idx_first = children.index(child)
                if idx_first > 0:
                    before = children[idx_first - 1]
                    if source[before.start_byte : before.end_byte] in SEPARATOR_TOKENS:
                        cuts.append([(before.start_byte, end)])
    return cuts


# How deep below a node to look for arbitrary named descendants to
# promote into its place. Same-type descendants are hoisted from any
# depth; for arbitrary descendants an unbounded search would generate
# quadratically many mostly-nonsensical candidates.
MAX_PROMOTION_DEPTH = 2


def lift_cuts(tree: tree_sitter.Tree, source: bytes) -> list[CutPatch]:
    """Cuts replacing a node with one of its named descendants.

    A descendant of the same node type is lifted from any depth (e.g.
    replacing an `if` with an `if` nested inside it); any named
    descendant is promoted from within MAX_PROMOTION_DEPTH levels (e.g.
    replacing a parenthesized expression with its contents).
    """
    cuts: list[CutPatch] = []
    for node in iter_nodes(tree):
        if not node.is_named:
            continue
        span = (node.start_byte, node.end_byte)
        stack = [(child, 1) for child in node.children]
        while stack:
            descendant, depth = stack.pop()
            d_span = (descendant.start_byte, descendant.end_byte)
            if d_span != span and descendant.is_named:
                same_type = descendant.type == node.type
                if same_type or depth <= MAX_PROMOTION_DEPTH:
                    cut = [(span[0], d_span[0]), (d_span[1], span[1])]
                    cuts.append([t for t in cut if t[0] < t[1]])
            stack.extend((child, depth + 1) for child in descendant.children)
    return [c for c in cuts if c]


def _is_import_like(node_type: str) -> bool:
    words = node_type.split("_")
    return any(w in ("import", "imports", "use") for w in words)


def _names_mentioned(node: tree_sitter.Node, source: bytes) -> frozenset[bytes]:
    """Names a node mentions: identifier tokens, plus final path
    segments of string literals (covering import paths like
    "path/filepath", whose usable name is `filepath`)."""
    names = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if "string" in n.type:
            text = source[n.start_byte : n.end_byte].strip(b"\"'`")
            if text:
                names.add(text.rsplit(b"/", 1)[-1])
        elif n.child_count == 0 and "identifier" in n.type:
            names.add(source[n.start_byte : n.end_byte])
        else:
            stack.extend(n.children)
    return frozenset(names)


def _occurs_outside(
    name: bytes, source: bytes, spans: list[tuple[int, int]]
) -> bool:
    pattern = rb"(?<![A-Za-z0-9_])" + re.escape(name) + rb"(?![A-Za-z0-9_])"
    for match in re.finditer(pattern, source):
        if not any(u <= match.start() < v for u, v in spans):
            return True
    return False


def _gc_candidates(tree: tree_sitter.Tree) -> list[tree_sitter.Node]:
    """Nodes eligible for orphan collection: top-level declarations
    (except package/module headers, which nothing references by name
    but whose deletion invalidates the file) and entries of import-like
    constructs (which are syntactically self-contained)."""
    result = []
    for child in tree.root_node.children:
        if (
            child.is_named
            and "package" not in child.type
            and "module" not in child.type
        ):
            result.append(child)
        stack = [(child, _is_import_like(child.type))]
        while stack:
            node, in_import = stack.pop()
            for grandchild in node.children:
                if grandchild.is_named and in_import:
                    result.append(grandchild)
                stack.append(
                    (grandchild, in_import or _is_import_like(grandchild.type))
                )
    return result


def orphaned_declaration_cuts(
    tree: tree_sitter.Tree, source: bytes
) -> list[CutPatch]:
    """Cuts deleting a node plus declarations it leaves unreferenced.

    For every named node, tentatively delete it, then repeatedly delete
    any eligible declaration none of whose names still occur outside
    the deleted regions. Only combinations that collect at least one
    extra declaration are returned: the plain single deletion is
    already generated by other passes. This is what makes dead code
    deletable in languages like Go that hard-error on unused imports.
    """
    gc_nodes = [(n, _names_mentioned(n, source)) for n in _gc_candidates(tree)]

    cuts: list[CutPatch] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for node in iter_nodes(tree):
        if not node.is_named or node.start_byte == node.end_byte:
            continue
        spans = [(node.start_byte, node.end_byte)]
        changed = True
        while changed:
            changed = False
            for candidate, names in gc_nodes:
                span = (candidate.start_byte, candidate.end_byte)
                if any(u < span[1] and span[0] < v for u, v in spans):
                    continue
                if not any(
                    _occurs_outside(name, source, [*spans, span]) for name in names
                ):
                    spans.append(span)
                    changed = True
        if len(spans) > 1:
            key = tuple(sorted(spans))
            if key not in seen:
                seen.add(key)
                cuts.append(sorted(spans))
    return cuts


def minimal_substitutions(
    tree: tree_sitter.Tree, source: bytes
) -> list[ReplacementPatch]:
    """Replace nodes with the smallest same-type text in the file.

    This is the "remember strings you've seen for each node type" idea:
    it cannot invent replacements, but a file being reduced usually
    already contains a small instance of most node types (a short call,
    a trivial block), and swapping one in unlocks further deletion.
    """
    minimal: dict[str, bytes] = {}
    named = [n for n in iter_nodes(tree) if n.is_named]
    for node in named:
        text = source[node.start_byte : node.end_byte]
        best = minimal.get(node.type)
        if best is None or (len(text), text) < (len(best), best):
            minimal[node.type] = text

    patches: list[ReplacementPatch] = []
    for node in named:
        text = source[node.start_byte : node.end_byte]
        replacement = minimal[node.type]
        if len(replacement) < len(text):
            patches.append(((node.start_byte, node.end_byte, replacement),))
    return patches


def _cut_pass(
    language: str,
    name: str,
    generator,
) -> ReductionPass[bytes]:
    async def run(problem: ReductionProblem[bytes]) -> None:
        source = problem.current_test_case
        cuts = generator(parse_tree(language, source), source)
        await apply_patches(problem, Cuts(), cuts)

    run.__name__ = f"treesitter({language})/{name}"
    return run


def _substitution_pass(language: str) -> ReductionPass[bytes]:
    async def run(problem: ReductionProblem[bytes]) -> None:
        source = problem.current_test_case
        patches = minimal_substitutions(parse_tree(language, source), source)
        await apply_patches(problem, Replacements(), patches)

    run.__name__ = f"treesitter({language})/substitute_nodes"
    return run


def treesitter_passes(language: str) -> list[ReductionPass[bytes]]:
    """Grammar-aware passes for a language, in rough order of value."""
    return [
        _cut_pass(language, "delete_children", child_deletion_cuts),
        _cut_pass(
            language, "delete_orphaned_declarations", orphaned_declaration_cuts
        ),
        _cut_pass(language, "lift_nodes", lift_cuts),
        _substitution_pass(language),
    ]
