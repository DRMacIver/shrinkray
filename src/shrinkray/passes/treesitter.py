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

import bisect
import os
import re
import sys
import traceback
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
    ".clj": "clojure",
    ".cljc": "clojure",
    ".cljs": "clojure",
    ".cmake": "cmake",
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
    ".f90": "fortran",
    ".f95": "fortran",
    ".go": "go",
    ".gql": "graphql",
    ".gradle": "groovy",
    ".graphql": "graphql",
    ".groovy": "groovy",
    ".h": "cpp",
    ".hcl": "hcl",
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
    ".less": "less",
    ".lua": "lua",
    ".m": "objc",
    ".mjs": "javascript",
    ".ml": "ocaml",
    ".nim": "nim",
    ".php": "php",
    ".pl": "perl",
    ".proto": "proto",
    ".ps1": "powershell",
    ".py": "python",
    ".r": "r",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".scss": "scss",
    ".sh": "bash",
    ".sol": "solidity",
    ".sql": "sql",
    ".svelte": "svelte",
    ".swift": "swift",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".vim": "vim",
    ".vue": "vue",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zig": "zig",
}


def language_for_filename(filename: str) -> str | None:
    """The tree-sitter language for a file, judged by its extension."""
    ext = os.path.splitext(filename)[1].lower()
    return EXTENSION_LANGUAGES.get(ext)


def loadable_language_for_filename(filename: str) -> str | None:
    """language_for_filename, restricted to grammars that actually load.

    tree-sitter-language-pack fetches grammars at runtime (and compiles
    them from source on platforms it publishes no prebuilt binaries
    for), so a mapped language can still be unavailable: a download can
    fail, or the grammar may not build on this platform. Reduction then
    runs without tree-sitter passes rather than crashing partway.
    """
    language = language_for_filename(filename)
    if language is None:
        return None
    try:
        tree_sitter_language_pack.get_language(language)
    except Exception as e:
        # The exception type says what failed: DownloadError for a
        # fetch, LanguageNotFoundError for a grammar this platform does
        # not have, DynamicLoadError for a broken build, and so on.
        # Loading also downloads, compiles and dlopens, so errors from
        # outside the pack's own hierarchy (OSError on a full cache
        # dir, a native load failure) are treated the same way. The
        # full traceback goes to stderr too (the run log in TUI mode) so
        # a broken grammar install can actually be debugged.
        traceback.print_exc()
        print(
            f"WARNING: could not load the tree-sitter grammar {language!r} "
            f"for {filename} ({type(e).__name__}: {e}); "
            "reducing without tree-sitter passes.",
            file=sys.stderr,
            flush=True,
        )
        return None
    return language


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
        # Positions of the named children within the full child list, so
        # adjacent anonymous tokens can be found without rescanning.
        named_indices = [j for j, c in enumerate(children) if c.is_named]
        for i, idx_first in enumerate(named_indices):
            child = children[idx_first]
            for run in range(1, MAX_CHILD_RUN + 1):
                if i + run > len(named_indices):
                    break
                idx_last = named_indices[i + run - 1]
                last = children[idx_last]
                start, end = child.start_byte, last.end_byte
                if end == start:
                    continue
                cuts.append([(start, end)])
                # A variant extending to the next named sibling's start,
                # eating the whitespace/newlines between them.
                if i + run < len(named_indices):
                    after_run = children[named_indices[i + run]]
                    cuts.append([(start, after_run.start_byte)])
                # Variants that also eat a separator token next to the
                # run, so deleting `bbb` from `f(aaa, bbb)` can remove
                # the now-dangling comma too.
                if idx_last + 1 < len(children):
                    after = children[idx_last + 1]
                    if source[after.start_byte : after.end_byte] in SEPARATOR_TOKENS:
                        cuts.append([(start, after.end_byte)])
                if idx_first > 0:
                    before = children[idx_first - 1]
                    if source[before.start_byte : before.end_byte] in SEPARATOR_TOKENS:
                        cuts.append([(before.start_byte, end)])
    return cuts


# How deep below a node to look for arbitrary named descendants to
# promote into its place. Same-type descendants are hoisted from
# arbitrarily far down (but only the nearest one along each path — the
# pass reapplies, so deeper ones are reached stepwise); for arbitrary
# descendants an unbounded search would generate quadratically many
# mostly-nonsensical candidates.
MAX_PROMOTION_DEPTH = 2


def lift_cuts(tree: tree_sitter.Tree, source: bytes) -> list[CutPatch]:
    """Cuts replacing a node with one of its named descendants.

    The nearest descendant of the same node type along each path is
    lifted regardless of depth (e.g. replacing an `if` with an `if`
    nested inside it); any named descendant is promoted from within
    MAX_PROMOTION_DEPTH levels (e.g. replacing a parenthesized
    expression with its contents).
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
            same_type = (
                descendant.is_named and d_span != span and descendant.type == node.type
            )
            if descendant.is_named and d_span != span:
                if same_type or depth <= MAX_PROMOTION_DEPTH:
                    cut = [(span[0], d_span[0]), (d_span[1], span[1])]
                    cuts.append([t for t in cut if t[0] < t[1]])
            # Stop below a same-type descendant: hoisting past it is
            # reachable by applying the pass repeatedly, and descending
            # regardless would make same-type chains quadratic.
            if not same_type:
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
                basename = text.rsplit(b"/", 1)[-1]
                names.add(basename)
                # A dotted basename like "yaml.v2" is referenced by its
                # leading word (Go imports gopkg.in/yaml.v2 as yaml).
                word = _WORD.match(basename)
                if word is not None:
                    names.add(word.group())
        elif n.child_count == 0 and "identifier" in n.type:
            # A MISSING identifier inserted by error recovery is zero
            # width; the empty bytes it covers are not a name.
            if n.end_byte > n.start_byte:
                names.add(source[n.start_byte : n.end_byte])
        else:
            stack.extend(n.children)
    return frozenset(names)


_WORD = re.compile(rb"[A-Za-z0-9_]+")


def _name_occurrences(source: bytes, names: set[bytes]) -> dict[bytes, list[int]]:
    """Start offsets of each name's whole-word occurrences in source.

    Names that are single word tokens (nearly all of them) are looked
    up in one tokenizing scan of the source; anything else (e.g. an
    import path basename like "yaml.v2") falls back to its own search.
    """
    tokens: dict[bytes, list[int]] = {}
    for match in _WORD.finditer(source):
        tokens.setdefault(match.group(), []).append(match.start())

    occurrences: dict[bytes, list[int]] = {}
    for name in names:
        if _WORD.fullmatch(name):
            occurrences[name] = tokens.get(name, [])
        else:
            pattern = rb"(?<![A-Za-z0-9_])" + re.escape(name) + rb"(?![A-Za-z0-9_])"
            occurrences[name] = [m.start() for m in re.finditer(pattern, source)]
    return occurrences


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


def _names_defined(node: tree_sitter.Node, source: bytes) -> frozenset[bytes]:
    """The names a declaration binds, as far as we can tell.

    Grammars conventionally expose a `name` field on declarations;
    references to that name are what should keep the declaration
    alive. Nodes without one (import entries in particular) fall back
    to every name they mention, which is conservative in the right
    direction: an import's mentioned names are exactly the names it
    binds, while for odd declaration shapes it just means fewer orphan
    candidates.
    """
    name_child = node.child_by_field_name("name")
    if name_child is not None and name_child.end_byte > name_child.start_byte:
        return frozenset({source[name_child.start_byte : name_child.end_byte]})
    return _names_mentioned(node, source)


def _covering_chain(tree: tree_sitter.Tree, lo: int, hi: int) -> list[tuple[int, int]]:
    """Spans of the named nodes whose span covers [lo, hi).

    Nodes covering an interval always form a chain from the root down,
    so this descends rather than scanning the whole tree.
    """
    spans = []
    node = tree.root_node
    while True:
        # Package/module headers are excluded for the same reason they
        # are not GC candidates: deleting one invalidates the file, so
        # cuts triggered by it (e.g. deleting `package main` orphans
        # `func main`) are wasted attempts.
        if (
            node.is_named
            and "package" not in node.type
            and node.start_byte <= lo
            and hi <= node.end_byte
        ):
            span = (node.start_byte, node.end_byte)
            if not spans or spans[-1] != span:
                spans.append(span)
        for child in node.children:
            if child.start_byte <= lo and hi <= child.end_byte:
                node = child
                break
        else:
            return spans


def orphaned_declaration_cuts(tree: tree_sitter.Tree, source: bytes) -> list[CutPatch]:
    """Cuts deleting a node plus declarations it leaves unreferenced.

    Deleting a node X orphans an eligible declaration when all the
    positions where the declaration's names occur (outside the
    declaration itself) fall inside X; the orphan can then be deleted
    along with X, cascading if that in turn orphans more declarations.
    Declarations that are unreferenced to begin with are not collected:
    a plain single cut, which other passes already generate, deletes
    those. This joint deletion is what makes dead code deletable in
    languages like Go that hard-error on unused imports.
    """
    # Zero-width nodes (from error recovery on garbage input) delete
    # nothing and can never be covered by a deleted span, so the
    # cascade below would re-collect them forever.
    gc_nodes = [
        (n, _names_defined(n, source))
        for n in _gc_candidates(tree)
        if n.end_byte > n.start_byte
    ]
    occurrences = _name_occurrences(
        source, {name for _, names in gc_nodes for name in names}
    )

    def external_extent(index: int) -> tuple[int, int] | None:
        """Smallest interval containing every position where the
        candidate's names occur outside its own span, or None if it is
        not referenced anywhere else at all."""
        candidate, names = gc_nodes[index]
        span = (candidate.start_byte, candidate.end_byte)
        lo: int | None = None
        hi: int | None = None
        for name in names:
            positions = occurrences[name]
            # Positions inside the candidate's own span are contiguous
            # in the sorted list; the external extremes sit just
            # outside that region.
            if positions and positions[0] < span[0]:
                lo = positions[0] if lo is None else min(lo, positions[0])
            if positions and positions[-1] >= span[1]:
                hi = positions[-1] if hi is None else max(hi, positions[-1])
            first_after = bisect.bisect_left(positions, span[1])
            if first_after < len(positions):
                lo = (
                    positions[first_after]
                    if lo is None
                    else min(lo, positions[first_after])
                )
            last_before = bisect.bisect_left(positions, span[0]) - 1
            if last_before >= 0:
                hi = (
                    positions[last_before]
                    if hi is None
                    else max(hi, positions[last_before])
                )
        if lo is None or hi is None:
            return None
        return lo, hi + 1

    def is_orphaned(index: int, spans: list[tuple[int, int]]) -> bool:
        """Whether every external reference to the candidate's names
        lies within the deleted spans."""
        candidate, names = gc_nodes[index]
        own = (candidate.start_byte, candidate.end_byte)
        for name in names:
            for position in occurrences[name]:
                if own[0] <= position < own[1]:
                    continue
                if not any(u <= position < v for u, v in spans):
                    return False
        return True

    # The nodes whose deletion orphans a candidate in one step are
    # exactly those whose span covers all its external references —
    # a chain from the root down.
    triggers: dict[tuple[int, int], list[int]] = {}
    extents: list[tuple[int, int] | None] = []
    for index in range(len(gc_nodes)):
        extent = external_extent(index)
        extents.append(extent)
        if extent is None:
            # Unreferenced to begin with: a plain single cut, which
            # other passes already generate, deletes it.
            continue
        for span in _covering_chain(tree, *extent):
            triggers.setdefault(span, []).append(index)

    cuts: list[CutPatch] = []
    for x_span, indices in sorted(triggers.items()):
        spans = [x_span]
        spans.extend(
            (g.start_byte, g.end_byte)
            for i in indices
            for g in [gc_nodes[i][0]]
            # A candidate overlapping the deleted node is already gone.
            if not (g.start_byte < x_span[1] and x_span[0] < g.end_byte)
        )
        if len(spans) < 2:
            continue
        # Cascade: deleting these regions may orphan further
        # declarations whose references lived inside them. Each
        # candidate is collected at most once, bounding the loop.
        collected: set[int] = set()
        changed = True
        while changed:
            changed = False
            for index, (candidate, _) in enumerate(gc_nodes):
                span = (candidate.start_byte, candidate.end_byte)
                if index in collected or extents[index] is None:
                    continue
                if any(u < span[1] and span[0] < v for u, v in spans):
                    continue
                if is_orphaned(index, spans):
                    spans.append(span)
                    collected.add(index)
                    changed = True
        cuts.append(sorted(set(spans)))
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
        _cut_pass(language, "delete_orphaned_declarations", orphaned_declaration_cuts),
        _cut_pass(language, "lift_nodes", lift_cuts),
        _substitution_pass(language),
    ]
