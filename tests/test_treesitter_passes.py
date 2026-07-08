import bisect

import pytest
import tree_sitter
from hypothesis import example, given, settings
from hypothesis import strategies as st
from tree_sitter_language_pack.exceptions import LanguageNotFoundError

from shrinkray.passes.patching import CutPatch
from shrinkray.passes.treesitter import (
    EXTENSION_LANGUAGES,
    _covering_chain,
    _gc_candidates,
    _name_occurrences,
    _names_defined,
    _names_mentioned,
    child_deletion_cuts,
    language_for_filename,
    lift_cuts,
    loadable_language_for_filename,
    minimal_substitutions,
    orphaned_declaration_cuts,
    parse_tree,
    treesitter_passes,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.work import WorkContext
from tests.helpers import reduce_with


def apply_cuts(source: bytes, cuts: list[tuple[int, int]]) -> bytes:
    parts = []
    prev = 0
    for u, v in sorted(cuts):
        parts.append(source[prev:u])
        prev = v
    parts.append(source[prev:])
    return b"".join(parts)


# === language detection ===


def test_language_for_known_extensions():
    assert language_for_filename("foo.go") == "go"
    assert language_for_filename("foo.rs") == "rust"
    assert language_for_filename("/some/path/foo.js") == "javascript"
    assert language_for_filename("FOO.GO") == "go"


@pytest.mark.parametrize(
    "filename, language",
    [
        ("styles.scss", "scss"),
        ("styles.less", "less"),
        ("schema.graphql", "graphql"),
        ("schema.gql", "graphql"),
        ("service.proto", "proto"),
        ("main.tf", "hcl"),
        ("vars.tfvars", "hcl"),
        ("config.hcl", "hcl"),
        ("Widget.vue", "vue"),
        ("Widget.svelte", "svelte"),
        ("Token.sol", "solidity"),
        ("build.gradle", "groovy"),
        ("app.clj", "clojure"),
        ("solver.f90", "fortran"),
        ("deploy.ps1", "powershell"),
        ("module.cmake", "cmake"),
    ],
)
def test_language_for_added_extensions(filename, language):
    assert language_for_filename(filename) == language


def test_language_for_unknown_extension():
    assert language_for_filename("foo.unknownext") is None
    assert language_for_filename("foo") is None


def test_loadable_language_matches_mapping_when_grammar_loads():
    assert loadable_language_for_filename("foo.go") == "go"


def test_loadable_language_none_for_unknown_extension():
    assert loadable_language_for_filename("foo.unknownext") is None


def test_loadable_language_none_when_grammar_unavailable(monkeypatch, capsys):
    # Grammars are fetched (and on some platforms compiled) at runtime,
    # so a mapped language can still fail to load; reduction must fall
    # back to running without tree-sitter passes instead of crashing,
    # and must say what actually went wrong (download failure, grammar
    # missing for this platform, build failure, ...) rather than
    # swallowing the error.
    def unavailable(name):
        raise LanguageNotFoundError(f"Language '{name}' not found")

    monkeypatch.setattr(
        "shrinkray.passes.treesitter.tree_sitter_language_pack.get_language",
        unavailable,
    )
    assert loadable_language_for_filename("foo.go") is None
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "'go'" in stderr
    assert "foo.go" in stderr
    assert "LanguageNotFoundError" in stderr
    assert "Language 'go' not found" in stderr
    assert "without tree-sitter passes" in stderr


def test_loadable_language_none_when_loading_fails_outside_the_pack(
    monkeypatch, capsys
):
    # Loading a grammar dlopens a shared library (after possibly
    # downloading or compiling it), so failures outside the pack's own
    # exception hierarchy can surface too — a full cache directory, a
    # native load error. Those must also downgrade to running without
    # tree-sitter passes, matching the background download path.
    def unavailable(name):
        raise OSError("read-only file system")

    monkeypatch.setattr(
        "shrinkray.passes.treesitter.tree_sitter_language_pack.get_language",
        unavailable,
    )
    assert loadable_language_for_filename("foo.go") is None
    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "OSError" in stderr
    assert "read-only file system" in stderr
    assert "without tree-sitter passes" in stderr


def test_all_mapped_languages_are_loadable_and_parse():
    for language in sorted(set(EXTENSION_LANGUAGES.values())):
        tree = parse_tree(language, b"")
        assert tree is not None, language


def test_parse_tree_handles_non_utf8_bytes():
    tree = parse_tree("go", b"package main\n\xff\xfe\nfunc f() {}\n")
    assert tree is not None


# === child deletion ===


def test_child_deletion_can_remove_call_argument_with_separator():
    source = b"f(aaa, bbb, ccc)"
    tree = parse_tree("javascript", source)
    cuts = child_deletion_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    # Deleting an inner argument must also eat a separating comma.
    assert b"f(aaa, ccc)" in results
    assert b"f(aaa, bbb)" in results


def test_child_deletion_generates_runs_of_statements():
    source = b"aa();\nbb();\ncc();\ndd();\n"
    tree = parse_tree("javascript", source)
    cuts = child_deletion_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    # A run of two adjacent statements is a single candidate.
    assert b"aa();\ndd();\n" in results


def test_child_deletion_skips_zero_width_nodes():
    # `a :` parses as a labeled statement whose body is a zero-width
    # empty_statement; deleting nothing is not a candidate.
    source = b"a :"
    tree = parse_tree("javascript", source)
    for cut in child_deletion_cuts(tree, source):
        for start, end in cut:
            assert end > start


# === lifting ===


def test_lift_hoists_same_type_descendant():
    source = b"if (a) { if (b) { body(); } }"
    tree = parse_tree("javascript", source)
    cuts = lift_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    assert b"if (b) { body(); }" in results


def test_lift_promotes_named_children():
    source = b"f((xyz));"
    tree = parse_tree("javascript", source)
    cuts = lift_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    assert b"f(xyz);" in results


# === orphaned declarations ===

GO_SOURCE = b"""package main

import (
\t"fmt"
\t"strings"
)

func helper(x string) string {
\treturn strings.ToUpper(x)
}

func main() {
\tfmt.Println(helper("hi"))
}
"""


def test_orphan_cuts_delete_function_along_with_its_only_import():
    tree = parse_tree("go", GO_SOURCE)
    cuts = orphaned_declaration_cuts(tree, GO_SOURCE)
    results = {apply_cuts(GO_SOURCE, cut) for cut in cuts}
    # Deleting helper() orphans the "strings" import, so a joint
    # candidate for both must exist.
    joint = [r for r in results if b"strings" not in r and b"fmt" in r]
    assert joint


def test_orphan_cuts_never_delete_the_package_clause():
    tree = parse_tree("go", GO_SOURCE)
    cuts = orphaned_declaration_cuts(tree, GO_SOURCE)
    for cut in cuts:
        assert b"package" in apply_cuts(GO_SOURCE, cut)


def test_orphan_cuts_ignore_non_import_clauses_containing_use():
    # Go's range_clause contains "use" as a substring of its node type;
    # nodes inside it must not be treated as import entries.
    source = (
        b"package main\n"
        b"func f(items []int) {\n"
        b"\tfor _, v := range items {\n"
        b"\t\tprintln(v)\n"
        b"\t}\n"
        b"}\n"
    )
    tree = parse_tree("go", source)
    cuts = orphaned_declaration_cuts(tree, source)
    for cut in cuts:
        result = apply_cuts(source, cut)
        # The blank identifier must never be deleted on its own (that
        # would leave `for , v :=` behind).
        assert b"for ," not in result


def test_orphan_names_ignore_empty_string_literals():
    # The empty string in the var declaration contributes no name, so
    # deleting main() leaves the var orphaned and jointly deletable.
    source = b'package main\nvar sentinel = ""\nfunc main() { _ = sentinel }\n'
    tree = parse_tree("go", source)
    cuts = orphaned_declaration_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    assert b"package main\n\n\n" in results


def test_orphan_names_from_dotted_import_paths():
    # The usable name of this import ("yaml.v2") is not a plain word
    # token, exercising the slow-path occurrence search.
    source = (
        b"package main\n"
        b'import "gopkg.in/yaml.v2"\n'
        b'import "fmt"\n'
        b"func dump(v any) { fmt.Println(yaml.Marshal(v)) }\n"
        b"func main() { dump(1) }\n"
    )
    tree = parse_tree("go", source)
    cuts = orphaned_declaration_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    # Deleting the dump function orphans both imports (the call site in
    # main survives; only the definition and imports are deleted).
    assert any(
        b"func dump" not in r and b"import" not in r and b"func main" in r
        for r in results
    )


def test_orphan_cascade_collects_transitively_dead_declarations():
    # Deleting the const statement orphans helper; helper's body plus
    # the const statement together hold every reference to util, so the
    # cascade collects util as well even though no single node covers
    # all of util's references.
    source = (
        b"function util(x) { return x + 1; }\n"
        b"function helper(y) { return util(y); }\n"
        b"function unused0() {}\n"
        b"const out = helper(util(2));\n"
    )
    tree = parse_tree("javascript", source)
    cuts = orphaned_declaration_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    assert any(
        b"util" not in r and b"helper" not in r and b"unused0" in r for r in results
    )


def test_orphan_extent_can_end_inside_an_anonymous_token():
    # The import's usable name is the keyword `return`, whose only
    # occurrence outside the import is a bare keyword token; the
    # covering-chain walk descends into that unnamed token.
    source = (
        b"package main\n"
        b'import "x/return"\n'
        b"func f() int { return 2 }\n"
        b"func main() { println(f()) }\n"
    )
    tree = parse_tree("go", source)
    cuts = orphaned_declaration_cuts(tree, source)
    results = {apply_cuts(source, cut) for cut in cuts}
    assert any(b"import" not in r and b"func main" in r for r in results)


def test_names_mentioned_handles_wordless_strings():
    source = b'x = "-";\n'
    tree = parse_tree("javascript", source)
    names = _names_mentioned(tree.root_node, source)
    assert b"-" in names
    assert b"x" in names


def test_orphan_cuts_empty_when_there_are_no_declarations():
    source = b"package main\n"
    tree = parse_tree("go", source)
    assert orphaned_declaration_cuts(tree, source) == []


def test_orphan_cuts_terminate_on_zero_width_nodes():
    # Garbage input makes tree-sitter emit a zero-width top-level node
    # containing a MISSING identifier. Its empty "name" occurs at every
    # non-word boundary and its zero-width span can never be covered by
    # the deleted spans, so the cascade used to re-append it forever.
    source = b"\x00\xc7pd"
    tree = parse_tree("go", source)
    for cut in orphaned_declaration_cuts(tree, source):
        for start, end in cut:
            assert end > start


def test_orphan_names_ignore_missing_identifiers():
    # A declaration whose name is a zero-width MISSING node defines no
    # usable name; it must be skipped rather than treated as a
    # declaration of the empty name (which "occurs" everywhere).
    source = b"package main\nfunc() {}\nfunc main() {}\n"
    tree = parse_tree("go", source)
    for cut in orphaned_declaration_cuts(tree, source):
        for start, end in cut:
            assert end > start


def test_orphan_names_ignore_zero_width_mentioned_identifiers():
    # This garbage input parses to a surviving (non-zero-width)
    # top-level node with no `name` field, so its names come from
    # _names_mentioned, whose traversal reaches a zero-width MISSING
    # identifier. That empty name must not be collected.
    source = b'/"efm;'
    tree = parse_tree("go", source)
    for cut in orphaned_declaration_cuts(tree, source):
        for start, end in cut:
            assert end > start


# === substitutions ===


def test_substitutes_minimal_same_type_text():
    source = b"if (aaaa) { long_call(1, 2, 3); } else { ok(); }"
    tree = parse_tree("javascript", source)
    subs = minimal_substitutions(tree, source)
    results = set()
    for patch in subs:
        for start, end, replacement in patch:
            results.add(source[:start] + replacement + source[end:])
    # The larger call expression can be replaced by the smaller one.
    assert b"if (aaaa) { ok(); } else { ok(); }" in results


def test_substitutions_empty_for_empty_source():
    tree = parse_tree("javascript", b"")
    assert minimal_substitutions(tree, b"") == []


# === passes ===


def test_treesitter_passes_are_named_after_the_language():
    passes = treesitter_passes("go")
    names = [p.__name__ for p in passes]
    assert names
    for name in names:
        assert name.startswith("treesitter(go)/")


def simulate_go_unused_import_rule(source: bytes) -> bool:
    """Textual approximation of the Go compiler's unused-import rule."""
    lines = source.split(b"\n")
    imports = [
        line.strip().strip(b'"')
        for line in lines
        if line.strip().startswith(b'"') or line.strip().startswith(b'import "')
    ]
    body = b"\n".join(line for line in lines if not line.strip().startswith(b'"'))
    return all(name.split(b"/")[-1] + b"." in body for name in imports)


def test_pass_deletes_function_and_orphaned_import_together():
    initial = GO_SOURCE

    def is_interesting(source: bytes) -> bool:
        if b"fmt.Println" not in source:
            return False
        return simulate_go_unused_import_rule(source)

    assert is_interesting(initial)
    result = reduce_with(treesitter_passes("go"), initial, is_interesting)
    # helper and the strings import can only go together; neither
    # single deletion is interesting on its own.
    assert b"strings" not in result
    assert b"helper" not in result
    assert b"fmt.Println" in result


def test_pass_lifts_nested_conditionals():
    initial = b"if (a) { if (b) { magic(); } }"

    def is_interesting(source: bytes) -> bool:
        return b"magic()" in source

    result = reduce_with(treesitter_passes("javascript"), initial, is_interesting)
    # Leading whitespace is outside the root node's span, so these
    # passes cannot remove it; the byte-level passes handle that.
    assert result.strip() == b"magic()"


def test_passes_no_op_on_unparseable_candidates():
    # tree-sitter parses anything (with ERROR nodes), so passes should
    # still run and just fail to find useful candidates.
    initial = b"}}}\x00\xff{{{"

    def is_interesting(source: bytes) -> bool:
        return source == initial

    result = reduce_with(treesitter_passes("go"), initial, is_interesting)
    assert result == initial


# === reducer wiring ===


def test_shrinkray_includes_treesitter_passes_when_language_set():
    async def is_interesting(data: bytes) -> bool:
        return True

    problem = BasicReductionProblem(
        b"package main\n",
        is_interesting,
        work=WorkContext(parallelism=1),
    )
    reducer = ShrinkRay(target=problem, treesitter_language="go")
    names = {p.__name__ for p in reducer.great_passes}
    assert any(name.startswith("treesitter(go)/") for name in names)


def test_shrinkray_has_no_treesitter_passes_without_language():
    async def is_interesting(data: bytes) -> bool:
        return True

    problem = BasicReductionProblem(
        b"package main\n",
        is_interesting,
        work=WorkContext(parallelism=1),
    )
    reducer = ShrinkRay(target=problem)
    names = {p.__name__ for p in reducer.great_passes}
    assert not any(name.startswith("treesitter(") for name in names)


# === orphan cascade differential test ===
#
# `orphaned_declaration_cuts` was rewritten from an O(n^3) repeated-rescan
# cascade to an event-driven worklist that computes the identical output.
# `_orphaned_declaration_cuts_reference` below is a verbatim copy of the
# original (zero-width-safe) implementation, kept so a Hypothesis
# differential test can assert the optimized version matches it byte for
# byte on every input.


def _orphaned_declaration_cuts_reference(
    tree: tree_sitter.Tree, source: bytes
) -> list[CutPatch]:
    """Reference implementation: the original repeated-rescan cascade."""
    gc_nodes = [
        (n, _names_defined(n, source))
        for n in _gc_candidates(tree)
        if n.end_byte > n.start_byte
    ]
    occurrences = _name_occurrences(
        source, {name for _, names in gc_nodes for name in names}
    )

    def external_extent(index: int) -> tuple[int, int] | None:
        candidate, names = gc_nodes[index]
        span = (candidate.start_byte, candidate.end_byte)
        lo: int | None = None
        hi: int | None = None
        for name in names:
            positions = occurrences[name]
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
        candidate, names = gc_nodes[index]
        own = (candidate.start_byte, candidate.end_byte)
        for name in names:
            for position in occurrences[name]:
                if own[0] <= position < own[1]:
                    continue
                if not any(u <= position < v for u, v in spans):
                    return False
        return True

    triggers: dict[tuple[int, int], list[int]] = {}
    extents: list[tuple[int, int] | None] = []
    for index in range(len(gc_nodes)):
        extent = external_extent(index)
        extents.append(extent)
        if extent is None:
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
            if not (g.start_byte < x_span[1] and x_span[0] < g.end_byte)
        )
        if len(spans) < 2:
            continue
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


# Small pools of identifiers/imports shared across generated declarations
# so references genuinely collide and drive cascades.
_IDENTS = ["a", "b", "c", "d", "f0", "f1", "f2", "main", "fmt", "strings"]
_IMPORT_PATHS = ["fmt", "strings", "os", "gopkg.in/yaml.v2", "x/return", "a/b/c"]


def _chain_source(n: int) -> bytes:
    lines = ["package main", "func f0() int { return 1 }"]
    for i in range(1, n):
        lines.append("func f%d() int { return f%d() }" % (i, i - 1))
    lines.append("func main() { println(f%d()) }" % (n - 1))
    return ("\n".join(lines) + "\n").encode()


@st.composite
def _go_source(draw: st.DrawFn) -> bytes:
    parts = ["package main"]
    for _ in range(draw(st.integers(0, 3))):
        parts.append('import "%s"' % draw(st.sampled_from(_IMPORT_PATHS)))
    for _ in range(draw(st.integers(0, 6))):
        name = draw(st.sampled_from(_IDENTS))
        calls = " ".join(
            "%s()" % draw(st.sampled_from(_IDENTS))
            for _ in range(draw(st.integers(0, 3)))
        )
        parts.append("func %s() { %s }" % (name, calls))
    return "\n".join(parts).encode()


@st.composite
def _js_source(draw: st.DrawFn) -> bytes:
    parts = []
    for _ in range(draw(st.integers(0, 3))):
        parts.append(
            'import {%s} from "%s";'
            % (draw(st.sampled_from(_IDENTS)), draw(st.sampled_from(_IMPORT_PATHS)))
        )
    for _ in range(draw(st.integers(0, 6))):
        name = draw(st.sampled_from(_IDENTS))
        calls = " ".join(
            "%s();" % draw(st.sampled_from(_IDENTS))
            for _ in range(draw(st.integers(0, 3)))
        )
        parts.append("function %s() { %s }" % (name, calls))
    return "\n".join(parts).encode()


@st.composite
def _python_source(draw: st.DrawFn) -> bytes:
    parts = []
    for _ in range(draw(st.integers(0, 3))):
        parts.append("import %s" % draw(st.sampled_from(_IDENTS)))
    for _ in range(draw(st.integers(0, 6))):
        name = draw(st.sampled_from(_IDENTS))
        calls = (
            "; ".join(
                "%s()" % draw(st.sampled_from(_IDENTS))
                for _ in range(draw(st.integers(0, 3)))
            )
            or "pass"
        )
        parts.append("def %s():\n    %s" % (name, calls))
    return "\n".join(parts).encode()


@st.composite
def _rust_source(draw: st.DrawFn) -> bytes:
    parts = []
    for _ in range(draw(st.integers(0, 3))):
        parts.append("use %s;" % draw(st.sampled_from(_IDENTS)))
    for _ in range(draw(st.integers(0, 6))):
        name = draw(st.sampled_from(_IDENTS))
        calls = " ".join(
            "%s();" % draw(st.sampled_from(_IDENTS))
            for _ in range(draw(st.integers(0, 3)))
        )
        parts.append("fn %s() { %s }" % (name, calls))
    return "\n".join(parts).encode()


@st.composite
def _c_source(draw: st.DrawFn) -> bytes:
    parts = []
    for _ in range(draw(st.integers(0, 3))):
        parts.append("#include <%s>" % draw(st.sampled_from(_IDENTS)))
    for _ in range(draw(st.integers(0, 6))):
        name = draw(st.sampled_from(_IDENTS))
        calls = " ".join(
            "%s();" % draw(st.sampled_from(_IDENTS))
            for _ in range(draw(st.integers(0, 3)))
        )
        parts.append("int %s() { %s return 0; }" % (name, calls))
    return "\n".join(parts).encode()


_SOURCE_BUILDERS = {
    "go": _go_source(),
    "javascript": _js_source(),
    "python": _python_source(),
    "rust": _rust_source(),
    "c": _c_source(),
}


@st.composite
def _language_and_source(draw: st.DrawFn) -> tuple[str, bytes]:
    language = draw(st.sampled_from(sorted(_SOURCE_BUILDERS)))
    structured = draw(_SOURCE_BUILDERS[language])
    # Sometimes splice in raw/garbage bytes to reach error-recovery and
    # zero-width-node paths.
    garbage = draw(st.binary(max_size=16))
    where = draw(st.integers(0, len(structured)))
    if draw(st.booleans()):
        source = structured[:where] + garbage + structured[where:]
    else:
        source = structured
    return language, source


@settings(max_examples=400)
@given(_language_and_source())
@example(("go", _chain_source(40)))
@example(("go", _chain_source(3)))
@example(("go", b"\x00\xc7pd"))
@example(("go", b'/"efm;'))
@example(("go", b"package main\nfunc() {}\nfunc main() {}\n"))
@example(("go", GO_SOURCE))
@example(("javascript", b"\x00\xff{{{}}}"))
@example(("python", b""))
def test_orphaned_declaration_cuts_matches_reference(
    language_and_source: tuple[str, bytes],
) -> None:
    language, source = language_and_source
    tree = parse_tree(language, source)
    assert orphaned_declaration_cuts(tree, source) == (
        _orphaned_declaration_cuts_reference(tree, source)
    )
