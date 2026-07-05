from shrinkray.passes.treesitter import (
    EXTENSION_LANGUAGES,
    _names_mentioned,
    child_deletion_cuts,
    language_for_filename,
    lift_cuts,
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


def test_language_for_unknown_extension():
    assert language_for_filename("foo.unknownext") is None
    assert language_for_filename("foo") is None


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
