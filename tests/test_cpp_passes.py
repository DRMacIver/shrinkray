import types
from pathlib import Path
from unittest.mock import patch

import trio
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.passes.cpp import (
    CHARACTER,
    COMMENT,
    CPP_PASSES,
    MAX_PUMP_ADOPTIONS,
    NAME,
    NUMBER,
    PREPROC,
    PUNCT,
    STRING,
    TokenView,
    _candidate_pump,
    _find_angle_close,
    _find_class_base_lists,
    _find_namespace_blocks,
    _qualifier_cuts_by_path,
    _split_on_top_level_commas,
    delete_function_definitions,
    find_function_definitions,
    find_typedefs,
    function_inlining_candidates,
    inline_function_calls,
    inline_typedefs,
    lex,
    match_brackets,
    remove_base_classes,
    remove_constructor_initializers,
    remove_namespaces,
    remove_template_parts,
    replace_function_bodies,
    replace_type_with_int,
    simplify_call_expressions,
    token_view,
    typedef_inlining_candidates,
)
from shrinkray.passes.genericlanguages import cut_comment_like_things
from shrinkray.passes.patching import Replacements
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import WorkContext
from tests.helpers import reduce_with


# === Lexer tests ===


def kinds_and_texts(source: bytes) -> list[tuple[str, bytes]]:
    return [(t.kind, t.text) for t in lex(source)]


def test_lexes_simple_function():
    assert kinds_and_texts(b"int main() { return 0; }") == [
        (NAME, b"int"),
        (NAME, b"main"),
        (PUNCT, b"("),
        (PUNCT, b")"),
        (PUNCT, b"{"),
        (NAME, b"return"),
        (NUMBER, b"0"),
        (PUNCT, b";"),
        (PUNCT, b"}"),
    ]


def test_lexes_line_comment_to_end_of_line():
    assert kinds_and_texts(b"x // hello\ny") == [
        (NAME, b"x"),
        (COMMENT, b"// hello"),
        (NAME, b"y"),
    ]


def test_lexes_line_comment_at_end_of_file():
    assert kinds_and_texts(b"// hello") == [(COMMENT, b"// hello")]


def test_lexes_block_comment():
    assert kinds_and_texts(b"x /* hi\nthere */ y") == [
        (NAME, b"x"),
        (COMMENT, b"/* hi\nthere */"),
        (NAME, b"y"),
    ]


def test_unterminated_block_comment_runs_to_end_of_file():
    assert kinds_and_texts(b"x /* oops") == [(NAME, b"x"), (COMMENT, b"/* oops")]


def test_lexes_string_with_escapes():
    assert kinds_and_texts(rb'"a\"b" x') == [(STRING, rb'"a\"b"'), (NAME, b"x")]


def test_unterminated_string_ends_at_newline():
    assert kinds_and_texts(b'"abc\ndef') == [(STRING, b'"abc'), (NAME, b"def")]


def test_unterminated_string_ends_at_end_of_file():
    assert kinds_and_texts(b'"abc') == [(STRING, b'"abc')]


def test_string_with_escaped_backslash_at_end_of_file():
    assert kinds_and_texts(b'"abc\\') == [(STRING, b'"abc\\')]


def test_lexes_char_literal():
    assert kinds_and_texts(rb"'\n' x") == [(CHARACTER, rb"'\n'"), (NAME, b"x")]


def test_lexes_raw_string():
    assert kinds_and_texts(b'R"(a "quoted" thing)" x') == [
        (STRING, b'R"(a "quoted" thing)"'),
        (NAME, b"x"),
    ]


def test_lexes_raw_string_with_delimiter():
    assert kinds_and_texts(b'R"xy(some )" stuff)xy" x') == [
        (STRING, b'R"xy(some )" stuff)xy"'),
        (NAME, b"x"),
    ]


def test_lexes_raw_string_with_prefix():
    assert kinds_and_texts(b'u8R"(hi)"') == [(STRING, b'u8R"(hi)"')]


def test_unterminated_raw_string_runs_to_end_of_file():
    assert kinds_and_texts(b'R"(oops') == [(STRING, b'R"(oops')]


def test_identifier_starting_with_r_is_not_a_raw_string():
    assert kinds_and_texts(b"Ready") == [(NAME, b"Ready")]


def test_lexes_preprocessor_directive():
    assert kinds_and_texts(b"#include <stdio.h>\nint x;") == [
        (PREPROC, b"#include <stdio.h>"),
        (NAME, b"int"),
        (NAME, b"x"),
        (PUNCT, b";"),
    ]


def test_preprocessor_directive_with_continuation():
    source = b"#define FOO \\\n  bar\nint x;"
    assert kinds_and_texts(source)[0] == (PREPROC, b"#define FOO \\\n  bar")


def test_preprocessor_directive_with_crlf_continuation():
    source = b"#define FOO \\\r\n  bar\r\nint x;"
    assert kinds_and_texts(source)[0] == (PREPROC, b"#define FOO \\\r\n  bar")


def test_preprocessor_directive_at_end_of_file():
    assert kinds_and_texts(b"#endif") == [(PREPROC, b"#endif")]


def test_hash_not_at_line_start_is_punctuation():
    assert kinds_and_texts(b"x # y") == [(NAME, b"x"), (PUNCT, b"#"), (NAME, b"y")]


def test_preprocessor_allowed_after_leading_whitespace():
    assert kinds_and_texts(b"  #pragma once") == [(PREPROC, b"#pragma once")]


def test_lexes_multibyte_punctuation():
    assert kinds_and_texts(b"a::b->c >> d <=> e") == [
        (NAME, b"a"),
        (PUNCT, b"::"),
        (NAME, b"b"),
        (PUNCT, b"->"),
        (NAME, b"c"),
        (PUNCT, b">>"),
        (NAME, b"d"),
        (PUNCT, b"<=>"),
        (NAME, b"e"),
    ]


def test_lexes_number_forms():
    assert kinds_and_texts(b"0x1fULL 1.5e-3 .5 1'000'000") == [
        (NUMBER, b"0x1fULL"),
        (NUMBER, b"1.5e-3"),
        (NUMBER, b".5"),
        (NUMBER, b"1'000'000"),
    ]


def test_digit_separator_requires_following_digit():
    assert kinds_and_texts(b"1' '") == [(NUMBER, b"1"), (CHARACTER, b"' '")]


def test_dot_not_followed_by_digit_is_punctuation():
    assert kinds_and_texts(b"a.b") == [(NAME, b"a"), (PUNCT, b"."), (NAME, b"b")]


def test_dollar_is_an_identifier_character():
    assert kinds_and_texts(b"$foo") == [(NAME, b"$foo")]


def test_division_is_not_a_comment():
    assert kinds_and_texts(b"a / b") == [(NAME, b"a"), (PUNCT, b"/"), (NAME, b"b")]


def test_slash_at_end_of_file():
    assert kinds_and_texts(b"a /") == [(NAME, b"a"), (PUNCT, b"/")]


@given(st.binary())
def test_lexer_tokens_tile_the_source(source: bytes):
    tokens = lex(source)
    prev = 0
    for token in tokens:
        assert prev <= token.start < token.end <= len(source)
        assert token.text == source[token.start : token.end]
        # Gaps between tokens are pure whitespace.
        assert source[prev : token.start].strip(b" \t\r\n\f\v") == b""
        prev = token.end
    assert source[prev:].strip(b" \t\r\n\f\v") == b""


@given(st.text(alphabet="abc{}()[]<>\"'\\/*#\n 0", max_size=30).map(str.encode))
def test_lexer_handles_confusing_punctuation_soup(source: bytes):
    lex(source)


# === Bracket matching tests ===


def test_matches_nested_brackets():
    tokens = lex(b"f(a[0], {1});")
    matches = match_brackets(tokens)
    texts = {tokens[i].text: tokens[j].text for i, j in matches.items()}
    assert texts == {
        b"(": b")",
        b")": b"(",
        b"[": b"]",
        b"]": b"[",
        b"{": b"}",
        b"}": b"{",
    }
    # The mapping is an involution.
    for i, j in matches.items():
        assert matches[j] == i


def test_tolerates_unbalanced_brackets():
    tokens = lex(b"f(x))} {")
    matches = match_brackets(tokens)
    # Only the one balanced pair of parens matches.
    matched_texts = sorted(tokens[i].text for i in matches)
    assert matched_texts == [b"(", b")"]


def test_brackets_in_comments_and_strings_do_not_match():
    tokens = lex(b'{ /* } */ "}" }')
    matches = match_brackets(tokens)
    assert len(matches) == 2
    i, j = sorted(matches)
    assert tokens[i].text == b"{"
    assert tokens[j].text == b"}"
    assert tokens[i].start == 0
    assert tokens[j].end == 15


# === Function definition discovery ===


def function_names(source: bytes) -> list[bytes]:
    return [f.name for f in find_function_definitions(token_view(source))]


def test_finds_simple_function():
    assert function_names(b"int add(int x, int y) { return x + y; }") == [b"add"]


def test_finds_adjacent_functions():
    assert function_names(b"void g() {}\nvoid f() {}") == [b"g", b"f"]


def test_finds_method_with_trailing_specifiers():
    assert function_names(b"struct A { int f() const noexcept { return 1; } };") == [
        b"f"
    ]


def test_finds_constructor_with_initializer_list():
    source = b"struct P { P(int a) : x(a), y{0} { init(); } int x, y; };"
    functions = find_function_definitions(token_view(source))
    real = [f for f in functions if f.init_colon is not None and f.name == b"P"]
    assert len(real) >= 1
    view = token_view(source)
    f = real[-1]
    body = source[view.tokens[f.body_open].start : view.tokens[f.body_close].end]
    assert body == b"{ init(); }"


def test_finds_destructor():
    assert function_names(b"Foo::~Foo() { delete p; }") == [b"Foo"]


def test_finds_operator_overload():
    assert function_names(b"int operator+(A a, A b) { return 0; }") == [b"operator"]


def test_control_flow_is_not_a_function():
    assert function_names(b"void f() { if (x) { g(); } while (y) { h(); } }") == [b"f"]


def test_lambda_is_not_a_function():
    assert function_names(b"auto l = [](int x) { return x; };") == []


def test_braced_initializer_is_not_a_function():
    assert function_names(b"int xs[] = {1, 2};") == []


def test_struct_is_not_a_function():
    assert function_names(b"struct Point { int x; int y; };") == []


def test_function_after_preprocessor_directive():
    assert function_names(b"#include <stdio.h>\nint main(void) { return 0; }") == [
        b"main"
    ]


def test_unbalanced_braces_tolerated():
    assert function_names(b"int f() { return 0;") == []


# === replace_function_bodies ===


def test_replaces_function_body_with_declaration():
    assert (
        reduce_with(
            [replace_function_bodies],
            b"int add(int x, int y) { return x + y; }",
            lambda x: b"add" in x,
        )
        == b"int add(int x, int y);"
    )


def test_replaces_constructor_including_initializers():
    assert (
        reduce_with(
            [replace_function_bodies],
            b"struct Foo { Foo() : x_(0) { } int x_; };",
            lambda x: b"Foo()" in x,
        )
        == b"struct Foo { Foo(); int x_; };"
    )


def test_can_keep_trailing_specifiers_when_replacing_body():
    result = reduce_with(
        [replace_function_bodies],
        b"struct A { virtual int f() const { return 1; } };",
        lambda x: b"const" in x,
    )
    assert result == b"struct A { virtual int f() const ; };"


# === delete_function_definitions ===


def test_deletes_unused_function():
    result = reduce_with(
        [delete_function_definitions],
        b"int used() { return 1; }\nint unused() { return 2; }\nint main() { return used(); }",
        lambda x: b"int used" in x and b"main" in x,
    )
    assert b"unused" not in result
    assert b"int used() { return 1; }" in result
    assert b"main" in result


def test_deletes_template_function_with_prefix():
    result = reduce_with(
        [delete_function_definitions],
        b"template <typename T>\nT twice(T x) { return x + x; }\nint keep;",
        lambda x: b"keep" in x,
    )
    assert result.strip() == b"int keep;"


# === remove_namespaces ===


def test_splices_namespace_contents():
    result = reduce_with(
        [remove_namespaces],
        b"namespace foo {\nint x = 1;\n}\n",
        lambda x: b"int x = 1;" in x,
    )
    assert b"namespace" not in result
    assert b"{" not in result
    assert b"int x = 1;" in result


def test_deletes_whole_namespace():
    result = reduce_with(
        [remove_namespaces],
        b"namespace foo {\nint x = 1;\n}\nint y;",
        lambda x: b"int y;" in x,
    )
    assert result.strip() == b"int y;"


def test_splices_nested_namespaces():
    result = reduce_with(
        [remove_namespaces],
        b"namespace a {\nnamespace b {\nint x = 1;\n}\n}\n",
        lambda x: b"int x = 1;" in x,
    )
    assert b"namespace" not in result


def test_splices_cpp17_nested_namespace():
    result = reduce_with(
        [remove_namespaces],
        b"namespace a::b {\nint x = 1;\n}\n",
        lambda x: b"int x = 1;" in x,
    )
    assert b"namespace" not in result


def test_splices_extern_c_block():
    result = reduce_with(
        [remove_namespaces],
        b'extern "C" {\nint f(void);\n}\n',
        lambda x: b"int f(void);" in x,
    )
    assert b"extern" not in result


def test_namespace_alias_is_left_alone():
    source = b"namespace a = b;\n"
    assert reduce_with([remove_namespaces], source, lambda x: True) == source


def test_removes_namespace_and_strips_qualified_references():
    # Splicing the namespace away must also drop the `ns::` qualifier
    # from references, or the reference dangles and won't compile. This
    # is the case shrink ray previously couldn't handle (kept the
    # namespace because `graph::Queue` blocked its removal).
    result = reduce_with(
        [remove_namespaces],
        b"namespace graph {\nstruct Queue {};\n}\ntemplate struct graph::Queue;\n",
        lambda x: b"struct Queue {};" in x and b"template struct" in x,
    )
    assert b"namespace" not in result
    assert b"graph::" not in result
    assert b"template struct Queue;" in result.replace(b"\n", b"")


def test_strips_self_qualified_references_inside_namespace_body():
    # A namespace whose body refers to its own members with a qualified
    # name (`ns::x`) can only be spliced away if those inner qualifiers
    # are stripped along with the ones outside the namespace.
    def is_interesting(x: bytes) -> bool:
        if b"int x" not in x or b"int z" not in x:
            return False
        # Simulate a compiler: `ns::` references only resolve while the
        # namespace still exists.
        return b"namespace ns" in x or b"ns::" not in x

    result = reduce_with(
        [remove_namespaces],
        b"namespace ns { int x; int y = ns::x; } int z = ns::x;\n",
        is_interesting,
    )
    assert b"namespace" not in result
    assert b"ns::" not in result


def test_strips_nested_namespace_path_qualifier():
    result = reduce_with(
        [remove_namespaces],
        b"namespace a::b {\nint v = 1;\n}\nint w = a::b::v;\n",
        lambda x: b"int v = 1;" in x and b"int w =" in x,
    )
    assert b"namespace" not in result
    assert b"a::b::" not in result
    assert b"int w = v;" in result.replace(b"\n", b"")


# === Namespace qualifier scanning ===
#
# All namespaces' qualifier cuts are found with a single indexed scan
# over the tokens. These tests check that the indexed scan agrees with
# the straightforward (but per-namespace, hence quadratic) scan it
# replaced, kept here as a reference implementation.


def _reference_namespace_qualifier_cuts(
    view: TokenView, name_path: tuple[int, int], decl_start: int
) -> list[tuple[int, int]]:
    """Pre-index implementation of namespace qualifier scanning: walk
    the whole token list for one namespace's path, skipping the
    declaration header, consuming matched qualifiers."""
    tokens = view.tokens
    lo, hi = name_path
    path_texts = [tokens[k].text for k in range(lo, hi)]
    n = len(path_texts)
    cuts: list[tuple[int, int]] = []
    p = 0
    limit = len(tokens) - n
    while p <= limit:
        if decl_start <= p <= hi:
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


NAMESPACEY_SOURCES = st.lists(
    st.sampled_from(
        ["namespace", "extern", "a", "b", "x", "::", "{", "}", ";", "=", '"C"']
    ),
    max_size=30,
).map(lambda parts: " ".join(parts).encode())


@given(NAMESPACEY_SOURCES)
def test_qualifier_cut_index_matches_reference_scan(source: bytes):
    view = token_view(source)
    tokens = view.tokens
    blocks = _find_namespace_blocks(view)
    paths: set[tuple[bytes, ...]] = set()
    for _, _, _, name_path in blocks:
        if name_path is not None:
            lo, hi = name_path
            paths.add(tuple(t.text for t in tokens[lo:hi]))
    index = _qualifier_cuts_by_path(view, paths)
    for decl, _, _, name_path in blocks:
        if name_path is None:
            continue
        lo, hi = name_path
        key = tuple(t.text for t in tokens[lo:hi])
        assert index[key] == _reference_namespace_qualifier_cuts(view, name_path, decl)


def test_qualifier_cuts_consume_overlapping_matches():
    # After matching `a::a` followed by `::`, the scan resumes after the
    # consumed qualifier, so the overlapping `a::a::` starting at the
    # second `a` is not also cut.
    source = b"namespace a::a { }\na::a::a::x;\n"
    view = token_view(source)
    [cuts] = _qualifier_cuts_by_path(view, {(b"a", b"::", b"a")}).values()
    assert [source[:s] + source[e:] for s, e in cuts] == [
        b"namespace a::a { }\na::x;\n"
    ]


# === remove_base_classes ===


def test_removes_all_base_classes():
    assert (
        reduce_with(
            [remove_base_classes],
            b"class A : public B, private C { };",
            lambda x: b"class A" in x,
        )
        == b"class A { };"
    )


def test_removes_individual_base_class():
    result = reduce_with(
        [remove_base_classes],
        b"class A : public B, private C { };",
        lambda x: b"private C" in x,
    )
    assert b"public B" not in result
    assert b"private C" in result


def test_removes_templated_base_class():
    assert (
        reduce_with(
            [remove_base_classes],
            b"struct A : Base<int, char> { };",
            lambda x: b"A" in x,
        )
        == b"struct A { };"
    )


def test_removes_enum_underlying_type():
    assert (
        reduce_with(
            [remove_base_classes],
            b"enum E : unsigned { X };",
            lambda x: b"X" in x,
        )
        == b"enum E { X };"
    )


def test_template_parameter_list_is_not_a_base_class():
    source = b"template <class T, class U> struct S { };"
    assert reduce_with([remove_base_classes], source, lambda x: True) == source


# === replace_type_with_int ===


def test_replaces_struct_type_with_int():
    # Delete the struct definition and turn its uses into int, in the
    # style of clang_delta's empty-struct-to-int.
    result = reduce_with(
        [replace_type_with_int],
        b"struct S { int a; int b; };\nS make();\n",
        lambda x: b"make()" in x,
    )
    assert b"struct S" not in result
    assert result.replace(b"\n", b"").strip() == b"int make();"


def test_replaces_forward_declared_class_template_with_int():
    # The udlit-style case: a class template used only as a return type.
    result = reduce_with(
        [replace_type_with_int],
        b"template <class T> class Box;\nBox<int> unwrap();\n",
        lambda x: b"unwrap()" in x,
    )
    assert b"Box" not in result
    assert result.replace(b"\n", b"").strip() == b"int unwrap();"


def test_replaces_type_use_consuming_template_arguments():
    result = reduce_with(
        [replace_type_with_int],
        b"struct S {};\nS<A, B<C>> f();\n",
        lambda x: b"f()" in x,
    )
    assert b"S<" not in result
    assert b"int f();" in result.replace(b"\n", b"")


def test_replace_type_leaves_unrelated_code_alone():
    source = b"int main() { return 0; }"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_ignores_anonymous_struct():
    # No name after the keyword: nothing to replace.
    source = b"struct { int x; } v;"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_ignores_elaborated_type_use():
    # `struct S s;` names an existing type; the token after the name is
    # not `;`, `{`, or `:`, so it is not treated as a definition.
    source = b"struct S s;"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_ignores_class_at_end_of_input():
    source = b"int x; class"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_ignores_unterminated_definition():
    # No terminating semicolon, so _declaration_end finds nothing.
    source = b"struct S { int a;"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_struct_without_semicolon_is_ignored():
    # Balanced braces but no terminating semicolon: _declaration_end
    # runs off the end without finding one.
    source = b"struct S {} S make()"
    assert reduce_with([replace_type_with_int], source, lambda x: True) == source


def test_replace_type_tolerates_unclosed_angle_in_base_clause():
    # A `<` in the base clause that never closes is stepped over rather
    # than derailing the scan for the terminating `;`.
    source = b"struct S : B< { int a; }; S x;"
    reduce_with([replace_type_with_int], source, lambda x: True)


def test_replace_type_tolerates_unclosed_angle_after_use():
    # A use of the type name followed by an unterminated `<` keeps just
    # the name in the replaced span.
    source = b"struct S {}; S<"
    reduce_with([replace_type_with_int], source, lambda x: True)


def test_replace_type_tolerates_unclosed_template_prefix():
    # An unterminated `template <` must not crash the prefix scan.
    source = b"template < struct S; S x;"
    reduce_with([replace_type_with_int], source, lambda x: True)


def test_replace_type_keeps_template_prefix_of_class_template():
    # Deleting a class template must take its `template<...>` prefix too,
    # or a dangling prefix is left behind.
    result = reduce_with(
        [replace_type_with_int],
        b"template <class T>\nstruct Wrapper { T value; };\nWrapper<int> w();\n",
        lambda x: b"w()" in x,
    )
    assert b"template" not in result
    assert b"Wrapper" not in result
    assert b"int w();" in result.replace(b"\n", b"")


def test_replace_type_handles_bases_and_nested_templates():
    # Exercises _declaration_end walking over a template argument list in
    # the base clause and a braced body.
    result = reduce_with(
        [replace_type_with_int],
        b"struct D : B<int, C> { int m; };\nD make();\n",
        lambda x: b"make()" in x,
    )
    assert b"struct D" not in result
    assert b"int make();" in result.replace(b"\n", b"")


def test_replace_type_skips_use_whose_span_overlaps_the_definition():
    # The first use of S consumes a `<...>` span that runs into the
    # first definition of S. Emitting an edit for it would make the
    # candidate self-conflicting (and so useless); the overlapping use
    # is skipped instead, so the definition can still be replaced.
    result = reduce_with(
        [replace_type_with_int],
        b"S< struct S : T > ; struct S : T { }; S y;",
        lambda x: b"{ }" in x and b"y;" in x,
    )
    assert b"struct S" not in result
    assert b"{ }" in result


def test_replace_type_does_not_touch_value_uses_that_break():
    # When the interestingness test needs the type to stay (here it
    # requires the `S s` declaration verbatim), the replace candidate is
    # rejected and the source is left unchanged.
    source = b"struct S { int f(); };\nint g() { S s; return s.f(); }\n"
    result = reduce_with([replace_type_with_int], source, lambda x: b"S s" in x)
    assert b"struct S" in result


# === remove_constructor_initializers ===


def test_removes_whole_initializer_list():
    assert (
        reduce_with(
            [remove_constructor_initializers],
            b"struct P { P() : x(1), y(2) { } int x, y; };",
            lambda x: b"P()" in x,
        )
        == b"struct P { P() { } int x, y; };"
    )


def test_removes_individual_initializer():
    result = reduce_with(
        [remove_constructor_initializers],
        b"struct P { P() : x(1), y(2) { } int x, y; };",
        lambda x: b"y(2)" in x,
    )
    assert b"x(1)" not in result
    assert b"y(2)" in result


# === remove_template_parts ===


def test_removes_template_machinery():
    result = reduce_with(
        [remove_template_parts],
        b"template <typename T, typename U>\nstruct Pair { T a; U b; };\nPair<int, char> p;",
        lambda x: b"Pair" in x and b"p;" in x,
    )
    assert b"template" not in result
    assert b"Pair p;" in result


def test_removes_individual_template_argument():
    result = reduce_with(
        [remove_template_parts],
        b"Pair<int, char> p;",
        lambda x: b"char" in x and b"Pair<" in x,
    )
    # Whitespace cleanup is left to other passes.
    assert result.replace(b" ", b"") == b"Pair<char>p;"


def test_leaves_comparisons_alone():
    source = b"int x = a < b;\nint y = c > d;"
    assert reduce_with([remove_template_parts], source, lambda x: True) == source


# === simplify_call_expressions ===


def test_replaces_call_with_argument():
    assert (
        reduce_with(
            [simplify_call_expressions],
            b"int main() { return wrap(inner(5)); }",
            lambda x: b"5" in x and b"main" in x,
        )
        == b"int main() { return 5; }"
    )


def test_replaces_call_with_zero():
    assert (
        reduce_with(
            [simplify_call_expressions],
            b"int main() { return f(); }",
            lambda x: b"main" in x and (b"return 0;" in x or b"f()" in x),
        )
        == b"int main() { return 0; }"
    )


def test_replaces_method_call_chain():
    assert (
        reduce_with(
            [simplify_call_expressions],
            b"int f() { return obj.method(1) + 2; }",
            lambda x: b"f()" in x and (b"0 + 2" in x or b"method" in x),
        )
        == b"int f() { return 0 + 2; }"
    )


def test_deletes_statement_call_entirely():
    result = reduce_with(
        [simplify_call_expressions],
        b"void f() { log_something(x); done = 1; }",
        lambda x: b"done = 1;" in x and b"void f()" in x,
    )
    assert b"log_something" not in result


def test_control_flow_calls_are_left_alone():
    source = b"void f() { while (g()) { } }"
    result = reduce_with([simplify_call_expressions], source, lambda x: b"while" in x)
    assert b"while (" in result


# === Typedef discovery and inlining ===


def test_finds_simple_typedef():
    typedefs = find_typedefs(token_view(b"typedef unsigned long ul;"))
    assert len(typedefs) == 1
    assert typedefs[0].name == b"ul"
    assert typedefs[0].definition == b"unsigned long"


def test_finds_using_alias():
    typedefs = find_typedefs(token_view(b"using V = std::vector<int>;"))
    assert len(typedefs) == 1
    assert typedefs[0].name == b"V"
    assert typedefs[0].definition == b"std::vector<int>"


def test_skips_function_pointer_typedef():
    assert find_typedefs(token_view(b"typedef int (*fp)(int);")) == []


def test_skips_array_typedef():
    assert find_typedefs(token_view(b"typedef int arr[3];")) == []


def test_skips_using_namespace():
    assert find_typedefs(token_view(b"using namespace std;")) == []


def test_finds_struct_typedef():
    typedefs = find_typedefs(token_view(b"typedef struct Foo Bar;"))
    assert len(typedefs) == 1
    assert typedefs[0].name == b"Bar"
    assert typedefs[0].definition == b"struct Foo"


def test_typedef_inlining_candidate_replaces_uses():
    candidates = list(
        typedef_inlining_candidates(
            b"typedef unsigned long ul;\nul f(ul x) { return x; }\n"
        )
    )
    assert candidates == [b"\nunsigned long f(unsigned long x) { return x; }\n"]


def test_typedef_with_no_uses_produces_no_candidates():
    assert list(typedef_inlining_candidates(b"typedef int unused_t;\nint x;\n")) == []


def test_typedef_inlining_candidates_are_lazy():
    # Each typedef with a use yields one candidate. Producing just the
    # first must not build every full-file rewrite: we count the whole
    # source rewrites (Replacements.apply) and check that the first
    # candidate costs far fewer of them than the whole run.
    source = b"".join(b"typedef int t%d; t%d x%d;\n" % (i, i, i) for i in range(50))
    calls = 0
    original_apply = Replacements.apply

    def counting_apply(self, patch, target):
        nonlocal calls
        calls += 1
        return original_apply(self, patch, target)

    with patch.object(Replacements, "apply", counting_apply):
        candidates = typedef_inlining_candidates(source)
        assert isinstance(candidates, types.GeneratorType)
        next(candidates)
        after_first = calls
        rest = list(candidates)
        after_all = calls
    assert len(rest) == 49
    assert after_first < after_all


def test_inline_typedefs_pump():
    initial = b"typedef unsigned long number;\nnumber f(number x) { return x; }\n"

    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"f(" in x

    async def run() -> bytes:
        problem = BasicReductionProblem(
            initial=initial,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await inline_typedefs(problem)

    assert trio.run(run) == b"\nunsigned long f(unsigned long x) { return x; }\n"


# === Function inlining ===


def test_function_inlining_substitutes_arguments():
    candidates = list(
        function_inlining_candidates(
            b"int sq(int x) { return x * x; }\nint main() { return sq(3); }\n"
        )
    )
    assert candidates == [
        b"int sq(int x) { return x * x; }\nint main() { return ((3) * (3)); }\n"
    ]


def test_function_inlining_produces_candidate_per_call():
    candidates = list(
        function_inlining_candidates(
            b"int sq(int x) { return x * x; }\nint main() { return sq(3) + sq(4); }\n"
        )
    )
    assert len(candidates) == 2


def test_function_inlining_candidates_are_lazy():
    # Producing the first inlined candidate must not build all of them.
    source = b"".join(
        b"int f%d(int x) { return x; }\nint m%d() { return f%d(1); }\n" % (i, i, i)
        for i in range(50)
    )
    calls = 0
    original_apply = Replacements.apply

    def counting_apply(self, patch, target):
        nonlocal calls
        calls += 1
        return original_apply(self, patch, target)

    with patch.object(Replacements, "apply", counting_apply):
        candidates = function_inlining_candidates(source)
        assert isinstance(candidates, types.GeneratorType)
        next(candidates)
        after_first = calls
        rest = list(candidates)
        after_all = calls
    assert len(rest) == 49
    assert after_first < after_all


def test_function_inlining_skips_arity_mismatch():
    assert (
        list(
            function_inlining_candidates(
                b"int sq(int x) { return x * x; }\nint main() { return sq(3, 4); }\n"
            )
        )
        == []
    )


def test_function_inlining_handles_void_parameter_list():
    candidates = list(
        function_inlining_candidates(
            b"int five(void) { return 5; }\nint main() { return five(); }\n"
        )
    )
    assert candidates == [b"int five(void) { return 5; }\nint main() { return (5); }\n"]


def test_function_inlining_skips_recursive_calls():
    assert (
        list(function_inlining_candidates(b"int f(int x) { return f(x - 1); }\n")) == []
    )


def test_function_inlining_skips_multi_statement_bodies():
    assert (
        list(
            function_inlining_candidates(
                b"int f(int x) { g(); return x; }\nint main() { return f(3); }\n"
            )
        )
        == []
    )


def test_inline_function_calls_pump():
    initial = b"int sq(int x) { return x * x; }\nint main() { return sq(3); }\n"

    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"main" in x

    async def run() -> bytes:
        problem = BasicReductionProblem(
            initial=initial,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await inline_function_calls(problem)

    assert (
        trio.run(run)
        == b"int sq(int x) { return x * x; }\nint main() { return ((3) * (3)); }\n"
    )


# === Corpus tests ===
#
# These run the whole C/C++ pass collection over realistic examples in
# tests/corpus/ and check that the critical clang_delta-style
# transformations actually fire on real-looking code.


def corpus(name: str) -> bytes:
    return (Path(__file__).parent / "corpus" / name).read_bytes()


# In real reductions the C/C++ passes always run alongside the generic
# comment removal pass, and asserting on the reduced output is much
# easier with comments gone.
CORPUS_PASSES = [cut_comment_like_things, *CPP_PASSES]


def test_corpus_widget_strips_cpp_structure():
    source = corpus("widget.cpp")

    def is_interesting(x: bytes) -> bool:
        return b"trigger_the_bug" in x

    result = reduce_with(CORPUS_PASSES, source, is_interesting)
    # The irrelevant class template should be gone entirely, along with
    # the namespaces wrapping everything and the inheritance and
    # initializer lists on Widget.
    assert b"namespace" not in result
    assert b"Buffer(size_t capacity, T fill)" not in result
    assert b": public Shape" not in result
    assert b"width_(width)" not in result
    assert b"trigger_the_bug" in result
    assert len(result) < len(source) / 3


def test_corpus_interpreter_reduces_helper_functions():
    source = corpus("interpreter.c")

    def is_interesting(x: bytes) -> bool:
        return b"overflow_site" in x and b"a * b + a" in x

    result = reduce_with(CORPUS_PASSES, source, is_interesting)
    # All the helpers that don't feed the marker can be deleted or
    # hollowed out, while the marker function keeps its body.
    assert b"reg_read" not in result
    assert b"add_wrapping" not in result
    assert b"a * b + a" in result
    assert len(result) < len(source) / 3


def test_corpus_templates_removes_template_machinery():
    source = corpus("templates.cpp")

    def is_interesting(x: bytes) -> bool:
        return b"explode" in x

    result = reduce_with(CORPUS_PASSES, source, is_interesting)
    assert b"sum_all" not in result
    assert b"cast" not in result
    assert b"explode" in result
    assert len(result) < len(source) / 3


def test_corpus_linkage_splices_all_wrappers():
    source = corpus("linkage.cpp")

    def is_interesting(x: bytes) -> bool:
        return b"poison_global = 42" in x

    result = reduce_with(CORPUS_PASSES, source, is_interesting)
    assert b"namespace" not in result
    assert b"extern" not in result
    assert b"helper_one" not in result
    assert b"helper_two" not in result
    assert b"helper_three" not in result
    assert b"poison_global = 42" in result


def test_corpus_interpreter_typedef_inlining():
    source = corpus("interpreter.c")

    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        # Keeping sizeof(VM) intact stops the pump from "inlining" the
        # struct typedef, which would not be valid C.
        return b"overflow_site" in x and b"sizeof(VM)" in x

    async def run() -> bytes:
        problem = BasicReductionProblem(
            initial=source,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await inline_typedefs(problem)

    result = trio.run(run)
    # reg_t -> word_t -> unsigned long, all the way down.
    assert b"reg_t" not in result
    assert b"word_t" not in result
    assert (
        b"static unsigned long overflow_site(unsigned long a, unsigned long b)"
        in result
    )


# === Edge cases in sloppy structure discovery ===


def angle_close_text(source: bytes, open_index: int) -> bytes | None:
    view = token_view(source)
    assert view.tokens[open_index].text == b"<"
    m = _find_angle_close(view, open_index)
    if m is None:
        return None
    return source[view.tokens[open_index].start : view.tokens[m].end]


def test_angle_close_handles_deeply_nested_shift_close():
    assert angle_close_text(b"A<B<C<int>> > x;", 1) == b"<B<C<int>> >"


def test_angle_close_jumps_parenthesised_groups():
    assert angle_close_text(b"A<(1 > 2)> x;", 1) == b"<(1 > 2)>"


def test_angle_close_rejects_unmatched_parens():
    assert angle_close_text(b"A<(1", 1) is None


def test_angle_close_rejects_strings():
    assert angle_close_text(b'a < "b" > c;', 1) is None


def test_angle_close_rejects_end_of_file():
    assert angle_close_text(b"a < b", 1) is None


def _reference_find_angle_close(view: TokenView, i: int) -> int | None:
    """Pre-precompute implementation of _find_angle_close: a fresh
    forward scan from each '<', kept as a reference for the equivalence
    test of the whole-stream construction."""
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


ANGLE_SOUP = st.lists(
    st.sampled_from(
        [
            "<",
            ">",
            ">>",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            ";",
            ",",
            "&&",
            "||",
            "?",
            "a",
            "0",
            '"s"',
            "'c'",
            "<=",
            "<<",
            "\n#d\n",
        ]
    ),
    max_size=40,
).map(lambda parts: " ".join(parts).encode())


@given(ANGLE_SOUP)
def test_angle_close_precompute_matches_reference_scan(source: bytes):
    view = token_view(source)
    for i, t in enumerate(view.tokens):
        if t.kind == PUNCT and t.text == b"<":
            assert view.angle_closes.get(i) == _reference_find_angle_close(view, i)


def test_angle_close_can_escape_crossed_brackets():
    # match_brackets pairs each bracket type independently, so a
    # bracketed jump can escape an enclosing group of a different type.
    source = b"( < [ ) x ] >"
    view = token_view(source)
    assert view.tokens[1].text == b"<"
    assert _find_angle_close(view, 1) == 6


def test_split_on_commas_ignores_unmatched_brackets():
    view = token_view(b"a, (b, c")
    pieces = _split_on_top_level_commas(view, 0, len(view.tokens))
    # The unmatched ( can't be jumped, so its commas still count.
    assert len(pieces) == 3


def test_function_found_after_unmatched_close_brace():
    assert function_names(b"}\nint f() {}") == [b"f"]


def test_function_found_after_access_specifier():
    assert function_names(b"public:\nvoid m() { }") == [b"m"]


def test_unmatched_paren_before_body_is_not_a_function():
    assert function_names(b"int f( { }") == []


def test_empty_template_argument_element_is_skipped():
    result = reduce_with(
        [remove_template_parts],
        b"A<int,,char> x;",
        lambda x: b"char" in x and b"A<" in x,
    )
    assert b"int" not in result


def test_base_list_with_alignas_attribute():
    assert (
        reduce_with(
            [remove_base_classes],
            b"struct alignas(8) A : B { };",
            lambda x: b"A" in x,
        )
        == b"struct alignas(8) A { };"
    )


def test_base_list_scan_tolerates_unmatched_paren():
    assert _find_class_base_lists(token_view(b"class A ( : B {")) == []


def test_base_list_scan_tolerates_unclosed_angle():
    assert _find_class_base_lists(token_view(b"struct A< : B { };")) == []


def test_base_list_scan_tolerates_end_of_file():
    assert _find_class_base_lists(token_view(b"struct A")) == []


def test_replaces_template_id_call():
    assert (
        reduce_with(
            [simplify_call_expressions],
            b"int y = f<int>(5);",
            lambda x: b"5" in x and b"y" in x,
        )
        == b"int y = 5;"
    )


def test_typedef_scan_stops_at_preprocessor_directive():
    assert find_typedefs(token_view(b"typedef int\n#define A 1\nfoo;")) == []


def test_typedef_scan_stops_at_unmatched_bracket():
    assert find_typedefs(token_view(b"typedef int (foo;")) == []


def test_typedef_without_semicolon_is_ignored():
    assert find_typedefs(token_view(b"typedef int foo")) == []


def test_typedef_with_empty_definition_is_ignored():
    assert find_typedefs(token_view(b"typedef foo;")) == []


def test_using_with_empty_definition_is_ignored():
    assert find_typedefs(token_view(b"using V = ;")) == []


def test_function_inlining_skips_empty_body():
    assert (
        list(function_inlining_candidates(b"void f() { }\nint main() { f(); }")) == []
    )


def test_function_inlining_skips_body_with_preprocessor_directive():
    source = b"int f() {\n#define A 1\nreturn 0; }\nint main() { return f(); }"
    assert list(function_inlining_candidates(source)) == []


def test_function_inlining_skips_body_with_unmatched_bracket():
    source = b"int f() { return (x; }\nint main() { return f(); }"
    assert list(function_inlining_candidates(source)) == []


def test_function_inlining_skips_body_with_stray_return():
    source = b"int f() { x return 0; }\nint main() { return f(); }"
    assert list(function_inlining_candidates(source)) == []


def test_function_inlining_skips_variadic_parameters():
    source = b"int f(...) { return 0; }\nint main() { return f(); }"
    assert list(function_inlining_candidates(source)) == []


def test_function_inlining_substitutes_pointer_parameters():
    candidates = list(
        function_inlining_candidates(
            b"int deref(int *p) { return *p; }\nint main() { return deref(q); }"
        )
    )
    assert candidates == [
        b"int deref(int *p) { return *p; }\nint main() { return (*(q)); }"
    ]


def test_candidate_pump_stops_after_adoption_limit():
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return True

    def derive(target: bytes) -> list[bytes]:
        return [target + b"x"]

    pump = _candidate_pump("test_pump", derive)

    async def run() -> bytes:
        problem = BasicReductionProblem(
            initial=b"start",
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await pump(problem)

    assert trio.run(run) == b"start" + b"x" * MAX_PUMP_ADOPTIONS


def test_function_after_unmatched_close_paren():
    assert function_names(b"x ) { }") == []


def test_template_id_without_call_is_left_alone():
    source = b"int x = f<int> + 2;"
    assert reduce_with([simplify_call_expressions], source, lambda x: True) == source


def test_function_inlining_skips_bare_return():
    source = b"void f() { return; }\nint main() { f(); return 0; }"
    assert list(function_inlining_candidates(source)) == []


def test_candidate_pump_skips_already_seen_candidates():
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return True

    calls = []

    def derive(target: bytes) -> list[bytes]:
        calls.append(target)
        if len(calls) == 1:
            return [target + b"x"]
        # The original test case is already in the seen set, so this
        # candidate is skipped and the pump stops.
        return [b"start"]

    pump = _candidate_pump("test_pump", derive)

    async def run() -> bytes:
        problem = BasicReductionProblem(
            initial=b"start",
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await problem.setup()
        return await pump(problem)

    assert trio.run(run) == b"startx"
    assert len(calls) == 2


# === Robustness ===
#
# The C/C++ passes run on whatever bytes the user gives shrink ray, so
# they must never crash, no matter how mangled the input.

CPPISH_SOUP = st.text(
    alphabet="ab<>(){}[],;:*&=#\"'\\/\n .0t~$",
    max_size=60,
).map(str.encode) | st.binary(max_size=60)


@given(CPPISH_SOUP)
def test_cpp_passes_never_crash_on_arbitrary_input(source: bytes):
    for reduction_pass in CPP_PASSES:
        reduce_with([reduction_pass], source, lambda x: x == source)


@given(CPPISH_SOUP)
def test_cpp_candidate_generators_never_crash_on_arbitrary_input(source: bytes):
    list(typedef_inlining_candidates(source))
    list(function_inlining_candidates(source))
