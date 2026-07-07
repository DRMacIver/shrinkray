"""Tests for the cheap reflow formatter used by the sort key."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.formatting import default_reformat_data
from shrinkray.reformat import basic_format, canonical_distance, detect_family


# An alphabet rich in the structural characters the formatter dispatches on, so
# random inputs exercise the brace / tag / indent branches.
STRUCTURAL = st.text(alphabet="{}[]()<>;#/*=+!&|,:\"'\\ \t\n" + "abcAB012", max_size=80)


# === family detection ===


@pytest.mark.parametrize(
    "text, family",
    [
        ("int main(void){}", "brace"),
        ("{}", "brace"),
        ("x = a + b", "brace"),
        ("SELECT a FROM t;", "brace"),
        ("def f(x):\n    return x", "indent"),
        ("if a:\n\tb", "indent"),
        ("<html><body></body></html>", "tag"),
        ("<!DOCTYPE html>", "tag"),
        # a dict literal with a block still counts as Python (indentation rules)
        ("def f():\n    d = {1: 2}\n    return d", "indent"),
        # tags with braces are treated as brace (templates / embedded code)
        ("<script>{a}</script>", "brace"),
    ],
)
def test_detect_family(text, family):
    assert detect_family(text) == family


# === canonicalisation: layout variants map to one string ===


@pytest.mark.parametrize(
    "a, b",
    [
        ("int main(void){}", "int\nmain(void)\n{}"),
        ("void f() {}", "void f()\n\n{}"),
        ("z = a + b", "z = a +\nb"),
        ("#define A 1\n#define B 2", "#define A 1 #define B 2"),
        ("def f(x):\n    a = 1\n    b = 2", "def f(x):\n\ta = 1\n\tb = 2"),
        ("<p>hi</p>", "<p>\n hi\n</p>"),
        ("<!DOCTYPE html>", "<!DOCTYPE\nhtml>"),
    ],
)
def test_canonicalises_layout_variants(a, b):
    assert basic_format(a) == basic_format(b)


# === brace family ===


def test_brace_braces_and_semicolons():
    out = basic_format("struct S{int a;int b;};")
    assert out == "struct S {\n  int a;\n  int b;\n};\n"


def test_brace_nested_indentation():
    # closing braces dedent correctly (indentation reflects depth after dedent)
    assert basic_format("{ { x; } }") == "{\n  {\n    x;\n  }\n}\n"


def test_brace_empty_braces_stay_together():
    assert basic_format("function() {}") == "function() {}\n"
    assert basic_format("f() { }") == "f() {}\n"


def test_brace_semicolon_attaches_to_close_brace():
    # a struct/class/enum trailing ';' stays on the '}' line
    assert basic_format("struct S{int a;};") == "struct S {\n  int a;\n};\n"
    assert basic_format("} ;") == "};\n"


def test_brace_drops_operator_trailing_space_at_line_break():
    # An operator emits a trailing space in case an operand follows; when a
    # line break arrives instead (here from '}'), newline() must pop that
    # dangling space rather than leave "a = \n". This input pins coverage of
    # that path, which hypothesis inputs only sometimes reach.
    assert basic_format("a = }") == "a =\n}\n"


def test_brace_preserves_string_literal_with_specials():
    # braces / ; / operators inside a string are untouched
    out = basic_format('x = "a{b;c+d}"')
    assert '"a{b;c+d}"' in out


def test_brace_preserves_escaped_quote_in_string():
    out = basic_format(r'x = "a\"b"')
    assert r'"a\"b"' in out


def test_brace_line_comment():
    out = basic_format("a; // trailing\nb;")
    assert "// trailing" in out
    assert out.count("\n") >= 2


def test_brace_block_comment_preserved():
    out = basic_format("a/* c ; { } */b")
    assert "/* c ; { } */" in out


def test_brace_unclosed_block_comment_preserved():
    # an unterminated block comment is kept verbatim (the formatter never adds
    # non-whitespace content such as a closing "*/")
    out = basic_format("a /* unclosed")
    assert "/* unclosed" in out and out.endswith("unclosed\n")


def test_brace_operator_spacing():
    assert basic_format("a==b") == "a == b\n"
    assert basic_format("a  +\n  b") == "a + b\n"


def test_brace_operator_trailing_space_stripped_before_newline():
    # An operator emits a trailing space; when the next token forces a
    # newline (here a closing brace), that dangling space is stripped
    # rather than left at the end of the line.
    assert basic_format("a+}") == "a +\n}\n"
    assert basic_format("a&&}") == "a &&\n}\n"


def test_brace_comma_spacing():
    assert basic_format("f(a,b,c)") == "f(a, b, c)\n"


def test_brace_directive_split():
    assert basic_format("#a 1 #b 2") == "#a 1\n#b 2\n"


def test_brace_directive_ends_at_newline():
    # A preprocessor directive must terminate at its own newline: the code that
    # follows on the next line is ordinary brace content, not part of the
    # directive. Previously the '#' swallowed everything up to the next '#'.
    assert (
        basic_format("#include <stdio.h>\nint main() { return 0; }\n")
        == "#include <stdio.h>\nint main() {\n  return 0;\n}\n"
    )
    assert (
        basic_format("#define FOO 1\n#define BAR 2\nint x = FOO;\n")
        == "#define FOO 1\n#define BAR 2\nint x = FOO;\n"
    )


def test_brace_directive_keeps_internal_space():
    # Inside a directive a run of whitespace between tokens is significant and
    # must be kept (collapsed to a single space), unlike the whitespace next to
    # punctuation the brace family drops elsewhere.
    assert basic_format("#define FOO   BAR") == "#define FOO BAR\n"
    # '#include <x>' keeps its space too (braces force the brace family here).
    assert basic_format("#include <x>\n{}") == "#include <x>\n{}\n"


def test_brace_directive_ends_at_newline_even_after_backslash():
    # A directive ends at its newline; a trailing backslash is copied verbatim
    # like any other character (no line-continuation join). Joining would make
    # the output re-read as a continuation on the next pass, breaking the fixed
    # point, so the continued line becomes ordinary content instead.
    assert basic_format("#define A \\\n  B") == "#define A \\\nB\n"


def test_brace_directive_preserves_string_literal():
    # A '#' inside a string in the directive body must not end the directive.
    out = basic_format('#error "a # b"\nx;')
    assert out == '#error "a # b"\nx;\n'


# === tag family ===


def test_tag_nesting_and_indent():
    out = basic_format("<a><b>x</b></a>")
    assert out == "<a>\n  <b>x</b>\n</a>\n"


def test_tag_self_closing_and_void():
    assert "<br/>" in basic_format("<div><br/></div>")
    # void element has no close tag; must not increase depth forever
    out = basic_format("<div><img src='x'><span>y</span></div>")
    assert "<img src='x'>" in out
    assert "<span>y</span>" in out


def test_tag_comment_and_declaration_and_pi():
    assert "<!-- c -->" in basic_format("<a><!-- c --></a>")
    assert basic_format("<!DOCTYPE   html>") == "<!DOCTYPE html>\n"
    assert "<?xml version='1.0'?>" in basic_format("<?xml version='1.0'?><a/>")


def test_tag_leaf_inline_vs_nested():
    # leaf (text only) stays inline; element with children expands
    assert basic_format("<p>hello world</p>") == "<p>hello world</p>\n"
    assert "\n" in basic_format("<ul><li>a</li></ul>").rstrip("\n")


def test_tag_text_node():
    out = basic_format("<p>a b   c</p>")
    assert out == "<p>a b c</p>\n"


def test_tag_unclosed():
    # a stray '<' with no '>' should not crash
    assert basic_format("<div").startswith("<div")


def test_tag_empty_produces_newline():
    assert basic_format("<>") == "<>\n" or basic_format("<>").endswith("\n")


# === indentation (Python) family ===


def test_indent_normalises_tabs_and_widths():
    a = basic_format("def f():\n    a = 1\n    b = 2")
    b = basic_format("def f():\n\ta = 1\n\tb = 2")
    assert a == b == "def f():\n  a = 1\n  b = 2\n"


def test_indent_dedent():
    out = basic_format("def f():\n    if a:\n        b\n    c")
    assert out == "def f():\n  if a:\n    b\n  c\n"


def test_indent_semicolon_split():
    # One statement per line, but the ';' is preserved (basic_format only
    # rewrites whitespace; it never deletes a non-whitespace character).
    out = basic_format("def f():\n    a = 1; b = 2")
    assert out == "def f():\n  a = 1;\n  b = 2\n"


def test_indent_semicolon_split_preserves_semicolons():
    # Every ';' present in the source survives the one-statement-per-line split,
    # including a trailing one and an empty statement.
    assert (
        basic_format("def f():\n    a = 1;;b = 2;")
        == "def f():\n  a = 1;\n  ;\n  b = 2;\n"
    )


def test_indent_comment_is_not_split_on_semicolon():
    # A ';' inside a '#' comment must not be treated as a statement separator:
    # the comment text stays on the comment's line rather than being moved onto
    # its own line as code.
    assert basic_format("if x:\n    y  # a; b") == "if x:\n  y#a;b\n"


def test_indent_string_spanning_brackets():
    # a newline inside brackets/strings does not split the logical line
    out = basic_format("def f():\n    x = (1 +\n         2)")
    assert out == "def f():\n  x = (1 + 2)\n"


def test_indent_string_literal_preserved():
    out = basic_format('def f():\n    s = "a:b;c"\n    return s')
    assert '"a:b;c"' in out


def test_indent_inconsistent_dedent():
    # dedent to a width that matches no prior level is treated as a new level
    out = basic_format("def f():\n        a\n     b")
    assert out.startswith("def f():\n")


# === canonical_distance ===


def test_distance_zero_when_equal():
    assert canonical_distance("abc", "abc") == 0


def test_distance_counts_whitespace_edits():
    # x has a newline where canonical has a space -> one substitution
    assert canonical_distance("a\nb", "a b") == 1
    # x missing a space the canonical has -> one insertion
    assert canonical_distance("ab", "a b") == 1
    # x has an extra space -> one deletion
    assert canonical_distance("a  b", "a b") == 1


def test_distance_non_whitespace_mismatch_and_tails():
    assert canonical_distance("abc", "abd") == 1  # substitution
    assert canonical_distance("abcd", "ab") == 2  # x longer
    assert canonical_distance("ab", "abcd") == 2  # canonical longer


# === robustness ===


@pytest.mark.parametrize(
    "text",
    ["\n\n", "   ", "{{{", "}}}", '"unclosed', "/* unclosed", "a<b", "\x00x"],
)
def test_never_crashes_and_ends_with_newline(text):
    out = basic_format(text)
    assert isinstance(out, str)
    assert out.endswith("\n")


@given(STRUCTURAL)
def test_property_robust_and_deterministic(s):
    out = basic_format(s)
    assert isinstance(out, str)
    if s:
        assert out.endswith("\n")
    else:
        assert out == ""
    assert basic_format(s) == out  # deterministic


@given(STRUCTURAL, STRUCTURAL)
def test_property_distance_nonnegative(a, b):
    d = canonical_distance(a, b)
    assert d >= 0
    assert (d == 0) == (a == b)


# === robustness against ARBITRARY textual input ===
#
# basic_format must behave sanely on any input, even text that corresponds to
# none of the known families (prose, unicode, binary-ish, control characters).
# These properties use unrestricted st.text()/st.binary(), not the structural
# alphabet, to hammer the formatter with inputs it was not designed around.

ARBITRARY = st.text(max_size=200)


@given(ARBITRARY)
def test_property_arbitrary_text_never_crashes(s):
    out = basic_format(s)
    assert isinstance(out, str)
    if s:
        assert out.endswith("\n")
    else:
        assert out == ""


def test_empty_input_formats_to_empty():
    # The empty string must be its own canonical form: if it mapped to "\n"
    # (like whitespace-only inputs do), the reflow sort key would rank ""
    # *above* whitespace-only strings, and the empty test case would no
    # longer be the global minimum of the reduction ordering.
    assert basic_format("") == ""


@given(ARBITRARY)
def test_property_arbitrary_text_deterministic(s):
    assert basic_format(s) == basic_format(s)


@given(ARBITRARY)
def test_property_arbitrary_text_is_idempotent(s):
    once = basic_format(s)
    assert basic_format(once) == once


def test_whitespace_collapse_does_not_fuse_operators():
    # Dropping the space between two single punctuation characters must not fuse
    # them into a longer operator token that a second pass would then re-space.
    # 'a& &b' is two '&' tokens; it must NOT become '&&' (which would re-detect
    # as the '&&' operator on the next pass, breaking idempotence).
    assert basic_format("a& &b") == "a& &b\n"
    assert basic_format("a& &b") == basic_format(basic_format("a& &b"))
    # The genuine '&&' operator (no intervening space) is still spaced.
    assert basic_format("a&&b") == "a && b\n"
    # Same mechanism in the inline (tag/Python) normaliser.
    assert basic_format("<p>a& &b</p>") == "<p>a& &b</p>\n"
    assert basic_format("<p>a&&b</p>") == "<p>a && b</p>\n"


def test_whitespace_collapse_does_not_fuse_comment_introducers():
    # Collapsing '/ /' -> '//' or '/ *' -> '/*' would start a comment that the
    # next pass tokenises differently, so those spaces are kept.
    assert basic_format("a/ /b") == "a/ /b\n"
    assert basic_format("a/ *b") == "a/ *b\n"


def test_whitespace_collapse_does_not_manufacture_a_tag():
    # Dropping the space between '<' and a tag-name character would splice them
    # into a tag that re-detects as the tag family on the next pass. When a '>'
    # follows (so a tag really would form) the space is kept.
    assert basic_format("< b>") == "< b>\n"
    assert basic_format("< b>") == basic_format(basic_format("< b>"))
    # With no following '>' there is no tag risk, so the space still collapses.
    assert basic_format("< b") == "<b\n"


def test_dict_colon_in_brackets_is_not_a_python_block():
    # A ':' inside brackets (a dict/slice literal) is not a block header colon,
    # so an input whose only colon is bracket-enclosed must stay in the brace
    # family. Otherwise it detects as Python once, flattens to brace-looking
    # output, and then re-detects as brace on the next pass (a family flip).
    src = 'd = {"x":\n     1}'
    assert detect_family(src) == "brace"
    once = basic_format(src)
    assert basic_format(once) == once


@given(ARBITRARY)
def test_property_arbitrary_text_preserves_non_whitespace(s):
    # The formatter only ever rewrites whitespace (and inserts spacing around
    # operators): the sequence of non-whitespace characters is never changed,
    # so no content is dropped, added, or reordered -- including for unicode
    # letters, which must not be merged across a space.
    assert "".join(basic_format(s).split()) == "".join(s.split())


@given(ARBITRARY, ARBITRARY)
def test_property_distance_arbitrary_text(a, b):
    d = canonical_distance(a, b)
    assert d >= 0
    assert (d == 0) == (a == b)


@given(st.binary(max_size=200))
def test_property_binary_reformat_never_crashes(data):
    # default_reformat_data is the language-agnostic output formatter; it must
    # accept arbitrary bytes (undecodable data is returned unchanged).
    out = default_reformat_data(data)
    assert isinstance(out, bytes)


def test_unicode_words_are_not_merged():
    # a space between two unicode identifier characters is significant
    assert basic_format("café résumé") == "café résumé\n"
    assert basic_format("день ночь") == "день ночь\n"


# extra targeted branches


def test_inline_string_escape_in_python():
    out = basic_format('def f():\n    s = "a\\"b"\n    return s')
    assert r'"a\"b"' in out


def test_inline_op_at_fragment_start():
    assert basic_format("<p>= x</p>") == "<p>= x</p>\n"


def test_brace_space_and_op_before_brace():
    assert basic_format("a={b}") == "a = {\n  b\n}\n"


def test_brace_op_after_newline():
    # an operator at the start of a line is not given a spurious leading space
    assert basic_format("a;==b") == "a;\n== b\n"


def test_brace_double_op_pops_trailing_space():
    # a second operator whose predecessor left a trailing space exercises the pop
    assert basic_format("a====b") == "a == == b\n"


def test_inline_double_op_pops_trailing_space():
    assert basic_format("<p>a====b</p>") == "<p>a == == b</p>\n"


def test_brace_line_comment_trailing_space():
    # the comment copies the trailing space; the following newline pops it
    out = basic_format("a// c \nb")
    assert "// c" in out and out.endswith("b\n")


def test_tag_whitespace_and_top_level_text():
    assert basic_format("<a/> <b/>") == "<a/>\n<b/>\n"
    assert basic_format("hi <a/>") == "hi\n<a/>\n"


def test_tag_stray_close_at_depth_zero():
    assert basic_format("</a>") == "</a>\n"


def test_indent_semicolon_in_string_not_split():
    out = basic_format('def f():\n    s = "a;b"; t = 1')
    assert '"a;b"' in out and "t = 1" in out


def test_indent_blank_line_and_trailing_semicolon_and_newline():
    # basic_format canonicalises whitespace only, so the trailing ';' is kept
    # (blank lines and the trailing newline are still normalised away).
    assert basic_format("def f():\n\n    a = 1;\n") == "def f():\n  a = 1;\n"


def test_indent_unterminated_string():
    assert basic_format('def f():\n    s = "unclosed').startswith("def f():\n")


def test_detect_brace_with_close_brace_only():
    assert detect_family("a}b") == "brace"


def test_detect_family_colon_needs_strictly_deeper_body():
    # A ':'-terminated line is only a Python block if the next non-blank line is
    # *more* indented than it. This keeps reflowed brace output (uniform
    # per-depth indentation) and C/C++ labels out of the indent family.
    assert detect_family("if a:\n    b") == "indent"  # deeper body -> block
    assert detect_family("a:\nb") == "brace"  # same indent -> not a block
    assert detect_family("{\n  :\n") == "brace"  # label colon, only blanks after
    assert detect_family("class C {\n  public:\n  int x;\n}") == "brace"
    # A header colon with only blank lines after it is not a block (no body).
    assert detect_family("x:\n\n") == "brace"


def test_colon_inside_multiline_string_is_not_a_block_header():
    # A ':' at the end of an (unterminated) string that spans physical lines is
    # inside the string, not a block header. Attributing it to an earlier line
    # would misdetect the input as Python and flip families on the next pass.
    src = ' |0]"cb*&|\n\t\t{B<}b0:'
    assert detect_family(src) == "brace"
    once = basic_format(src)
    assert basic_format(once) == once


@pytest.mark.parametrize(
    "text",
    [
        "{:",  # the Hypothesis-found idempotency counterexample
        "{\n  :\n",  # its own once-reflowed output
        "class C{public:int x;};",  # C++ access specifier
        "case 1:\nx;",  # C label colon at end of line
    ],
)
def test_brace_output_is_idempotent(text):
    # Reflowed brace output must not re-detect as Python, or a second pass would
    # change it (basic_format must be a fixed point).
    once = basic_format(text)
    assert basic_format(once) == once
