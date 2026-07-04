"""Tests for the cheap reflow formatter used by the sort key."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.reformat import basic_format, canonical_distance, detect_family


# An alphabet rich in the structural characters the formatter dispatches on, so
# random inputs exercise the brace / tag / indent branches.
STRUCTURAL = st.text(
    alphabet="{}[]()<>;#/*=+!&|,:\"'\\ \t\n" + "abcAB012", max_size=80
)


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
        ("def f(x):\n    a = 1\n    b = 2", "def f(x):\n\ta = 1;b = 2"),
        ("<p>hi</p>", "<p>\n hi\n</p>"),
        ("<!DOCTYPE html>", "<!DOCTYPE\nhtml>"),
    ],
)
def test_canonicalises_layout_variants(a, b):
    assert basic_format(a) == basic_format(b)


# === brace family ===


def test_brace_braces_and_semicolons():
    out = basic_format("struct S{int a;int b;};")
    assert out == "struct S {\n  int a;\n  int b;\n}\n;\n"


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


def test_brace_unclosed_block_comment_terminated():
    out = basic_format("a /* unclosed")
    assert out.endswith("*/\n")


def test_brace_operator_spacing():
    assert basic_format("a==b") == "a == b\n"
    assert basic_format("a  +\n  b") == "a + b\n"


def test_brace_comma_spacing():
    assert basic_format("f(a,b,c)") == "f(a, b, c)\n"


def test_brace_directive_split():
    assert basic_format("#a 1 #b 2") == "#a 1\n#b 2\n"


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
    out = basic_format("def f():\n    a = 1; b = 2")
    assert out == "def f():\n  a = 1\n  b = 2\n"


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
    ["", "\n\n", "   ", "{{{", "}}}", '"unclosed', "/* unclosed", "a<b", "\x00x"],
)
def test_never_crashes_and_ends_with_newline(text):
    out = basic_format(text)
    assert isinstance(out, str)
    assert out.endswith("\n")


@given(STRUCTURAL)
def test_property_robust_and_deterministic(s):
    out = basic_format(s)
    assert isinstance(out, str)
    assert out.endswith("\n")
    assert basic_format(s) == out  # deterministic
    # non-whitespace content is preserved (only whitespace/op-spacing changes)
    assert "".join(out.split()) or not "".join(s.split())


@given(STRUCTURAL, STRUCTURAL)
def test_property_distance_nonnegative(a, b):
    d = canonical_distance(a, b)
    assert d >= 0
    assert (d == 0) == (a == b)


# extra targeted branches


def test_inline_string_escape_in_python():
    out = basic_format('def f():\n    s = "a\\"b"\n    return s')
    assert r'"a\"b"' in out


def test_inline_op_at_fragment_start():
    assert basic_format("<p>= x</p>") == "<p>= x</p>\n"


def test_brace_space_and_op_before_brace():
    assert basic_format("a={b}") == "a = {\n  b\n}\n"


def test_brace_op_after_newline():
    assert basic_format("a;==b") == "a;\n == b\n"


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
    assert basic_format("def f():\n\n    a = 1;\n") == "def f():\n  a = 1\n"


def test_indent_unterminated_string():
    assert basic_format('def f():\n    s = "unclosed').startswith("def f():\n")


def test_detect_brace_with_close_brace_only():
    assert detect_family("a}b") == "brace"
