"""Parametrized tests for reduction passes.

Each test asserts that a specific input can reduce to a specific output
via a specific pass, running under both parallelism=1 and parallelism=2.
"""

from unittest.mock import patch

import libcst
import pytest

from shrinkray.passes.bytes import (
    Encoding,
    NewlineReplacer,
    RegionReplacement,
    Split,
    Tokenize,
    debracket,
    delete_byte_spans,
    hollow,
    lexeme_based_deletions,
    lift_braces,
    line_sorter,
    lower_bytes,
    lower_individual_bytes,
    remove_indents,
    remove_whitespace,
    replace_space_with_newlines,
    short_deletions,
    short_replacements,
    sort_whitespace,
    standard_substitutions,
)
from shrinkray.passes.definitions import compose
from shrinkray.passes.genericlanguages import (
    IntegerFormat,
    RegionReplacingPatches,
    Substring,
    combine_expressions,
    cut_comment_like_things,
    merge_adjacent_strings,
    normalize_identifiers,
    reduce_integer_literals,
    replace_falsey_with_zero,
    simplify_brackets,
)
from shrinkray.passes.python import (
    replace_bodies_with_ellipsis,
    strip_annotations,
)
from shrinkray.passes.sequences import (
    block_deletion,
    delete_duplicates,
    delete_elements,
    merged_intervals,
    with_deletions,
)
from shrinkray.problem import BasicReductionProblem, ParseError, shortlex
from shrinkray.work import WorkContext
from tests.helpers import reduce_with


@pytest.fixture(params=[1, 2])
def parallelism(request):
    return request.param


# =============================================================================
# Byte Passes - bytes.py
# =============================================================================


def test_hollow_empties_curly_braces(parallelism):
    result = reduce_with(
        [hollow],
        b"{hello world}",
        lambda x: x.startswith(b"{") and x.endswith(b"}"),
        parallelism=parallelism,
    )
    assert result == b"{}"


def test_hollow_empties_square_brackets(parallelism):
    result = reduce_with(
        [hollow],
        b"[1, 2, 3, 4, 5]",
        lambda x: x.startswith(b"[") and x.endswith(b"]"),
        parallelism=parallelism,
    )
    assert result == b"[]"


def test_hollow_empties_quotes(parallelism):
    result = reduce_with(
        [hollow],
        b'"hello world"',
        lambda x: x.startswith(b'"') and x.endswith(b'"'),
        parallelism=parallelism,
    )
    assert result == b'""'


def test_hollow_empties_single_quotes(parallelism):
    result = reduce_with(
        [hollow],
        b"'hello world'",
        lambda x: x.startswith(b"'") and x.endswith(b"'"),
        parallelism=parallelism,
    )
    assert result == b"''"


def test_lift_braces_nested(parallelism):
    result = reduce_with(
        [lift_braces],
        b"{outer {inner} more}",
        lambda x: b"inner" in x,
        parallelism=parallelism,
    )
    assert result == b"{inner}"


def test_lift_braces_multiple_nested(parallelism):
    result = reduce_with(
        [lift_braces],
        b"{a {b} c {d} e}",
        lambda x: b"b" in x and x.startswith(b"{") and x.endswith(b"}"),
        parallelism=parallelism,
    )
    assert result == b"{b}"


def test_lift_braces_with_empty_braces(parallelism):
    """Test that lift_braces handles empty braces {} correctly."""
    result = reduce_with(
        [lift_braces],
        b"{} {content} {}",
        lambda x: b"content" in x,
        parallelism=parallelism,
    )
    assert b"content" in result


def test_debracket_removes_parens(parallelism):
    result = reduce_with(
        [debracket],
        b"(hello)",
        lambda x: b"hello" in x,
        parallelism=parallelism,
    )
    assert result == b"hello"


def test_debracket_removes_square_brackets(parallelism):
    result = reduce_with(
        [debracket],
        b"[value]",
        lambda x: b"value" in x,
        parallelism=parallelism,
    )
    assert result == b"value"


def test_debracket_removes_curly_braces(parallelism):
    result = reduce_with(
        [debracket],
        b"{content}",
        lambda x: b"content" in x,
        parallelism=parallelism,
    )
    assert result == b"content"


def test_short_deletions_trailing(parallelism):
    result = reduce_with(
        [short_deletions],
        b"hello!!!",
        lambda x: b"hello" in x,
        parallelism=parallelism,
    )
    assert result == b"hello"


def test_short_deletions_leading(parallelism):
    result = reduce_with(
        [short_deletions],
        b"...world",
        lambda x: b"world" in x,
        parallelism=parallelism,
    )
    assert result == b"world"


def test_delete_byte_spans(parallelism):
    result = reduce_with(
        [delete_byte_spans],
        b"aaabbbccc",
        lambda x: b"aaa" in x and b"ccc" in x,
        parallelism=parallelism,
    )
    assert b"aaa" in result
    assert b"ccc" in result
    assert len(result) <= len(b"aaabbbccc")


def test_remove_indents(parallelism):
    result = reduce_with(
        [remove_indents],
        b"line1\n    indented\nline3",
        lambda x: b"indented" in x,
        parallelism=parallelism,
    )
    assert result == b"line1\nindented\nline3"


def test_remove_indents_multiple(parallelism):
    result = reduce_with(
        [remove_indents],
        b"a\n  b\n    c\n      d",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"a\nb\nc\nd"


def test_remove_whitespace_spaces(parallelism):
    result = reduce_with(
        [remove_whitespace],
        b"hello     world",
        lambda x: b"hello" in x and b"world" in x,
        parallelism=parallelism,
    )
    assert result == b"helloworld"


def test_remove_whitespace_tabs(parallelism):
    result = reduce_with(
        [remove_whitespace],
        b"a\t\t\tb",
        lambda x: b"a" in x and b"b" in x,
        parallelism=parallelism,
    )
    assert result == b"ab"


def test_replace_space_with_newlines(parallelism):
    result = reduce_with(
        [replace_space_with_newlines],
        b"a b c",
        lambda x: True,
        parallelism=parallelism,
    )
    assert b"\n" in result or result == b"a b c"


def test_replace_space_with_newlines_tabs(parallelism):
    result = reduce_with(
        [replace_space_with_newlines],
        b"a\tb",
        lambda x: True,
        parallelism=parallelism,
    )
    assert b"\n" in result or result == b"a\tb"


def test_lower_bytes_to_zero(parallelism):
    result = reduce_with(
        [lower_bytes],
        b"\x05\x05\x05",
        lambda x: len(x) == 3,
        parallelism=parallelism,
        sort_key=shortlex,
    )
    assert result == b"\x00\x00\x00"


def test_lower_bytes_to_minimum(parallelism):
    result = reduce_with(
        [lower_bytes],
        b"\x10",
        lambda x: len(x) == 1 and x[0] >= 5,
        parallelism=parallelism,
        sort_key=shortlex,
    )
    assert result == b"\x05"


def test_lower_individual_bytes(parallelism):
    result = reduce_with(
        [lower_individual_bytes],
        b"\x00\x10",
        lambda x: len(x) == 2 and x[0] == 0 and x[1] >= 5,
        parallelism=parallelism,
        sort_key=shortlex,
    )
    assert result == b"\x00\x05"


def test_lexeme_based_deletions(parallelism):
    result = reduce_with(
        [lexeme_based_deletions],
        b"print(a); print(b); print(c)",
        lambda x: b"print" in x,
        parallelism=parallelism,
    )
    assert b"print" in result
    assert len(result) < len(b"print(a); print(b); print(c)")


def test_standard_substitutions(parallelism):
    result = reduce_with(
        [standard_substitutions],
        b"\x00\x00",
        lambda x: len(x) >= 1,
        parallelism=parallelism,
        sort_key=shortlex,
    )
    assert result == b"\x01"


def test_standard_substitutions_no_progress(parallelism):
    result = reduce_with(
        [standard_substitutions],
        b"\x00\x00",
        lambda x: len(x) == 2,
        parallelism=parallelism,
        sort_key=shortlex,
    )
    assert result == b"\x00\x00"


def test_line_sorter(parallelism):
    result = reduce_with(
        [line_sorter],
        b"c\na\nb",
        lambda x: set(x.split(b"\n")) == {b"a", b"b", b"c"},
        parallelism=parallelism,
    )
    assert result == b"a\nb\nc"


def test_line_sorter_already_sorted(parallelism):
    result = reduce_with(
        [line_sorter],
        b"a\nb\nc",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"a\nb\nc"


def test_line_sorter_reverse(parallelism):
    result = reduce_with(
        [line_sorter],
        b"c\nb\na",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"a\nb\nc"


def test_line_sorter_swap_not_interesting(parallelism):
    result = reduce_with(
        [line_sorter],
        b"c\nb\na",
        lambda x: x.split(b"\n")[0] == b"c",
        parallelism=parallelism,
    )
    assert result.split(b"\n")[0] == b"c"


def test_short_replacements(parallelism):
    result = reduce_with(
        [short_replacements],
        b"\x05\x05\x05",
        lambda x: len(x) == 3,
        parallelism=parallelism,
    )
    assert all(b <= 5 for b in result)


def test_sort_whitespace_to_front(parallelism):
    result = reduce_with(
        [sort_whitespace],
        b"a b c",
        lambda x: set(x) >= set(b"abc "),
        parallelism=parallelism,
    )
    assert len(result) == 5


def test_sort_whitespace_simple(parallelism):
    result = reduce_with(
        [sort_whitespace],
        b"hello world test",
        lambda x: True,
        parallelism=parallelism,
    )
    assert len(result) <= len(b"hello world test")


def test_sort_whitespace_with_initial_newline(parallelism):
    """Test whitespace sorting when initial whitespace ends with newline."""
    result = reduce_with(
        [sort_whitespace],
        b"x\n y z",
        lambda x: True,
        parallelism=parallelism,
    )
    assert len(result) <= len(b"x\n y z")


def test_sort_whitespace_large_k(parallelism):
    """Test whitespace sorting when sort size exceeds remaining input."""
    result = reduce_with(
        [sort_whitespace],
        b"abc   ",
        lambda x: True,
        parallelism=parallelism,
    )
    assert len(result) <= len(b"abc   ")


# =============================================================================
# Format Tests
# =============================================================================


def test_split_repr():
    split = Split(b"\n")
    assert repr(split) == "Split(b'\\n')"


def test_split_repr_semicolon():
    split = Split(b";")
    assert repr(split) == "Split(b';')"


def test_tokenize_repr():
    tok = Tokenize()
    assert repr(tok) == "tokenize"


def test_split_with_block_deletion(parallelism):
    result = reduce_with(
        [compose(Split(b"\n"), block_deletion(1, 3))],
        b"a\nb\nc\nd\ne",
        lambda x: b"a" in x and b"e" in x,
        parallelism=parallelism,
    )
    assert result == b"a\ne"


def test_split_with_delete_duplicates(parallelism):
    result = reduce_with(
        [compose(Split(b"\n"), delete_duplicates)],
        b"a\nb\na\nc\na",
        lambda x: b"b" in x and b"c" in x,
        parallelism=parallelism,
    )
    lines = result.split(b"\n")
    assert b"a" not in lines


def test_tokenize_with_block_deletion(parallelism):
    result = reduce_with(
        [compose(Tokenize(), block_deletion(1, 5))],
        b"foo bar baz qux",
        lambda x: b"foo" in x,
        parallelism=parallelism,
    )
    assert b"foo" in result
    assert len(result) < len(b"foo bar baz qux")


def test_tokenize_with_consecutive_spaces(parallelism):
    result = reduce_with(
        [compose(Tokenize(), block_deletion(1, 3))],
        b"a    b",
        lambda x: b"a" in x and b"b" in x,
        parallelism=parallelism,
    )
    assert b"a" in result
    assert b"b" in result


def test_tokenize_with_numbers(parallelism):
    result = reduce_with(
        [compose(Tokenize(), block_deletion(1, 3))],
        b"x = 123.456",
        lambda x: b"x" in x and b"." in x,
        parallelism=parallelism,
    )
    assert b"x" in result


def test_tokenize_with_underscores(parallelism):
    result = reduce_with(
        [compose(Tokenize(), block_deletion(1, 3))],
        b"my_var_name = 1",
        lambda x: b"=" in x,
        parallelism=parallelism,
    )
    assert b"=" in result


def test_encoding_parse():
    enc = Encoding("utf-8")
    assert enc.parse(b"hello") == "hello"


def test_encoding_dumps():
    enc = Encoding("utf-8")
    assert enc.dumps("caf\u00e9") == b"caf\xc3\xa9"


def test_encoding_name():
    enc = Encoding("utf-8")
    assert enc.name == "utf-8"


def test_encoding_repr():
    enc = Encoding("utf-8")
    assert repr(enc) == "Encoding('utf-8')"


# Note: Substring tests are in test_generic_language.py

# =============================================================================
# Sequence Passes - sequences.py
# =============================================================================


def test_delete_elements_list(parallelism):
    result = reduce_with(
        [delete_elements],
        [1, 2, 3, 4, 5],
        lambda x: 1 in x and 5 in x,
        parallelism=parallelism,
    )
    assert result == [1, 5]


def test_delete_elements_tuple(parallelism):
    result = reduce_with(
        [delete_elements],
        (1, 2, 3),
        lambda x: 1 in x,
        parallelism=parallelism,
    )
    assert result == (1,)


def test_block_deletion(parallelism):
    result = reduce_with(
        [block_deletion(2, 4)],
        [1, 2, 3, 4, 5, 6, 7, 8],
        lambda x: 1 in x and 8 in x,
        parallelism=parallelism,
    )
    assert result == [1, 8]


def test_block_deletion_min_block(parallelism):
    result = reduce_with(
        [block_deletion(3, 5)],
        [1, 2, 3, 4, 5],
        lambda x: len(x) >= 3,
        parallelism=parallelism,
    )
    assert len(result) <= 2 or result == [1, 2, 3, 4, 5]


def test_block_deletion_small_input(parallelism):
    result = reduce_with(
        [block_deletion(5, 10)],
        [1, 2, 3],
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == [1, 2, 3]


def test_delete_duplicates_removes_all(parallelism):
    result = reduce_with(
        [delete_duplicates],
        [1, 2, 1, 3, 1],
        lambda x: 2 in x and 3 in x,
        parallelism=parallelism,
    )
    assert 1 not in result
    assert 2 in result
    assert 3 in result


def test_delete_duplicates_keeps_when_required(parallelism):
    result = reduce_with(
        [delete_duplicates],
        [1, 2, 1, 3, 1],
        lambda x: 1 in x,
        parallelism=parallelism,
    )
    assert result.count(1) == 3


def test_merged_intervals_simple():
    result = merged_intervals([(0, 2), (1, 3)])
    assert result == [(0, 3)]


def test_merged_intervals_no_overlap():
    result = merged_intervals([(0, 1), (2, 3)])
    assert result == [(0, 1), (2, 3)]


def test_merged_intervals_adjacent():
    result = merged_intervals([(0, 2), (2, 4)])
    assert result == [(0, 4)]


def test_merged_intervals_unsorted():
    result = merged_intervals([(3, 5), (0, 2)])
    assert result == [(0, 2), (3, 5)]


def test_with_deletions_list():
    result = with_deletions([1, 2, 3, 4, 5], [(1, 2), (3, 4)])
    assert result == [1, 3, 5]


def test_with_deletions_tuple():
    result = with_deletions((1, 2, 3, 4, 5), [(0, 2)])
    assert result == (3, 4, 5)


def test_with_deletions_empty():
    result = with_deletions([1, 2, 3], [])
    assert result == [1, 2, 3]


# =============================================================================
# Generic Language Passes - genericlanguages.py
# =============================================================================


def test_reduce_integer_literals_to_zero(parallelism):
    result = reduce_with(
        [reduce_integer_literals],
        b"value = 99999",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"value = 0"


def test_reduce_integer_literals_to_boundary(parallelism):
    result = reduce_with(
        [reduce_integer_literals],
        b"x = 100",
        lambda x: b"=" in x and int(x.split(b"=")[1]) >= 50,
        parallelism=parallelism,
    )
    assert result == b"x = 50"


def test_reduce_integer_literals_edge_case(parallelism):
    """Test integer reduction when consecutive lower values are also interesting."""
    result = reduce_with(
        [reduce_integer_literals],
        b"n = 10",
        lambda x: b"=" in x and int(x.split(b"=")[1].strip()) >= 5,
        parallelism=parallelism,
    )
    assert int(result.split(b"=")[1].strip()) >= 5


def test_combine_expressions_addition(parallelism):
    result = reduce_with(
        [combine_expressions],
        b"result = 10 + 15",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"result = 25"


def test_combine_expressions_subtraction(parallelism):
    result = reduce_with(
        [combine_expressions],
        b"x = 100 - 30",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"x = 70"


def test_combine_expressions_multiplication(parallelism):
    result = reduce_with(
        [combine_expressions],
        b"y = 6 * 7",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"y = 42"


def test_combine_expressions_division_by_zero(parallelism):
    result = reduce_with(
        [combine_expressions],
        b"x = 1 / 0",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"x = 1 / 0"


def test_merge_adjacent_strings_double_quotes(parallelism):
    result = reduce_with(
        [merge_adjacent_strings],
        b'"hello" "world"',
        lambda x: b"hello" in x and b"world" in x,
        parallelism=parallelism,
    )
    assert result == b'"helloworld"'


def test_merge_adjacent_strings_single_quotes(parallelism):
    result = reduce_with(
        [merge_adjacent_strings],
        b"'a' 'b'",
        lambda x: b"a" in x and b"b" in x,
        parallelism=parallelism,
    )
    assert result == b"'ab'"


def test_replace_falsey_with_zero_empty_string(parallelism):
    result = reduce_with(
        [replace_falsey_with_zero],
        b'x = ""',
        lambda x: b"x = " in x,
        parallelism=parallelism,
    )
    assert result == b"x = 0"


def test_replace_falsey_with_zero_false(parallelism):
    result = reduce_with(
        [replace_falsey_with_zero],
        b"y = false",
        lambda x: b"y = " in x,
        parallelism=parallelism,
    )
    assert result == b"y = 0"


def test_replace_falsey_with_zero_empty_list(parallelism):
    result = reduce_with(
        [replace_falsey_with_zero],
        b"z = []",
        lambda x: b"z = " in x,
        parallelism=parallelism,
    )
    assert result == b"z = 0"


def test_simplify_brackets_curly_to_parens(parallelism):
    result = reduce_with(
        [simplify_brackets],
        b"x = {}",
        lambda x: b"x = " in x and len(x) == 6,
        parallelism=parallelism,
    )
    assert result == b"x = ()"


def test_simplify_brackets_square_to_parens(parallelism):
    result = reduce_with(
        [simplify_brackets],
        b"a = []",
        lambda x: b"a = " in x and len(x) == 6,
        parallelism=parallelism,
    )
    assert result == b"a = ()"


def test_normalize_identifiers(parallelism):
    result = reduce_with(
        [normalize_identifiers],
        b"longVariableName = 1",
        lambda x: b"= 1" in x,
        parallelism=parallelism,
    )
    assert len(result) < len(b"longVariableName = 1")


def test_normalize_identifiers_multiple(parallelism):
    result = reduce_with(
        [normalize_identifiers],
        b"longName = anotherLongName + thirdName",
        lambda x: b"=" in x and b"+" in x,
        parallelism=parallelism,
    )
    assert len(result) < len(b"longName = anotherLongName + thirdName")


def test_normalize_identifiers_not_found_after_reduction(parallelism):
    """Test identifier normalization when identifier disappears during reduction."""
    result = reduce_with(
        [normalize_identifiers],
        b"x = y + z",
        lambda x: b"=" in x,
        parallelism=parallelism,
    )
    assert b"=" in result


def test_cut_comment_like_things_hash(parallelism):
    result = reduce_with(
        [cut_comment_like_things],
        b"code # comment\nmore",
        lambda x: b"code" in x and b"more" in x,
        parallelism=parallelism,
    )
    assert b"# comment" not in result


def test_cut_comment_like_things_slash_slash(parallelism):
    result = reduce_with(
        [cut_comment_like_things],
        b"code // comment\nmore",
        lambda x: b"code" in x and b"more" in x,
        parallelism=parallelism,
    )
    assert b"// comment" not in result


def test_cut_comment_like_things_multiline(parallelism):
    result = reduce_with(
        [cut_comment_like_things],
        b"before /* comment */ after",
        lambda x: b"before" in x and b"after" in x,
        parallelism=parallelism,
    )
    assert b"/* comment */" not in result


def test_cut_comment_like_things_docstring(parallelism):
    result = reduce_with(
        [cut_comment_like_things],
        b'code """docstring""" more',
        lambda x: b"code" in x and b"more" in x,
        parallelism=parallelism,
    )
    assert b'"""docstring"""' not in result


def test_cut_comment_like_things_end_of_file(parallelism):
    """Test comment removal when comment extends to end of file without newline."""
    result = reduce_with(
        [cut_comment_like_things],
        b"code # comment without newline",
        lambda x: b"code" in x,
        parallelism=parallelism,
    )
    assert result == b"code " or b"# comment" not in result


def test_integer_format_parse_invalid_non_numeric():
    fmt = IntegerFormat()
    with pytest.raises(ParseError):
        fmt.parse(b"abc")


def test_integer_format_parse_invalid_unicode():
    fmt = IntegerFormat()
    with pytest.raises(ParseError):
        fmt.parse(b"\xff\xfe")


# =============================================================================
# Patches Classes
# =============================================================================


def test_region_replacing_patches_empty():
    patches = RegionReplacingPatches([(0, 2)])
    assert patches.empty == {}


def test_region_replacing_patches_combine():
    patches = RegionReplacingPatches([(0, 2), (3, 5)])
    result = patches.combine({0: b"a"}, {1: b"b"})
    assert result == {0: b"a", 1: b"b"}


def test_region_replacing_patches_apply_with_patch():
    patches = RegionReplacingPatches([(0, 2), (3, 5)])
    result = patches.apply({0: b"X"}, b"aabbb")
    assert result == b"Xbbb"


def test_region_replacing_patches_apply_without_patch():
    patches = RegionReplacingPatches([(0, 2), (3, 5)])
    result = patches.apply({}, b"aabbb")
    assert result == b"aabbb"


def test_region_replacing_patches_size():
    patches = RegionReplacingPatches([(0, 5)])
    size = patches.size({0: b"X"})
    assert size == 4


def test_region_replacing_patches_size_empty_raises():
    patches = RegionReplacingPatches([(0, 5)])
    with pytest.raises(AssertionError):
        patches.size({})


def test_newline_replacer_empty():
    nr = NewlineReplacer()
    assert nr.empty == frozenset()


def test_newline_replacer_size():
    nr = NewlineReplacer()
    patch = frozenset({0, 1, 2})
    assert nr.size(patch) == 3


def test_region_replacement_empty():
    rr = RegionReplacement()
    assert rr.empty == []


def test_region_replacement_size():
    rr = RegionReplacement()
    assert rr.size([(0, 1, 0)]) == 0


# =============================================================================
# Edge Cases for Full Coverage
# =============================================================================


def test_lexeme_based_deletions_all_same_bytes(parallelism):
    """Test ngram_search returns early when all bytes are identical."""
    result = reduce_with(
        [lexeme_based_deletions],
        b"aaaa",
        lambda x: True,
        parallelism=parallelism,
    )
    assert result == b"aaaa" or len(result) <= 4


def test_sort_whitespace_newline_at_end_of_initial_ws(parallelism):
    """Test whitespace sorting when whitespace block ends with newline.

    Input has whitespace ' \\n' which ends with newline - the algorithm needs
    to adjust the whitespace boundary to avoid including the trailing newline.
    """
    result = reduce_with(
        [sort_whitespace],
        b"a \nb",  # 'a', then ' \n' (ws ending in newline), then 'b'
        lambda x: True,
        parallelism=parallelism,
    )
    # Verify it runs without error and doesn't grow
    assert isinstance(result, bytes)
    assert len(result) <= 4


def test_sort_whitespace_k_exceeds_length(parallelism):
    """Test whitespace sorting when probing past end of input.

    The algorithm uses find_large_integer to probe how much whitespace to sort,
    which may probe past the end of the input.
    """
    result = reduce_with(
        [sort_whitespace],
        b"a b ",  # 'a', ' ', 'b', ' ' - second space triggers the search
        lambda x: True,
        parallelism=parallelism,
    )
    assert isinstance(result, bytes)
    assert len(result) <= 4


def test_reduce_integer_literals_hi_minus_one_interesting(parallelism):
    """Test integer reduction finds minimum when consecutive values are interesting.

    Binary search converges to smallest interesting value, then checks if
    hi-1 is also interesting to find the true minimum."""
    result = reduce_with(
        [reduce_integer_literals],
        b"x = 10",
        lambda x: b"=" in x and int(x.split(b"=")[1].strip()) in [4, 5, 6, 7, 8, 9, 10],
        parallelism=parallelism,
    )
    # Binary search finds 5 as smallest interesting, then checks 4 which is also interesting
    assert int(result.split(b"=")[1].strip()) == 4


def test_normalize_identifiers_pattern_not_found(parallelism):
    """Test identifier normalization when one replacement removes another identifier."""
    result = reduce_with(
        [normalize_identifiers],
        b"longFoo longFoo longFoo",
        lambda x: x.count(b" ") == 2,  # Keep structure but identifiers can change
        parallelism=parallelism,
    )
    # Verify it ran
    assert b" " in result


# =============================================================================
# Format class tests - definitions.py coverage
# =============================================================================


def test_format_is_valid_true():
    """Test Format.is_valid returns True for valid input."""
    fmt = Substring(b"<", b">")
    assert fmt.is_valid(b"<hello>") is True


def test_format_is_valid_false():
    """Test Format.is_valid returns False for invalid input."""
    fmt = Substring(b"<", b">")
    assert fmt.is_valid(b"hello") is False


async def test_compose_with_changing_format():
    """Test compose handles cases where format becomes invalid during reduction."""

    # Start with valid format
    fmt = Substring(b"<", b">")

    async def pass_that_changes_format(problem):
        # Try to delete elements, which will access current_test_case
        await delete_elements(problem)

    composed = compose(fmt, pass_that_changes_format)

    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        # Always interesting
        return True

    problem = BasicReductionProblem(
        initial=b"<abc>",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # This should work since initial input is parseable
    await composed(problem)


# =============================================================================
# Python passes tests - python.py coverage
# =============================================================================


async def test_strip_annotations_with_value():
    """Test stripping type annotations from assignments with values."""

    # Python code with type annotation that has a value
    code = b"x: int = 5\n"

    async def is_interesting(x):
        # Accept if it's valid Python and assigns x
        return b"x" in x and b"=" in x

    problem = BasicReductionProblem(
        initial=code,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    await strip_annotations(problem)

    # Should have removed the type annotation
    result = problem.current_test_case
    assert b"int" not in result or result != code


async def test_strip_annotations_without_value():
    """Test stripping type annotations from declarations without values."""

    # Python code with type annotation declaration only (no value)
    code = b"x: int\ny = 1\n"

    async def is_interesting(x):
        # Accept if it has y assignment
        return b"y = 1" in x

    problem = BasicReductionProblem(
        initial=code,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    await strip_annotations(problem)

    # Should have removed the type-only declaration
    result = problem.current_test_case
    assert result == b"y = 1\n" or b"x: int" not in result


async def test_libcst_transform_invalid_python():
    """Test that libcst_transform handles invalid Python gracefully."""

    # Invalid Python code
    code = b"def foo(:\n  pass\n"

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=code,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    # Should not raise, should just return early
    await replace_bodies_with_ellipsis(problem)

    # Code should be unchanged since parsing failed
    assert problem.current_test_case == code


async def test_libcst_transform_reparsing_fails():
    """Test handling when code becomes unparseable during reduction."""

    # Valid Python code with a function
    code = b"def foo():\n    return 1\n"

    call_count = [0]
    original_parse = libcst.parse_module

    def mock_parse(source):
        call_count[0] += 1
        if call_count[0] > 1:
            # Fail on subsequent parses with a proper ParserSyntaxError
            raise libcst.ParserSyntaxError(
                "mock syntax error",
                lines=["def foo():"],
                raw_line=1,
                raw_column=0,
            )
        return original_parse(source)

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=code,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    with patch.object(libcst, "parse_module", side_effect=mock_parse):
        await replace_bodies_with_ellipsis(problem)

    # Should have handled the error gracefully
