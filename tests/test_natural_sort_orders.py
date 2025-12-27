"""Tests illustrating the natural sort order heuristics for strings.

The natural sort order for strings uses a chain of heuristics:
1. Total length (fewer characters is better)
2. Average squared line length (penalizes long lines)
   Formula: sum(len(line)² for each line) / count(lines)²
3. Number of lines (fewer lines is better)
4. List of line lengths lexicographically (shorter lines first)
5. Natural character order (whitespace < digits < lowercase < uppercase)

Each section below demonstrates one heuristic in isolation.
"""

import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.problem import (
    LazyChainedSortKey,
    natural_key,
)


# =============================================================================
# Helper for clear comparisons
# =============================================================================


def is_preferred(a: str, b: str) -> bool:
    """Return True if `a` is preferred (smaller) than `b` in natural order."""
    return natural_key(a) < natural_key(b)


def are_equivalent(a: str, b: str) -> bool:
    """Return True if `a` and `b` are equivalent in natural order."""
    return not (natural_key(a) < natural_key(b)) and not (
        natural_key(b) < natural_key(a)
    )


# =============================================================================
# Heuristic 1: Total Length
# =============================================================================


def test_length_empty_preferred_over_any_content():
    """Shorter strings are always preferred, regardless of content."""
    assert is_preferred("", "a")
    assert is_preferred("", " ")
    assert is_preferred("", "Z")


def test_length_single_char_preferred_over_two_chars():
    assert is_preferred("a", "aa")
    assert is_preferred("Z", "aa")  # Even uppercase over lowercase
    assert is_preferred("~", "aa")  # Even high ASCII over alpha


def test_length_dominates_character_quality():
    # "ZZ" is worse chars but shorter than "aaa"
    assert is_preferred("ZZ", "aaa")
    # Single space preferred over any two-char string
    assert is_preferred(" ", "aa")


def test_length_same_length_goes_to_next_heuristic():
    # Same length, so character content matters
    assert is_preferred("a", "b")
    assert is_preferred("aa", "ab")


@given(st.text(min_size=1))
def test_length_empty_always_preferred(s):
    """Empty string is the global minimum."""
    assert is_preferred("", s)


@given(st.text(), st.text())
def test_length_shorter_always_wins_or_ties(a, b):
    """If a is strictly shorter, it's preferred. If same length, depends on content."""
    if len(a) < len(b):
        assert is_preferred(a, b)
    elif len(a) > len(b):
        assert is_preferred(b, a)
    # Equal length: could go either way based on later heuristics


# =============================================================================
# Heuristic 2: Average Squared Line Length
# =============================================================================


def test_squared_balanced_lines_preferred():
    """Among equal-length strings, prefer those with balanced line lengths.

    This heuristic minimizes average squared line length, which penalizes
    very long lines. The formula is: sum(len(line)²) / count(lines)².
    """
    # Both have 7 chars total (including newlines)
    # "aaa\naaa" has lines [3,3], avg sq = 18/4 = 4.5
    # "aa\naaaa" has lines [2,4], avg sq = 20/4 = 5.0
    assert is_preferred("aaa\naaa", "aa\naaaa")


def test_squared_one_long_line_vs_balanced():
    # "aaaa\nbb" has lines [4,2], avg sq = 20/4 = 5.0, total = 7 chars
    # "aaa\nbbb" has lines [3,3], avg sq = 18/4 = 4.5, total = 7 chars
    assert is_preferred("aaa\nbbb", "aaaa\nbb")


def test_squared_multiple_short_lines_vs_one_long():
    # Both 11 chars
    a = "aaaa\naaaa\na"  # 11 chars, 3 lines, avg sq = 33/9 ≈ 3.67
    b = "aaaaaaaaaa\n"  # 11 chars, 1 line, avg sq = 100/1 = 100
    assert len(a) == len(b) == 11
    assert is_preferred(a, b)


def test_squared_same_avg_goes_to_line_count():
    # "ab" vs "ba" - same length (2), same avg sq (4), same line count (1)
    # Goes to later heuristics (character order)
    assert is_preferred("ab", "ba")


# =============================================================================
# Heuristic 3: Number of Lines
# =============================================================================


def test_line_count_controlled_examples():
    """Among strings with equal length and equal avg-squared, fewer lines wins.

    The constraint of equal avg-sq + equal length + different line count is
    mathematically constrained, so we test what we can.
    """
    # These don't have equal avg-sq, so avg-sq takes precedence
    # [1,0,1] avg sq = 2/9, [2,1] avg sq = 5/4
    # So "a\n\nb" is preferred (lower avg sq)
    assert is_preferred("a\n\nb", "ab\nc")


# =============================================================================
# Heuristic 4: List of Line Lengths
# =============================================================================


def test_line_lengths_shorter_first_line_preferred():
    """Among equal strings up to heuristic 3, compare line lengths lexicographically."""
    # Need same avg-sq. [1,2] avg sq = 5/4, [2,1] avg sq = 5/4, both 2 lines
    # "a\nbb" (4 chars) vs "aa\nb" (4 chars)
    # [1,2] < [2,1] lexicographically
    assert is_preferred("a\nbb", "aa\nb")


def test_line_lengths_lexicographic():
    # [1,1,2] vs [1,2,1] - first differs at index 1
    # "a\na\nbb" (6 chars) vs "a\nbb\na" (6 chars)
    # sums: 1+1+4=6 vs 1+4+1=6
    assert is_preferred("a\na\nbb", "a\nbb\na")


def test_line_lengths_more_balanced_prefix_preferred():
    # [0,3] vs [1,2] - [0,3] has avg sq 9/4, [1,2] has avg sq 5/4
    # Avg-sq differs! [1,2] has lower avg-sq, so it's preferred
    assert is_preferred("a\naa", "\naaa")


# =============================================================================
# Heuristic 5: Natural Character Order
# =============================================================================


def test_char_order_priority():
    """Final tiebreaker: character ordering. Whitespace < digits < lowercase < uppercase."""
    # Space comes first
    assert is_preferred(" ", "0")
    assert is_preferred("0", "a")
    assert is_preferred("a", "A")


def test_char_order_whitespace():
    """Within whitespace, order by position in string.whitespace."""
    # string.whitespace = ' \t\n\r\x0b\x0c' - space is first
    ws_order = list(string.whitespace)
    # Compare first two whitespace chars
    if len(ws_order) >= 2:
        assert is_preferred(ws_order[0], ws_order[1])


def test_char_order_digits():
    """Digits are ordered 0-9."""
    assert is_preferred("0", "1")
    assert is_preferred("1", "9")


def test_char_order_lowercase():
    """Lowercase letters a-z."""
    assert is_preferred("a", "b")
    assert is_preferred("b", "z")


def test_char_order_uppercase():
    """Uppercase letters A-Z come after lowercase."""
    assert is_preferred("A", "B")
    assert is_preferred("a", "A")  # lowercase preferred over uppercase
    assert is_preferred("z", "A")  # all lowercase before all uppercase


def test_char_order_unknown_come_last():
    """Characters not in NATURAL_CHARACTER_ORDER sort by ord() after known chars."""
    # '~' has ord 126, not in NATURAL_CHARACTER_ORDER
    # 'Z' is in the order, so comes before '~'
    assert is_preferred("Z", "~")
    assert is_preferred("A", "!")


def test_char_order_punctuation_after_alphanumeric():
    """Punctuation (not in NATURAL_CHARACTER_ORDER) comes after letters."""
    assert is_preferred("a", "!")
    assert is_preferred("Z", "!")
    assert is_preferred("9", "!")


def test_char_order_full_string_comparison():
    """Character comparison is done position by position."""
    assert is_preferred("aaa", "aab")
    assert is_preferred("aab", "aba")
    assert is_preferred("  a", "  b")


# =============================================================================
# Interaction Between Heuristics
# =============================================================================


def test_interaction_length_beats_good_characters():
    """A short string with 'bad' chars beats a long string with 'good' chars."""
    # "~" is a bad character, " " is the best
    # But "~" (1 char) beats "  " (2 chars)
    assert is_preferred("~", "  ")


def test_interaction_avg_squared_beats_line_count():
    """Lower average squared line length beats fewer lines."""
    # "ab\ncd\nef" (8 chars) has lines [2,2,2], avg sq = 12/9 ≈ 1.33, 3 lines
    # "abcdefgh" (8 chars) has lines [8], avg sq = 64/1 = 64, 1 line
    # Lower avg-sq wins, so the 3-line version is preferred
    assert is_preferred("ab\ncd\nef", "abcdefgh")


def test_interaction_character_order_as_final_tiebreaker():
    """When all else is equal, character content decides."""
    # Same length, same structure, different characters
    assert is_preferred("abc", "abd")
    assert is_preferred("aaa", "aab")


# =============================================================================
# LazyChainedSortKey Tests
# =============================================================================


def test_lazy_key_equality():
    """Equal values produce equal keys."""
    assert natural_key("abc") == natural_key("abc")


def test_lazy_key_inequality():
    """Different values produce ordered keys."""
    assert natural_key("a") < natural_key("ab")
    assert natural_key("ab") > natural_key("a")


def test_lazy_key_not_equal_to_non_key():
    """Comparison with non-LazyChainedSortKey returns NotImplemented."""
    key = natural_key("abc")
    assert key.__eq__("abc") is NotImplemented
    assert key.__lt__("abc") is NotImplemented


def test_lazy_key_all_functions_equal_returns_false():
    """When all comparison functions return equal, __lt__ returns False."""
    # This tests the edge case where different values produce equal keys
    # for all comparison functions (line 103 in problem.py).
    # We use custom functions that always return a constant.
    key1 = LazyChainedSortKey(functions=[lambda x: 0], value="a")
    key2 = LazyChainedSortKey(functions=[lambda x: 0], value="b")
    # Values differ, but functions all return equal -> neither is less
    assert not (key1 < key2)
    assert not (key2 < key1)


# =============================================================================
# Hypothesis-Discovered Examples
# =============================================================================


@given(st.text(min_size=0, max_size=20))
def test_hypothesis_reflexive(s):
    """A string is equivalent to itself."""
    assert are_equivalent(s, s)


@given(st.text(min_size=0, max_size=20), st.text(min_size=0, max_size=20))
def test_hypothesis_antisymmetric(a, b):
    """If a < b, then not b < a."""
    if natural_key(a) < natural_key(b):
        assert not natural_key(b) < natural_key(a)


@given(
    st.text(min_size=0, max_size=10),
    st.text(min_size=0, max_size=10),
    st.text(min_size=0, max_size=10),
)
def test_hypothesis_transitive(a, b, c):
    """If a < b and b < c, then a < c."""
    if natural_key(a) < natural_key(b) and natural_key(b) < natural_key(c):
        assert natural_key(a) < natural_key(c)


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


def test_edge_empty_string():
    """Empty string is the absolute minimum."""
    assert is_preferred("", " ")
    assert is_preferred("", "\n")
    assert is_preferred("", "a")


def test_edge_only_newlines():
    """Strings of only newlines: fewer is better."""
    assert is_preferred("\n", "\n\n")
    assert is_preferred("", "\n")


def test_edge_trailing_newline():
    """Trailing newline affects line structure."""
    # "a\n" has length 2, "a" has length 1 - length takes precedence
    assert is_preferred("a", "a\n")


def test_edge_leading_newline():
    """Leading newline affects line structure."""
    # "\na" is length 2, "a" is length 1 - length takes precedence
    assert is_preferred("a", "\na")


def test_edge_only_spaces():
    """Spaces are preferred characters, but length still matters."""
    assert is_preferred(" ", "  ")
    assert is_preferred("  ", "   ")


def test_edge_mixed_whitespace():
    """Different whitespace characters."""
    # All whitespace is before digits/letters
    assert is_preferred("\t", "0")
    assert is_preferred(" ", "a")


def test_edge_unicode_characters():
    """Unicode characters not in NATURAL_CHARACTER_ORDER."""
    # Unicode comes after ASCII since it's sorted by ord() as fallback
    assert is_preferred("a", "\u00e9")  # 'a' vs 'é'
    assert is_preferred("Z", "\u00e9")  # 'Z' vs 'é'


# =============================================================================
# Trailing vs Leading Newlines (splitlines behavior)
# =============================================================================


def test_newline_trailing_vs_leading_same_length():
    """Trailing newline preferred over leading newline (fewer lines).

    Key insight: splitlines() treats trailing newlines differently from leading.
    - "a\\n".splitlines() = ['a']       (1 line)
    - "\\na".splitlines() = ['', 'a']   (2 lines)
    """
    # Both have length 2
    # "a\n" has avg sq = 1/1 = 1, "\na" has avg sq = 1/4 = 0.25
    # BUT "\na" has 2 lines while "a\n" has 1 line with different structure
    assert is_preferred("a\n", "\na")


def test_newline_trailing_dont_add_lines():
    """splitlines() ignores trailing newlines."""
    # "abc" (length 3) < "abc\n" (length 4)
    assert is_preferred("abc", "abc\n")


def test_newline_leading_add_empty_lines():
    """Leading newlines create empty lines."""
    # Length: 4 vs 3, so "abc" wins on length
    assert is_preferred("abc", "\nabc")


def test_newline_middle_create_splits():
    """Newlines in the middle always create splits."""
    # "a\nb" has 2 lines: avg sq = 2/4 = 0.5
    # "ab\n" has 1 line: avg sq = 4/1 = 4
    # "a\nb" is preferred (lower avg-sq)
    assert is_preferred("a\nb", "ab\n")


def test_newline_multiple_trailing():
    """Multiple trailing newlines don't add to line count consistently."""
    # Lengths differ: 3 vs 2
    assert is_preferred("a\n", "a\n\n")


# =============================================================================
# Minimal Strings by Length
# =============================================================================


def test_minimal_single_char():
    """Single space is the minimal 1-character string."""
    assert is_preferred(" ", "0")
    assert is_preferred(" ", "a")
    assert is_preferred(" ", "\t")  # space comes before tab


def test_minimal_two_chars():
    """Newline gives different structure than two spaces."""
    # "  ".split("\n") = ['  '], avg sq = 4/1 = 4
    # " \n".split("\n") = [' ', ''], avg sq = 1/4 = 0.25
    # So " \n" has lower avg-sq!
    assert is_preferred(" \n", "  ")


def test_minimal_three_chars_structure_matters():
    """At 3 chars, line structure can beat character quality."""
    # "   " (3 spaces): lines ['   '], avg sq = 9/1 = 9
    # "  \n" (2 spaces + newline): lines ['  '], avg sq = 4/1 = 4
    # " \n " (space, newline, space): lines [' ', ' '], avg sq = 2/4 = 0.5
    assert is_preferred(" \n ", "  \n")
    assert is_preferred("  \n", "   ")


@pytest.mark.parametrize(
    "length,expected_minimal",
    [
        pytest.param(1, " ", id="length-1"),
        pytest.param(2, " \n", id="length-2"),
        pytest.param(3, " \n ", id="length-3"),
    ],
)
def test_minimal_by_length(length, expected_minimal):
    """Verify the expected minimal string for each length."""
    assert len(expected_minimal) == length
    # The expected minimal should beat alternatives
    for c in [" ", "0", "a", "A", "\t"]:
        alternative = c * length
        if alternative != expected_minimal:
            assert is_preferred(expected_minimal, alternative) or are_equivalent(
                expected_minimal, alternative
            )


# =============================================================================
# Comparison Chain Verification
# =============================================================================


def test_chain_heuristic_1_length():
    """Length is checked first."""
    # Even with worst characters, shorter wins
    assert is_preferred("~", "  ")
    assert is_preferred("!!!", "    ")


def test_chain_heuristic_2_avg_squared():
    """Average squared line length is second."""
    # Same length, balanced beats unbalanced
    a = "ab\ncd"  # 5 chars, lines [2,2], avg sq = 8/4 = 2
    b = "abcd\n"  # 5 chars, lines [4], avg sq = 16/1 = 16
    assert is_preferred(a, b)


def test_chain_heuristic_4_line_lengths_list():
    """Line length list comparison is fourth."""
    # Same length, same avg-sq, same line count, different distribution
    # [1,2] vs [2,1] - both have avg sq = 5/4, both 2 lines
    a = "a\nbb"  # lines [1,2]
    b = "aa\nb"  # lines [2,1]
    assert is_preferred(a, b)


def test_chain_heuristic_5_character_order():
    """Character order is the final tiebreaker."""
    # Same structure, different characters
    assert is_preferred("aa", "ab")
    assert is_preferred(" a", " b")
    assert is_preferred("a\na", "a\nb")


# =============================================================================
# Real-World Motivating Examples
# =============================================================================


def test_real_code_simplification():
    """Simpler code is preferred."""
    complex_code = "if x:\n    return 1\nelse:\n    return 0"
    simple_code = "return x"
    assert is_preferred(simple_code, complex_code)


def test_real_shorter_identifier():
    """Shorter identifiers are preferred."""
    assert is_preferred("x", "foo")
    assert is_preferred("a", "variable_name")


def test_real_lowercase_preferred_in_code():
    """Lowercase identifiers preferred over uppercase (more common in Python)."""
    assert is_preferred("foo", "FOO")
    assert is_preferred("x", "X")


def test_real_numeric_simplification():
    """Smaller numbers (by character) are preferred."""
    assert is_preferred("0", "1")
    assert is_preferred("1", "99")


def test_real_balanced_function_body():
    """Well-formatted code with balanced lines preferred."""
    unbalanced = "def f(): return verylongexpression"
    balanced = "def f():\n    return x"
    # Length: unbalanced=35, balanced=22
    # Balanced wins on length alone
    assert is_preferred(balanced, unbalanced)


def test_real_removing_comments():
    """Code without comments is shorter, thus preferred."""
    with_comment = "x = 1  # set x"
    without_comment = "x = 1"
    assert is_preferred(without_comment, with_comment)
