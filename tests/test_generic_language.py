import pytest

from shrinkray.passes.definitions import ParseError
from shrinkray.passes.genericlanguages import (
    IntegerFormat,
    RegionReplacingPatches,
    Substring,
    combine_expressions,
    cut_comment_like_things,
    iter_indices,
    merge_adjacent_strings,
    normalize_identifiers,
    reduce_integer_literals,
    replace_falsey_with_zero,
    shortlex,
    simplify_brackets,
)
from tests.helpers import reduce_with


def test_can_reduce_an_integer_in_the_middle_of_a_string() -> None:
    assert (
        reduce_with([reduce_integer_literals], b"bobcats99999hello", lambda x: True)
        == b"bobcats0hello"
    )


def test_can_reduce_integers_to_boundaries() -> None:
    assert (
        reduce_with([reduce_integer_literals], b"100", lambda x: eval(x) >= 73) == b"73"
    )


def test_can_combine_expressions() -> None:
    assert reduce_with([combine_expressions], b"10 + 10", lambda x: True) == b"20"


def test_does_not_error_on_bad_expression() -> None:
    assert reduce_with([combine_expressions], b"1 / 0", lambda x: True) == b"1 / 0"


def test_can_combine_expressions_with_no_expressions() -> None:
    assert (
        reduce_with([combine_expressions], b"hello world", lambda x: True)
        == b"hello world"
    )


LOTS_OF_COMMENTS = b"""
hello # this is a single line comment
/* this
is
a
multiline
comment */ world // some extraneous garbage
"""


def test_comment_removal():
    x = reduce_with([cut_comment_like_things], LOTS_OF_COMMENTS, lambda x: True)
    lines = [line.strip() for line in x.splitlines() if line.strip()]
    assert lines == [b"hello", b"world"]


# === Substring tests ===


def test_substring_parse_valid():
    sub = Substring(prefix=b"<", suffix=b">")
    assert sub.parse(b"<hello>") == b"hello"


def test_substring_parse_invalid_prefix():
    sub = Substring(prefix=b"<", suffix=b">")
    with pytest.raises(ParseError):
        sub.parse(b"hello>")


def test_substring_parse_invalid_suffix():
    sub = Substring(prefix=b"<", suffix=b">")
    with pytest.raises(ParseError):
        sub.parse(b"<hello")


def test_substring_dumps():
    sub = Substring(prefix=b"<", suffix=b">")
    assert sub.dumps(b"hello") == b"<hello>"


def test_substring_name():
    sub = Substring(prefix=b"<", suffix=b">")
    assert sub.name == "Substring(1, 1)"


# === RegionReplacingPatches tests ===


def test_region_replacing_patches_size():
    patches = RegionReplacingPatches([(0, 5)])
    # Replacing 5 chars with 3 chars = size savings of 2
    assert patches.size({0: b"abc"}) == 2


def test_region_replacing_patches_size_empty_patch_raises():
    patches = RegionReplacingPatches([(0, 5)])
    with pytest.raises(AssertionError):
        patches.size({})


# === IntegerFormat tests ===


def test_integer_format_parse_valid():
    fmt = IntegerFormat()
    assert fmt.parse(b"123") == 123


def test_integer_format_parse_invalid():
    fmt = IntegerFormat()
    with pytest.raises(ParseError):
        fmt.parse(b"abc")


def test_integer_format_parse_invalid_unicode():
    fmt = IntegerFormat()
    with pytest.raises(ParseError):
        fmt.parse(b"\xff\xfe")


def test_integer_format_dumps():
    fmt = IntegerFormat()
    assert fmt.dumps(42) == b"42"


# === shortlex tests ===


def test_shortlex_shorter_wins():
    assert shortlex(b"ab") < shortlex(b"abc")


def test_shortlex_same_length_lex_order():
    assert shortlex(b"ab") < shortlex(b"ba")


# === iter_indices tests ===


def test_iter_indices_finds_all():
    result = list(iter_indices(b"abcabc", b"abc"))
    assert result == [0, 3]


def test_iter_indices_no_match():
    result = list(iter_indices(b"hello", b"xyz"))
    assert result == []


def test_iter_indices_single_match():
    result = list(iter_indices(b"hello", b"ell"))
    assert result == [1]


# === merge_adjacent_strings tests ===


def test_merge_adjacent_strings():
    result = reduce_with([merge_adjacent_strings], b"'' ''", lambda x: True)
    assert result == b" "


def test_merge_adjacent_double_quotes():
    result = reduce_with([merge_adjacent_strings], b'"" ""', lambda x: True)
    assert result == b" "


# === replace_falsey_with_zero tests ===


def test_replace_falsey_empty_string():
    result = reduce_with([replace_falsey_with_zero], b"''", lambda x: True)
    assert result == b"0"


def test_replace_falsey_false():
    result = reduce_with([replace_falsey_with_zero], b"false", lambda x: True)
    assert result == b"0"


def test_replace_falsey_empty_list():
    result = reduce_with([replace_falsey_with_zero], b"[]", lambda x: True)
    assert result == b"0"


def test_replace_falsey_empty_parens():
    result = reduce_with([replace_falsey_with_zero], b"()", lambda x: True)
    assert result == b"0"


# === simplify_brackets tests ===


def test_simplify_brackets_curly_to_square():
    result = reduce_with([simplify_brackets], b"{}", lambda x: True)
    assert result == b"()"


def test_simplify_brackets_square_to_parens():
    result = reduce_with([simplify_brackets], b"[]", lambda x: True)
    assert result == b"()"


# === normalize_identifiers tests ===


def test_normalize_identifiers_basic():
    result = reduce_with([normalize_identifiers], b"hello hello", lambda x: True)
    # Tries uppercase first, then lowercase
    assert result == b"A A"


def test_normalize_identifiers_multiple():
    result = reduce_with([normalize_identifiers], b"foo bar foo bar", lambda x: True)
    # Should replace with shorter names
    assert len(result) < len(b"foo bar foo bar")


def test_normalize_identifiers_target_not_found():
    """Test the continue branch when target is not in current source."""
    # This tests line 235 - when a target identifier is no longer in the source
    # after a previous replacement
    result = reduce_with(
        [normalize_identifiers],
        b"abc ab abc",  # Both abc and ab are identifiers
        lambda x: b"ab" in x,  # Keep any version with "ab" in it
    )
    # "abc" should be replaceable since it's larger and "ab" is preserved
    assert b"ab" in result


def test_normalize_identifiers_identifier_disappears():
    """Test when an identifier disappears after a replacement.

    When 'hello' is replaced with 'A', the identifier 'hello' no longer exists,
    so when we try to find 'hello' again in a later iteration it won't be found.
    """
    # Create a situation where one identifier gets replaced and then
    # we look for another identifier that doesn't exist in the new source
    result = reduce_with(
        [normalize_identifiers],
        b"longname x longname",  # longname and x are identifiers
        lambda x: x.count(b" ") == 2,  # Keep 2 spaces
    )
    # longname should be replaced with something shorter
    assert b"longname" not in result


def test_regex_pass_with_compiled_pattern():
    """Test regex_pass with an already compiled pattern."""
    import re

    from shrinkray.passes.genericlanguages import regex_pass

    # Create a pass with a pre-compiled pattern
    pattern = re.compile(rb"[0-9]+")

    @regex_pass(pattern)
    async def reduce_numbers(problem):
        await problem.is_interesting(b"0")

    result = reduce_with([reduce_numbers], b"abc123def", lambda x: True)
    assert result == b"abc0def"


def test_reduce_integer_early_termination():
    """Test reduce_integer when lo + 1 is interesting.

    The reduce_integer function does binary search with special handling
    for early termination when lo + 1 is interesting. This happens when
    the search narrows down to where lo + 1 equals the minimum interesting value.
    """
    # Test case: reduce 100 where only values >= 5 are interesting
    # 0 is not interesting (so we enter the loop)
    # Eventually lo becomes 4, and checking lo+1=5 returns True -> line 152
    result = reduce_with(
        [reduce_integer_literals],
        b"100",
        lambda x: int(x) >= 5,  # Only values >= 5 are interesting
    )
    assert result == b"5"


def test_cut_comments_no_end_found():
    """Test cut_comments when include_end=True but end marker not found."""
    from shrinkray.passes.genericlanguages import cut_comments

    # Test with include_end=True and no end marker
    result = reduce_with(
        [lambda p: cut_comments(p, b"/*", b"*/", include_end=True)],
        b"hello /* comment without end",
        lambda x: True,
    )
    # Comment should not be removed since end marker wasn't found
    assert b"/*" in result


def test_cut_comments_no_end_include_false():
    """Test cut_comments when include_end=False and end not found."""
    from shrinkray.passes.genericlanguages import cut_comments

    # Test with include_end=False and no end marker - should cut to end
    result = reduce_with(
        [lambda p: cut_comments(p, b"#", b"\n", include_end=False)],
        b"hello # comment to end",  # No newline at end
        lambda x: True,
    )
    # Comment should be removed to end of string
    assert result == b"hello "


def test_normalize_identifiers_all_letters_used():
    """Test when all single-letter identifiers are already used.

    When all lowercase and uppercase letters exist as identifiers,
    the inner loop completes without breaking, triggering the 221->220 branch.
    """
    # Create a source with all 26 lowercase and 26 uppercase letters as identifiers
    from string import ascii_lowercase, ascii_uppercase

    all_letters = " ".join(ascii_lowercase + ascii_uppercase).encode("ascii")
    result = reduce_with(
        [normalize_identifiers],
        all_letters,
        lambda x: True,
    )
    # Result should still be valid - letters might be replaced with numbers etc
    assert len(result) <= len(all_letters)


async def test_normalize_identifiers_target_disappears():
    """Test line 235: when an identifier disappears before its turn in the loop.

    This tests the defensive continue statement when an identifier that was
    in the original source is no longer present when we try to process it.
    This can happen in concurrent scenarios or with modified problem classes.
    """
    from unittest.mock import MagicMock, PropertyMock

    from shrinkray.passes.genericlanguages import normalize_identifiers
    from shrinkray.work import WorkContext

    # Create a mock problem that changes its current_test_case
    mock_problem = MagicMock()
    mock_problem.work = WorkContext(parallelism=1)

    # Track call count to simulate changing source
    # Access pattern:
    # 1. Line 217: get identifiers from initial source
    # 2. Line 233 (first iter): get source for first target (longer one)
    # 3. Line 233 (second iter): get source for second target - should NOT contain it
    access_count = [0]
    original_source = b"aaa bb"  # 'aaa' is longer, processed first

    def get_current_test_case():
        access_count[0] += 1
        # After first two accesses, return source without 'bb'
        # Access 1: identifiers (contains both aaa, bb)
        # Access 2: first iteration source check (for 'aaa')
        # Access 3+: return source without 'bb' to trigger continue
        if access_count[0] > 2:
            return b"aaa only"  # 'bb' is gone but 'aaa' remains
        return original_source

    type(mock_problem).current_test_case = PropertyMock(
        side_effect=get_current_test_case
    )

    # Make is_interesting always return False so no replacements happen
    async def mock_is_interesting(x):
        return False

    mock_problem.is_interesting = mock_is_interesting

    # This should hit the continue branch when 'bb' can't be found
    await normalize_identifiers(mock_problem)
