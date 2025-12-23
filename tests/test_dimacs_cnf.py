"""Tests for DimacsCNF format parsing and dumping.

These tests do NOT require minisat and can run anywhere.
"""

import pytest

from shrinkray.passes.definitions import DumpError, ParseError
from shrinkray.passes.sat import DimacsCNF


def test_dimacs_parse_with_comments():
    """Test that comment lines are skipped during parsing."""
    cnf = b"c This is a comment\np cnf 2 1\n1 -2 0\n"
    result = DimacsCNF.parse(cnf)
    assert result == [[1, -2]]


def test_dimacs_parse_skips_p_line():
    """Test that p line is skipped during parsing."""
    cnf = b"p cnf 2 2\n1 0\n2 0\n"
    result = DimacsCNF.parse(cnf)
    assert result == [[1], [2]]


def test_dimacs_parse_skips_empty_lines():
    """Test that empty lines are skipped during parsing."""
    cnf = b"p cnf 2 1\n\n1 2 0\n\n"
    result = DimacsCNF.parse(cnf)
    assert result == [[1, 2]]


def test_dimacs_parse_raises_on_missing_zero():
    """Test that ParseError is raised when clause doesn't end with 0."""
    cnf = b"p cnf 2 1\n1 2\n"
    with pytest.raises(ParseError, match="did not end with 0"):
        DimacsCNF.parse(cnf)


def test_dimacs_parse_raises_on_empty_clauses():
    """Test that ParseError is raised when no clauses are found."""
    cnf = b"c just a comment\n"
    with pytest.raises(ParseError, match="No clauses found"):
        DimacsCNF.parse(cnf)


def test_dimacs_parse_raises_on_invalid_int():
    """Test that ParseError is raised on non-integer literals."""
    cnf = b"p cnf 2 1\n1 abc 0\n"
    with pytest.raises(ParseError):
        DimacsCNF.parse(cnf)


def test_dimacs_parse_raises_on_non_utf8():
    """Test that ParseError is raised on non-UTF8 input."""
    cnf = b"\xff\xfe"  # Invalid UTF-8
    with pytest.raises(ParseError):
        DimacsCNF.parse(cnf)


def test_dimacs_dumps_basic():
    """Test basic dump functionality."""
    clauses = [[1, 2], [-1, 3]]
    result = DimacsCNF.dumps(clauses)
    assert b"p cnf" in result
    assert b"1 2 0" in result
    assert b"-1 3 0" in result


def test_dimacs_dumps_raises_on_empty_input():
    """Test that DumpError is raised on empty input."""
    with pytest.raises(DumpError):
        DimacsCNF.dumps([])


def test_dimacs_dumps_raises_on_empty_clause():
    """Test that DumpError is raised when a clause is empty."""
    with pytest.raises(DumpError):
        DimacsCNF.dumps([[1], []])


def test_dimacs_is_valid_returns_true_for_valid():
    """Test is_valid returns True for valid DIMACS."""
    cnf = b"p cnf 2 1\n1 -2 0\n"
    assert DimacsCNF.is_valid(cnf) is True


def test_dimacs_is_valid_returns_false_for_invalid():
    """Test is_valid returns False for invalid DIMACS."""
    cnf = b"not a dimacs file"
    assert DimacsCNF.is_valid(cnf) is False


def test_dimacs_roundtrip():
    """Test that parsing dumped output produces the same clauses."""
    original = [[1, 2], [-1, -2], [3]]
    dumped = DimacsCNF.dumps(original)
    loaded = DimacsCNF.parse(dumped)
    assert loaded == original
