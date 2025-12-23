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


# Tests for UnionFind and related classes


def test_union_find_copy():
    """Test UnionFind __copy__ method."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 2)
    uf.merge(2, 3)

    uf_copy = uf.__copy__()
    assert uf_copy.find(1) == uf_copy.find(3)
    # Modifying copy shouldn't affect original
    uf_copy.merge(4, 5)
    assert 4 not in uf.table


def test_union_find_clone():
    """Test UnionFind clone method."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 2)
    cloned = uf.clone()
    assert cloned.find(1) == cloned.find(2)


def test_union_find_mapping():
    """Test UnionFind mapping method."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 2)
    uf.merge(3, 4)
    mapping = uf.mapping()
    # Keys should be non-representative elements that point to their representatives
    assert 2 in mapping or 1 in mapping


def test_union_find_extend():
    """Test UnionFind extend method."""
    from shrinkray.passes.sat import UnionFind

    uf1 = UnionFind()
    uf1.merge(1, 2)

    uf2 = UnionFind()
    uf2.merge(3, 4)

    uf1.extend(uf2)
    assert uf1.find(3) == uf1.find(4)


def test_union_find_extend_self():
    """Test UnionFind extend with self (no-op)."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 2)
    uf.extend(uf)
    # Should be unchanged
    assert uf.find(1) == uf.find(2)


def test_union_find_repr():
    """Test UnionFind __repr__ method."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 2)
    uf.merge(3, 4)
    r = repr(uf)
    assert "UnionFind" in r
    assert "components" in r


def test_boolean_equivalence_find_zero_error():
    """Test BooleanEquivalence.find raises on zero."""
    from shrinkray.passes.sat import BooleanEquivalence

    be = BooleanEquivalence()
    with pytest.raises(ValueError, match="Invalid variable"):
        be.find(0)


def test_boolean_equivalence_merge_same():
    """Test BooleanEquivalence.merge with same values."""
    from shrinkray.passes.sat import BooleanEquivalence

    be = BooleanEquivalence()
    be.merge(1, 1)  # Should be a no-op
    # Should not raise


def test_boolean_equivalence_merge_contradiction():
    """Test BooleanEquivalence.merge with contradicting values."""
    from shrinkray.passes.sat import BooleanEquivalence, Inconsistent

    be = BooleanEquivalence()
    be.merge(1, 2)
    # Now 1 and 2 are equivalent
    # Trying to merge 1 with -2 should fail
    with pytest.raises(Inconsistent):
        be.merge(1, -2)


def test_negating_map_copy():
    """Test NegatingMap __copy__ method."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 5
    nm[2] = 6

    nm_copy = nm.__copy__()
    assert nm_copy[1] == 5
    assert nm_copy[2] == 6


def test_negating_map_repr():
    """Test NegatingMap __repr__ method."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 2
    r = repr(nm)
    assert "1" in r
    assert "2" in r


def test_negating_map_iter():
    """Test NegatingMap iteration."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 2
    nm[3] = 4
    keys = list(nm)
    # Should have both positive and negative keys
    assert 1 in keys
    assert -1 in keys
    assert 3 in keys
    assert -3 in keys


def test_negating_map_items():
    """Test NegatingMap items method."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 2
    items = list(nm.items())
    # Should include (1, 2) and (-1, -2)
    assert (1, 2) in items
    assert (-1, -2) in items


def test_negating_map_positive_keys():
    """Test NegatingMap positive_keys method."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 2
    nm[3] = 4
    pkeys = list(nm.positive_keys())
    assert 1 in pkeys
    assert 3 in pkeys
    assert -1 not in pkeys


# Tests for UnitPropagator


def test_unit_propagator_freeze():
    """Test UnitPropagator freeze method."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1, 2], [-1, 3]])
    up.freeze()
    with pytest.raises(ValueError, match="Frozen"):
        up.add_clauses([[4, 5]])


def test_unit_propagator_add_clauses():
    """Test UnitPropagator add_clauses method."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1, 2]])
    up.add_clauses([[-1, 3]])
    result = up.propagated_clauses()
    # Should have both clauses
    assert any(1 in c or 2 in c for c in result)


def test_unit_propagator_add_unit():
    """Test UnitPropagator add_unit method."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1, 2], [-1, 3]])
    up.add_unit(1)
    # After adding 1 as unit, first clause is satisfied
    result = up.propagated_clauses()
    # Should have 1 as a unit clause
    assert [1] in result


def test_unit_propagator_add_units_frozen():
    """Test UnitPropagator add_units on frozen propagator."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1, 2]])
    up.freeze()
    with pytest.raises(ValueError, match="Frozen"):
        up.add_units([1])


def test_unit_propagator_clause_count():
    """Test UnitPropagator clause_count method."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1, 2], [1, 3], [2, 3]])
    # Variable 1 appears in 2 clauses
    assert up.clause_count(1) == 2
    # Variable 2 appears in 2 clauses
    assert up.clause_count(2) == 2
    # Variable 3 appears in 2 clauses
    assert up.clause_count(3) == 2


def test_unit_propagator_clause_count_forced():
    """Test UnitPropagator clause_count for forced variable."""
    from shrinkray.passes.sat import UnitPropagator

    up = UnitPropagator([[1], [1, 2]])
    # Variable 1 is forced (unit clause [1])
    assert 1 in up.forced_variables
    assert up.clause_count(1) == 1


def test_unit_propagator_inconsistent_units():
    """Test UnitPropagator with contradicting units."""
    from shrinkray.passes.sat import Inconsistent, UnitPropagator

    # Should raise Inconsistent because 1 and -1 can't both be true
    with pytest.raises(Inconsistent):
        UnitPropagator([[1], [-1]])


def test_unit_propagator_empty_clause():
    """Test UnitPropagator with empty clause."""
    from shrinkray.passes.sat import Inconsistent, UnitPropagator

    with pytest.raises(Inconsistent, match="empty clause"):
        UnitPropagator([[]])


def test_union_find_merge_same():
    """Test UnionFind.merge with same elements."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind()
    uf.merge(1, 1)  # Should be a no-op
    # Should have only 1 representative
    assert uf.find(1) == 1


def test_union_find_initial_merges():
    """Test UnionFind with initial merges in constructor."""
    from shrinkray.passes.sat import UnionFind

    uf = UnionFind(initial_merges=[(1, 2), (3, 4)])
    assert uf.find(1) == uf.find(2)
    assert uf.find(3) == uf.find(4)
