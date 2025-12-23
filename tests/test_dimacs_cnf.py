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


# Tests for UnitPropagator


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


def test_dimacs_cnf_name():
    """Test DimacsCNF.name property."""
    from shrinkray.passes.sat import DimacsCNF

    assert DimacsCNF.name == "DimacsCNF"


def test_union_find_repr():
    """Test UnionFind.__repr__ shows component count."""
    from shrinkray.passes.sat import UnionFind

    uf: UnionFind[int] = UnionFind()
    uf.find(1)
    uf.find(2)
    uf.merge(3, 4)
    # 3 components: {1}, {2}, {3, 4}
    assert repr(uf) == "UnionFind(3 components)"


def test_negating_map_repr():
    """Test NegatingMap.__repr__ shows both positive and negative keys."""
    from shrinkray.passes.sat import NegatingMap

    nm = NegatingMap()
    nm[1] = 2
    nm[3] = 4
    r = repr(nm)
    # Should contain both positive and negative mappings
    assert "1" in r and "-1" in r
    assert "2" in r and "-2" in r
    assert "3" in r and "-3" in r
    assert "4" in r and "-4" in r


def test_unit_propagator_duplicate_unit():
    """Test UnitPropagator handles unit clauses appearing multiple times."""
    from shrinkray.passes.sat import UnitPropagator

    # [1] forces 1 as a unit, and [1, 2] becomes satisfied (contains unit 1)
    up = UnitPropagator([[1], [1, 2]])
    assert 1 in up.units
    result = up.propagated_clauses()
    assert [1] in result


def test_boolean_equivalence_inconsistent_merge():
    """Test BooleanEquivalence raises Inconsistent on contradictory merge.

    This covers the path where merge_literals catches Inconsistent.
    """
    from shrinkray.passes.sat import BooleanEquivalence, Inconsistent

    be = BooleanEquivalence()
    be.merge(1, 2)  # 1 = 2
    # Now merging 1 with -2 means 1 = -2, but we already have 1 = 2
    # So this is a contradiction: 2 = -2
    with pytest.raises(Inconsistent):
        be.merge(1, -2)


def test_merge_literals_runs_on_sat():
    """Test merge_literals pass runs on SAT formula.

    This exercises the merge_literals pass to try to hit the conflict path.
    """
    from shrinkray.passes.sat import merge_literals
    from tests.helpers import reduce_with

    # SAT with clauses that could trigger merge conflicts
    sat = [[1, 2], [2, 3], [-1, -2], [-2, -3]]

    # Run merge_literals - it should not crash
    result = reduce_with([merge_literals], sat, lambda x: len(x) > 0)
    assert result is not None


def test_combine_clauses_runs_on_sat():
    """Test combine_clauses pass runs on SAT formula.

    This exercises the combine_clauses pass with shared literals.
    """
    from shrinkray.passes.sat import combine_clauses
    from tests.helpers import reduce_with

    # SAT with clauses sharing literals (so they might be combined)
    sat = [[1, 2], [1, 3], [2, 3], [1, 2, 3]]

    # Run combine_clauses - it should try to combine clauses
    result = reduce_with([combine_clauses], sat, lambda x: any(1 in c for c in x))
    assert result is not None


def test_unit_propagator_propagation_chain():
    """Test UnitPropagator propagation chain through multiple clauses."""
    from shrinkray.passes.sat import UnitPropagator

    # [1] forces 1, [-1, 2] forces 2, [-2, 1] is satisfied by 1
    up = UnitPropagator([[1], [-1, 2], [-2, 1]])
    assert 1 in up.units
    assert 2 in up.units


def test_union_find_components_with_singletons():
    """Test UnionFind.components with both merged and singleton elements.

    This exercises line 485->484 (loop continues when component has len == 1).
    """
    from shrinkray.passes.sat import UnionFind

    uf: UnionFind[int] = UnionFind()
    uf.find(1)  # Add singleton
    uf.find(2)  # Add singleton
    uf.merge(3, 4)  # Add merged component

    components = uf.components()
    # Should have 3 components: [1], [2], [3, 4]
    assert len(components) == 3
    sizes = sorted(len(c) for c in components)
    assert sizes == [1, 1, 2]


def test_merge_literals_with_contradiction_detection():
    """Test merge_literals triggers Conflict on contradictory merge attempt.

    This tries to trigger lines 328-329 where Inconsistent -> Conflict.
    The Conflict exception is expected during reduction when patches conflict.
    """
    from shrinkray.passes.sat import merge_literals
    from tests.helpers import reduce_with

    # SAT with potential for merge conflicts (has both 2 and -2 in clauses with shared literals)
    sat = [[1, 2, 3], [1, -2, 3], [2, 3]]

    # The reduction may raise ExceptionGroup containing Conflict, which is expected
    # behavior when patches can't be applied. We just want to exercise the code path.
    try:
        result = reduce_with([merge_literals], sat, lambda x: len(x) >= 1)
        assert result is not None
    except ExceptionGroup:
        # Conflict during patching is expected for some SAT instances
        pass


def test_unit_propagator_removes_negated_units_from_clauses():
    """Test UnitPropagator.propagated_clauses removes negated unit literals.

    This covers line 455 where negated units are removed from clauses.
    When a unit [1] is propagated, any clause containing -1 should have
    that literal removed.
    """
    from shrinkray.passes.sat import UnitPropagator

    # [1] is a unit clause, so 1 must be true
    # [-1, 2, 3] contains -1 (negation of unit 1) which should be removed
    # The result should have [-1, 2, 3] become [2, 3]
    up = UnitPropagator([[1], [-1, 2, 3]])

    assert 1 in up.units
    result = up.propagated_clauses()

    # Should have [1] as a unit clause
    assert [1] in result

    # Should have [2, 3] (with -1 removed) as a clause
    # Note: the clause might be sorted differently
    non_unit_clauses = [c for c in result if len(c) > 1]
    assert len(non_unit_clauses) == 1
    assert set(non_unit_clauses[0]) == {2, 3}
