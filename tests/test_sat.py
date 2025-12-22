import operator
import os
import subprocess
import tempfile
from shutil import which
from typing import Callable

import pytest
from hypothesis import assume, example, given, settings, strategies as st

from shrinkray.passes.sat import SAT, SAT_PASSES, DimacsCNF

from .helpers import reduce_with

HAS_MINISAT = which("minisat") is not None
pytestmark = pytest.mark.skipif(not HAS_MINISAT, reason="not installed")

sat_settings = settings(deadline=None)


class MinisatNotInstalled(Exception): ...


def check_minisat():
    if not HAS_MINISAT:
        raise MinisatNotInstalled()


def is_satisfiable(clauses):
    check_minisat()
    if not clauses:
        return True
    if not all(clauses):
        return False

    f, sat_file = tempfile.mkstemp()
    os.close(f)

    with open(sat_file, "wb") as o:
        o.write(DimacsCNF.dumps(clauses))
    try:
        result = subprocess.run(
            ["minisat", sat_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        assert result in (10, 20)
        return result == 10
    finally:
        os.unlink(sat_file)


def find_solution(clauses):
    if not clauses:
        return []
    if not all(clauses):
        return None

    f, sat_file = tempfile.mkstemp()
    os.close(f)

    f, out_file = tempfile.mkstemp()
    os.close(f)

    with open(sat_file, "wb") as o:
        o.write(DimacsCNF.dumps(clauses))
    try:
        result = subprocess.run(
            ["minisat", sat_file, out_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        assert result in (10, 20)
        if result == 20:
            return None
        with open(out_file) as i:
            satline, resultline = i
        assert satline == "SAT\n"
        result = list(map(int, resultline.strip().split()))
        assert result[-1] == 0
        result.pop()
        return result
    finally:
        os.unlink(sat_file)
        os.unlink(out_file)


@st.composite
def sat_clauses(draw, min_clause_size=1):
    n_variables = draw(st.integers(min_clause_size, min(10, min_clause_size * 2)))
    variables = range(1, n_variables + 1)

    literal = st.builds(
        operator.mul, st.sampled_from(variables), st.sampled_from((-1, 1))
    )

    return draw(
        st.lists(st.lists(literal, unique_by=abs, min_size=min_clause_size), min_size=1)
    )


@st.composite
def unsatisfiable_clauses(draw, min_clause_size=1):
    clauses = draw(sat_clauses(min_clause_size=min_clause_size))
    assume(clauses)

    while True:
        sol = find_solution(clauses)
        if sol is None:
            return clauses
        assert len(sol) >= min_clause_size, (sol, clauses)
        subset = draw(
            st.lists(st.sampled_from(sol), min_size=min_clause_size, unique=True)
        )
        clauses.append([-n for n in subset])


@st.composite
def has_unique_solution(draw):
    clauses = draw(sat_clauses(min_clause_size=2))
    sol = find_solution(clauses)
    assume(sol is not None)
    assert sol is not None

    while True:
        other_sol = find_solution(clauses + [[-literal for literal in sol]])
        if other_sol is None:
            assert is_satisfiable(clauses)
            return clauses

        to_rule_out = sorted(set(other_sol) - set(sol))
        assert to_rule_out
        subset = draw(
            st.lists(
                st.sampled_from(to_rule_out),
                min_size=min(2, len(to_rule_out)),
                unique=True,
            )
        )
        clauses.append([-n for n in subset])


def shrink_sat(clauses: SAT, test_function: Callable[[SAT], bool]) -> SAT:
    return reduce_with(SAT_PASSES, clauses, test_function)


@sat_settings
@example([[1]])
@given(sat_clauses())
def test_shrink_to_one_single_literal_clause(clauses):
    result = shrink_sat(clauses, any)
    assert result == [[1]]


@pytest.mark.parametrize("n", range(2, 11))
def test_can_shrink_chain_to_two(n):
    chain = [[-i, i + 1] for i in range(1, n + 1)]

    def test(clauses):
        clauses = list(clauses)
        return (
            is_satisfiable(clauses)
            and is_satisfiable(clauses + [[1], [n]])
            and is_satisfiable(clauses + [[-1], [-n]])
            and not is_satisfiable(clauses + [[1], [-n]])
        )

    assert test(chain)

    shrunk = shrink_sat(chain, test)

    assert shrunk == [[-1, n]]


@pytest.mark.slow
@sat_settings
@given(unsatisfiable_clauses())
def test_reduces_unsatisfiable_to_trivial(unsat):
    def test(clauses):
        return clauses and all(clauses) and not is_satisfiable(clauses)

    shrunk = shrink_sat(unsat, test)

    assert shrunk == [
        [-1],
        [
            1,
        ],
    ]


@pytest.mark.slow
@sat_settings
@example([[-1], [-2], [-3], [-4, -5], [4, 5], [-6], [4, -5]])
@given(has_unique_solution())
def test_reduces_unique_satisfiable_to_trivial(unique_sat):
    def test(clauses):
        if not clauses:
            return False
        sol = find_solution(clauses)
        if sol is None:
            return False
        return not is_satisfiable(list(clauses) + [[-literal for literal in sol]])

    shrunk = shrink_sat(unique_sat, test)
    assert test(shrunk)

    assert shrunk == [[1]]
