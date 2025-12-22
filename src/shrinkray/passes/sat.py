from collections import Counter, defaultdict
from copy import copy, deepcopy
from typing import Iterable

from shrinkray.passes.definitions import (
    DumpError,
    Format,
    ParseError,
    ReductionPass,
)
from shrinkray.passes.patching import Conflict, SetPatches, apply_patches
from shrinkray.passes.sequences import delete_elements
from shrinkray.problem import ReductionProblem

Clause = list[int]
SAT = list[Clause]


class _DimacsCNF(Format[bytes, SAT]):
    @property
    def name(self) -> str:
        return "DimacsCNF"

    def parse(self, input: bytes) -> SAT:
        try:
            contents = input.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ParseError(*e.args)
        clauses = []
        for line in contents.splitlines():
            line = line.strip()
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                continue
            if not line.strip():
                continue
            try:
                clause = list(map(int, line.strip().split()))
            except ValueError as e:
                raise ParseError(*e.args)
            if clause[-1] != 0:
                raise ParseError(f"{line} did not end with 0")
            clause.pop()
            clauses.append(clause)
        if not clauses:
            raise ParseError("No clauses found")
        return clauses

    def dumps(self, input: SAT) -> bytes:
        if not input or not all(input):
            raise DumpError(input)
        n_variables = max(abs(literal) for clause in input for literal in clause)

        parts = [f"p cnf {n_variables} {len(input)}"]

        for c in input:
            parts.append(" ".join(map(repr, list(c) + [0])))

        return "\n".join(parts).encode("utf-8")


DimacsCNF = _DimacsCNF()


async def flip_literal_signs(problem: ReductionProblem[SAT]):
    def flip_terms(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        result = list(map(list, sat))
        for i, j in terms:
            result[i][j] = abs(result[i][j])
        return result

    await apply_patches(
        problem,
        SetPatches(flip_terms),
        [
            frozenset({(i, j)})
            for i, clause in enumerate(problem.current_test_case)
            for j, v in enumerate(clause)
            if v < 0
        ],
    )
    await unit_propagate(problem)


async def remove_redundant_clauses(problem: ReductionProblem[SAT]):
    attempt = []
    seen = set()
    for clause in problem.current_test_case:
        if len(set(map(abs, clause))) < len(set(clause)):
            continue
        key = tuple(clause)
        if key in seen:
            continue
        seen.add(key)
        attempt.append(clause)
    await problem.is_interesting(attempt)


def literals_in(sat: SAT) -> frozenset[int]:
    return frozenset({literal for clause in sat for literal in clause})


async def delete_literals(problem: ReductionProblem[SAT]):
    def remove_literals(literals: frozenset[int], sat: SAT) -> SAT:
        result = []
        for clause in sat:
            new_clause = [v for v in clause if v not in literals]
            if new_clause:
                result.append(new_clause)
        return result

    await apply_patches(
        problem,
        SetPatches(remove_literals),
        [frozenset({v}) for v in literals_in(problem.current_test_case)],
    )


async def delete_single_terms(problem: ReductionProblem[SAT]):
    def remove_terms(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        result = list(map(list, sat))
        grouped = defaultdict(set)
        for i, j in terms:
            grouped[i].add(j)
        for i, js in grouped.items():
            for j in sorted(js, reverse=True):
                del result[i][j]
        return [c for c in result if c]

    await apply_patches(
        problem,
        SetPatches(remove_terms),
        [
            frozenset({(i, j)})
            for i, clause in enumerate(problem.current_test_case)
            for j in range(len(clause))
        ],
    )
    await unit_propagate(problem)


async def renumber_variables(problem: ReductionProblem[SAT]):
    variables = sorted(
        {abs(lit) for clause in problem.current_test_case for lit in clause}
    )

    def renumber(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        renumbering = {}
        for i, j in sorted(terms):
            if j not in renumbering:
                renumbering[j] = i
        result = []
        for clause in sat:
            new_clause = sorted(
                set(
                    [
                        (renumbering[lit] if lit > 0 else -renumbering[-lit])
                        if abs(lit) in renumbering
                        else lit
                        for lit in clause
                    ]
                )
            )
            if len(set(map(abs, new_clause))) == len(new_clause):
                result.append(new_clause)
        return result

    ideal_number = {v: i for i, v in enumerate(variables, 1)}
    backup_number = {}
    used = set(variables)
    i = 1
    for v in variables:
        while i in used and i <= v:
            i += 1
        if i < v:
            backup_number[v] = i

    await apply_patches(
        problem,
        SetPatches(renumber),
        [
            frozenset({(u, v)})
            for v in variables
            for u in {
                1,
                2,
                3,
                4,
                5,
                v // 3,
                v // 2,
                v - 3,
                v - 2,
                v - 1,
                ideal_number[v],
                backup_number.get(v, v),
            }
            if 0 < u < v
        ],
    )
    await unit_propagate(problem)


class UnionFind(object):
    def __init__(self, initial_merges=(), key=None):
        self.table = {}
        self.key = key
        self.generation = 0
        self.representatives = 0
        for k, v in initial_merges:
            self.merge(k, v)

    def __copy__(self):
        result = type(self).__new__(type(self))
        result.table = copy(self.table)
        result.key = self.key
        result.generation = 0
        result.representatives = self.representatives
        return result

    def clone(self):
        return self.__copy__()

    def mapping(self):
        sources = [k for k, k2 in self.table.items() if k != k2]
        return {k: self.find(k) for k in sources}

    def components(self):
        groupings = defaultdict(list)
        for k in list(self.table):
            groupings[self.find(k)].append(k)
        return list(groupings.values())

    def extend(self, other):
        if other is self:
            return
        for k, v in other.table.items():
            self.merge(k, v)

    def find(self, value):
        try:
            if self.table[value] == value:
                return value
        except KeyError:
            self.representatives += 1
            self.table[value] = value
            return value

        trail = []
        while value != self.table[value]:
            trail.append(value)
            value = self.table[value]
        for t in trail:
            self.table[t] = value
        return value

    def merge(self, left, right):
        if left == right:
            return
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        self.representatives -= 1
        self.generation += 1
        left, right = sorted((left, right), key=self.key)
        self.table[right] = left

    def merge_all(self, values):
        if len(values) > 1:
            a, *rest = sorted(values, key=self.key)
            for b in rest:
                self.merge(a, b)

    def __repr__(self):
        classes = {}
        for k in self.table:
            trail = [k]
            v = k
            while self.table[v] != v:
                v = self.table[v]
                trail.append(v)
            classes.setdefault(v, set()).update(trail)
        return "%s(%d components)" % (
            type(self).__name__,
            len(self.components()),
        )


class BooleanEquivalence(UnionFind):
    def __init__(self, initial_merges=()):
        UnionFind.__init__(self, initial_merges, key=abs)
        self.table = NegatingMap()

    def find(self, a):
        if not a:
            raise ValueError("Invalid variable %r" % (a,))
        return UnionFind.find(self, a)

    def merge(self, a, b):
        if a == b:
            return
        a2 = self.find(a)
        b2 = self.find(b)
        if a2 == b2:
            return
        if a2 == -b2:
            raise Inconsistent(
                "Attempted to merge %d (=%d) with %d (=%d)" % (a, a2, b, b2)
            )
        UnionFind.merge(self, a, b)


class NegatingMap(object):
    def __init__(self):
        self.__data = {}

    def __copy__(self):
        result = NegatingMap.__new__(NegatingMap)
        result.__data = dict(self.__data)
        return result

    def __repr__(self):
        m = {}
        for k, v in self.__data.items():
            m[k] = v
            m[-k] = -v
        return repr(m)

    def positive_keys(self):
        return self.__data.keys()

    def __iter__(self):
        yield from self.__data.keys()
        for k in self.__data.keys():
            yield -k

    def items(self):
        for k in self:
            yield (k, self[k])

    def __getitem__(self, key):
        assert key != 0
        if key < 0:
            return -self.__data[-key]
        else:
            return self.__data[key]

    def __setitem__(self, key, value):
        assert key != 0
        assert value != 0
        if key < 0:
            self.__data[-key] = -value
        else:
            self.__data[key] = value


async def merge_literals(problem: ReductionProblem[SAT]):
    variables = {abs(lit) for clause in problem.current_test_case for lit in clause}

    def apply_merges(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        uf = BooleanEquivalence()
        try:
            for u, v in terms:
                uf.merge(u, v)
        except Inconsistent:
            raise Conflict()

        result = set()
        for clause in sat:
            new_clause = frozenset(map(uf.find, clause))
            result.add(new_clause)
        return sorted([sorted(clause, key=abs) for clause in result], key=len)

    await apply_patches(
        problem,
        SetPatches(apply_merges),
        [
            frozenset({(u, -v)})
            for clause in problem.current_test_case
            for u in clause
            for v in clause
            if u != v
        ],
    )
    await unit_propagate(problem)


async def pass_to_component(problem: ReductionProblem[SAT]):
    groups = UnionFind()
    clauses = problem.current_test_case
    for clause in clauses:
        groups.merge_all(list(map(abs, clause)))
    partitions = defaultdict(list)
    for clause in clauses:
        partitions[groups.find(abs(clause[0]))].append(clause)
    if len(partitions) > 1:
        for p in sorted(partitions.values(), key=len):
            await problem.is_interesting(p)


async def sort_clauses(problem: ReductionProblem[SAT]):
    await problem.is_interesting(sorted(map(sorted, problem.current_test_case)))


class Inconsistent(Exception):
    pass


class UnitPropagator:
    def __init__(self, clauses):
        self.__clauses = list(map(tuple, clauses))
        self.__clause_counts = Counter()
        for clause in self.__clauses:
            for literal in clause:
                self.__clause_counts[abs(literal)] += 1
        self.__watches = defaultdict(frozenset)
        self.__watched_by = [frozenset() for _ in self.__clauses]

        self.units = set()
        self.forced_variables = set()
        self.__dirty = set(range(len(clauses)))
        self.__clean_dirty_clauses()
        self.__frozen = False

    def freeze(self):
        self.__frozen = True

    def add_clauses(self, clauses):
        if self.__frozen:
            raise ValueError("Frozen")
        i = len(self.__clauses)
        self.__clauses.extend(map(tuple, clauses))
        while len(self.__watched_by) < len(self.__clauses):
            self.__watched_by.append(set())
        self.__dirty.update(range(i, len(self.__clauses)))
        self.__clean_dirty_clauses()

    def add_units(self, units: Iterable[int]) -> None:
        if self.__frozen:
            raise ValueError("Frozen")
        assert not self.__dirty, self.__dirty
        for u in units:
            self.__enqueue_unit(u)
        self.__clean_dirty_clauses()

    def add_unit(self, unit: int) -> None:
        self.add_units((unit,))

    def clause_count(self, variable):
        if variable in self.forced_variables:
            return 1
        else:
            return self.__clause_counts[variable]

    def __enqueue_unit(self, unit):
        if unit not in self.units:
            if -unit in self.units:
                raise Inconsistent(
                    f"Tried to add {unit} as a unit but {-unit} is already a unit."
                )
            self.units.add(unit)
            self.forced_variables.add(abs(unit))
            self.__dirty.update(self.__watches.pop(-unit, ()))

    def __clean_dirty_clauses(self):
        iters = 0
        while self.__dirty:
            iters += 1
            assert iters <= 10**6
            dirty = self.__dirty
            self.__dirty = set()

            for i in dirty:
                clause = self.__clauses[i]
                if not clause:
                    raise Inconsistent("Clauses contain an empty clause")
                if any(literal in self.units for literal in clause):
                    for literal in self.__watched_by[i]:
                        if literal in self.__watches:
                            self.__watches[literal] -= {i}
                    self.__watched_by[i] = frozenset()
                    for literal in clause:
                        self.__clause_counts[abs(literal)] -= 1
                else:
                    for literal in list(self.__watched_by[i]):
                        if -literal in self.units:
                            self.__watched_by[i] -= {literal}
                    for literal in clause:
                        if len(self.__watched_by[i]) == 2:
                            break
                        if -literal not in self.units:
                            self.__watches[literal] |= {i}
                            self.__watched_by[i] |= {literal}
                    if len(self.__watched_by[i]) == 0:
                        raise Inconsistent(
                            f"Clause {' '.join(map(str, clause))} can no longer be satisfied"
                        )
                    elif len(self.__watched_by[i]) == 1:
                        self.__enqueue_unit(*self.__watched_by[i])

    def propagated_clauses(self):
        results = set()
        neg_units = {-v for v in self.units}
        for clause in self.__clauses:
            if any(literal in self.units for literal in clause):
                continue
            if not neg_units.isdisjoint(clause):
                clause = tuple(sorted(set(clause) - neg_units))
            results.add(clause)
        return [[literal] for literal in self.units] + sorted(
            results, key=lambda c: (len(c), list(map(abs, c)), c)
        )


async def unitize(problem: ReductionProblem[SAT]):
    clauses = list(problem.current_test_case)
    problem.work.random.shuffle(clauses)
    clauses.sort(key=len)
    i = 0
    while i < len(problem.current_test_case):
        clause = clauses[i]
        if len(clause) > 1:
            for v in sorted(clause, key=abs):
                attempt = list(clauses)
                attempt[i] = [v]
                if await problem.is_interesting(attempt):
                    break
        i += 1


async def unit_propagate(problem: ReductionProblem[SAT]):
    propagated = UnitPropagator(problem.current_test_case).propagated_clauses()
    if not await problem.is_interesting([c for c in propagated if len(c) > 1]):
        await problem.is_interesting(propagated)


async def delete_units(problem: ReductionProblem[SAT]):
    await problem.is_interesting([c for c in problem.current_test_case if len(c) > 1])


async def flip_signs(problem: ReductionProblem[SAT]):
    await problem.is_interesting(
        [[-lit for lit in clause] for clause in problem.current_test_case]
    )


async def force_literals(problem: ReductionProblem[SAT]):
    literals = literals_in(problem.current_test_case)
    for lit in literals:
        try:
            await problem.is_interesting(
                UnitPropagator(problem.current_test_case + [[lit]]).propagated_clauses()
            )
        except Inconsistent:
            pass


async def combine_clauses(problem: ReductionProblem[SAT]):
    def apply_merges(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        uf = UnionFind()
        for u, v in terms:
            uf.merge(u, v)

        result = list(map(list, sat))
        for c in uf.components():
            if len(c) > 1:
                combined = sorted({lit for i in c for lit in sat[i]}, key=abs)
                for i in c:
                    result[i] = None
                if len(combined) == len(set(map(abs, combined))):
                    result.append(combined)
        return [clause for clause in result if clause is not None]

    by_literal = defaultdict(list)
    for i, clause in enumerate(problem.current_test_case):
        for lit in clause:
            by_literal[lit].append(i)

    await apply_patches(
        problem,
        SetPatches(apply_merges),
        [
            frozenset({(i, j)})
            for group in by_literal.values()
            for i in group
            for j in group
            if i != j
        ]
        + [frozenset({(i, i + 1)}) for i in range(len(problem.current_test_case) - 1)],
    )
    await unit_propagate(problem)


SAT_PASSES: list[ReductionPass[SAT]] = [
    combine_clauses,
    force_literals,
    flip_signs,
    renumber_variables,
    pass_to_component,
    unit_propagate,
    delete_literals,
    delete_single_terms,
    sort_clauses,
    delete_elements,
    flip_literal_signs,
    merge_literals,
]
