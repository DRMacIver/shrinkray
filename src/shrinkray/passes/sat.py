from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator

from shrinkray.passes.definitions import ReductionPass
from shrinkray.passes.patching import Conflict, SetPatches, apply_patches
from shrinkray.passes.sequences import delete_elements
from shrinkray.problem import DumpError, Format, ParseError, ReductionProblem


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
        clauses: SAT = []
        for line in contents.splitlines():
            line = line.strip()
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                continue
            if not line.strip():
                continue
            try:
                clause: Clause = list(map(int, line.strip().split()))
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
    """Make negative literals positive.

    Tries to replace negative literals (-x) with positive ones (x).
    This normalizes the formula toward using positive literals only.
    """

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


def literals_in(sat: SAT) -> frozenset[int]:
    return frozenset({literal for clause in sat for literal in clause})


async def delete_literals(problem: ReductionProblem[SAT]) -> None:
    """Remove entire literals from the formula.

    Tries to remove all occurrences of a literal (both positive and
    negative forms) from all clauses. Clauses that become empty are
    removed entirely.
    """

    def remove_literals(literals: frozenset[int], sat: SAT) -> SAT:
        result: SAT = []
        for clause in sat:
            new_clause: Clause = [v for v in clause if v not in literals]
            if new_clause:
                result.append(new_clause)
        return result

    await apply_patches(
        problem,
        SetPatches(remove_literals),
        [frozenset({v}) for v in literals_in(problem.current_test_case)],
    )


async def delete_single_terms(problem: ReductionProblem[SAT]) -> None:
    """Remove individual literal occurrences from specific clauses.

    Unlike delete_literals (which removes a literal everywhere), this
    tries removing literals from individual positions, allowing different
    clauses to keep or lose the same literal independently.
    """

    def remove_terms(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        result: list[list[int]] = [list(c) for c in sat]
        grouped: defaultdict[int, set[int]] = defaultdict(set)
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


async def renumber_variables(problem: ReductionProblem[SAT]) -> None:
    """Renumber variables to use smaller indices.

    Tries to replace variable numbers with smaller ones (1, 2, 3, etc.)
    to minimize the variable indices used. This normalizes the formula
    toward using the smallest possible variable numbers.
    """
    variables = sorted(
        {abs(lit) for clause in problem.current_test_case for lit in clause}
    )

    def renumber(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        renumbering: dict[int, int] = {}
        for i, j in sorted(terms):
            if j not in renumbering:
                renumbering[j] = i
        result: SAT = []
        for clause in sat:
            new_clause: Clause = sorted(
                {
                    (
                        (renumbering[lit] if lit > 0 else -renumbering[-lit])
                        if abs(lit) in renumbering
                        else lit
                    )
                    for lit in clause
                }
            )
            if len(set(map(abs, new_clause))) == len(new_clause):
                result.append(new_clause)
        return result

    ideal_number: dict[int, int] = {v: i for i, v in enumerate(variables, 1)}
    backup_number: dict[int, int] = {}
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


class UnionFind[T]:
    table: dict[T, T]
    key: Callable[[T], object] | None
    generation: int
    representatives: int

    def __init__(
        self,
        initial_merges: Iterable[tuple[T, T]] = (),
        key: Callable[[T], object] | None = None,
    ) -> None:
        self.table = {}
        self.key = key
        self.generation = 0
        self.representatives = 0
        for k, v in initial_merges:
            self.merge(k, v)

    def components(self) -> list[list[T]]:
        groupings: defaultdict[T, list[T]] = defaultdict(list)
        for k in list(self.table):
            groupings[self.find(k)].append(k)
        return list(groupings.values())

    def find(self, value: T) -> T:
        try:
            if self.table[value] == value:
                return value
        except KeyError:
            self.representatives += 1
            self.table[value] = value
            return value

        trail: list[T] = []
        while value != self.table[value]:
            trail.append(value)
            value = self.table[value]
        for t in trail:
            self.table[t] = value
        return value

    def merge(self, left: T, right: T) -> None:
        if left == right:
            return
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        self.representatives -= 1
        self.generation += 1
        left, right = sorted((left, right), key=self.key)  # type: ignore[arg-type]
        self.table[right] = left

    def merge_all(self, values: list[T]) -> None:
        if len(values) > 1:
            sorted_values: list[T] = sorted(values, key=self.key)  # type: ignore[arg-type]
            a: T = sorted_values[0]  # type: ignore[reportUnknownVariableType]
            for b in sorted_values[1:]:  # type: ignore[reportUnknownVariableType]
                self.merge(a, b)  # type: ignore[reportUnknownArgumentType]

    def __repr__(self) -> str:
        return "%s(%d components)" % (
            type(self).__name__,
            len(self.components()),
        )


class BooleanEquivalence(UnionFind[int]):
    table: "NegatingMap"  # type: ignore[reportIncompatibleVariableOverride]

    def __init__(self, initial_merges: Iterable[tuple[int, int]] = ()) -> None:
        super().__init__(initial_merges, key=abs)
        self.table = NegatingMap()  # pyright: ignore[reportIncompatibleVariableOverride]

    def find(self, value: int) -> int:
        if not value:
            raise ValueError(f"Invalid variable {value!r}")
        return super().find(value)

    def merge(self, left: int, right: int) -> None:
        if left == right:
            return
        left2 = self.find(left)
        right2 = self.find(right)
        if left2 == right2:
            return
        if left2 == -right2:
            raise Inconsistent(
                "Attempted to merge %d (=%d) with %d (=%d)"
                % (left, left2, right, right2)
            )
        super().merge(left, right)


class NegatingMap:
    _data: dict[int, int]

    def __init__(self) -> None:
        self._data = {}

    def __repr__(self) -> str:
        m: dict[int, int] = {}
        for k, v in self._data.items():
            m[k] = v
            m[-k] = -v
        return repr(m)

    def __iter__(self) -> Iterator[int]:
        yield from self._data.keys()
        for k in self._data.keys():
            yield -k

    def __getitem__(self, key: int) -> int:
        assert key != 0
        if key < 0:
            return -self._data[-key]
        else:
            return self._data[key]

    def __setitem__(self, key: int, value: int) -> None:
        assert key != 0
        assert value != 0
        if key < 0:
            self._data[-key] = -value
        else:
            self._data[key] = value


async def merge_literals(problem: ReductionProblem[SAT]) -> None:
    """Merge pairs of literals into single variables.

    Tries to identify pairs of literals that can be treated as equivalent
    (or negations of each other) and replaces them with a single variable.
    This reduces the number of distinct variables in the formula.
    """

    def apply_merges(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        uf = BooleanEquivalence()
        try:
            for u, v in terms:
                uf.merge(u, v)
        except Inconsistent:
            raise Conflict()

        result: set[frozenset[int]] = set()
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


async def pass_to_component(problem: ReductionProblem[SAT]) -> None:
    """Try to reduce to a single connected component.

    If the formula can be split into independent components (clauses that
    share no variables), tries each component individually to see if any
    single component is sufficient to maintain interestingness.
    """
    groups: UnionFind[int] = UnionFind()
    clauses = problem.current_test_case
    for clause in clauses:
        groups.merge_all(list(map(abs, clause)))
    partitions: defaultdict[int, SAT] = defaultdict(list)
    for clause in clauses:
        partitions[groups.find(abs(clause[0]))].append(clause)
    if len(partitions) > 1:
        for p in sorted(partitions.values(), key=len):
            await problem.is_interesting(p)


async def sort_clauses(problem: ReductionProblem[SAT]) -> None:
    """Sort clauses and literals into canonical order.

    Sorts literals within each clause and sorts clauses themselves.
    This normalizes the formula representation for consistent output.
    """
    await problem.is_interesting(sorted(map(sorted, problem.current_test_case)))


class Inconsistent(Exception):
    pass


class UnitPropagator:
    __clauses: list[tuple[int, ...]]
    __clause_counts: Counter[int]
    __watches: defaultdict[int, frozenset[int]]
    __watched_by: list[frozenset[int]]
    units: set[int]
    forced_variables: set[int]
    __dirty: set[int]

    def __init__(self, clauses: Iterable[Iterable[int]]) -> None:
        self.__clauses = [tuple(c) for c in clauses]
        self.__clause_counts = Counter()
        for clause in self.__clauses:
            for literal in clause:
                self.__clause_counts[abs(literal)] += 1
        self.__watches = defaultdict(frozenset)
        self.__watched_by = [frozenset() for _ in self.__clauses]

        self.units = set()
        self.forced_variables = set()
        self.__dirty = set(range(len(self.__clauses)))
        self.__clean_dirty_clauses()

    def __enqueue_unit(self, unit: int) -> None:
        # Invariant: unit should not already be in self.units because satisfied
        # clauses are skipped at line 424 before we try to enqueue their units.
        assert unit not in self.units, f"unit {unit} already enqueued"
        # Invariant: -unit should not be in self.units because we only add
        # literals to watched_by if their negation is not in units (line 438).
        assert -unit not in self.units, (
            f"Tried to add {unit} as a unit but {-unit} is already a unit"
        )
        self.units.add(unit)
        self.forced_variables.add(abs(unit))
        self.__dirty.update(self.__watches.pop(-unit, ()))

    def __clean_dirty_clauses(self) -> None:
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

    def propagated_clauses(self) -> SAT:
        results: set[tuple[int, ...]] = set()
        neg_units = {-v for v in self.units}
        for clause in self.__clauses:
            if any(literal in self.units for literal in clause):
                continue
            if not neg_units.isdisjoint(clause):
                clause = tuple(sorted(set(clause) - neg_units))
            results.add(clause)
        return [[literal] for literal in self.units] + [
            list(c)
            for c in sorted(results, key=lambda c: (len(c), list(map(abs, c)), c))
        ]


async def unit_propagate(problem: ReductionProblem[SAT]) -> None:
    """Apply unit propagation to simplify the formula.

    Finds unit clauses (single-literal clauses) and propagates their
    implications: removes satisfied clauses and removes the negated
    literal from other clauses. This is a standard SAT preprocessing step.
    """
    try:
        propagated = UnitPropagator(problem.current_test_case).propagated_clauses()
    except Inconsistent:
        # Clauses are unsatisfiable, nothing to propagate
        return
    if not await problem.is_interesting([c for c in propagated if len(c) > 1]):
        await problem.is_interesting(propagated)


async def force_literals(problem: ReductionProblem[SAT]) -> None:
    """Try forcing each literal to a specific value.

    For each literal in the formula, tries adding it as a unit clause
    and propagating. If the result is interesting, the formula is
    simplified by that forced assignment.
    """
    literals = literals_in(problem.current_test_case)
    for lit in literals:
        try:
            await problem.is_interesting(
                UnitPropagator(problem.current_test_case + [[lit]]).propagated_clauses()
            )
        except Inconsistent:
            pass


async def combine_clauses(problem: ReductionProblem[SAT]) -> None:
    """Merge pairs of clauses into single clauses.

    Tries to combine clauses that share literals, creating a single
    clause containing all literals from both. This reduces clause count
    while potentially creating larger but fewer clauses.
    """

    def apply_merges(terms: frozenset[tuple[int, int]], sat: SAT) -> SAT:
        uf: UnionFind[int] = UnionFind()
        for u, v in terms:
            uf.merge(u, v)

        result: list[Clause | None] = [list(c) for c in sat]
        for c in uf.components():
            # Note: len(c) == 1 can't occur because every element in uf
            # came from a merge(u, v) pair where u != v, so all
            # components have size >= 2.
            combined: Clause = sorted({lit for i in c for lit in sat[i]}, key=abs)
            for i in c:
                result[i] = None
            if len(combined) == len(set(map(abs, combined))):
                result.append(combined)
        return [clause for clause in result if clause is not None]

    by_literal: defaultdict[int, list[int]] = defaultdict(list)
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
    sort_clauses,
    force_literals,
    pass_to_component,
    unit_propagate,
    delete_literals,
    delete_single_terms,
    delete_elements,
    flip_literal_signs,
    combine_clauses,
    merge_literals,
    renumber_variables,
]
