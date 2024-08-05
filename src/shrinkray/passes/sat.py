from shrinkray.passes.definitions import Format, ParseError, ReductionPass
from shrinkray.passes.patching import SetPatches, apply_patches
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
        n_variables = max(abs(literal) for clause in input for literal in clause)

        parts = [f"p cnf {n_variables} {len(input)}"]

        for c in input:
            parts.append(" ".join(map(repr, list(c) + [0])))

        return "\n".join(parts).encode("utf-8")


DimacsCNF = _DimacsCNF()


async def renumber_variables(problem: ReductionProblem[SAT]):
    renumbering = {}

    def renumber(l):
        if l < 0:
            return -renumber(-l)
        try:
            return renumbering[l]
        except KeyError:
            pass
        result = len(renumbering) + 1
        renumbering[l] = result
        return result

    renumbered = [
        [renumber(literal) for literal in clause]
        for clause in problem.current_test_case
    ]

    await problem.is_interesting(renumbered)


async def flip_literal_signs(problem: ReductionProblem[SAT]):
    seen_variables = set()
    target = problem.current_test_case
    for i in range(len(target)):
        for j, v in enumerate(target[i]):
            if abs(v) not in seen_variables and v < 0:
                attempt = []
                for clause in target:
                    new_clause = []
                    for literal in clause:
                        if abs(literal) == abs(v):
                            new_clause.append(-literal)
                        else:
                            new_clause.append(literal)
                    attempt.append(new_clause)
                if await problem.is_interesting(attempt):
                    target = attempt
            seen_variables.add(abs(v))


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


async def merge_variables(problem: ReductionProblem[SAT]):
    i = 0
    j = 1
    while True:
        variables = sorted({abs(l) for c in problem.current_test_case for l in c})
        if j >= len(variables):
            i += 1
            j = i + 1
        if j >= len(variables):
            return

        target = variables[i]
        to_replace = variables[j]

        new_clauses = []
        for c in problem.current_test_case:
            c = set(c)
            if to_replace in c:
                c.discard(to_replace)
                c.add(target)
            if -to_replace in c:
                c.discard(-to_replace)
                c.add(-target)
            if len(set(map(abs, c))) < len(c):
                continue
            new_clauses.append(sorted(c))

        assert new_clauses != problem.current_test_case
        await problem.is_interesting(new_clauses)
        if new_clauses != problem.current_test_case:
            j += 1


async def sort_clauses(problem: ReductionProblem[SAT]):
    await problem.is_interesting(sorted(map(sorted, problem.current_test_case)))


SAT_PASSES: list[ReductionPass[SAT]] = [
    sort_clauses,
    renumber_variables,
    flip_literal_signs,
    remove_redundant_clauses,
    delete_elements,
    delete_literals,
    merge_variables,
]
