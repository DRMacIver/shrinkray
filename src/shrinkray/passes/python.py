from typing import Any, AnyStr, Callable

import libcst
import libcst.matchers as m
from libcst import CSTNode, codemod

from shrinkray.problem import ReductionProblem
from shrinkray.work import NotFound


def is_python(source: AnyStr) -> bool:
    try:
        libcst.parse_module(source)
        return True
    except (SyntaxError, UnicodeDecodeError, libcst.ParserSyntaxError, Exception):
        return False


Replacement = CSTNode | libcst.RemovalSentinel | libcst.FlattenSentinel[Any]
called = 0


async def libcst_transform(
    problem: ReductionProblem[bytes],
    matcher: m.BaseMatcherNode,
    transformer: Callable[
        [CSTNode],
        Replacement,
    ],
) -> None:
    class CM(codemod.VisitorBasedCodemodCommand):
        def __init__(self, context: codemod.CodemodContext, target_indices: list[int]):
            super().__init__(context)
            self.target_indices = set(target_indices)
            self.current_index = 0
            self.fired_indices = set()

        @m.leave(matcher)
        def maybe_change_node(self, _, updated_node):
            global called
            print(f"{called=}")
            called += 1
            if self.current_index in self.target_indices:
                self.fired_indices.add(self.current_index)
                return transformer(updated_node)
            else:
                self.current_index += 1
                return updated_node

    context = codemod.CodemodContext()

    def get_node_count() -> int:
        try:
            module = libcst.parse_module(problem.current_test_case)
            counting_mod = CM(context, [])
            counting_mod.transform_module(module)
            return counting_mod.current_index + 1
        except Exception:
            return 0

    async def apply_transformation(i: int) -> bool:
        try:
            module = libcst.parse_module(problem.current_test_case)
        except libcst.ParserSyntaxError:
            return False

        codemod_i = CM(context, [i])
        try:
            transformed = codemod_i.transform_module(module)
        except (libcst.CSTValidationError, TypeError):
            return False

        if not codemod_i.fired_indices:
            return False

        transformed_test_case = transformed.code.encode(transformed.encoding)
        if problem.sort_key(transformed_test_case) >= problem.sort_key(problem.current_test_case):
            return False

        if await problem.is_interesting(transformed_test_case):
            problem.current_test_case = transformed_test_case
            return True
        return False

    n = get_node_count()
    i = 0
    while i < n:
        if await apply_transformation(i):
            n = get_node_count()  # Recount nodes after successful transformation
            i = 0  # Start over to catch new opportunities
        else:
            i += 1


async def lift_indented_constructs(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.OneOf(m.While(), m.If(), m.Try()),
        lambda x: x.with_changes(orelse=None),
    )

    await libcst_transform(
        problem,
        m.OneOf(m.While(), m.If(), m.Try(), m.With()),
        lambda x: libcst.FlattenSentinel(x.body.body),  # type: ignore
    )


async def delete_statements(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.SimpleStatementLine(),
        lambda x: libcst.RemoveFromParent(),  # type: ignore
    )


async def replace_statements_with_pass(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.SimpleStatementLine(),
        lambda x: x.with_changes(body=[libcst.Pass()]),  # type: ignore
    )


ELLIPSIS_STATEMENT = libcst.parse_statement("...")


async def replace_bodies_with_ellipsis(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.IndentedBlock(),
        lambda x: x.with_changes(body=[ELLIPSIS_STATEMENT]),  # type: ignore
    )


async def strip_annotations(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.FunctionDef(),
        lambda x: x.with_changes(returns=None),
    )
    await libcst_transform(
        problem,
        m.Param(),
        lambda x: x.with_changes(annotation=None),
    )
    await libcst_transform(
        problem,
        m.AnnAssign(),
        lambda x: (
            libcst.Assign(
                targets=[libcst.AssignTarget(target=x.target)],
                value=x.value,
                semicolon=x.semicolon,
            )
            if x.value
            else libcst.RemoveFromParent()
        ),
    )


PYTHON_PASSES = [
    replace_bodies_with_ellipsis,
    # strip_annotations,
    # lift_indented_constructs,
    # delete_statements,
    # replace_statements_with_pass,
]
