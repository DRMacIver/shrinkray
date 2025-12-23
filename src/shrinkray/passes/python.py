from collections.abc import Callable
from typing import Any, AnyStr, cast

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


async def libcst_transform(
    problem: ReductionProblem[bytes],
    matcher: m.BaseMatcherNode,
    transformer: Callable[
        [CSTNode],
        Replacement,
    ],
) -> None:
    class CM(codemod.VisitorBasedCodemodCommand):
        def __init__(
            self, context: codemod.CodemodContext, start_index: int, end_index: int
        ):
            super().__init__(context)
            self.start_index = start_index
            self.end_index = end_index
            self.current_index = 0
            self.fired = False

        # We have to have an ignore on the return type because if we don't LibCST
        # will do some stupid bullshit with checking if the return type is correct
        # and we use this generically in a way that makes it hard to type correctly.
        @m.leave(matcher)
        def maybe_change_node(self, _, updated_node):  # type: ignore
            if self.start_index <= self.current_index < self.end_index:
                self.fired = True
                self.current_index += 1
                return transformer(updated_node)
            else:
                self.current_index += 1
                return updated_node

    try:
        module = libcst.parse_module(problem.current_test_case)
    except Exception:
        return

    context = codemod.CodemodContext()

    counting_mod = CM(context, -1, -1)
    counting_mod.transform_module(module)

    n = counting_mod.current_index + 1

    async def can_apply(start: int, end: int) -> bool:
        nonlocal n
        if start >= n:
            return False
        initial_test_case = problem.current_test_case
        try:
            module = libcst.parse_module(initial_test_case)
        except libcst.ParserSyntaxError:
            n = 0
            return False

        codemod_i = CM(context, start, end)
        try:
            transformed = codemod_i.transform_module(module)
        except libcst.CSTValidationError:
            return False
        except TypeError as e:
            if "does not allow for it" in e.args[0]:
                return False
            raise

        if not codemod_i.fired:
            n = start
            return False

        transformed_test_case = transformed.code.encode(transformed.encoding)

        if problem.sort_key(transformed_test_case) >= problem.sort_key(
            initial_test_case
        ):
            return False

        return await problem.is_interesting(transformed_test_case)

    i = 0
    while i < n:
        try:
            i = await problem.work.find_first_value(
                range(i, n), lambda i: can_apply(i, i + 1)
            )
            await problem.work.find_large_integer(lambda k: can_apply(i + 1, i + 1 + k))
            i += 1
        except NotFound:
            break


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

    def ann_assign_to_assign(x: libcst.CSTNode) -> Replacement:
        ann = cast(libcst.AnnAssign, x)
        if ann.value is not None:
            return libcst.Assign(
                targets=[libcst.AssignTarget(target=ann.target)],
                value=ann.value,
                semicolon=ann.semicolon,
            )
        else:
            return libcst.RemoveFromParent()

    await libcst_transform(problem, m.AnnAssign(), ann_assign_to_assign)


PYTHON_PASSES = [
    replace_bodies_with_ellipsis,
    strip_annotations,
    lift_indented_constructs,
    delete_statements,
    replace_statements_with_pass,
]
