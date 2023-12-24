import ast
from typing import Any, AnyStr, Callable

import libcst
import libcst.matchers as m
from libcst import CSTNode, codemod

from shrinkray.problem import ReductionProblem
from shrinkray.work import NotFound


def is_python(source: AnyStr) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
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
        def __init__(self, context: codemod.CodemodContext, target_index: int):
            super().__init__(context)
            self.target_index = target_index
            self.current_index = 0
            self.fired = False

        # We have to have an ignore on the return type because if we don't LibCST
        # will do some stupid bullshit with checking if the return type is correct
        # and we use this generically in a way that makes it hard to type correctly.
        @m.leave(matcher)
        def maybe_change_node(self, _, updated_node):  # type: ignore
            if self.current_index == self.target_index:
                self.fired = True
                return transformer(updated_node)
            else:
                self.current_index += 1
                return updated_node

    try:
        module = libcst.parse_module(problem.current_test_case)
    except libcst.ParserSyntaxError:
        return

    context = codemod.CodemodContext()

    counting_mod = CM(context, -1)
    counting_mod.transform_module(module)

    n = counting_mod.current_index + 1

    async def can_apply(i: int) -> bool:
        nonlocal n
        if i >= n:
            return False

        try:
            module = libcst.parse_module(problem.current_test_case)
        except libcst.ParserSyntaxError:
            n = 0
            return False

        codemod_i = CM(context, i)
        try:
            transformed = codemod_i.transform_module(module)
        except libcst.CSTValidationError:
            return False
        except TypeError as e:
            if "does not allow for it to be replaced" in e.args[0]:
                return False
            raise

        if not codemod_i.fired:
            n = i
            return False

        return await problem.is_interesting(
            transformed.code.encode(transformed.encoding)
        )

    i = 0
    while i < n:
        try:
            i = await problem.work.find_first_value(range(i, n), can_apply)
        except NotFound:
            break


async def lift_indented_constructs(problem: ReductionProblem[bytes]) -> None:
    await libcst_transform(
        problem,
        m.OneOf(m.While(), m.If(), m.Try(), m.With()),
        lambda x: libcst.FlattenSentinel(x.body.body),  # type: ignore
    )
