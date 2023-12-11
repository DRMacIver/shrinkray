import ast
from asyncio import _leave_task
from typing import Callable
import libcst

from shrinkray.problem import ReductionProblem
import libcst.matchers as m
from libcst import codemod

from shrinkray.work import NotFound


def is_python(source):
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        pass


async def libcst_transform(
    problem: ReductionProblem[bytes],
    matcher: m.BaseMatcherNode,
    transformer: Callable[
        [libcst.CSTNode],
        libcst.CSTNode | libcst.RemovalSentinel | libcst.FlattenSentinel,
    ],
):
    class CM(codemod.VisitorBasedCodemodCommand):
        def __init__(self, context: codemod.CodemodContext, target_index: int):
            super().__init__(context)
            self.target_index = target_index
            self.current_index = 0
            self.fired = False

        @m.leave(matcher)
        def maybe_change_node(self, _, updated_node):
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

    async def can_apply(i):
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


async def lift_indented_constructs(problem: ReductionProblem[bytes]):
    await libcst_transform(
        problem,
        m.OneOf(m.While(), m.If(), m.Try(), m.With()),
        lambda x: libcst.FlattenSentinel(x.body.body),
    )
