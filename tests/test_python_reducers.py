import os
from glob import glob

import libcst
import libcst.matchers as m
import pytest
import trio

from shrinkray.passes.python import (
    PYTHON_PASSES,
    is_python,
    libcst_transform,
    lift_indented_constructs,
    replace_bodies_with_ellipsis,
    replace_statements_with_pass,
    strip_annotations,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import WorkContext
from tests.helpers import reduce, reduce_with


def test_can_replace_blocks_with_body() -> None:
    body = reduce_with(
        [lift_indented_constructs], b"if True:\n    x = 1", lambda t: b"x" in t
    )
    assert body == b"x = 1"


ELIF_BLOCK = b"""
if True:
    x = 1
elif True:
    x = 2
"""


def test_lifts_bodies_of_elif():
    assert (
        reduce_with([lift_indented_constructs], ELIF_BLOCK, lambda x: True).strip()
        == b"x = 1"
    )


def test_does_not_error_on_elif():
    assert (
        reduce_with([lift_indented_constructs], ELIF_BLOCK, lambda x: b"elif" in x)
        == ELIF_BLOCK
    )


def test_lifts_bodies_of_with():
    assert (
        reduce_with(
            [lift_indented_constructs], b"with ...:\n    x = 1", lambda x: True
        ).strip()
        == b"x = 1"
    )


def test_can_replace_statements_with_pass() -> None:
    body = reduce_with(
        [replace_statements_with_pass], b"from x import *", lambda t: len(t) > 1
    )
    assert body == b"pass"


def test_can_reduce_an_example_that_crashes_lib_cst():
    reduce(b"() if 0 else(lambda:())", lambda x: len(x) >= 5)


ANNOTATED = b"""
def has_an_annotation(x: list[int]) -> list[int]:
    y: list[int] = list(reversed(x))
    return x + y
"""

DEANNOTATED = b"""
def has_an_annotation(x):
    y = list(reversed(x))
    return x + y
"""


def test_strip_annotations():
    assert reduce_with([strip_annotations], ANNOTATED, lambda x: True) == DEANNOTATED


def test_single_annotation():
    x = b"x:A\n"
    assert reduce_with(PYTHON_PASSES, x, lambda y: True).strip() == b""


IF_BLOCK = b"""
if True:
    x = 1
    y = 2
    assert x + y
"""


def test_body_replacement_of_if():
    assert (
        reduce_with([replace_bodies_with_ellipsis], IF_BLOCK, lambda x: True).strip()
        == b"if True:\n    ..."
    )


ROOT = os.path.dirname(os.path.dirname(__file__))


PYTHON_FILES = glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True) + glob(
    os.path.join(ROOT, "tests", "**", "*.py"), recursive=True
)


@pytest.mark.slow
@pytest.mark.parametrize("pyfile", PYTHON_FILES)
def test_reduce_all(pyfile):
    with open(pyfile, "rb") as i:
        code = i.read()

    def is_interesting(x):
        return True

    reduce_with(PYTHON_PASSES, code, is_interesting)


ISSUE_12_INPUT = b"""
import asyncio
import _lsprof

if True:
    a = 1
    b = 2
    c = 3

if True:
    obj = _lsprof.Profiler()
    obj.enable()
    obj._pystart_callback(lambda: 0, 0)
    obj = None
    loop = asyncio.get_event_loop()
"""

ISSUE_12_OUTPUT = b"""
import asyncio
import _lsprof

if True:
    ...

if True:
    obj = _lsprof.Profiler()
    obj.enable()
    obj._pystart_callback(lambda: 0, 0)
    obj = None
    loop = asyncio.get_event_loop()
"""


def test_reduce_with_ellipsis_can_reduce_single_block():
    reduced = reduce_with(
        [replace_bodies_with_ellipsis],
        ISSUE_12_INPUT,
        lambda x: b"Profiler" in x,
        parallelism=1,
    )
    assert reduced == ISSUE_12_OUTPUT


def test_python_passes_handle_invalid_syntax():
    """Test that PYTHON_PASSES handle invalid Python syntax gracefully."""
    invalid_python = b"def foo(:\n    pass"  # Missing parameter

    # Should not raise, just return the original
    result = reduce_with(PYTHON_PASSES, invalid_python, lambda x: True)
    assert result == invalid_python


def test_python_passes_handle_non_python():
    """Test that PYTHON_PASSES handle non-Python content gracefully."""
    not_python = b"This is not Python code at all {{{}}}"

    # Should not raise, just return the original
    result = reduce_with(PYTHON_PASSES, not_python, lambda x: True)
    assert result == not_python


def test_libcst_validation_error_handling():
    """Test that CSTValidationError is handled gracefully.

    The Python passes can encounter situations where a transformation
    would create an invalid CST structure. These should be caught
    and handled gracefully.
    """
    # Code that when reduced might try to create invalid structures
    # This tests the CSTValidationError exception handling (line 81)
    code = b"""
class A:
    def method(self):
        pass
"""
    # Force various reductions to exercise validation paths
    result = reduce_with(PYTHON_PASSES, code, lambda x: len(x) > 10)
    assert len(result) <= len(code)


def test_python_reduction_with_complex_annotations():
    """Test reduction of code with complex type annotations."""
    code = b"""
def complex_function(
    x: list[tuple[int, str]],
    y: dict[str, list[int]] | None = None
) -> tuple[int, ...]:
    z: set[int] = set()
    return (1, 2, 3)
"""
    result = reduce_with([strip_annotations], code, lambda x: b"def" in x)
    # Annotations should be stripped
    assert b": list[tuple" not in result
    assert b"-> tuple" not in result


def test_is_python_returns_true_for_valid_python():
    """Test that is_python returns True for valid Python code."""
    valid_code = b"x = 1"
    assert is_python(valid_code) is True


def test_is_python_returns_false_for_invalid_python():
    """Test that is_python returns False for invalid Python code."""
    invalid_code = b"def foo(:"
    assert is_python(invalid_code) is False


def test_libcst_transform_handles_test_case_becoming_invalid():
    """Test that libcst_transform handles the case where the test case becomes invalid Python mid-reduction.

    This exercises lines 73-75: ParserSyntaxError when parsing during can_apply.
    The test case is valid initially but becomes invalid between calls.
    """
    # Track how many times current_test_case is accessed
    access_count = [0]
    valid_code = b"x = 1\ny = 2\nz = 3\n"
    invalid_code = b"def foo(:\n"

    class MockProblem:
        def __init__(self):
            self.work = WorkContext(parallelism=1)
            self._test_case = valid_code

        @property
        def current_test_case(self):
            access_count[0] += 1
            # First access returns valid code (for initial parse)
            # Later accesses return invalid code (to trigger ParserSyntaxError in can_apply)
            if access_count[0] <= 2:
                return valid_code
            return invalid_code

        def sort_key(self, x):
            return (len(x), x)

        async def is_interesting(self, x):
            return True

    problem = MockProblem()

    async def run_test():
        await libcst_transform(
            problem,  # type: ignore
            m.SimpleStatementLine(),
            lambda x: libcst.RemoveFromParent(),
        )

    trio.run(run_test)
    # The function should complete without raising, handling the ParserSyntaxError internally


def test_libcst_transform_handles_cst_validation_error():
    """Test that libcst_transform handles CSTValidationError gracefully.

    This exercises line 80-81: CSTValidationError during transformation.
    """
    # Create a transformer that causes validation errors
    def bad_transformer(node):
        # Try to create an invalid CST node by adding a node in wrong context
        # This should trigger CSTValidationError
        raise libcst.CSTValidationError("Test validation error")

    code = b"x = 1\n"

    async def is_interesting(x: bytes) -> bool:
        return True

    async def run_test():
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=code,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await libcst_transform(
            problem,
            m.SimpleStatementLine(),
            bad_transformer,
        )

    trio.run(run_test)
    # Should complete without raising


def test_libcst_transform_handles_type_error_does_not_allow():
    """Test that libcst_transform handles TypeError with 'does not allow for it'.

    This exercises lines 82-84: TypeError handling when transformation isn't allowed.
    """
    # Create a transformer that raises the specific TypeError
    def bad_transformer(node):
        raise TypeError("The parent node does not allow for it")

    code = b"x = 1\n"

    async def is_interesting(x: bytes) -> bool:
        return True

    async def run_test():
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=code,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        await libcst_transform(
            problem,
            m.SimpleStatementLine(),
            bad_transformer,
        )

    trio.run(run_test)
    # Should complete without raising


def test_libcst_transform_reraises_other_type_error():
    """Test that libcst_transform re-raises TypeError without 'does not allow for it'.

    This exercises line 85: the re-raise path for other TypeErrors.
    """
    # Create a transformer that raises a different TypeError
    def bad_transformer(node):
        raise TypeError("Some other type error")

    code = b"x = 1\n"

    async def is_interesting(x: bytes) -> bool:
        return True

    async def run_test():
        problem: BasicReductionProblem[bytes] = BasicReductionProblem(
            initial=code,
            is_interesting=is_interesting,
            work=WorkContext(parallelism=1),
        )
        # The TypeError gets wrapped in ExceptionGroups by trio nurseries
        with pytest.raises(ExceptionGroup):
            await libcst_transform(
                problem,
                m.SimpleStatementLine(),
                bad_transformer,
            )

    trio.run(run_test)
