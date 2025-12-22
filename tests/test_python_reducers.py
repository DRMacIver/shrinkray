import os
from glob import glob

import pytest

from shrinkray.passes.python import (
    PYTHON_PASSES,
    lift_indented_constructs,
    replace_bodies_with_ellipsis,
    replace_statements_with_pass,
    strip_annotations,
)
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
            [lift_indented_constructs], "with ...:\n    x = 1", lambda x: True
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


IF_BLOCK = """
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
