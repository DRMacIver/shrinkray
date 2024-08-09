from shrinkray.passes.python import (
    lift_indented_constructs,
    replace_bodies_with_ellipsis,
    replace_statements_with_pass,
    strip_annotations,
    PYTHON_PASSES,
)

from tests.helpers import reduce, reduce_with


def test_can_replace_blocks_with_body() -> None:
    body = reduce_with(
        [lift_indented_constructs], b"if True:\n    x = 1", lambda t: b"x" in t
    )
    assert body == b"x = 1"


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


def test_body_replacement():
    assert (
        reduce_with([replace_bodies_with_ellipsis], IF_BLOCK, lambda x: True).strip()
        == b"if True:\n    ..."
    )
