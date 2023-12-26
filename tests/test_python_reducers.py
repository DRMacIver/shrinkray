from shrinkray.passes.python import (
    lift_indented_constructs,
    replace_statements_with_pass,
)

from tests.helpers import reduce_with


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
