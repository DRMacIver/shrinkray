from shrinkray.passes.python import lift_indented_constructs

from tests.helpers import reduce_with


def test_can_replace_blocks_with_body():
    body = reduce_with(
        [lift_indented_constructs], b"if True:\n    x = 1", lambda t: b"x" in t
    )
    assert body == b"x = 1"
