import ast

from shrinkray.passes.bytes import short_deletions

from tests.helpers import reduce_with


def is_hello(data: bytes) -> bool:
    try:
        tree = ast.parse(data)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "Hello world!":
            return True

    return False


def test_short_deletions_can_delete_brackets() -> None:
    assert (
        reduce_with([short_deletions], b'"Hello world!"()', is_hello)
        == b'"Hello world!"'
    )
