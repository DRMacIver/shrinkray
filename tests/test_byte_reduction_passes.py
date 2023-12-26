import ast
from collections import Counter

import pytest
from hypothesis import assume, example, given, strategies as st

from shrinkray.passes.bytes import (
    WHITESPACE,
    find_ngram_endpoints,
    short_deletions,
    sort_whitespace,
    debracket,
)
from shrinkray.passes.python import is_python

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


@example(b"")
@example(b"\x00")
@example(b"\x00\x00")
@given(st.binary())
def test_ngram_endpoints(b):
    find_ngram_endpoints(b)


def count_whitespace(b):
    return len([c for c in b if c in WHITESPACE])


def count_regions(b):
    n = 0
    is_whitespace_last = True
    for c in b:
        typ = c in WHITESPACE
        if not typ and is_whitespace_last:
            n += 1
        is_whitespace_last = typ
    return n


@given(st.builds(bytes, st.lists(st.sampled_from(b"\t\n\r 0123."))))
def test_sorting_whitespace(initial):
    initial_count = count_whitespace(initial)

    assume(initial_count) > 0

    def is_interesting(tc):
        assert count_whitespace(tc) == initial_count
        assert Counter(tc) == Counter(initial)
        return True

    result = reduce_with([sort_whitespace], initial, is_interesting)

    for i, c in enumerate(result):
        assert (c in WHITESPACE) == (i < initial_count)


@given(st.builds(bytes, st.lists(st.sampled_from(b"\t\n\r 0123."))))
def test_sorting_whitespace_preserving_regions(initial):
    initial_count = count_whitespace(initial)
    initial_regions = count_regions(initial)

    assume(initial_count) > 0
    assume(initial_regions) > 1

    def is_interesting(tc):
        assert count_whitespace(tc) == initial_count
        assert Counter(tc) == Counter(initial)
        return count_regions(tc) == initial_regions

    result = reduce_with([sort_whitespace], initial, is_interesting)

    first_run = True
    run_length = 0
    for i, c in enumerate(result):
        if c in WHITESPACE:
            run_length += 1
        else:
            run_length = 0
            first_run = False
        if not first_run:
            assert run_length <= 1


@pytest.mark.parametrize(
    "initial",
    [
        b"\t\t\t\t\nfrom\t\t\t\t\t\t\t\t\t\t.\timport\tA\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\nfrom\t\t\t.\t\t\t\t\t\t\t\t\t\timport\t\to\t\t\t\t\t\t\t\t\nfrom\t\t.\t\t\t\t\t\t\t\timport\t\tr\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
        b"from\t\t.\t\t\t\t\t\t\t\t\timport\ta\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\nclass\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\ta\t\t\t\t\t\t\t\t\t\t\t(\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t)\t\t\t\t\t\t\t\t:\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t()\ndef\tr\t\t\t\t\t\t\t\t\t\t\t\t\t\t(\t\t\t\t\t\t\t\t\t\t\t\t\t\t)\t\t\t\t\t\t\t\t:\t\t\t\t\t\t\t\t\t\t\t\t\t\t...\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
    ],
)
def test_sorting_whitespace_preserving_python(initial):
    initial_count = count_whitespace(initial)

    def is_interesting(tc):
        assert count_whitespace(tc) == initial_count
        assert Counter(tc) == Counter(initial)
        return is_python(tc)

    result = reduce_with([sort_whitespace], initial, is_interesting)

    whitespace_runs = []
    i = 0
    while i < len(result):
        c = result[i]
        if c not in WHITESPACE:
            i += 1
            continue
        j = i + 1
        while j < len(result) and result[j] in WHITESPACE:
            j += 1
        whitespace_runs.append(result[i:j])
        i = j

    for run in whitespace_runs[1:]:
        assert len(run) <= 1

def test_debracket():
    assert (
        reduce_with([debracket], b"(1 + 2) + (3 + 4)", lambda x: b"(3 + 4)" in x)
        == b"1 + 2 + (3 + 4)"
    )
