import ast
from collections import Counter

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

from shrinkray.passes.bytes import (
    WHITESPACE,
    ByteReplacement,
    debracket,
    find_ngram_endpoints,
    line_sorter,
    lower_bytes,
    lower_individual_bytes,
    short_deletions,
    sort_whitespace,
)
from shrinkray.passes.patching import apply_patches
from shrinkray.passes.python import is_python
from shrinkray.problem import BasicReductionProblem, shortlex
from shrinkray.work import WorkContext
from tests.helpers import assert_reduces_to, reduce_with


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
@example(b"aaab")  # Triggers branch 103->102 (overlapping indices skipped)
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


@pytest.mark.skip
@example(initial=b"\n0\r")
@given(st.builds(bytes, st.lists(st.sampled_from(b"\t\n\r 0123."))))
def test_sorting_whitespace(initial):
    initial_count = count_whitespace(initial)

    assume(initial_count > 0)

    def is_interesting(tc):
        assert count_whitespace(tc) == initial_count
        assert Counter(tc) == Counter(initial)
        return True

    result = reduce_with([sort_whitespace], initial, is_interesting)

    for i, c in enumerate(result):
        assert (c in WHITESPACE) == (i < initial_count)


@pytest.mark.skip
@given(st.builds(bytes, st.lists(st.sampled_from(b"\t\n\r 0123."))))
def test_sorting_whitespace_preserving_regions(initial):
    initial_count = count_whitespace(initial)
    initial_regions = count_regions(initial)

    assume(initial_count > 0)
    assume(initial_regions > 1)

    def is_interesting(tc):
        assert count_whitespace(tc) == initial_count
        assert Counter(tc) == Counter(initial)
        return count_regions(tc) == initial_regions

    result = reduce_with([sort_whitespace], initial, is_interesting)

    for run in runs_of_whitespace(result)[1:]:
        assert len(run) <= 1


@pytest.mark.skip
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

    for run in runs_of_whitespace(result)[1:]:
        assert len(run) <= 1


def runs_of_whitespace(result):
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
    return whitespace_runs


def test_debracket():
    assert (
        reduce_with([debracket], b"(1 + 2) + (3 + 4)", lambda x: b"(3 + 4)" in x)
        == b"1 + 2 + (3 + 4)"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_byte_reduction_example_1(parallelism):
    assert (
        reduce_with(
            [lower_bytes],
            b"\x00\x03",
            lambda x: (len(x) == 2 and x[1] >= 2),
            parallelism=parallelism,
        )
        == b"\x00\x02"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_byte_reduction_example_2(parallelism):
    assert (
        reduce_with(
            [lower_bytes, lower_individual_bytes],
            b"\x03\x00",
            lambda x: (len(x) == 2 and x[0] >= 2),
            parallelism=parallelism,
        )
        == b"\x02\x00"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_byte_reduction_example_3(parallelism):
    assert_reduces_to(
        origin=b"1200\x00\x01\x02",
        target=b"120",
        parallelism=parallelism,
        sort_key=shortlex,
    )


@st.composite
def lowering_problem(draw):
    initial = bytes(draw(st.lists(st.integers(0, 255), unique=True, min_size=1)))
    target = bytes([draw(st.integers(0, c)) for c in initial])
    patches = draw(
        st.permutations([{c: d} for c, d in zip(initial, target, strict=True)])
    )

    if len(initial) > 1:
        n_pair_patches = draw(st.integers(0, len(initial) * (len(initial) - 1)))
        pair_patch_indexes = draw(
            st.lists(
                st.lists(
                    st.integers(0, len(initial) - 1),
                    min_size=2,
                    max_size=2,
                    unique=True,
                ),
                min_size=n_pair_patches,
                max_size=n_pair_patches,
            )
        )
        patches += [{initial[i]: target[i] for i in ls} for ls in pair_patch_indexes]

    return (initial, target, patches)


async def always_true(x):
    return True


@given(lowering_problem(), st.integers(1, 5))
@example(lowering=(b"\x01\x02", b"\x00\x00", [{1: 0}, {2: 0}]), parallelism=2).via(
    "discovered failure"
)
@example(
    lowering=(
        b"\x00\x01\x02\x03\x04\x05\x06\x07",
        b"\x00\x00\x00\x00\x00\x00\x00\x00",
        [{0: 0}, {1: 0}, {2: 0}, {3: 0}, {4: 0}, {5: 0}, {6: 0}, {7: 0}],
    ),
    parallelism=3,
).via("discovered failure")
async def test_apply_byte_replacement_patches(lowering, parallelism):
    initial, target, patches = lowering

    trivial_problem = BasicReductionProblem(
        initial, always_true, work=WorkContext(parallelism=parallelism)
    )

    await apply_patches(trivial_problem, ByteReplacement(), patches)

    assert trivial_problem.current_test_case == target


@pytest.mark.parametrize("parallelism", [1, 2])
def test_line_sorting_can_put_shorter_line_first(parallelism):
    assert_reduces_to(
        origin=b"aaa\nbb",
        target=b"bb\naaa",
        parallelism=parallelism,
        passes=[line_sorter],
    )
