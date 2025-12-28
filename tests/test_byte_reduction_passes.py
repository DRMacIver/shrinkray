import ast

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from shrinkray.passes.bytes import (
    ByteReplacement,
    debracket,
    find_ngram_endpoints,
    line_sorter,
    lower_bytes,
    lower_individual_bytes,
    short_deletions,
)
from shrinkray.passes.patching import apply_patches
from shrinkray.problem import BasicReductionProblem, shortlex
from shrinkray.work import WorkContext
from tests.helpers import assert_reduces_to, direct_reductions, reduce_with


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
def test_line_sorting_can_put_shorter_line_first_natural(parallelism):
    assert_reduces_to(
        origin=b"aaa\nbb",
        target=b"bb\naaa",
        parallelism=parallelism,
        passes=[line_sorter],
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_line_sorting_can_put_shorter_line_first_shortlex(parallelism):
    assert_reduces_to(
        origin=b"bb\naaa",
        target=b"aaa\nbb",
        parallelism=parallelism,
        passes=[line_sorter],
        sort_key=shortlex,
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_line_sorting_does_not_change_already_sorted(parallelism):
    reductions = direct_reductions(
        origin=b"bb\naaa",
        parallelism=parallelism,
        passes=[line_sorter],
    )
    assert not reductions


@pytest.mark.parametrize("parallelism", [1, 2])
def test_line_sorting_no_progress(parallelism):
    initial = b"aaa\nbb"
    outcome = reduce_with(
        [line_sorter], initial, lambda x: x == initial, parallelism=parallelism
    )
    assert initial == outcome
