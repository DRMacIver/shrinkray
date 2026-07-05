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
from tests.helpers import (
    assert_reduces_to,
    direct_reductions,
    latin1_text_sort_key,
    reduce_with,
)


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
    for k, indices in find_ngram_endpoints(b):
        # Each entry is a non-empty ngram at in-bounds, ordered positions,
        # and every position holds the same ngram.
        assert k >= 1
        assert indices
        assert indices == sorted(indices)
        assert all(0 <= i and i + k <= len(b) for i in indices)
        assert len({b[i : i + k] for i in indices}) == 1


def test_debracket():
    assert (
        reduce_with([debracket], b"(1 + 2) + (3 + 4)", lambda x: b"(3 + 4)" in x)
        == b"1 + 2 + (3 + 4)"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_byte_reduction_example_1(parallelism):
    # Uses shortlex so the expected minimum is the numerically lowered
    # byte; under natural text ordering whitespace replacements sort lower.
    assert (
        reduce_with(
            [lower_bytes],
            b"\x00\x03",
            lambda x: len(x) == 2 and x[1] >= 2,
            parallelism=parallelism,
            sort_key=shortlex,
        )
        == b"\x00\x02"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_byte_reduction_example_2(parallelism):
    assert (
        reduce_with(
            [lower_bytes, lower_individual_bytes],
            b"\x03\x00",
            lambda x: len(x) == 2 and x[0] >= 2,
            parallelism=parallelism,
            sort_key=shortlex,
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


@pytest.mark.parametrize("parallelism", [1, 2])
def test_lower_individual_bytes_descends_in_sort_key_order(parallelism):
    """Natural text ordering ranks b"z" below b"\\x00", so lowering must be
    able to move a byte to a numerically larger but lower-sorting value,
    converging on the smallest interesting one."""
    sort_key = latin1_text_sort_key
    assert (
        reduce_with(
            [lower_individual_bytes],
            b"a\x00",
            lambda x: sort_key(x) >= sort_key(b"az"),
            parallelism=parallelism,
            sort_key=sort_key,
        )
        == b"az"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_lower_bytes_descends_in_sort_key_order(parallelism):
    """As above, but for the pass that replaces all occurrences of a byte."""
    sort_key = latin1_text_sort_key
    assert (
        reduce_with(
            [lower_bytes],
            b"\x00\x00",
            lambda x: sort_key(x) >= sort_key(b"zz"),
            parallelism=parallelism,
            sort_key=sort_key,
        )
        == b"zz"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_restart_phase_escapes_greedy_corner(parallelism):
    """Greedy reduction from this origin deletes bytes first and wanders
    into whitespace-layout states from which the pair-replacement target
    cannot be reached by any single edit. The restart phase re-reduces from
    the original input constrained below the fixpoint, which excludes that
    basin and finds the target."""
    assert_reduces_to(
        origin=b"\x93\xe3.\xc5",
        target=b"\x93\t\t\xc5",
        parallelism=parallelism,
        language_restrictions=False,
        sort_key=latin1_text_sort_key,
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_can_reach_whitespace_padded_target(parallelism):
    """The reflow text ordering is not length-monotone: b"\\t\\t" sorts below
    the single byte b"0", so reduction can adopt a state from which the
    target is only reachable by temporarily growing the test case. The
    restart phase must recover this."""
    assert_reduces_to(
        origin=b"ab",
        target=b"\t\t",
        parallelism=parallelism,
        language_restrictions=False,
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_can_reach_content_preserving_padded_target(parallelism):
    """As above, but the padded target keeps some of the original content:
    b"a\\t\\t" is longer than the intermediate state b"ab" yet sorts below
    it, so it needs padding followed by byte lowering."""
    assert_reduces_to(
        origin=b"abc",
        target=b"a\t\t",
        parallelism=parallelism,
        language_restrictions=False,
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
def test_line_sorting_sorts_lines_natural(parallelism):
    # The reflow sort key canonicalises a plain two-line file by content, so the
    # sorted order is "aaa" before "bb" (like shortlex here).
    assert_reduces_to(
        origin=b"bb\naaa",
        target=b"aaa\nbb",
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
        origin=b"aaa\nbb",
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
