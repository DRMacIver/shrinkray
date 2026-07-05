import ast

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from shrinkray.passes.bytes import (
    ByteReplacement,
    SuffixRaises,
    debracket,
    find_ngram_endpoints,
    line_sorter,
    lower_bytes,
    lower_individual_bytes,
    lower_with_suffix_raises,
    short_deletions,
    whitespace_layout_candidates,
)
from shrinkray.passes.patching import Conflict, apply_patches
from shrinkray.problem import BasicReductionProblem, shortlex, sort_key_for_initial
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
    # Uses shortlex so the expected minimum is the numerically lowered
    # byte; under natural text ordering whitespace replacements sort lower.
    assert (
        reduce_with(
            [lower_bytes],
            b"\x00\x03",
            lambda x: (len(x) == 2 and x[1] >= 2),
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
            lambda x: (len(x) == 2 and x[0] >= 2),
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
    sort_key = sort_key_for_initial(b"a\x00")
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
    sort_key = sort_key_for_initial(b"\x00\x00")
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
def test_lowering_can_raise_suffix_natural_order(parallelism):
    """Position-by-position orderings can require lowering one byte while
    raising the bytes after it: reaching b"qqz" from b"qrq" needs position 1
    lowered from "r" at the same time as position 2 is raised above "z"."""
    sort_key = sort_key_for_initial(b"qrq")
    assert (
        reduce_with(
            [lower_with_suffix_raises, lower_individual_bytes],
            b"qrq",
            lambda x: sort_key(x) >= sort_key(b"qqz"),
            parallelism=parallelism,
            sort_key=sort_key,
        )
        == b"qqz"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_lowering_can_raise_suffix_shortlex(parallelism):
    """The shortlex version of suffix raising is numeric carrying across a
    run of positions: b"\\x98\\x00\\x00" can only descend towards
    b"\\x97\\xff\\x08" by lowering position 0 while raising the rest."""
    assert (
        reduce_with(
            [lower_with_suffix_raises, lower_individual_bytes],
            b"\x98\x00\x00",
            lambda x: shortlex(x) >= shortlex(b"\x97\xff\x08"),
            parallelism=parallelism,
            sort_key=shortlex,
        )
        == b"\x97\xff\x08"
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_can_lower_while_stripping_trailing_whitespace(parallelism):
    """Reaching b"qzzz" from b"r 00\\n" needs a chain of coupled edits: a
    suffix raise that preserves the trailing newline, then lowering the
    raised bytes, and finally a lowering step combined with stripping the
    trailing whitespace (b"qzzz\\n" sorts below the target and b"qzz~"
    sorts above the intermediate, so the last two edits must land as one)."""
    assert_reduces_to(
        origin=b"\x00\x00\x00\x00",
        target=b"qzzz",
        parallelism=parallelism,
        language_restrictions=False,
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
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_restart_replays_original_attempt_sequence(parallelism):
    """The pair replacement needed to reach this target is only attempted
    within the early-abort budget under the run's initial shuffle order, so
    the restart phase must replay the original random state for the same
    candidates to stay reachable (found by the generic shrinking
    properties)."""
    assert_reduces_to(
        origin=b"1N\x94\xcd\xb5\x13\x10hr\x87\x9b\xa0'yI",
        target=b"1N\x94\xcd\xb5\x13\x1ehr\x87\x9b\x1e'yI",
        parallelism=parallelism,
        language_restrictions=False,
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_can_reach_whitespace_padded_target(parallelism):
    """The reflow text ordering is not length-monotone: b"\\t\\t" sorts below
    the single byte b"0", so reduction can adopt a state from which the
    target is only reachable by temporarily growing the test case. The
    whitespace padding pump must recover this."""
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


def test_suffix_raises_patch_algebra():
    patches = SuffixRaises()
    a = frozenset([(0, 1, 2, 3, b"")])
    b = frozenset([(1, 2, 3, 4, b"")])
    assert patches.combine(a) == a
    assert patches.combine(a, patches.empty) == a
    with pytest.raises(Conflict):
        patches.combine(a, b)
    assert patches.apply(patches.empty, b"xyz") == b"xyz"
    assert patches.apply(a, b"xyz") == b"\x01\x02\x02"
    assert patches.size(a) == 0


async def test_whitespace_layout_skips_length_monotone_orderings():
    calls = []

    async def is_interesting(x):
        calls.append(x)
        return True

    problem = BasicReductionProblem(
        initial=b"abc",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
        sort_key=shortlex,
    )
    await whitespace_layout_candidates(problem)
    assert calls == []


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
