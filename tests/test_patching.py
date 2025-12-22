"""Unit tests for patching module."""

from random import Random

import pytest

from shrinkray.passes.patching import (
    Conflict,
    Cuts,
    LazyMutableRange,
    ListPatches,
    SetPatches,
    apply_patches,
    lazy_shuffle,
)
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import WorkContext


# =============================================================================
# Cuts class tests
# =============================================================================


def test_cuts_empty():
    cuts = Cuts()
    assert cuts.empty == []


def test_cuts_combine_single():
    cuts = Cuts()
    result = cuts.combine([(0, 2)])
    assert result == [(0, 2)]


def test_cuts_combine_multiple_disjoint():
    cuts = Cuts()
    result = cuts.combine([(0, 2)], [(5, 7)])
    assert result == [(0, 2), (5, 7)]


def test_cuts_combine_overlapping():
    cuts = Cuts()
    result = cuts.combine([(0, 3)], [(2, 5)])
    assert result == [(0, 5)]


def test_cuts_combine_adjacent():
    cuts = Cuts()
    result = cuts.combine([(0, 2)], [(2, 4)])
    assert result == [(0, 4)]


def test_cuts_combine_contained():
    cuts = Cuts()
    result = cuts.combine([(0, 10)], [(2, 5)])
    assert result == [(0, 10)]


def test_cuts_combine_unsorted():
    cuts = Cuts()
    result = cuts.combine([(5, 7)], [(0, 2)])
    assert result == [(0, 2), (5, 7)]


def test_cuts_combine_multiple_merges():
    cuts = Cuts()
    result = cuts.combine([(0, 2), (4, 6)], [(1, 5)])
    assert result == [(0, 6)]


def test_cuts_apply_bytes():
    cuts = Cuts()
    result = cuts.apply([(2, 4)], b"abcdef")
    assert result == b"abef"


def test_cuts_apply_list():
    cuts = Cuts()
    result = cuts.apply([(1, 3)], [1, 2, 3, 4, 5])
    assert result == [1, 4, 5]


def test_cuts_apply_multiple_cuts():
    cuts = Cuts()
    result = cuts.apply([(0, 2), (4, 5)], b"abcdef")
    assert result == b"cdf"


def test_cuts_apply_empty():
    cuts = Cuts()
    result = cuts.apply([], b"hello")
    assert result == b"hello"


def test_cuts_size():
    cuts = Cuts()
    assert cuts.size([(0, 5)]) == 5
    assert cuts.size([(0, 2), (5, 10)]) == 7
    assert cuts.size([]) == 0


# =============================================================================
# SetPatches class tests
# =============================================================================


def test_set_patches_empty():
    def apply_fn(patch, target):
        return target - patch

    sp = SetPatches(apply_fn)
    assert sp.empty == frozenset()


def test_set_patches_combine():
    def apply_fn(patch, target):
        return target - patch

    sp = SetPatches(apply_fn)
    result = sp.combine(frozenset({1, 2}), frozenset({3, 4}))
    assert result == frozenset({1, 2, 3, 4})


def test_set_patches_combine_overlap():
    def apply_fn(patch, target):
        return target - patch

    sp = SetPatches(apply_fn)
    result = sp.combine(frozenset({1, 2}), frozenset({2, 3}))
    assert result == frozenset({1, 2, 3})


def test_set_patches_apply():
    def apply_fn(patch, target):
        return target - patch

    sp = SetPatches(apply_fn)
    result = sp.apply(frozenset({2, 4}), frozenset({1, 2, 3, 4, 5}))
    assert result == frozenset({1, 3, 5})


def test_set_patches_size():
    def apply_fn(patch, target):
        return target

    sp = SetPatches(apply_fn)
    assert sp.size(frozenset({1, 2, 3})) == 3
    assert sp.size(frozenset()) == 0


# =============================================================================
# ListPatches class tests
# =============================================================================


def test_list_patches_empty():
    def apply_fn(patch, target):
        return [x for x in target if x not in patch]

    lp = ListPatches(apply_fn)
    assert lp.empty == []


def test_list_patches_combine():
    def apply_fn(patch, target):
        return [x for x in target if x not in patch]

    lp = ListPatches(apply_fn)
    result = lp.combine([1, 2], [3, 4])
    assert result == [1, 2, 3, 4]


def test_list_patches_apply():
    def apply_fn(patch, target):
        return [x for x in target if x not in patch]

    lp = ListPatches(apply_fn)
    result = lp.apply([2, 4], [1, 2, 3, 4, 5])
    assert result == [1, 3, 5]


def test_list_patches_size():
    def apply_fn(patch, target):
        return target

    lp = ListPatches(apply_fn)
    assert lp.size([1, 2, 3]) == 3
    assert lp.size([]) == 0


# =============================================================================
# LazyMutableRange class tests
# =============================================================================


def test_lazy_mutable_range_len():
    r = LazyMutableRange(5)
    assert len(r) == 5


def test_lazy_mutable_range_getitem_unmodified():
    r = LazyMutableRange(5)
    assert r[0] == 0
    assert r[2] == 2
    assert r[4] == 4


def test_lazy_mutable_range_setitem():
    r = LazyMutableRange(5)
    r[2] = 10
    assert r[2] == 10
    assert r[0] == 0  # Others unchanged


def test_lazy_mutable_range_pop():
    r = LazyMutableRange(5)
    assert r.pop() == 4
    assert len(r) == 4
    assert r.pop() == 3
    assert len(r) == 3


def test_lazy_mutable_range_pop_after_swap():
    r = LazyMutableRange(5)
    r[4] = 0
    r[0] = 4
    assert r.pop() == 0  # Was swapped to position 4
    assert len(r) == 4


def test_lazy_mutable_range_pop_all():
    r = LazyMutableRange(3)
    assert r.pop() == 2
    assert r.pop() == 1
    assert r.pop() == 0
    assert len(r) == 0


# =============================================================================
# lazy_shuffle function tests
# =============================================================================


def test_lazy_shuffle_produces_all_elements():
    rnd = Random(42)
    seq = [1, 2, 3, 4, 5]
    result = list(lazy_shuffle(seq, rnd))
    assert sorted(result) == [1, 2, 3, 4, 5]


def test_lazy_shuffle_empty_sequence():
    rnd = Random(42)
    result = list(lazy_shuffle([], rnd))
    assert result == []


def test_lazy_shuffle_single_element():
    rnd = Random(42)
    result = list(lazy_shuffle([1], rnd))
    assert result == [1]


def test_lazy_shuffle_different_seeds_different_order():
    seq = list(range(10))
    result1 = list(lazy_shuffle(seq, Random(1)))
    result2 = list(lazy_shuffle(seq, Random(2)))
    # Very likely to be different with different seeds
    assert result1 != result2 or len(seq) == 1


def test_lazy_shuffle_same_seed_same_order():
    seq = list(range(10))
    result1 = list(lazy_shuffle(seq, Random(42)))
    result2 = list(lazy_shuffle(seq, Random(42)))
    assert result1 == result2


def test_lazy_shuffle_preserves_original():
    rnd = Random(42)
    seq = [1, 2, 3, 4, 5]
    original = seq.copy()
    _ = list(lazy_shuffle(seq, rnd))
    assert seq == original


# =============================================================================
# Conflict exception tests
# =============================================================================


def test_conflict_exception():
    with pytest.raises(Conflict):
        raise Conflict("test conflict")


# =============================================================================
# apply_patches async function tests
# =============================================================================


async def test_apply_patches_all_applicable():
    """Test apply_patches when all patches can be applied."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    patches = [[(1, 2)], [(3, 4)]]  # Delete 'b' and 'd'

    await apply_patches(problem, cuts, patches)
    # Should have applied some cuts
    assert len(problem.current_test_case) < 6


async def test_apply_patches_some_not_applicable():
    """Test apply_patches when some patches fail interestingness."""

    async def is_interesting(x):
        # Only interesting if starts with 'a'
        return len(x) > 0 and x[0:1] == b"a"

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    patches = [[(0, 1)], [(2, 3)]]  # Try to delete 'a' (will fail) and 'c'

    await apply_patches(problem, cuts, patches)
    # 'a' should still be there
    assert problem.current_test_case[0:1] == b"a"


async def test_apply_patches_empty():
    """Test apply_patches with no patches."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"hello",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    await apply_patches(problem, cuts, [])
    assert problem.current_test_case == b"hello"


async def test_apply_patches_all_at_once():
    """Test when all patches can be combined at once."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdefgh",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    # All small non-overlapping cuts
    patches = [[(1, 2)], [(3, 4)], [(5, 6)]]

    await apply_patches(problem, cuts, patches)
    # All should be applied
    assert len(problem.current_test_case) <= 5


async def test_apply_patches_with_parallelism():
    """Test apply_patches with parallelism > 1."""
    import trio

    async def is_interesting(x):
        await trio.sleep(0)  # Yield to allow parallelism
        return True

    problem = BasicReductionProblem(
        initial=b"abcdefghij",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    cuts = Cuts()
    patches = [[(i, i + 1)] for i in range(0, 10, 2)]  # Delete every other char

    await apply_patches(problem, cuts, patches)
    assert len(problem.current_test_case) < 10
