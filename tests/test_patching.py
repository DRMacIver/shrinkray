"""Unit tests for patching module."""

from random import Random

import pytest
import trio

from shrinkray.passes.patching import (
    Conflict,
    Cuts,
    LazyMutableRange,
    ListPatches,
    PatchApplier,
    Patches,
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

    async def is_interesting(x):
        await trio.lowlevel.checkpoint()  # Yield to allow parallelism
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


async def test_apply_patches_conflict_in_combine():
    """Test apply_patches when patches conflict during combine."""

    class ConflictingPatches(Patches[tuple[int, ...], bytes]):
        @property
        def empty(self):
            return ()

        def combine(self, *patches):
            # Raise conflict if any patch has overlapping elements
            all_items = []
            for p in patches:
                for item in p:
                    if item in all_items:
                        raise Conflict("Overlapping")
                    all_items.append(item)
            return tuple(sorted(all_items))

        def apply(self, patch, target):
            result = list(target)
            for i in sorted(patch, reverse=True):
                if i < len(result):
                    del result[i]
            return bytes(result)

        def size(self, patch):
            return len(patch)

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    applier = PatchApplier(ConflictingPatches(), problem)
    # Apply first patch
    result = await applier.try_apply_patch((1,))
    assert result is True  # First one succeeds
    # Try to apply a conflicting patch - should fail due to Conflict exception
    result = await applier.try_apply_patch((1,))
    # Second application returns False because of Conflict in combine
    assert result is False


async def test_apply_patches_patch_already_in_current():
    """Test when patch would result in same as current test case."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Apply a cut
    result = await applier.try_apply_patch([(1, 2)])
    assert result is True

    # Apply the same cut again - should return True (already applied)
    result = await applier.try_apply_patch([(1, 2)])
    assert result is True


async def test_patch_applier_concurrent_patches(autojump_clock):
    """Test PatchApplier with concurrent patch applications."""
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        await trio.sleep(0.01)
        return len(x) > 0

    problem = BasicReductionProblem(
        initial=b"abcdefghij",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=4),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    async def try_patch(i):
        await applier.try_apply_patch([(i, i + 1)])

    async with trio.open_nursery() as nursery:
        for i in range(5):
            nursery.start_soon(try_patch, i)

    # Some patches should have been applied
    assert len(problem.current_test_case) < 10


async def test_apply_patches_partial_merge():
    """Test when only some patches can be merged."""

    async def is_interesting(x):
        # Only interesting if keeps first 3 chars
        return len(x) >= 3 and x[:3] == b"abc"

    problem = BasicReductionProblem(
        initial=b"abcdefghij",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    cuts = Cuts()
    # Some of these will fail (the ones that cut the first 3 chars)
    patches = [[(0, 1)], [(4, 5)], [(6, 7)]]

    await apply_patches(problem, cuts, patches)
    # First 3 chars should still be there
    assert problem.current_test_case[:3] == b"abc"


async def test_apply_patches_conflict_in_initial_combine():
    """Test when combining all patches at once raises Conflict."""

    class ConflictOnAllPatches(Cuts):
        def combine(self, *patches):
            # Conflict only when combining 3 or more patches
            if len(patches) >= 3:
                raise Conflict("Too many patches")
            return super().combine(*patches)

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdefgh",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    patches_info = ConflictOnAllPatches()
    patches = [[(1, 2)], [(3, 4)], [(5, 6)]]

    # Should not raise - the conflict is caught and individual patches are tried
    await apply_patches(problem, patches_info, patches)


async def test_patch_applier_merge_finds_partial(capfd):
    """Test merge logic when only some patches can be merged."""
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        # Only interesting if length >= 5
        return len(x) >= 5

    problem = BasicReductionProblem(
        initial=b"abcdefghij",  # 10 chars
        is_interesting=is_interesting,
        work=WorkContext(parallelism=4),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Apply multiple patches concurrently
    # Some will succeed (remove 1 char each), but can't remove all
    async def apply_cuts():
        async with trio.open_nursery() as nursery:
            for i in range(5):
                nursery.start_soon(applier.try_apply_patch, [(i * 2, i * 2 + 1)])

    await apply_cuts()
    # Some patches should have been applied
    assert len(problem.current_test_case) < 10


async def test_patch_applier_patch_equals_current():
    """Test when applying patch results in same as current."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # First, apply a cut to change the current state
    result = await applier.try_apply_patch([(1, 2)])  # Remove 'b'
    assert result is True

    # Now apply empty patch - patch applied equals current
    result = await applier.try_apply_patch([])
    assert result is True


async def test_patch_applier_fast_path():
    """Test the fast path in try_apply_patch for single patches."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Single patch application - should use fast path
    result = await applier.try_apply_patch([(1, 2)])
    assert result is True
    assert problem.current_test_case == b"acdef"


async def test_merge_with_conflict_in_queue():
    """Test merge when patches in queue conflict."""

    class ConflictingPatches(Patches[frozenset[int], bytes]):
        @property
        def empty(self):
            return frozenset()

        def combine(self, *patches):
            all_items = set()
            for p in patches:
                if all_items & p:
                    raise Conflict("Overlapping items")
                all_items.update(p)
            return frozenset(all_items)

        def apply(self, patch, target):
            result = list(target)
            for i in sorted(patch, reverse=True):
                if i < len(result):
                    del result[i]
            return bytes(result)

        def size(self, patch):
            return len(patch)

    async def is_interesting(x):
        await trio.lowlevel.checkpoint()
        return len(x) > 0

    problem = BasicReductionProblem(
        initial=b"abcdefgh",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    patches = ConflictingPatches()
    applier = PatchApplier(patches, problem)

    # Apply overlapping patches concurrently - will conflict
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, frozenset({0, 1}))
        nursery.start_soon(
            applier.try_apply_patch, frozenset({1, 2})
        )  # Conflicts with first

    # Something should have been applied
    assert len(problem.current_test_case) < 8


async def test_merge_patch_becomes_base():
    """Test when attempted_patch equals base_patch after combine."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Apply a cut
    await applier.try_apply_patch([(1, 2)])

    # Now try to apply the same cut again - combining with current should be same
    result = await applier.try_apply_patch([(1, 2)])
    assert result is True


async def test_is_reduction_fails_in_merge(autojump_clock):
    """Test when is_reduction returns False during merge."""

    async def is_interesting(x):
        await trio.lowlevel.checkpoint()
        # Only interesting if starts with 'a'
        return len(x) > 0 and x[0:1] == b"a"

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Try to apply a cut that would remove 'a' - will be interesting but not a reduction
    # because it would make the test case start with 'b' which fails
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, [(0, 1)])  # Remove 'a' - will fail
        await trio.sleep(0.01)
        nursery.start_soon(
            applier.try_apply_patch, [(2, 3)]
        )  # Remove 'c' - should succeed

    # 'a' should still be there
    assert problem.current_test_case[0:1] == b"a"


async def test_patch_applied_equals_problem_current():
    """Test when applying patch to initial equals problem's current."""
    # Problem where current_test_case changes during patch application

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # First, reduce the problem directly to match what a patch would produce
    await problem.is_interesting(b"acdef")  # This changes problem.current_test_case

    # Now try to apply the same patch - with_patch_applied should equal current
    result = await applier.try_apply_patch([(1, 2)])  # Would also produce b"acdef"
    assert result is True


async def test_merge_can_merge_k_greater_than_to_merge(autojump_clock):
    """Test can_merge returns False when k > number of patches to merge.

    This is called by find_large_integer which probes beyond to_merge.
    """
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        # Only interesting for specific sizes to force partial merges
        if len(x) == 6:  # Original
            return True
        if len(x) == 5:  # One deletion OK
            return True
        return False  # Two or more deletions fail

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=3),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Apply multiple patches that can only partially merge
    # This forces find_large_integer to be called, which probes k > to_merge
    async with trio.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(applier.try_apply_patch, [(i, i + 1)])
            await trio.sleep(0.01)

    # One should have succeeded
    assert len(problem.current_test_case) == 5


async def test_can_merge_attempted_equals_base():
    """Test when attempted_patch == base_patch after combine."""

    async def is_interesting(x):
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Apply a patch first
    await applier.try_apply_patch([(1, 2)])

    # Now apply the same patch via the merge path (concurrent)
    # The combine will produce the same result as base_patch
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, [(1, 2)])  # Already applied
        nursery.start_soon(applier.try_apply_patch, [(3, 4)])  # New patch

    assert len(problem.current_test_case) < 6


async def test_merge_empty_patches_equals_base(autojump_clock):
    """Test empty patch in queue equals base after combine."""
    call_count = [0]

    async def is_interesting(x):
        call_count[0] += 1
        await trio.sleep(0.01)
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Submit empty patches - combining them with base gives base
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, [])  # Empty patch
        await trio.sleep(0.01)
        nursery.start_soon(applier.try_apply_patch, [(1, 2)])  # Real patch

    # Real patch should succeed
    assert len(problem.current_test_case) <= 5


async def test_is_reduction_returns_false(autojump_clock):
    """Test is_reduction returns False during merge."""

    # is_interesting always True, but is_reduction will fail for larger results
    async def is_interesting(x):
        await trio.sleep(0.01)
        return True

    problem = BasicReductionProblem(
        initial=b"ab",  # Start small
        is_interesting=is_interesting,
        work=WorkContext(parallelism=2),
    )

    # Patch that makes things larger
    class GrowingPatches(Cuts):
        def apply(self, patch, target):
            # Instead of cutting, add bytes
            for start, _end in patch:
                target = target[:start] + b"xxx" + target[start:]
            return target

    patches = GrowingPatches()
    applier = PatchApplier(patches, problem)

    # Apply patches that make things larger - is_reduction will fail
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, [(0, 1)])
        await trio.sleep(0.02)
        nursery.start_soon(applier.try_apply_patch, [(1, 2)])

    # Nothing should have been applied since results are larger


async def test_find_large_integer_probes_beyond(autojump_clock):
    """Test find_large_integer probes beyond to_merge.

    find_large_integer probes k=1,2,3,4, then exponentially (5,10,20...).
    When to_merge < k for some probed value, can_merge returns False.
    """

    # Accept merging up to 5 patches, but not 6
    async def is_interesting(x):
        await trio.sleep(0.001)
        # Original is 20 chars, allow removing up to 5 chars
        return len(x) >= 15

    problem = BasicReductionProblem(
        initial=b"abcdefghijklmnopqrst",  # 20 chars
        is_interesting=is_interesting,
        work=WorkContext(parallelism=8),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Submit 6+ patches to queue them for merge
    # The first 5 should succeed, the 6th should fail
    async with trio.open_nursery() as nursery:
        for i in range(7):

            async def submit_patch(idx):
                await trio.sleep(0.002 * idx)
                await applier.try_apply_patch([(idx * 2, idx * 2 + 1)])

            nursery.start_soon(submit_patch, i)

    # At least 5 patches should have been applied
    assert len(problem.current_test_case) <= 15


async def test_empty_patch_equals_base_in_merge(autojump_clock):
    """Test empty patch in queue means attempted == base."""

    async def is_interesting(x):
        await trio.sleep(0.01)
        return True

    problem = BasicReductionProblem(
        initial=b"abcdef",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=4),
    )

    cuts = Cuts()
    applier = PatchApplier(cuts, problem)

    # Submit patches including empty ones
    async with trio.open_nursery() as nursery:
        nursery.start_soon(applier.try_apply_patch, [])  # Empty
        nursery.start_soon(applier.try_apply_patch, [])  # Empty
        await trio.sleep(0.02)
        nursery.start_soon(applier.try_apply_patch, [(1, 2)])  # Real patch

    assert len(problem.current_test_case) <= 5
