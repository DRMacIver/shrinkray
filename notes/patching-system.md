# The Patching System

This document explains how Shrink Ray efficiently applies multiple changes in parallel.

## The Problem

When reducing, we often want to try many potential changes:
- Delete each of 1000 lines
- Delete each token
- Replace each integer with a smaller value

Trying these one-at-a-time is slow. But we can often apply multiple changes simultaneously if they don't conflict.

## Patches Abstract Interface

The `Patches[Patch, Target]` class defines how patches work:

```python
class Patches[Patch, Target]:
    @property
    def empty(self) -> Patch: ...           # The identity patch
    def combine(self, *patches) -> Patch: ...  # Merge patches
    def apply(self, patch, target) -> Target: ...  # Apply patch to target
    def size(self, patch) -> int: ...       # How much the patch reduces
```

### Cuts (Most Common)

`Cuts` patches are lists of `[start, end)` half-open intervals to delete:
```python
patch = [(0, 5), (10, 15)]  # Delete bytes [0, 5) and [10, 15)
```

Combining cuts merges their intervals. The `size` is total bytes deleted.

### Other Patch Types

The codebase has several other `Patches` implementations for different use cases:

- **SetPatches[T]**: A frozenset of items (for sets of changes that combine via union)
- **ListPatches[T]**: A list of items (for ordered changes that combine via concatenation)
- **ByteReplacement**: Dict mapping byte positions to replacement values
- **IndividualByteReplacement**: Like ByteReplacement but for single-byte changes
- **RegionReplacement**: Replaces byte regions with new content
- **RegionReplacingPatches**: Generic region replacement for any string-like type
- **NewlineReplacer**: Replaces spaces with newlines at specific positions
- **DeleteIdentifiers**: Removes keys from JSON objects
- **UpdateKeys**: Updates key-value pairs in dictionaries

Patches must be idempotent and combinable. The key requirement is that if patches A and B can each be applied independently, combining them should also be valid (though it may fail if they conflict).

## PatchApplier

The core engine that applies patches in parallel with intelligent merging.

### Merge Master Pattern

Workers call `try_apply_patch()` in parallel. When a patch passes its interestingness test, the worker attempts to acquire the merge lock and add it to a merge queue.

**Important:** While a merge master is active, all other workers must wait for the merge to complete or reject their patch. This serialises the actual state updates while allowing parallel testing.

One worker becomes the "merge master" and processes the queue:

1. Try to apply all queued patches at once
2. If that fails, use `find_large_integer` to find how many patches from the front of the queue can be applied together
3. Apply that subset, notify waiting workers of success/failure
4. Repeat until queue is empty

**Important caveats:**
- The subset found is **not necessarily maximal** - we don't retry patches once they fail (until the next pass invocation), and applying some patches might unlock others (e.g., deleting a variable's use allows deleting its assignment)
- The only guarantee is: if any patches can be applied, at least one will be applied

### Finding Compatible Patches

The merge master uses `find_large_integer` to probe how many queued patches can be applied together. This is essentially a parallelisation of the obvious one-patch-at-a-time linear algorithm, with binary search as an optimisation for when there are many successes at once.

In practice, the binary search rarely matters much - the merge queue is usually small (rarely larger than the parallelism count), so we're typically in linear probe territory. But since we have `find_large_integer` available, we use it.

### The apply_patches Function

The main entry point:
```python
await apply_patches(problem, Cuts(), [
    [(i, i+10)] for i in range(0, len(data), 10)
])
```

This tries to delete 10-byte blocks in parallel, finding large (but not necessarily maximal) sets of compatible deletions.

## Why This Works

The key insight is that we can run concurrently **while making progress**. In naive parallel reduction, failing to reduce is embarrassingly parallel, but whenever you succeed you have to throw away your parallel work because the test case changed. The merge master pattern avoids this:

1. **Independent reductions compose**: If deleting A works and deleting B works independently, deleting both often works too
2. **Speculative testing**: Multiple candidates are tested in parallel against the current test case
3. **Batched updates**: Successful patches are queued and applied together, so parallel work isn't wasted

This allows near-linear speedups from parallelism even while actively reducing.

## Example: Block Deletion

```python
blocks = [[(i, i+10)] for i in range(0, 1000, 10)]  # 100 blocks
await apply_patches(problem, Cuts(), blocks)
```

The algorithm works roughly like this:

1. In parallel, try deleting each block. This finds k deletable blocks in an embarrassingly parallel manner.
2. Try to delete all k of those blocks at once. If that works, great.
3. If not, perform (sequentially) an adaptive algorithm that runs in worst-case O(k) but in the happy path O(log(k)) interestingness tests to find a large (but not necessarily maximal) set of the k blocks that can be simultaneously deleted.

The actual implementation pipelines these steps together, which improves matters mostly because smaller test cases tend to be faster to run the interestingness test on.
