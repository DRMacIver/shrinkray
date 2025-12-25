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

`Cuts` patches are lists of `(start, end)` intervals to delete:
```python
patch = [(0, 5), (10, 15)]  # Delete bytes 0-5 and 10-15
```

Combining cuts merges their intervals. The `size` is total bytes deleted.

### Other Patch Types

- **SetPatches[T]**: A frozenset of items to remove (for JSON keys, etc.)
- **ByteReplacement**: Dict mapping byte positions to replacement values

## PatchApplier

The core engine that applies patches in parallel with intelligent merging.

### Merge Master Pattern

Workers call `try_apply_patch()` in parallel. When a patch succeeds its interestingness test, it goes into a merge queue. One worker becomes the "merge master" and processes the queue:

1. Try to apply all queued patches at once
2. If that fails, use `find_large_integer` to binary-search for how many patches from the front of the queue can be applied together
3. Apply that subset, notify waiting workers of success/failure
4. Repeat until queue is empty

```python
# Simplified from __possibly_become_merge_master():
async def can_merge(k):
    attempted_patch = combine(base_patch, *queue[:k])
    return await problem.is_reduction(apply(attempted_patch, initial))

if await can_merge(len(queue)):
    merged = len(queue)  # All patches work together
else:
    merged = await work.find_large_integer(can_merge)
```

### Binary Search for Maximal Sets

`find_large_integer` uses binary search: if `can_merge(k)` is true, try larger k; if false, try smaller k. This finds a large working subset in O(log n) interestingness tests.

### The apply_patches Function

The main entry point:
```python
await apply_patches(problem, Cuts(), [
    [(i, i+10)] for i in range(0, len(data), 10)
])
```

This tries to delete 10-byte blocks in parallel, finding maximal sets of non-conflicting deletions.

## Why This Works

1. **Independent reductions compose**: If deleting A works and deleting B works independently, deleting both often works too.

2. **Binary search is cheap**: Each test tells us about many potential reductions.

3. **Parallelism**: Multiple interestingness tests run concurrently.

## Example: Block Deletion

```python
blocks = [[(i, i+10)] for i in range(0, 1000, 10)]  # 100 blocks
await apply_patches(problem, Cuts(), blocks)
```

Instead of 100 sequential tests:
1. First test: try deleting all 100 blocks
2. If fails: binary search to find maximal subset
3. Apply working subset, continue with remainder
4. Typically finishes in ~10-20 tests
