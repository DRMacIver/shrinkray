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

When applying multiple patches:
1. Try to apply all patches at once
2. If that fails, binary-search to find a maximal compatible subset
3. Apply that subset, record which patches succeeded
4. Repeat with remaining patches

```python
async def try_apply_patches(patches):
    combined = combine_all(patches)
    if await is_interesting(apply(combined, target)):
        return True  # All patches work together
    else:
        return await find_maximal_subset(patches)
```

### Binary Search for Maximal Sets

`find_maximal_set` uses binary search: if patches `[0:mid]` work, try `[0:mid2]` where `mid2 > mid`. If they don't work, try `[0:mid2]` where `mid2 < mid`.

This finds a large working subset in O(log n) interestingness tests.

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
