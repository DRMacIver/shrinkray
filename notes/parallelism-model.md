# Parallelism Model

Shrink Ray uses Trio for structured concurrency and achieves parallelism through speculative execution of interestingness tests.

## WorkContext

`WorkContext` (in `work.py`) is the parallelism coordinator. Key attributes:

- **parallelism**: Max concurrent tasks
- **random**: RNG for shuffling (reproducibility via seed)
- **volume**: Logging verbosity

### Core Methods

**map(fn, items)**
Parallel map with lazy evaluation and backpressure. Returns an async iterator.
```python
async for result in work.map(process_item, items):
    handle(result)
```

**filter(predicate, items)**
Parallel filter, returns items where predicate is true.

**find_first_value(items, predicate)**
Returns the first item where predicate is true.
```python
# Find first deletion that works
i = await work.find_first_value(range(n), lambda i: can_delete(i))
```

**find_large_integer(predicate)**
Binary search for largest k where predicate(k) is true.
```python
# Find how many consecutive items we can delete
k = await work.find_large_integer(lambda k: can_delete_range(i, i+k))
```

## Speculative Execution

The key insight: interestingness tests can run in parallel even though only one result "wins."

### Example: Block Deletion

```python
blocks = [[(i, i+10)] for i in range(0, 1000, 10)]
await apply_patches(problem, Cuts(), blocks)
```

This doesn't test blocks sequentially. Instead:
1. Multiple block deletions run in parallel
2. First successful reduction updates `current_test_case`
3. Other in-flight tests may become stale (their target changed)
4. The system handles this gracefully - stale successes are "wasted" but harmless

### Wasted Work

When multiple parallel tests succeed, all but one are "wasted" - the reduction was already achieved. `problem.stats.wasted_interesting_calls` tracks this.

The system always runs at the maximum parallelism supported by the current reduction pass and the configured parallelism limit.

## Backpressure

`LazyParallelMap` uses bounded queues to prevent unbounded work creation. The channel buffer size is set to `parallelism`, which limits in-flight work and prevents memory exhaustion on large inputs.

## Structured Concurrency

All parallelism uses Trio's nursery pattern:
```python
async with trio.open_nursery() as nursery:
    for item in items:
        nursery.start_soon(process, item)
```

Benefits:
- Automatic cancellation when parent scope exits
- Exception propagation
- Clean resource cleanup

### Cancellation in Reduction Stages

Some stages (like the initial cuts stage) use timeout-based cancellation:
```python
async with trio.open_nursery() as nursery:
    @nursery.start_soon
    async def watcher():
        while True:
            await trio.sleep(5)
            if no_progress_recently():
                nursery.cancel_scope.cancel()

    await run_pass(rp)
    nursery.cancel_scope.cancel()  # Cancel watcher when done
```

## The Merge Master Pattern

The merge master pattern is how `PatchApplier` coordinates parallel patch testing while making progress. See [patching-system.md](patching-system.md) for a detailed explanation.

The key benefit is that we can run concurrently *while making progress*. Normally, parallelism in reduction is embarrassingly parallel only when failing to reduce - whenever you make progress, you have to throw away your parallel work because the test case changed. The merge master pattern allows near-linear speedups from parallelism even while actively reducing.

## Parallelism Limits

The `is_interesting_limiter` (a `trio.CapacityLimiter`) in `ShrinkRayState` limits concurrent interestingness tests:
```python
# In ShrinkRayState.__attrs_post_init__():
self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))

# In ShrinkRayState.is_interesting():
async def is_interesting(self, test_case):
    async with self.is_interesting_limiter:
        return await self.run_for_exit_code(test_case) == 0
```

This prevents overwhelming the system with subprocess spawns.

## Random Ordering

Patches are shuffled before testing using `context.random`. This helps avoid repeating work: if a pass runs multiple times and patches fail, shuffling means we're less likely to try the same patches in the same order and waste effort on patches that failed last time.
