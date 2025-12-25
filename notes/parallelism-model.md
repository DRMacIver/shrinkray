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

High parallelism on easy reductions = more waste. The system auto-tunes based on success rates.

## Backpressure

`LazyParallelMap` uses bounded queues to prevent unbounded work creation:
```python
self.send_channel, self.recv_channel = trio.open_memory_channel(
    max_buffer_size=context.parallelism
)
```

This limits in-flight work to `parallelism` items, preventing memory exhaustion on large inputs.

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

### Cancellation in Passes

Some passes (like `initial_cut`) use timeout-based cancellation:
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

`PatchApplier` coordinates parallel patch testing:

1. **Controller** (`try_apply_patches`): Decides which patches to try
2. **Workers**: Run interestingness tests in parallel
3. **State synchronization**: Through `problem.current_test_case`

When a patch succeeds:
1. Problem state updates atomically
2. Other workers' tests may now be invalid (testing old version)
3. Workers check `problem.current_test_case` before reporting success
4. Invalid successes are logged as "wasted"

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

Patches are shuffled before testing:
```python
if context.random is not None:
    patches = list(patches)
    context.random.shuffle(patches)
```

This prevents systematic bias and improves parallelism utilization when early patches tend to fail.
