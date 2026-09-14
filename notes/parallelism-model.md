# Parallelism Model

Shrink Ray uses Trio for structured concurrency and achieves parallelism through speculative execution of interestingness tests.

## WorkContext

`WorkContext` (in `work.py`) is the parallelism coordinator. Key attributes:

- **parallelism**: Max concurrent tasks
- **random**: RNG for shuffling (reproducibility via seed)
- **volume**: Logging verbosity

### Core Methods

**map(items, fn)**
Parallel map with lazy evaluation and backpressure. An async context manager that yields a receive channel of results:
```python
async with work.map(items, process_item) as results:
    async for result in results:
        handle(result)
```

**filter(items, predicate)**
Parallel filter; an async context manager yielding the items where predicate is true.

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

Patch application (`apply_patches`) runs at the full configured parallelism. The `map`/`find_first_value` path instead ramps up prefetching gradually (processing 1 item, then batches of 2, 4, 8, ... up to the parallelism limit) so that searches which expect an early hit don't speculate too far ahead.

## Backpressure

`parallel_map` uses a semaphore whose tokens cover both running evaluations and results waiting for their turn in the ordered output. A slow early result therefore cannot allow later results to accumulate without bound. Tokens are released when results reach the bounded output channel. `WorkContext.map` adds a result buffer of `parallelism + 1`; `filter` also uses a bounded channel.

Directory reduction starts at most `parallelism` per-file reducers. Each may speculate internally, while the state-level oracle limiter continues to cap the total number of running interestingness tests.

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

## Adoption and persistence

Oracle calls remain concurrent. Calls for the same candidate share an in-flight event so only one task drives that candidate's evidence at a time. If that task is cancelled, waiters retry.

A problem-level lock serializes accepted-result commits and periodic incumbent verification. Verification keeps its evidence attached to the incumbent it actually replayed. Before adoption, candidates recheck whether nondeterminism handling engaged or the required reproduction rate rose while they were running.

Once a commit starts, it shields its callbacks from cancellation. History snapshots run in a worker thread; single-file targets use atomic replacement preserving symlinks and permissions. Directory writes also run outside the Trio loop. Progress reporting waits for the commit before exposing its history entry. In-place basename attempts share the target-write lock and restore the incumbent in a cancellation-safe `finally` block.

## Worker lifecycle

A history restart first validates the requested entry (and is refused once the reduction is no longer running), cancels the old reducer, and waits for all its cleanup and commits to finish. The replacement reducer is installed, and the worker marked running again, before the run loop is released to start it; the target write follows and is best effort. A restart that fails after the old reducer was cancelled ends the worker with that error rather than a completion.

Directory reduction runs the initial test case through the problem's setup before any candidate, as single-file reduction does, so the calibration call, the default memory-limit retry and startup nondeterminism detection all see the initial input.

Worker stdout uses a Trio file-descriptor stream, so a full pipe can be cancelled without blocking the scheduler. The asyncio client coalesces pending progress messages, preserving incremental graph samples, and resolves outstanding commands on EOF or reader failure. Validation subprocesses have bounded runtimes and process-group cleanup as well.

## The Merge Master Pattern

The merge master pattern is how `PatchApplier` coordinates parallel patch testing while making progress. See [patching-system.md](patching-system.md) for a detailed explanation.

The key benefit is that we can run concurrently *while making progress*. Normally, parallelism in reduction is embarrassingly parallel only when failing to reduce - whenever you make progress, you have to throw away your parallel work because the test case changed. The merge master pattern allows near-linear speedups from parallelism even while actively reducing.

## Parallelism Limits

The `is_interesting_limiter` (a `trio.CapacityLimiter`) in `ShrinkRayState` limits concurrent interestingness tests:
```python
# In ShrinkRayState.__attrs_post_init__():
self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))

# Simplified from ShrinkRayState.is_interesting() (the real method also
# handles exclusion sets, --also-interesting, and output capture):
async def is_interesting(self, test_case):
    async with self.is_interesting_limiter:
        result = await self.run_for_result(test_case)
        return result.exit_code == 0 and not result.timed_out
```

This prevents overwhelming the system with subprocess spawns.

## Random Ordering

Patches are shuffled before testing using `context.random`. This helps avoid repeating work: if a pass runs multiple times and patches fail, shuffling means we're less likely to try the same patches in the same order and waste effort on patches that failed last time.
