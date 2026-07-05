# Caching in Shrink Ray

Historically most test-case reducers implement caching of their interestingness test,
so that if you generate the same test case variant multiple times you don't need to
call the underlying interestingness test multiple times.

Shrink Ray caches results in `BasicReductionProblem` by content hash, and keeps them
for the whole reduction. The cache used to be cleared whenever a successful reduction
was found, on the theory that candidates derived from the old test case were no longer
useful; measured hit rates were near 0% and removing the cache entirely was considered
(see [issue #31](https://github.com/DRMacIver/shrinkray/issues/31), which concluded it
should stay as basically harmless).

I suspect caching in general is not that useful for test-case reduction, but this is
likely particularly the case with Shrink Ray which has a very large number of fine-grained
transformations, many of which it tries in a random order, so it's actually quite unlikely
to hit duplicates. The popularity of caching with test-case reduction is likely a
historical artefact from delta debugging, which has a very high chance of generating
duplicates due to the way its coarse grained passes decompose into multiple operations
from its fine grained passes. Shrink Ray basically never does that so has few
opportunities to generate duplicates.

Two later features gave the cache real jobs, which is why it is now kept for the whole
run rather than cleared on each reduction:

## Adaptive timeouts

When reduction stalls with tests timing out, the reducer raises the timeout and
re-runs a full round of passes; candidates that previously *timed out* are
retried (their cache entries carry a validity condition tied to the timeout
they ran under), while candidates that completed are served from cache. Without
the cache, each of those retry rounds would re-execute every candidate the
passes regenerate, which is exactly the situation where test runs are at their
most expensive.

## The restart phase

When reduction reaches a fixpoint it re-runs from the original input, constrained to
sort below the fixpoint, and each restart round deliberately replays the attempt
sequence of earlier rounds (same random state, same shuffles) until it finds its first
improvement. Those replayed attempts are exact duplicates of earlier calls, so they are
answered from the cache instead of re-running the interestingness test. The cache
stores only small content hashes, so keeping it for the whole run is cheap.
