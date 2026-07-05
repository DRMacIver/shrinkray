# Caching in Shrink Ray

Historically most test-case reducers implement caching of their interestingness test,
so that if you generate the same test case variant multiple times you don't need to
call the underlying interestingness test multiple times.

Shrink Ray caches results in `BasicReductionProblem` by content hash, and keeps them
for the whole reduction. The cache used to be cleared whenever a successful reduction
was found, on the theory that candidates derived from the old test case were no longer
useful; measured hit rates were near 0% and removing the cache entirely was considered
(see [issue #31](https://github.com/DRMacIver/shrinkray/issues/31)).

The restart phase changed that calculus: when reduction reaches a fixpoint it re-runs
from the original input, constrained to sort below the fixpoint, and each restart round
deliberately replays the attempt sequence of earlier rounds (same random state, same
shuffles) until it finds its first improvement. Those replayed attempts are exact
duplicates of earlier calls, so they are answered from the cache instead of re-running
the interestingness test. The cache stores only small content hashes, so keeping it for
the whole run is cheap.

Ordinary forward reduction still rarely generates duplicates — Shrink Ray has a very
large number of fine-grained transformations, many tried in a random order — so outside
the restart phase the cache continues to do little. The popularity of caching with
test-case reduction is likely a historical artefact from delta debugging, whose coarse
grained passes decompose into operations that its fine grained passes retry verbatim.
