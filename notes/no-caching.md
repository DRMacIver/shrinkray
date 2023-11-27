# Caching in Shrink Ray

Historically most test-case reducers implement caching of their interestingness test,
so that if you generate the same test case variant multiple times you don't need to
call the underlying interestingness test multiple times.

Shrink Ray doesn't do this, because it doesn't seem worth it. In fairly natural test
cases I was noticing cache hit rates of literally 0%.

I suspect caching in general is not that useful for test-case reduction, but this is
likely particularly the case with Shrink Ray which has a very large number of fine-grained
transformations, many of which it tries in a random order, so it's actually quite unlikely
to hit duplicates.

I suspect the popularity of caching with test-case reduction is a historical artefact from
delta debugging, which has a very high chance of generating duplicates due to the way
its coarse grained passes decompose into multiple operations from its fine grained passes.
Shrink Ray basically never does that (I don't think it actually works) so has few opportunities
to generate duplicates.