# Parallelism in Shrink Ray

How to do highly parallel test-case reduction is one of the active research problems I'm working on in Shrink Ray,
so I'm still experimenting with it and changing it. I think the current approach works very well[^1], but it lacks a good unifying
abstraction, and it will doubtless change a lot as the reducer develops.

This document describes the approach taken at the time of writing, but I can't promise it's always 100% up to date.

## The problem to be solved

At its most basic, test-case reduction consists of repeatedly running the following algorithm:

1. Take your current best test-case and generate `N` smaller variants of it.
2. Find one of those `N` variants that is still interesting, and replace your current best test-case with it.

It is hopefully clear that step 2 can be run in an embarrassingly parallel manner, making use of as many threads
as you want to give it.

The problem is that you can only really take advantage of this parallelism in the case where the test-case reducer
isn't making progress. The best case scenario for it is when the reducer is done and you just have to evaluate that
none of the `N` test cases are interesting, this will be able to take full advantage of parallelism. In the case
where the reducer is making easy progress, you end up not being able to take much advantage of parallelism, because
you search a few of the variants, find a successful reduction, and then immediately have to start the process again.
The result is that you end up mostly sequential, or at least significantly less parallel than you could be.[^2]

When the variants found are *much* smaller (as in the early steps of a classic delta debugging[^3]), this is largely
fine and you end up in a situation where you're making large progress (and thus fast) or you're highly parallel 
(and thus still fast). The big problem is when you've got to the point where there are many small reductions that
can be made but few large ones. At that point this sort of algorithm does not manage to take as much advantage of
parallelism as one might hope.

## Shrink Ray's solution

Note: This section makes it sound like this approach is better abstracted in Shrink Ray than it actually is right now.
In reality, there are at least two independent implementations of this sort of idea in Shrink Ray that work subtly
differently in important ways. This is more of a high level description of the idea than the actual implementation.

The essential idea is this: Rather than generating a list of variants, generate a list of edits. For example,
"delete this region of the test case". Now, in parallel, try applying each of these edits with the following
algorithm:

1. If the edit is redundant or conflicts[^4] with edits already successfully applied, skip it.
2. Check if applying this edit on top of the already applied edits would lead to an interesting test case.
      a. If it would not, then skip this edit.
      b. If it *would* then check whether any other edits have been applied while we were checking. If they
         have not, then add this edit to the list of successfully applied edits. Otherwise, try this edit again,
         after sleeping briefly to give the other task that's working some time to proceed.

This approach means that you don't have to stop when you find an interesting variant - you just apply the edit
and keep going. In the cases where successful variants are *very* common this will still not be fully parallel,
because it needs to do retries that can result in wasted work, but this only occurs when two successful edits
race with each other, rather than every successful edit immediately halting everything else. In the case where
the probability of a successful edit is significantly under 1 / number of parallel threads, this should scale
linearly with the amount of parallelism.

Strictly it doesn't need to do those retries, but it seems to be helpful. The idea of the retry is that edits
that result in interesting test cases are likely (though not certain) to be independent,
and are worth the effort of finding out if they are, so when there's evidence that a patch is likely to be good,
we try it again in the hopes that it still is. The algorithm's correctness doesn't rely on this assumption - it
would still provide the same guarantees (when run repeatedly to a fixed point) without it, but when the heuristic
holds it is likely to be signfiicantly faster. When it doesn't, in theory it can produce arbitrary slowdowns in
the case where the reducer produces many successful edits many of which block other edits, but this seems like a
relatively unusual case.

The sleeping wasn't in earlier versions, and it plays badly with the adaptive nature of a number of shrink ray
passes, where once you start succeeding you find many other similar ways to delete. This means that you can
potentially end up with very long retry loops as the other task keeps sniping the work out from under you.
The sleep thus assumes that if you got sniped once you're likely to get sniped again and it's better to just
back off and let the other task do its thing. Currently we sleep for an exponential variate with an average of
1s. This is unlikely to be optimal but seems to work reasonable well.

## Bounding the amount of parallelism

Shrink Ray gives you an option of how much parallelism to use, though defaults to the number of cores available.

The way that this is implemented is that internally Shrink Ray is written as if it had access to unlimited
parallelism and just spawns as many tasks (using trio, so these are light weight tasks rather than OS threads),
which rather than calling the underlying interestingness test directly instead communicate with a number of
worker tasks, each of which reads a result off the queue, runs the underlying interestingness test, and then
replies. This means that most of Shrink Ray can be written in a way that is agnostic to the amount of parallelism
it's actually been given (although it does have access to this information when useful).


[^1]: It is possibly currently the best test-case reducer for making effective use of parallelism. It should be,
      but I don't have benchmarks to back this up at present and it's very likely I've screwed something up or missed
      some of the existing state of the art, so you should not believe this claim right now even though I think it's probably true.

[^2]: Another problem with this is that if generating variants is expensive (which it can be - some of Shrink Ray's reduction passes
      certainly have an expensive variant generation step), you end up spending a lot of time on that. As a result the
      algorithms described should generally be faster even with no parallelism.

[^3]: Although, I don't actually think these steps usually work. They're fine for very forgiving parsers, such as HTML,
      but for something with quite strict syntactic correctness requirements like a programming language, I think the
      early steps largely just waste time.

[^4]: In fact currently Shrink Ray has no passes where conflicts are possible, but redundancy still is.