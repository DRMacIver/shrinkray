# Pumps and Passes

Shrink Ray makes a distinction between two different types of transformation: A reduction pass, and a reduction pump.

A reduction pass is a set of transformations of the current test case of a reduction problem which is designed to produce a smaller interesting test case. Reduction passes are always greedy, in the sense that you only care about transforming the test case into a strictly better one.

A reduction pump on the other hand is any transformation of the current test case into another interesting test case. It can, and typically will,
result in a test case larger than the starting one.

The general idea of a reduction pump is that you only run it when you're stuck. If you hit a point where none of the reduction passes are making progress, you start running reduction pumps, which may unlock further progress. For example, if you inline a function, this will typically increase the size of the test case. However, this may allow significantly more deletions, as the function definition may now be deleted, and each individual
call site may be reduced independently.