# Shrink Ray

Shrink Ray is a modern multiformat test-case reducer.

It's designed to be highly parallel, and work with a very wide variety of formats, through a mix of good generic algorithms and format-specific reduction passes.

## Disclaimers about quality

Currently shrink ray in a very alpha quality state. I expect it to mostly work for most use cases, but also is probably super buggy in ways I've not discovered yet. I encourage people to try it out, but also please let me know when it inevitably breaks, or if it doesn't produce very good results for your use cases.

Also, the output as it runs is probably mostly incomprehensible if you're not me right now. Sorry. I tried to make a better UI for it and fell down a rabbit hole of annoying details that stalled development for more than a month because I didn't want to touch it, so I ripped all of that out and decided to release a version for general use with incomprehensible output. You mostly don't need this output anyway, it's just there to give you something to look at while it runs.

## Usage

There is currently no official release for shrink ray, and I recommend running off main. You can install it as follows:

```
pipx install git+https://github.com/DRMacIver/shrinkray.git
```

(if you don't have or want [pipx](https://pypa.github.io/pipx/) you could also do this with pip and it would work fine)

You then run it as follows:

```
shrinkray is_interesting.sh my-test-case
```

Where `my-test-case` is some file you want to reduce and `is_interesting.sh` can be any executable that exits with `0` when a test case passed to it is interesting and non-zero otherwise.

Variant test cases are passed to the interestingness test both on STDIN and as a file name passed as an argument. Additionally for creduce compatibility, the file has the same base name as the original test case and is in the current working directory the script is run with. This behaviour can be customised with the `--input-type` argument.

`shrinkray --help` will give more usage instructions.

## Supported formats

Shrink Ray is fully generic in the sense that it will work with literally any file you give it in any format. However, some formats will work a lot better than others.

It has a generic reduction algorithm that should work pretty well with any textual format, and an architecture that is designed to make it easy to add specialised support for specific formats as needed. Currently there are no formats special cased in this way.

If you run into a test case and interestingness test that you care about that shrink ray handles badly please let me know and I'll likely see about improving its handling of that format.

My main advice on format support is that if your format is C, C++, or a language with C-like syntax, you should consider using [creduce](https://github.com/csmith-project/creduce) instead. It's possible that shrinkray will be faster (especially when running in parallel) and might be a good first pass, but creduce will definitely get you better quality reduced test cases if you run it afterwards.

## Parallelism

You can control the number of parallel tasks shrinkray will run with the `--parallelism` flag. By default this will be the number of CPU cores you have available

Shrink Ray is designed to be able to run heavily in parallel, with a basic heuristic of aiming to be embarrassingly parallel when making no progress, mostly sequential when making progress, and smoothly scaling in between the two. It mostly succeeds at this.

Currently the bottleneck on scaling to a very large number of cores is how fast the controlling Python program can generate variant test cases to try and pass them to the interestingness test. This isn't well optimised at present and I don't currently have good benchmarks for it, but I'd expect you to be able to get linear speedups on most workflows while running 10-20 test cases in parallel, and to start to struggle past that.

This also depends on the performance of the interestingness test - the slower your test is to run, the more you'll be able to scale linearly with the number of cores available.

I'm quite interested in getting this part to scale well, so please let me know if you find examples where it doesn't seem to work.

## Bug Reports

Shrink Ray is still pretty new and under-tested software, so it definitely has bugs. If you run into any, [please file an issue](https://github.com/DRMacIver/shrinkray/issues).

As well as obvious bugs (crashes, etc) I'm also very interested in hearing about usability issues and cases where the reduced test case isn't very good.

Requests for new features, new supported formats, etc. also welcome although I'm less likely to jump right on them.