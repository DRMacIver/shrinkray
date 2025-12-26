# Shrink Ray

Shrink Ray is a modern multiformat test-case reducer.

## What is test-case reduction?

Test-case reduction is the process of automatically taking a *test case* and *reducing* it to something close to a [minimal reproducible example](https://en.wikipedia.org/wiki/Minimal_reproducible_example).

That is, you have some file that has some interesting property (usually that it triggers a bug in some software),
but it is large and complicated and as a result you can't figure out what about the file actually matters.
You want to be able to trigger the bug with a small, simple, version of it that contains only the features of interest.

For example, the following is some Python code that [triggered a bug in libcst](https://github.com/Instagram/LibCST/issues/1061):

```python
() if 0 else(lambda:())
```

This was extracted from a large Python file (probably several thousand lines of code) and systematically reduced down to this example.

You would obtain this by running `shrinkray breakslibcst.py mytestcase.py`, where `breakslibcst.py` looks something like this:

```python
import libcst
import sys

if __name__  == '__main__':
    try:
        libcst.parse_module(sys.stdin.read())
    except TypeError:
        sys.exit(0)
    sys.exit(1)
```

This script exits with 0 if the code passed to it on standard input triggers the relevant bug (that libcst raises a TypeError when parsing this code), and with a non-zero exit code otherwise.

shrinkray (or any other test-case reducer) then systematically tries smaller and simpler variants of your original source file until it reduces it to something as small as it can manage.

While it runs, you will see the following user interface:

![Demo of shrink ray running](demo.png)

When it finishes you will be left with the reduced test case in `mytestcase.py`.

Test-case reducers are useful for any tools that handle files with complex formats that can trigger bugs in them. Historically this has been particularly useful for compilers and other programming tools, but in principle it can be used for anything.

Most test-case reducers only work well on a few formats. Shrink Ray is designed to be able to support a wide variety of formats, including binary ones, although it's currently best tuned for "things that look like programming languages".

## What makes Shrink Ray distinctive?

It's designed to be highly parallel, and work with a very wide variety of formats, through a mix of good generic algorithms and format-specific reduction passes.

## Versioning and Releases

Shrink Ray uses calendar versioning (calver) in the format YY.M.D (e.g., 25.12.26 for December 26, 2025).

New releases are (in theory, this is still something of a work in progress) published automatically every day if there are any changes to the source code since the previous release.

Shrinkray makes no particularly strong backwards compatibility guarantees. I aim to keep its behaviour relatively stable between releases, but for example will not be particularly shy about dropping old versions of Python or adding new dependencies. The basic workflow of running a simple reduction will rarely, if ever, change, but the UI is likely to be continuously evolving for some time.

## Installation

Shrink Ray requires Python 3.12 or later, and can be installed using pip or uv like any other python package.

You can install the latest release from PyPI or run directly from the main branch:

```
pipx install shrinkray
# or
pipx install git+https://github.com/DRMacIver/shrinkray.git
```

(if you don't have or want [pipx](https://pypa.github.io/pipx/) you could also do this with pip or `uv pip` and it would work fine)

Shrink Ray requires Python 3.12 or later and won't work on earlier versions. If everything is working correctly, it should refuse to install
on versions it's incompatible with. If you do not have Python 3.12 installed, I recommend [pyenv](https://github.com/pyenv/pyenv) for managing
Python installs.

If you want to use it from the git repo directly, you can do the following:

```
git clone https://github.com/DRMacIver/shrinkray.git
cd shrinkray
python -m venv .venv
.venv/bin/pip install -e .
```

You will now have a shrinkray executable in .venv/bin, which you can also put on your path by running `source .venv/bin/activate`.

## Usage

Shrink Ray is run as follows:

```
shrinkray is_interesting.sh my-test-case
```

Where `my-test-case` is some file you want to reduce and `is_interesting.sh` can be any executable that exits with `0` when a test case passed to it is interesting and non-zero otherwise.

Variant test cases are passed to the interestingness test both on STDIN and as a file name passed as an argument. Additionally for creduce compatibility, the file has the same base name as the original test case and is in the current working directory the script is run with. This behaviour can be customised with the `--input-type` argument.

`shrinkray --help` will give more usage instructions.

## Supported formats

Shrink Ray is fully generic in the sense that it will work with literally any file you give it in any format. However, some formats will work a lot better than others.

It has a generic reduction algorithm that should work pretty well with any textual format, and an architecture that is designed to make it easy to add specialised support for specific formats as needed.

Additionally, Shrink Ray has special support for the following formats:

* C and C++ (via `clang_delta`, which you will have if creduce is installed)
* Python
* JSON
* Dimacs CNF format for SAT problems

Most of this support is quite basic and is just designed to deal with specific cases that the generic logic is known
not to handle well, but it's easy to extend with additional transformations.
It is also fairly easy to add support for new formats as needed.

If you run into a test case and interestingness test that you care about that shrink ray handles badly please let me know and I'll likely see about improving its handling of that format.

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

## Sponsorship

Shrink Ray is something of a labour of love - I wanted to have a tool that actually put into practice many of my ideas about test-case reduction, as I think the previous state of the art was well behind where I'd like it to be.

That being said, it is first and foremost designed to be a useful tool for practical engineering problems.
If you find it useful as such, please [consider sponsoring my development of it](https://github.com/sponsors/DRMacIver).
