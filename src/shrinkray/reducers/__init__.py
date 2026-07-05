"""External reducer support for shrink ray.

An *external reducer* is a subprocess that reduces a test case by talking to
shrink ray over its stdin/stdout using a small newline-delimited JSON protocol
(see :mod:`shrinkray.reducers.protocol`). Its stderr is redirected to a log
file.

This package contains:

- ``protocol``: the wire protocol (message encoding/decoding + a line reader).
- ``driver``: a :class:`~shrinkray.problem.ReductionProblem` implementation that
  speaks the protocol, plus a helper for writing external reducers in Python.
- ``python``: the built-in libcst-based Python reducer, implemented as an
  external reducer.

The shrink-ray side of the protocol lives in
:mod:`shrinkray.passes.external`.
"""
