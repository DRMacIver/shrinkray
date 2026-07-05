"""The built-in libcst-based Python reducer, as an external reducer.

This runs the ordinary :data:`~shrinkray.passes.python.PYTHON_PASSES` against a
:class:`~shrinkray.reducers.driver.RemoteReductionProblem`, so shrink ray's
Python-specific reductions execute in a subprocess that talks to shrink ray over
the external-reducer protocol.

It is launched by shrink ray as ``python -m shrinkray.reducers.python`` (see
:func:`shrinkray.passes.python.python_reducer_command`). Parallelism and the
random seed are taken from the environment.
"""

import os

import trio

from shrinkray.passes.python import PYTHON_PASSES
from shrinkray.reducers.driver import run_reducer


async def serve(parallelism: int, seed: int) -> None:
    """Run the reducer over this process's stdin/stdout."""
    stdin_stream = trio.lowlevel.FdStream(os.dup(0))
    stdout_stream = trio.lowlevel.FdStream(os.dup(1))
    try:
        await run_reducer(
            PYTHON_PASSES,
            stdin_stream=stdin_stream,
            stdout_stream=stdout_stream,
            parallelism=parallelism,
            seed=seed,
        )
    finally:
        await stdin_stream.aclose()
        await stdout_stream.aclose()


def main() -> None:
    """Entry point for the ``shrinkray.reducers.python`` module."""
    parallelism = int(os.environ.get("SHRINKRAY_REDUCER_PARALLELISM", "1"))
    seed = int(os.environ.get("SHRINKRAY_REDUCER_SEED", "0"))
    trio.run(serve, parallelism, seed)


if __name__ == "__main__":
    main()
