import os
import random
import shlex
import signal
import subprocess
import sys
import time
import traceback
from enum import Enum
from enum import EnumMeta
from enum import IntEnum
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any
from typing import Generic
from typing import TypeVar

import click
import trio

from shrinkray.passes.bytes import byte_passes
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import Reducer
from shrinkray.work import Volume
from shrinkray.work import WorkContext


def validate_command(ctx: Any, param: Any, value: str) -> list[str]:
    parts = shlex.split(value)
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter("%s: command not found" % (command,))
        command = os.path.abspath(what)
    return [command] + parts[1:]


def signal_group(sp: "trio.Process", signal: int) -> None:
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, signal)


async def interrupt_wait_and_kill(sp: "trio.Process", timeout: float = 0.1) -> None:
    if sp.returncode is None:
        try:
            # In case the subprocess forked. Python might hang if you don't close
            # all pipes.
            for pipe in [sp.stdout, sp.stderr, sp.stdin]:
                if pipe:
                    await pipe.aclose()
            signal_group(sp, signal.SIGINT)
            for _ in range(10):
                if sp.poll() is not None:
                    return
                await trio.sleep(timeout)
            signal_group(sp, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover
            # This is incredibly hard to trigger reliably, because it only happens
            # if the process exits at exactly the wrong time.
            pass

        with trio.move_on_after(timeout):
            await sp.wait()

        if sp.returncode is not None:
            raise ValueError(
                f"Could not kill subprocess with pid {sp.pid}. Something has gone seriously wrong."
            )


EnumType = TypeVar("EnumType", bound=Enum)


class EnumChoice(click.Choice, Generic[EnumType]):
    def __init__(self, enum: type[EnumType]):
        self.enum = enum
        choices = [str(e.name) for e in enum]
        self.__values = {e.name: e for e in enum}
        super().__init__(choices)

    def convert(self, value: str, param: Any, ctx: Any) -> EnumType:
        return self.__values[value]


class InputType(IntEnum):
    all = 0
    stdin = 1
    arg = 2
    basename = 3

    def enabled(self, value: "InputType") -> bool:
        if self == InputType.all:
            return True
        return self == value


@click.command(
    help="""
""".strip()
)
@click.version_option()
@click.option(
    "--backup",
    default="",
    help=(
        "Name of the backup file to create. Defaults to adding .bak to the "
        "name of the source file"
    ),
)
@click.option(
    "--timeout",
    default=1,
    type=click.FLOAT,
    help=(
        "Time out subprocesses after this many seconds. If set to <= 0 then "
        "no timeout will be used. Any commands that time out will be treated "
        "as failing the test"
    ),
)
@click.option(
    "--seed",
    default=0,
    type=click.INT,
    help=("Random seed to use for any non-deterministic reductions."),
)
@click.option(
    "--parallelism",
    default=os.cpu_count(),
    type=click.INT,
    help="Number of tests to run in parallel.",
)
@click.option(
    "--volume",
    default="normal",
    type=EnumChoice(Volume),
    help="Level of output to provide.",
)
@click.option(
    "--input-type",
    default="all",
    type=EnumChoice(InputType),
    help=""""
How to pass input to the test function. Options are:

1. basename writes it to a file of the same basename as the original, in the current working directory where the test is run.
2. arg passes it in a file whose name is provided as an argument to the test.
3. stdin passes its contents on stdin.
4. all (the default) does all of the above.
    """.strip(),
)
@click.argument("test", callback=validate_command)
@click.argument(
    "filename",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, allow_dash=False),
)
def main(
    input_type: InputType,
    backup: str,
    filename: str,
    test: list[str],
    timeout: float,
    parallelism: int,
    seed: int,
    volume: Volume,
) -> None:
    if timeout <= 0:
        timeout = float("inf")
    debug = volume == Volume.debug

    if debug:
        # This is a debugging option so that when the reducer seems to be taking
        # a long time you can Ctrl-\ to find out what it's up to. I have no idea
        # how to test it in a way that shows up in coverage.
        def dump_trace(signum: int, frame: Any) -> None:  # pragma: no cover
            traceback.print_stack()

        signal.signal(signal.SIGQUIT, dump_trace)

    if not backup:
        backup = filename + os.extsep + "bak"

    try:
        os.remove(backup)
    except FileNotFoundError:
        pass

    base = os.path.basename(filename)
    first_call = True

    async def is_interesting(test_case: bytes) -> bool:
        nonlocal first_call
        with TemporaryDirectory() as d:
            working = os.path.join(d, base)
            with open(working, "wb") as o:
                o.write(test_case)

            if input_type.enabled(InputType.arg):
                command = test + [working]
            else:
                command = test

            kwargs = dict(
                universal_newlines=False,
                preexec_fn=os.setsid,
                cwd=d,
                check=False,
            )
            if input_type.enabled(InputType.stdin):
                kwargs["stdin"] = test_case
            else:
                kwargs["stdin"] = b""

            if not debug:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL

            async with trio.open_nursery() as nursery:

                def call_with_kwargs(task_status=trio.TASK_STATUS_IGNORED):  # type: ignore
                    return trio.run_process(command, **kwargs, task_status=task_status)  # type: ignore

                sp = await nursery.start(call_with_kwargs)

                try:
                    with trio.move_on_after(timeout):
                        await sp.wait()

                    if sp.returncode is None:
                        await interrupt_wait_and_kill(sp)
                        if first_call:
                            raise ValueError(
                                f"Initial test call exceeded timeout of {timeout}s. Try raising or disabling timeout."
                            )
                        return False
                finally:
                    first_call = False

                succeeded: bool = sp.returncode == 0
                return succeeded

    with open(filename, "rb") as reader:
        initial = reader.read()

    with open(backup, "wb") as writer:
        writer.write(initial)

    @trio.run
    async def _() -> None:
        work = WorkContext(
            random=random.Random(seed),
            volume=volume,
            parallelism=parallelism,
        )

        problem: BasicReductionProblem[bytes] = await BasicReductionProblem(  # type: ignore
            is_interesting=is_interesting,
            initial=initial,
            work=work,
        )

        @problem.on_reduce
        async def _(test_case: bytes) -> None:
            with open(filename, "wb") as o:
                o.write(test_case)

        reducer = Reducer(target=problem, reduction_passes=byte_passes(problem))

        async with problem.work.pb(
            total=lambda: len(initial),
            current=lambda: len(initial) - len(problem.current_test_case),
            desc="Bytes deleted",
        ):
            await reducer.run()


if __name__ == "__main__":
    main(prog_name="shrinkray")  # pragma: no cover
