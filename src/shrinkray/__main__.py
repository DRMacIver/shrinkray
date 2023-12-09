from difflib import unified_diff
import os
import random
import shlex
import signal
import subprocess
import traceback
from enum import Enum, IntEnum
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any, Generic, TypeVar
import warnings

import click

import trio

warnings.filterwarnings("ignore", category=trio.TrioDeprecationWarning)

from shrinkray.passes import common_passes

from shrinkray.passes.bytes import byte_passes
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import Reducer
from shrinkray.work import Volume, WorkContext
from typing import Any

import urwid
import urwid.raw_display
import trio


def validate_command(ctx: Any, param: Any, value: str) -> list[str]:
    parts = shlex.split(value)
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter(f"{command}: command not found")
        command = os.path.abspath(what)
    return [command] + parts[1:]


def signal_group(sp: "trio.Process", signal: int) -> None:
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, signal)


async def interrupt_wait_and_kill(sp: "trio.Process", delay: float = 0.1) -> None:
    await trio.lowlevel.checkpoint()
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
                await trio.sleep(delay)
            signal_group(sp, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover
            # This is incredibly hard to trigger reliably, because it only happens
            # if the process exits at exactly the wrong time.
            pass

        with trio.move_on_after(delay):
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
    help="""
How to pass input to the test function. Options are:

1. `basename` writes it to a file of the same basename as the original, in the current working directory where the test is run.

2. `arg` passes it in a file whose name is provided as an argument to the test.

3. `stdin` passes its contents on stdin.

4. `all` (the default) does all of the above.
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

    async def is_interesting_do_work(test_case: bytes) -> bool:
        nonlocal first_call
        with TemporaryDirectory() as d:
            working = os.path.join(d, base)
            async with await trio.open_file(working, "wb") as o:
                await o.write(test_case)  # type: ignore

            if input_type.enabled(InputType.arg):
                command = test + [working]
            else:
                command = test

            kwargs: dict[str, Any] = dict(
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
                    return trio.run_process(command, **kwargs, task_status=task_status)  # type: ignore  # noqa

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

    text_header = "Shrink Ray. Press q to exit."
    blank = urwid.Divider()

    details_text = urwid.Text("")
    reducer_status = urwid.Text("")
    parallelism_status = urwid.Text("")
    diff_to_display = urwid.Text("")

    parallel_samples = 0
    parallel_total = 0

    listbox_content = [
        urwid.Divider("-"),
        details_text,
        reducer_status,
        parallelism_status,
        urwid.Divider("-"),
        diff_to_display,
    ]

    header = urwid.AttrMap(urwid.Text(text_header, align="center"), "header")
    listbox = urwid.ListBox(urwid.SimpleListWalker(listbox_content))
    frame = urwid.Frame(urwid.AttrMap(listbox, "body"), header=header)

    palette = []

    screen = urwid.raw_display.Screen()

    def unhandled(key: Any) -> bool:
        if key == "q":
            raise urwid.ExitMainLoop()
        return False

    event_loop = urwid.TrioEventLoop()

    ui_loop = urwid.MainLoop(
        frame,
        palette,
        screen,
        unhandled_input=unhandled,
        event_loop=event_loop,
    )

    @trio.run
    async def _() -> None:
        work = WorkContext(
            random=random.Random(seed),
            volume=volume,
            parallelism=parallelism,
        )

        async with trio.open_nursery() as nursery:
            send_test_cases: trio.MemorySendChannel[
                tuple[bytes, trio.MemorySendChannel[bool]]
            ]
            receive_test_cases: trio.MemoryReceiveChannel[
                tuple[bytes, trio.MemorySendChannel[bool]]
            ]
            send_test_cases, receive_test_cases = trio.open_memory_channel(
                max(100, 10 * max(parallelism, 1))
            )

            parallel_tasks_running = 0

            async def is_interesting_worker():
                nonlocal parallel_tasks_running
                try:
                    while True:
                        test_case, reply = await receive_test_cases.receive()
                        parallel_tasks_running += 1
                        result = await is_interesting_do_work(test_case)
                        parallel_tasks_running -= 1
                        await reply.send(result)
                except trio.EndOfChannel:
                    pass

            for _i in range(max(parallelism, 1)):
                nursery.start_soon(is_interesting_worker)

            async def is_interesting(test_case: bytes) -> bool:
                send_result, receive_result = trio.open_memory_channel(1)
                await send_test_cases.send((test_case, send_result))
                return await receive_result.receive()

            problem: BasicReductionProblem[bytes] = BasicReductionProblem(
                is_interesting=is_interesting,
                initial=initial,
                work=work,
            )

            @nursery.start_soon
            async def _():
                while True:
                    await trio.sleep(0.1)

                    details_text.set_text(problem.stats.display_stats())
                    reducer_status.set_text(f"Reducer status: {reducer.status}")

            @nursery.start_soon
            async def _():
                while True:
                    await trio.sleep(random.expovariate(10.0))
                    nonlocal parallel_samples, parallel_total
                    parallel_samples += 1
                    parallel_total += parallel_tasks_running
                    parallelism_status.set_text(
                        f"Current parallel workers: {parallel_tasks_running} (Average {parallel_total / parallel_samples:.2f})"
                    )

            def to_lines(test_case: bytes) -> list[str]:
                result = []
                for line in test_case.split(b"\n"):
                    try:
                        result.append(line.decode("utf-8"))
                    except UnicodeDecodeError:
                        result.append(line.hex())
                return result

            def format_diff(diff):
                results = []
                start_writing = False
                for line in diff:
                    if not start_writing and line.startswith("@@"):
                        start_writing = True
                    if start_writing:
                        results.append(line)
                        if len(results) > 500:
                            results.append("...")
                            break
                return "\n".join(results)

            @nursery.start_soon
            async def _():
                prev = problem.current_test_case
                while True:
                    current = problem.current_test_case
                    if prev == current:
                        await trio.sleep(0.1)
                        continue
                    diff = format_diff(unified_diff(to_lines(prev), to_lines(current)))
                    diff_to_display.set_text(diff)
                    prev = current
                    await trio.sleep(2)

            @problem.on_reduce
            async def _(test_case: bytes) -> None:
                async with await trio.open_file(filename, "wb") as o:
                    await o.write(test_case)  # type: ignore

            reducer = Reducer(
                target=problem,
                reduction_passes=common_passes(problem),
            )

            @nursery.start_soon
            async def _():
                await reducer.run()
                nursery.cancel_scope.cancel()

            with ui_loop.start():
                await event_loop.run_async()

            nursery.cancel_scope.cancel()


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="shrinkray")
