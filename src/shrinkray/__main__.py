import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from difflib import unified_diff
from enum import Enum, IntEnum
from glob import glob
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any, Generic, Iterable, TypeVar

import chardet
import click
import humanize
import trio
import urwid
import urwid.raw_display

from shrinkray.passes.clangdelta import ClangDelta
from shrinkray.problem import BasicReductionProblem, InvalidInitialExample
from shrinkray.reducer import ShrinkRay
from shrinkray.work import Volume, WorkContext


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
            for n in range(10):
                if sp.poll() is not None:
                    return
                await trio.sleep(delay * 1.5**n * random.random())
        except ProcessLookupError:  # pragma: no cover
            # This is incredibly hard to trigger reliably, because it only happens
            # if the process exits at exactly the wrong time.
            pass

        if sp.returncode is None:
            signal_group(sp, signal.SIGKILL)

        with trio.move_on_after(delay):
            await sp.wait()

        if sp.returncode is not None:
            raise ValueError(
                f"Could not kill subprocess with pid {sp.pid}. Something has gone seriously wrong."
            )


EnumType = TypeVar("EnumType", bound=Enum)


class EnumChoice(click.Choice, Generic[EnumType]):
    def __init__(self, enum: type[EnumType]) -> None:
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


def try_decode(data: bytes) -> tuple[str | None, str]:
    for guess in chardet.detect_all(data):
        try:
            enc = guess["encoding"]
            if enc is not None:
                return enc, data.decode(enc)
        except UnicodeDecodeError:
            pass
    return None, ""


class TimeoutExceededOnInitial(InvalidInitialExample):
    def __init__(self, runtime: float, timeout: float) -> None:
        self.runtime = runtime
        self.timeout = timeout
        super().__init__(
            f"Initial test call exceeded timeout of {timeout}s. Try raising or disabling timeout."
        )


def reformat_data(data: bytes) -> bytes:
    encoding, decoded = try_decode(data)
    if encoding is None:
        return data
    result = []
    indent = 0

    def newline() -> None:
        result.append("\n" + indent * " ")

    start_of_newline = True
    for i, c in enumerate(decoded):
        if c == "\n":
            start_of_newline = True
            newline()
            continue
        elif c == " ":
            if start_of_newline:
                continue
        else:
            start_of_newline = False
        if c == "{":
            result.append(c)
            indent += 4
            if i + 1 == len(decoded) or decoded[i + 1] != "}":
                newline()
        elif c == "}":
            if len(result) > 1 and result[-1].endswith("    "):
                result[-1] = result[-1][:-4]
            result.append(c)
            indent -= 4
            newline()
        elif c == ";":
            result.append(c)
            newline()
        else:
            result.append(c)

    output = "".join(result)
    prev = None
    while prev != output:
        prev = output

        output = output.replace(" \n", "\n")
        output = output.replace("\n\n", "\n")

    return output.encode(encoding)


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
@click.option(
    "--reformat/--no-reformat",
    default=True,
    help="""
Internally Shrink Ray tries to delete as much data as possible. This results in very small
test cases, but not always very pleasant ones. If --format is set, Shrink Ray will
try to reformat the test case at the end to be a bit more manageable.

Note that this is not as good as a dedicated formatter for your format, and you should
probably use that after reduction if you have one.
""",
)
@click.option(
    "--no-clang-delta",
    is_flag=True,
    default=False,
    help="Pass this if you do not want to use clang delta for C/C++ transformations.",
)
@click.option(
    "--clang-delta",
    default="",
    help="Path to your clang_delta executable.",
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
    reformat: bool,
    no_clang_delta: bool,
    clang_delta: str,
) -> None:
    if timeout <= 0:
        timeout = float("inf")

    if not os.access(test[0], os.X_OK):
        print(
            f"Interestingness test {os.path.relpath(test[0])} is not executable.",
            file=sys.stderr,
        )
        sys.exit(1)

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
    initial_exit_code = None

    async def run_script_on_file(
        working: str, cwd: str, test_case: bytes, debug: bool = False
    ) -> int:
        nonlocal first_call, initial_exit_code
        if input_type.enabled(InputType.arg):
            command = test + [working]
        else:
            command = test

        kwargs: dict[str, Any] = dict(
            universal_newlines=False,
            preexec_fn=os.setsid,
            cwd=cwd,
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
                return trio.run_process(command, **kwargs, task_status=task_status)

            start_time = time.time()
            sp = await nursery.start(call_with_kwargs)

            try:
                with trio.move_on_after(timeout * 10 if first_call else timeout):
                    await sp.wait()

                runtime = time.time() - start_time

                if sp.returncode is None:
                    await interrupt_wait_and_kill(sp)

                if runtime >= timeout and first_call:
                    raise TimeoutExceededOnInitial(
                        timeout=timeout,
                        runtime=runtime,
                    )
            finally:
                if first_call:
                    initial_exit_code = sp.returncode
                first_call = False

            result: int | None = sp.returncode
            assert result is not None
            return result

    async def run_for_exit_code(test_case: bytes, debug: bool = False) -> int:
        with TemporaryDirectory() as d:
            working = os.path.join(d, base)
            async with await trio.open_file(working, "wb") as o:
                await o.write(test_case)

            return await run_script_on_file(
                working=working,
                test_case=test_case,
                debug=debug,
                cwd=d,
            )

    async def is_interesting_do_work(test_case: bytes, debug: bool = False) -> bool:
        return await run_for_exit_code(test_case, debug=debug) == 0

    with open(filename, "rb") as reader:
        initial = reader.read()

    with open(backup, "wb") as writer:
        writer.write(initial)

    text_header = "Shrink Ray. Press q to exit."

    details_text = urwid.Text("")
    reducer_status = urwid.Text("")
    parallelism_status = urwid.Text("")
    diff_to_display = urwid.Text("")

    try:
        text = initial.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        diff_to_display.set_text("\n".join(text.splitlines()[:1000]))

    parallel_samples = 0
    parallel_total = 0

    line = urwid.Divider("─")

    listbox_content = [
        line,
        details_text,
        reducer_status,
        parallelism_status,
        line,
        diff_to_display,
    ]

    header = urwid.AttrMap(urwid.Text(text_header, align="center"), "header")
    listbox = urwid.ListBox(urwid.SimpleListWalker(listbox_content))
    frame = urwid.Frame(urwid.AttrMap(listbox, "body"), header=header)

    screen = urwid.raw_display.Screen()

    def unhandled(key: Any) -> bool:
        if key == "q":
            raise urwid.ExitMainLoop()
        return False

    event_loop = urwid.TrioEventLoop()

    ui_loop = urwid.MainLoop(
        frame,
        [],
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

            async def is_interesting_worker() -> None:
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
                if first_call:
                    return await is_interesting_do_work(test_case)

                receive_result: trio.MemoryReceiveChannel[bool]
                send_result: trio.MemorySendChannel[bool]
                send_result, receive_result = trio.open_memory_channel(1)
                await send_test_cases.send((test_case, send_result))
                return await receive_result.receive()

            problem: BasicReductionProblem[bytes] = BasicReductionProblem(
                is_interesting=is_interesting,
                initial=initial,
                work=work,
            )

            reducer: ShrinkRay | None

            try:
                await problem.setup()
            except InvalidInitialExample as e:
                print(
                    "Shrink ray cannot proceed because the initial call of the interestingness test resulted in an uninteresting test case.",
                    file=sys.stderr,
                )
                if isinstance(e, TimeoutExceededOnInitial):
                    print(
                        f"This is because your initial test case took {e.runtime:.2f}s exceeding your timeout setting of {timeout}.",
                        file=sys.stderr,
                    )
                    print(
                        f"Try rerunning with --timeout={math.ceil(e.runtime * 2)}.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Rerunning the initerestingness test for debugging purposes...",
                        file=sys.stderr,
                    )
                    exit_code = await run_for_exit_code(problem.current_test_case)
                    if exit_code != 0:
                        print(
                            f"This exited with code {exit_code}, but the script should return 0 for interesting test cases.",
                            file=sys.stderr,
                        )
                        local_exit_code = await run_script_on_file(
                            working=filename,
                            test_case=problem.current_test_case,
                            debug=False,
                            cwd=os.getcwd(),
                        )
                        if local_exit_code == 0:
                            print(
                                "Note that Shrink Ray runs your script on a copy of the file in a temporary directory. "
                                "Here are the results of running it in the current directory directory...",
                                file=sys.stderr,
                            )
                            other_exit_code = await run_script_on_file(
                                working=filename,
                                test_case=problem.current_test_case,
                                debug=True,
                                cwd=os.getcwd(),
                            )
                            if other_exit_code != local_exit_code:
                                print(
                                    f"This interestingness is probably flaky as the first time we reran it locally it exited with {local_exit_code}, "
                                    f"but the second time it exited with {other_exit_code}. Please make sure your interestingness test is deterministic.",
                                    file=sys.stderr,
                                )
                            else:
                                print(
                                    "This suggests that your script depends on being run from the current working directory. Please fix it to be directory independent.",
                                    file=sys.stderr,
                                )
                    else:
                        assert initial_exit_code not in (None, 0)
                        print(
                            f"This exited with code 0, but previously the script exited with {initial_exit_code}. "
                            "This suggests your interestingness test exhibits nondeterministic behaviour.",
                            file=sys.stderr,
                        )
                reducer = None
                sys.exit(1)

            @nursery.start_soon
            async def _() -> None:
                while True:
                    await trio.sleep(0.1)

                    details_text.set_text(problem.stats.display_stats())

                    if reducer is not None:
                        reducer_status.set_text(f"Reducer status: {reducer.status}")

            @nursery.start_soon
            async def _() -> None:
                while True:
                    await trio.sleep(random.expovariate(10.0))
                    nonlocal parallel_samples, parallel_total
                    parallel_samples += 1
                    parallel_total += parallel_tasks_running
                    stats = problem.stats
                    if stats.calls > 0:
                        wasteage = stats.wasted_interesting_calls / stats.calls
                    else:
                        wasteage = 0.0

                    average_parallelism = parallel_total / parallel_samples

                    parallelism_status.set_text(
                        f"Current parallel workers: {parallel_tasks_running} (Average {average_parallelism:.2f}) "
                        f"(effective parallelism: {average_parallelism * (1.0 - wasteage):.2f})"
                    )

            def to_lines(test_case: bytes) -> list[str]:
                result = []
                for line in test_case.split(b"\n"):
                    try:
                        result.append(line.decode("utf-8"))
                    except UnicodeDecodeError:
                        result.append(line.hex())
                return result

            def format_diff(diff: Iterable[str]) -> str:
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

            can_format = reformat

            async def attempt_format(data: bytes) -> bytes:
                nonlocal can_format
                if not can_format:
                    return data
                attempt = reformat_data(data)
                if attempt == data or await problem.is_interesting(attempt):
                    return attempt
                else:
                    can_format = False
                    return data

            @nursery.start_soon
            async def _() -> None:
                prev_unformatted = problem.current_test_case
                prev = await attempt_format(prev_unformatted)

                time_of_last_update = time.time()
                while True:
                    if problem.current_test_case.count(b"\n") <= 50:
                        diff_to_display.set_text(problem.current_test_case)
                        await trio.sleep(0.1)
                        continue

                    if problem.current_test_case == prev_unformatted:
                        await trio.sleep(0.1)
                        continue
                    current = await attempt_format(problem.current_test_case)
                    if prev == current:
                        await trio.sleep(0.1)
                        continue
                    diff = format_diff(unified_diff(to_lines(prev), to_lines(current)))
                    # When running in parallel sometimes we can produce diffs that have
                    # a lot of insertions because we undo some work and then immediately
                    # redo it. this can be quite confusing when it happens in the UI
                    # (and makes Shrink Ray look bad), so when this happens we pause a
                    # little bit to try to get a better diff.
                    if (
                        diff.count("\n+") > 2 * diff.count("\n-")
                        and time.time() <= time_of_last_update + 10
                    ):
                        await trio.sleep(0.5)
                        continue
                    diff_to_display.set_text(diff)
                    prev = current
                    prev_unformatted = problem.current_test_case
                    time_of_last_update = time.time()
                    if can_format:
                        await trio.sleep(4)
                    else:
                        await trio.sleep(2)

            @problem.on_reduce
            async def _(test_case: bytes) -> None:
                async with await trio.open_file(filename, "wb") as o:
                    await o.write(test_case)

            cd_exec = None
            if (
                os.path.splitext(filename)[1] in (".c", ".cpp", ".h", ".hpp")
                and not no_clang_delta
            ):
                nonlocal clang_delta
                if not clang_delta:
                    clang_delta = which("clang_delta") or ""
                if not clang_delta:
                    possible_locations = glob(
                        "/opt/homebrew//Cellar/creduce/*/libexec/clang_delta"
                    ) + glob("/usr/libexec/clang_delta")
                    if not possible_locations:
                        raise click.UsageError(
                            "Attempting to reduce a C or C++ file, but clang_delta is not installed. "
                            "Please run with --no-clang-delta, or install creduce on your system. "
                            "If creduce is already installed and you wish to use clang_delta, please "
                            "pass its path with the --clang-delta argument."
                        )
                    clang_delta = max(possible_locations)

                cd_exec = ClangDelta(clang_delta)

            reducer = ShrinkRay(target=problem, clang_delta=cd_exec)

            @nursery.start_soon
            async def _() -> None:
                await reducer.run()
                nursery.cancel_scope.cancel()

            with ui_loop.start():
                await event_loop.run_async()

            nursery.cancel_scope.cancel()

        formatting_increase = 0
        final_result = problem.current_test_case
        if reformat:
            reformatted = reformat_data(final_result)
            if reformatted != final_result:
                if await is_interesting_do_work(reformatted):
                    async with await trio.open_file(filename, "wb") as o:
                        await o.write(reformatted)
                formatting_increase = max(0, len(reformatted) - len(final_result))
                final_result = reformatted

        print("Reduction completed!")
        stats = problem.stats
        if initial == final_result:
            print("Test case was already maximally reduced.")
        elif len(final_result) < len(initial):
            print(
                f"Deleted {humanize.naturalsize(stats.initial_test_case_size - len(final_result))} "
                f"out of {humanize.naturalsize(stats.initial_test_case_size)} "
                f"({(1.0 - len(final_result) / stats.initial_test_case_size) * 100:.2f}% reduction) "
                f"in {humanize.precisedelta(timedelta(seconds=time.time() - stats.start_time))}"
            )
        elif len(final_result) == len(initial):
            print("Some changes were made but no bytes were deleted")
        else:
            assert reformat
            print(
                f"Running reformatting resulted in an increase of {humanize.naturalsize(formatting_increase)}."
            )


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="shrinkray")
