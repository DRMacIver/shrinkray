import math
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from datetime import timedelta
from difflib import unified_diff
from enum import Enum, IntEnum, auto
from shutil import which
from tempfile import TemporaryDirectory
from typing import Any, Generic, Iterable, TypeVar

import chardet
import click
import humanize
import trio
import urwid
import urwid.raw_display
from attrs import define
from binaryornot.check import is_binary_string

from shrinkray.passes.clangdelta import C_FILE_EXTENSIONS, ClangDelta, find_clang_delta
from shrinkray.problem import (
    BasicReductionProblem,
    InvalidInitialExample,
    ReductionProblem,
    shortlex,
)
from shrinkray.reducer import DirectoryShrinkRay, Reducer, ShrinkRay
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
            try:
                signal_group(sp, signal.SIGKILL)
            except ProcessLookupError:
                pass

        with trio.move_on_after(delay):
            await sp.wait()

        if sp.returncode is None:
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


class DisplayMode(IntEnum):
    auto = 0
    text = 1
    hex = 2


class UIType(Enum):
    urwid = auto()
    basic = auto()


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


def find_python_command(name: str) -> str | None:
    first_attempt = which(name)
    if first_attempt is not None:
        return first_attempt
    second_attempt = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(second_attempt):
        return second_attempt
    return None


def default_formatter_command_for(filename):
    *_, ext = os.path.splitext(filename)

    if ext in (".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"):
        return which("clang-format")

    if ext == ".py":
        black = find_python_command("black")
        if black is not None:
            return [black, "-"]


def default_reformat_data(data: bytes) -> bytes:
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


TestCase = TypeVar("TestCase")


@define(slots=False)
class ShrinkRayState(Generic[TestCase], ABC):
    input_type: InputType
    in_place: bool
    test: list[str]
    filename: str
    timeout: float
    base: str
    parallelism: int
    initial: TestCase
    formatter: str
    trivial_is_error: bool
    seed: int
    volume: Volume
    clang_delta_executable: ClangDelta | None

    first_call: bool = True
    initial_exit_code: int | None = None
    parallel_tasks_running: int = 0
    can_format: bool = True
    formatter_command: list[str] | None = None

    first_call_time: float | None = None

    def __attrs_post_init__(self):
        self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))
        self.setup_formatter()

    @abstractmethod
    def setup_formatter(self): ...

    @abstractmethod
    def new_reducer(self, problem: ReductionProblem[TestCase]) -> Reducer[TestCase]: ...

    @abstractmethod
    async def write_test_case_to_file_impl(self, working: str, test_case: TestCase): ...

    async def write_test_case_to_file(self, working: str, test_case: TestCase):
        await self.write_test_case_to_file_impl(working, test_case)

    async def run_script_on_file(
        self, working: str, cwd: str, debug: bool = False
    ) -> int:
        if not os.path.exists(working):
            raise ValueError(f"No such file {working}")
        if self.input_type.enabled(InputType.arg):
            command = self.test + [working]
        else:
            command = self.test

        kwargs: dict[str, Any] = dict(
            universal_newlines=False,
            preexec_fn=os.setsid,
            cwd=cwd,
            check=False,
        )
        if self.input_type.enabled(InputType.stdin) and not os.path.isdir(working):
            with open(working, "rb") as i:
                kwargs["stdin"] = i.read()
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
                with trio.move_on_after(
                    self.timeout * 10 if self.first_call else self.timeout
                ):
                    await sp.wait()

                runtime = time.time() - start_time

                if sp.returncode is None:
                    await interrupt_wait_and_kill(sp)

                if runtime >= self.timeout and self.first_call:
                    raise TimeoutExceededOnInitial(
                        timeout=self.timeout,
                        runtime=runtime,
                    )
            finally:
                if self.first_call:
                    self.initial_exit_code = sp.returncode
                self.first_call = False

            result: int | None = sp.returncode
            assert result is not None
            return result

    async def run_for_exit_code(self, test_case: TestCase, debug: bool = False) -> int:
        if self.in_place:
            if self.input_type == InputType.basename:
                working = self.filename
                await self.write_test_case_to_file(working, test_case)

                return await self.run_script_on_file(
                    working=working,
                    debug=debug,
                    cwd=os.getcwd(),
                )
            else:
                base, ext = os.path.splitext(self.filename)
                working = base + '-' + os.urandom(16).hex() + ext
                assert not os.path.exists(working)
                try:
                    await self.write_test_case_to_file(working, test_case)

                    return await self.run_script_on_file(
                        working=working,
                        debug=debug,
                        cwd=os.getcwd(),
                    )
                finally:
                    if os.path.exists(working):
                        if os.path.isdir(working):
                            shutil.rmtree(working)
                        else:
                            os.unlink(working)
        else:
            with TemporaryDirectory() as d:
                working = os.path.join(d, self.base)
                await self.write_test_case_to_file(working, test_case)

                return await self.run_script_on_file(
                    working=working,
                    debug=debug,
                    cwd=d,
                )

    @abstractmethod
    async def format_data(self, test_case: TestCase) -> TestCase | None: ...

    @abstractmethod
    async def run_formatter_command(
        self, command: str | list[str], input: TestCase
    ) -> subprocess.CompletedProcess: ...

    @abstractmethod
    async def print_exit_message(self, problem): ...

    @property
    def reducer(self):
        try:
            return self.__reducer
        except AttributeError:
            pass

        work = WorkContext(
            random=random.Random(self.seed),
            volume=self.volume,
            parallelism=self.parallelism,
        )

        problem: BasicReductionProblem[TestCase] = BasicReductionProblem(
            is_interesting=self.is_interesting,
            initial=self.initial,
            work=work,
            **self.extra_problem_kwargs,
        )

        # Writing the file back can't be guaranteed atomic, so we put a lock around
        # writing successful reductions back to the original file so we don't
        # write some confused combination of reductions.
        write_lock = trio.Lock()

        @problem.on_reduce
        async def _(test_case: TestCase):
            async with write_lock:
                await self.write_test_case_to_file(self.filename, test_case)

        self.__reducer = self.new_reducer(problem)
        return self.__reducer

    @property
    def extra_problem_kwargs(self):
        return {}

    @property
    def problem(self):
        return self.reducer.target

    async def is_interesting(self, test_case: TestCase) -> bool:
        if self.first_call_time is None:
            self.first_call_time = time.time()
        async with self.is_interesting_limiter:
            try:
                self.parallel_tasks_running += 1
                return await self.run_for_exit_code(test_case) == 0
            finally:
                self.parallel_tasks_running -= 1

    async def attempt_format(self, data: TestCase) -> TestCase:
        if not self.can_format:
            return data
        attempt = await self.format_data(data)
        if attempt is None:
            self.can_format = False
            return data
        if attempt == data or await self.is_interesting(attempt):
            return attempt
        else:
            self.can_format = False
            return data

    async def check_formatter(self):
        if self.formatter_command is None:
            return
        formatter_result = await self.run_formatter_command(
            self.formatter_command, self.initial
        )

        if formatter_result.returncode != 0:
            print(
                "Formatter exited unexpectedly on initial test case. If this is expected, please run with --formatter=none.",
                file=sys.stderr,
            )
            print(
                formatter_result.stderr.decode("utf-8").strip(),
                file=sys.stderr,
            )
            sys.exit(1)
        reformatted = formatter_result.stdout
        if not await self.is_interesting(reformatted) and await self.is_interesting(
            self.initial
        ):
            print(
                "Formatting initial test case made it uninteresting. If this is expected, please run with --formatter=none.",
                file=sys.stderr,
            )
            print(
                formatter_result.stderr.decode("utf-8").strip(),
                file=sys.stderr,
            )
            sys.exit(1)

    async def report_error(self, e):
        print(
            "Shrink ray cannot proceed because the initial call of the interestingness test resulted in an uninteresting test case.",
            file=sys.stderr,
        )
        if isinstance(e, TimeoutExceededOnInitial):
            print(
                f"This is because your initial test case took {e.runtime:.2f}s exceeding your timeout setting of {self.timeout}.",
                file=sys.stderr,
            )
            print(
                f"Try rerunning with --timeout={math.ceil(e.runtime * 2)}.",
                file=sys.stderr,
            )
        else:
            print(
                "Rerunning the interestingness test for debugging purposes...",
                file=sys.stderr,
            )
            exit_code = await self.run_for_exit_code(self.initial, debug=True)
            if exit_code != 0:
                print(
                    f"This exited with code {exit_code}, but the script should return 0 for interesting test cases.",
                    file=sys.stderr,
                )
                local_exit_code = await self.run_script_on_file(
                    working=self.filename,
                    debug=False,
                    cwd=os.getcwd(),
                )
                if local_exit_code == 0:
                    print(
                        "Note that Shrink Ray runs your script on a copy of the file in a temporary directory. "
                        "Here are the results of running it in the current directory directory...",
                        file=sys.stderr,
                    )
                    other_exit_code = await self.run_script_on_file(
                        working=self.filename,
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
                assert self.initial_exit_code not in (None, 0)
                print(
                    f"This exited with code 0, but previously the script exited with {self.initial_exit_code}. "
                    "This suggests your interestingness test exhibits nondeterministic behaviour.",
                    file=sys.stderr,
                )
        sys.exit(1)


@define(slots=False)
class ShrinkRayStateSingleFile(ShrinkRayState[bytes]):
    def new_reducer(self, problem: ReductionProblem[bytes]) -> Reducer[bytes]:
        return ShrinkRay(problem, clang_delta=self.clang_delta_executable)

    def setup_formatter(self):
        if self.formatter.lower() == "none":

            async def format_data(test_case: bytes) -> bytes | None:
                await trio.lowlevel.checkpoint()
                return test_case

            self.can_format = False

        else:
            formatter_command = determine_formatter_command(
                self.formatter, self.filename
            )
            if formatter_command is not None:
                self.formatter_command = formatter_command

                async def format_data(test_case: bytes) -> bytes | None:
                    result = await self.run_formatter_command(
                        formatter_command, test_case
                    )
                    if result.returncode != 0:
                        return None
                    return result.stdout

            else:

                async def format_data(test_case: bytes) -> bytes | None:
                    await trio.lowlevel.checkpoint()
                    return default_reformat_data(test_case)

        self.__format_data = format_data

    async def format_data(self, test_case: bytes) -> bytes | None:
        return await self.__format_data(test_case)

    async def run_formatter_command(
        self, command: str | list[str], input: bytes
    ) -> subprocess.CompletedProcess:
        return await trio.run_process(
            command,
            stdin=input,
            capture_stdout=True,
            capture_stderr=True,
            check=False,
        )

    async def write_test_case_to_file_impl(self, working: str, test_case: bytes):
        async with await trio.open_file(working, "wb") as o:
            await o.write(test_case)

    async def is_interesting(self, test_case: bytes) -> bool:
        async with self.is_interesting_limiter:
            try:
                self.parallel_tasks_running += 1
                return await self.run_for_exit_code(test_case) == 0
            finally:
                self.parallel_tasks_running -= 1

    async def print_exit_message(self, problem):
        formatting_increase = 0
        final_result = problem.current_test_case
        reformatted = await self.attempt_format(final_result)
        if reformatted != final_result and reformatted is not None:
            if await self.is_interesting(reformatted):
                async with await trio.open_file(self.filename, "wb") as o:
                    await o.write(reformatted)
            formatting_increase = max(0, len(reformatted) - len(final_result))
            final_result = reformatted

        if len(problem.current_test_case) <= 1 and self.trivial_is_error:
            print(
                f"Reduced to a trivial test case of size {len(problem.current_test_case)}"
            )
            print(
                "This probably wasn't what you intended. If so, please modify your interestingness test "
                "to be more restrictive.\n"
                "If you intended this behaviour, you can run with '--trivial-is-not-error' to "
                "suppress this message."
            )
            sys.exit(1)

        else:
            print("Reduction completed!")
            stats = problem.stats
            if self.initial == final_result:
                print("Test case was already maximally reduced.")
            elif len(final_result) < len(self.initial):
                print(
                    f"Deleted {humanize.naturalsize(stats.initial_test_case_size - len(final_result))} "
                    f"out of {humanize.naturalsize(stats.initial_test_case_size)} "
                    f"({(1.0 - len(final_result) / stats.initial_test_case_size) * 100:.2f}% reduction) "
                    f"in {humanize.precisedelta(timedelta(seconds=time.time() - stats.start_time))}"
                )
            elif len(final_result) == len(self.initial):
                print("Some changes were made but no bytes were deleted")
            else:
                print(
                    f"Running reformatting resulted in an increase of {humanize.naturalsize(formatting_increase)}."
                )


def to_lines(test_case: bytes) -> list[str]:
    result = []
    for line in test_case.split(b"\n"):
        if is_binary_string(line):
            result.append(line.hex())
        else:
            try:
                result.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                result.append(line.hex())
    return result


def to_blocks(test_case: bytes) -> list[str]:
    return [test_case[i : i + 80].hex() for i in range(0, len(test_case), 80)]


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


class ShrinkRayDirectoryState(ShrinkRayState[dict[str, bytes]]):
    def setup_formatter(self): ...

    @property
    def extra_problem_kwargs(self):
        def dict_size(test_case: dict[str, bytes]) -> int:
            return sum(len(v) for v in test_case.values())

        def dict_sort_key(test_case: dict[str, bytes]) -> Any:
            return (
                len(test_case),
                dict_size(test_case),
                sorted((k, shortlex(v)) for k, v in test_case.items()),
            )

        return dict(
            sort_key=dict_sort_key,
            size=dict_size,
        )

    def new_reducer(
        self, problem: ReductionProblem[dict[str, bytes]]
    ) -> Reducer[dict[str, bytes]]:
        return DirectoryShrinkRay(
            target=problem, clang_delta=self.clang_delta_executable
        )

    async def write_test_case_to_file_impl(
        self, working: str, test_case: dict[str, bytes]
    ):
        shutil.rmtree(working, ignore_errors=True)
        os.makedirs(working, exist_ok=True)
        for k, v in test_case.items():
            f = os.path.join(working, k)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            async with await trio.open_file(f, "wb") as o:
                await o.write(v)

    async def format_data(self, test_case: dict[str, bytes]) -> dict[str, bytes] | None:
        # TODO: Implement
        return None

    async def run_formatter_command(
        self, command: str | list[str], input: TestCase
    ) -> subprocess.CompletedProcess:
        raise AssertionError

    async def print_exit_message(self, problem):
        print("All done!")


@define(slots=False)
class ShrinkRayUI(Generic[TestCase], ABC):
    state: ShrinkRayState[TestCase]

    @property
    def reducer(self):
        return self.state.reducer

    @property
    def problem(self) -> BasicReductionProblem:
        return self.reducer.target  # type: ignore

    def install_into_nursery(self, nursery: trio.Nursery): ...

    async def run(self, nursery: trio.Nursery): ...


class BasicUI(ShrinkRayUI[TestCase]):
    async def run(self, nursery: trio.Nursery):
        prev_reduction = 0
        while True:
            initial = self.state.initial
            current = self.state.problem.current_test_case
            size = self.state.problem.size
            reduction = size(initial) - size(current)
            if reduction > prev_reduction:
                print(
                    f"Reduced test case to {humanize.naturalsize(size(current))} "
                    f"(deleted {humanize.naturalsize(reduction)}, "
                    f"{humanize.naturalsize(reduction - prev_reduction)} since last time)"
                )
                prev_reduction = reduction
                await trio.sleep(5)
            else:
                await trio.sleep(0.1)


@define(slots=False)
class UrwidUI(ShrinkRayUI[TestCase]):
    parallel_samples: int = 0
    parallel_total: int = 0

    def __attrs_post_init__(self):
        frame = self.create_frame()

        screen = urwid.raw_display.Screen()

        def unhandled(key: Any) -> bool:
            if key == "q":
                raise urwid.ExitMainLoop()
            return False

        self.event_loop = urwid.TrioEventLoop()

        self.loop = urwid.MainLoop(
            frame,
            [],
            screen,
            unhandled_input=unhandled,
            event_loop=self.event_loop,
        )

    async def regularly_clear_screen(self):
        while True:
            self.loop.screen.clear()
            await trio.sleep(1)

    async def update_parallelism_stats(self) -> None:
        while True:
            await trio.sleep(random.expovariate(10.0))
            self.parallel_samples += 1
            self.parallel_total += self.state.parallel_tasks_running
            stats = self.problem.stats
            if stats.calls > 0:
                wasteage = stats.wasted_interesting_calls / stats.calls
            else:
                wasteage = 0.0

            average_parallelism = self.parallel_total / self.parallel_samples

            self.parallelism_status.set_text(
                f"Current parallel workers: {self.state.parallel_tasks_running} (Average {average_parallelism:.2f}) "
                f"(effective parallelism: {average_parallelism * (1.0 - wasteage):.2f})"
            )

    async def update_reducer_stats(self) -> None:
        while True:
            await trio.sleep(0.1)
            if self.problem is None:
                continue

            self.details_text.set_text(self.problem.stats.display_stats())
            self.reducer_status.set_text(f"Reducer status: {self.reducer.status}")

    def install_into_nursery(self, nursery: trio.Nursery):
        nursery.start_soon(self.regularly_clear_screen)
        nursery.start_soon(self.update_parallelism_stats)
        nursery.start_soon(self.update_reducer_stats)

    async def run(self, nursery: trio.Nursery):
        with self.loop.start():
            await self.event_loop.run_async()
        nursery.cancel_scope.cancel()

    def create_frame(self) -> urwid.Frame:
        text_header = "Shrink Ray. Press q to exit."
        self.parallelism_status = urwid.Text("")

        self.details_text = urwid.Text("")
        self.reducer_status = urwid.Text("")

        line = urwid.Divider("─")

        listbox_content = [
            line,
            self.details_text,
            self.reducer_status,
            self.parallelism_status,
            line,
            *self.create_main_ui_elements(),
        ]

        header = urwid.AttrMap(urwid.Text(text_header, align="center"), "header")
        listbox = urwid.ListBox(urwid.SimpleFocusListWalker(listbox_content))
        return urwid.Frame(urwid.AttrMap(listbox, "body"), header=header)

    def create_main_ui_elements(self) -> list[Any]:
        return []


class ShrinkRayUIDirectory(UrwidUI[dict[str, bytes]]):
    def create_main_ui_elements(self) -> list[Any]:
        self.col1 = urwid.Text("")
        self.col2 = urwid.Text("")
        self.col3 = urwid.Text("")

        columns = urwid.Columns(
            [
                ("weight", 1, self.col1),
                ("weight", 1, self.col2),
                ("weight", 1, self.col3),
            ]
        )

        return [columns]

    async def update_file_list(self):
        while True:
            if self.state.first_call_time is None:
                await trio.sleep(0.05)
                continue
            data = sorted(self.problem.current_test_case.items())

            runtime = time.time() - self.state.first_call_time

            col1_bits = []
            col2_bits = []
            col3_bits = []

            for k, v in data:
                col1_bits.append(k)
                col2_bits.append(humanize.naturalsize(len(v)))
                reduction_percentage = (1.0 - len(v) / len(self.state.initial[k])) * 100
                reduction_rate = (len(self.state.initial[k]) - len(v)) / runtime
                reduction_msg = f"{reduction_percentage:.2f}% reduction, {humanize.naturalsize(reduction_rate)} / second"
                col3_bits.append(reduction_msg)

            self.col1.set_text("\n".join(col1_bits))
            self.col2.set_text("\n".join(col2_bits))
            self.col3.set_text("\n".join(col3_bits))
            await trio.sleep(0.5)

    def install_into_nursery(self, nursery: trio.Nursery):
        super().install_into_nursery(nursery)
        nursery.start_soon(self.update_file_list)


@define(slots=False)
class ShrinkRayUISingleFile(UrwidUI[bytes]):
    hex_mode: bool = False

    def create_main_ui_elements(self) -> list[Any]:
        self.diff_to_display = urwid.Text("")
        return [self.diff_to_display]

    def file_to_lines(self, test_case: bytes) -> list[str]:
        if self.hex_mode:
            return to_blocks(test_case)
        else:
            return to_lines(test_case)

    async def update_diffs(self):
        initial = self.problem.current_test_case
        self.diff_to_display.set_text("\n".join(self.file_to_lines(initial)[:1000]))
        prev_unformatted = self.problem.current_test_case
        prev = await self.state.attempt_format(prev_unformatted)

        time_of_last_update = time.time()
        while True:
            if self.problem.current_test_case == prev_unformatted:
                await trio.sleep(0.1)
                continue
            current = await self.state.attempt_format(self.problem.current_test_case)
            lines = self.file_to_lines(current)
            if len(lines) <= 50:
                display_text = "\n".join(lines)
                self.diff_to_display.set_text(display_text)
                await trio.sleep(0.1)
                continue

            if prev == current:
                await trio.sleep(0.1)
                continue
            diff = format_diff(
                unified_diff(self.file_to_lines(prev), self.file_to_lines(current))
            )
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
            self.diff_to_display.set_text(diff)
            prev = current
            prev_unformatted = self.problem.current_test_case
            time_of_last_update = time.time()
            if self.state.can_format:
                await trio.sleep(4)
            else:
                await trio.sleep(2)

    def install_into_nursery(self, nursery: trio.Nursery):
        super().install_into_nursery(nursery)
        nursery.start_soon(self.update_diffs)


def determine_formatter_command(formatter: str, filename: str) -> list[str] | None:
    if formatter.lower() == "default":
        formatter_command = default_formatter_command_for(filename)
    elif formatter.lower() != "none":
        formatter_command = formatter
    else:
        formatter_command = None
    if isinstance(formatter_command, str):
        formatter_command = [formatter_command]
    return formatter_command


async def run_shrink_ray(
    state: ShrinkRayState[TestCase],
    ui: ShrinkRayUI[TestCase],
):
    async with trio.open_nursery() as nursery:
        problem = state.problem
        try:
            await problem.setup()
        except InvalidInitialExample as e:
            await state.report_error(e)

        reducer = state.reducer

        @nursery.start_soon
        async def _() -> None:
            await reducer.run()
            nursery.cancel_scope.cancel()

        ui.install_into_nursery(nursery)

        await ui.run(nursery)

    await state.print_exit_message(problem)


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
    "--volume",
    default="normal",
    type=EnumChoice(Volume),
    help="Level of output to provide.",
)
@click.option(
    "--display-mode",
    default="auto",
    type=EnumChoice(DisplayMode),
    help="Determines whether ShrinkRay displays files as a textual or hex representation of binary data.",
)
@click.option(
    "--in-place/--not-in-place",
    default=False,
    help="""
If `--in-place` is passed, shrinkray will run in the current working directory instead of
creating a temporary subdirectory. Note that this requires you to either run with no
parallelism or be very careful about files created in your interestingness
test not conflicting with each other.
""",
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

If --in-place is specified, all will not include basename by default, only arg and stdin.
If you want basename with --in-place you may pass it explicitly, but note that this is incompatible
with any parallelism.
    """.strip(),
)
@click.option(
    "--parallelism",
    type=click.INT,
    help="Number of tests to run in parallel. If set to 0 will default to either 1 or number of cpus depending on other options.",
    default=0,
)
@click.option(
    "--ui",
    "ui_type",
    default="urwid",
    type=EnumChoice(UIType),
    help="""
By default shrinkray runs with a terminal UI based on urwid. If you want a more basic UI
(e.g. for running in a script), you can specify --ui=basic instead.
    """.strip(),
)
@click.option(
    "--formatter",
    default="default",
    help="""
Path to a formatter for Shrink Ray to use. This is mostly used for display purposes,
and to format the final test case.

A formatter should accept input on stdin and write to stdout, and exit with a status
code of 0. If the formatter exits with a non-zero status code its output will be
ignored.

Special values for this:

* 'none' turns off formatting.
* 'default' causes Shrink Ray to use its default behaviour, which is to look for
  formatters it knows about on PATH and use one of those if found, otherwise to
  use a very simple language-agnostic formatter.
""",
)
@click.option(
    "--trivial-is-error/--trivial-is-not-error",
    default=True,
    help="""
It's easy to write interestingness tests which accept too much, and one common way this
happens is if they accept empty or otherwise trivial files. By default Shrink Ray will
print an error message at the end of reduction and exit with non-zero status in this case.
This behaviour can be disabled by passing --trivial-is-not-error.
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
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, allow_dash=False),
)
def main(
    input_type: InputType,
    display_mode: DisplayMode,
    backup: str,
    filename: str,
    test: list[str],
    timeout: float,
    in_place: bool,
    parallelism: int,
    seed: int,
    volume: Volume,
    formatter: str,
    no_clang_delta: bool,
    clang_delta: str,
    trivial_is_error: bool,
    ui_type: UIType,
) -> None:
    if timeout <= 0:
        timeout = float("inf")

    if not os.access(test[0], os.X_OK):
        print(
            f"Interestingness test {os.path.relpath(test[0])} is not executable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if in_place and input_type == InputType.basename and parallelism > 1:
        raise click.BadParameter(
            f"parallelism cannot be greater than 1 when --in-place and --input-type=basename (got {parallelism})"
        )

    if parallelism == 0:
        if in_place and input_type == InputType.basename:
            parallelism = 1
        else:
            parallelism = os.cpu_count()

    clang_delta_executable: ClangDelta | None = None
    if os.path.splitext(filename)[1] in C_FILE_EXTENSIONS and not no_clang_delta:
        if not clang_delta:
            clang_delta = find_clang_delta()
        if not clang_delta:
            raise click.UsageError(
                "Attempting to reduce a C or C++ file, but clang_delta is not installed. "
                "Please run with --no-clang-delta, or install creduce on your system. "
                "If creduce is already installed and you wish to use clang_delta, please "
                "pass its path with the --clang-delta argument."
            )

        clang_delta_executable = ClangDelta(clang_delta)

    # This is a debugging option so that when the reducer seems to be taking
    # a long time you can Ctrl-\ to find out what it's up to. I have no idea
    # how to test it in a way that shows up in coverage.
    def dump_trace(signum: int, frame: Any) -> None:  # pragma: no cover
        traceback.print_stack()

    signal.signal(signal.SIGQUIT, dump_trace)

    if not backup:
        backup = filename + os.extsep + "bak"

    state_kwargs: dict[str, Any] = dict(
        input_type=input_type,
        in_place=in_place,
        test=test,
        timeout=timeout,
        base=os.path.basename(filename),
        parallelism=parallelism,
        filename=filename,
        formatter=formatter,
        trivial_is_error=trivial_is_error,
        seed=seed,
        volume=volume,
        clang_delta_executable=clang_delta_executable,
    )

    state: ShrinkRayState[Any]
    ui: ShrinkRayUI[Any]

    if os.path.isdir(filename):
        if input_type == InputType.stdin:
            raise click.UsageError("Cannot pass a directory input on stdin.")

        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(filename, backup)

        files = [os.path.join(d, f) for d, _, fs in os.walk(filename) for f in fs]

        initial = {}
        for f in files:
            with open(f, "rb") as i:
                initial[os.path.relpath(f, filename)] = i.read()

        state = ShrinkRayDirectoryState(initial=initial, **state_kwargs)

        trio.run(state.check_formatter)

        ui = ShrinkRayUIDirectory(state)

    else:
        try:
            os.remove(backup)
        except FileNotFoundError:
            pass

        with open(filename, "rb") as reader:
            initial = reader.read()

        with open(backup, "wb") as writer:
            writer.write(initial)

        if display_mode == DisplayMode.auto:
            hex_mode = is_binary_string(initial)
        else:
            hex_mode = display_mode == DisplayMode.hex

        state = ShrinkRayStateSingleFile(initial=initial, **state_kwargs)

        trio.run(state.check_formatter)

        ui = ShrinkRayUISingleFile(state, hex_mode=hex_mode)

    if ui_type == UIType.basic:
        ui = BasicUI(state)

    try:
        trio.run(
            lambda: run_shrink_ray(
                state=state,
                ui=ui,
            )
        )
    # If you try to sys.exit from within an exception handler, trio will instead
    # put it in an exception group. I wish to register the complaint that this is
    # incredibly fucking stupid, but anyway this is a workaround for it.
    except *SystemExit as eg:
        raise eg.exceptions[0]


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="shrinkray")
