"""State management for shrink ray reduction sessions."""

import math
import os
import random
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import humanize
import trio
from attrs import define

from shrinkray.formatting import try_decode
from shrinkray.passes.clangdelta import ClangDelta
from shrinkray.problem import (
    BasicReductionProblem,
    InvalidInitialExample,
    LazyChainedSortKey,
    ReductionProblem,
    natural_key,
    shortlex,
)
from shrinkray.reducer import DirectoryShrinkRay, Reducer, ShrinkRay
from shrinkray.work import Volume, WorkContext


T = TypeVar("T")


def sort_key_for_initial(initial: Any) -> Callable[[Any], Any]:
    if isinstance(initial, bytes):
        encoding, _ = try_decode(initial)
        if encoding is None:
            return shortlex
        else:

            def natural_for_encoding(b: T) -> Any:
                try:
                    s = b.decode(encoding)
                    return (0, natural_key(s))
                except UnicodeDecodeError:
                    return (1, shortlex(b))

            return natural_for_encoding
    else:
        assert isinstance(initial, dict)
        keys = sorted(initial, key=lambda k: shortlex(initial[k]), reverse=True)
        natural_keys = {k: sort_key_for_initial(v) for k, v in initial.items()}

        def dict_total_size(s):
            return sum(len(v) for v in s.values())

        def key_sort_key(k):
            def f(x):
                try:
                    v = x[k]
                except KeyError:
                    return (0,)
                else:
                    return (1, natural_keys[k](v))

            return f

        functions = [
            dict_total_size,
            len,
        ] + [key_sort_key(k) for k in keys]

        def dict_sort_key(v):
            return LazyChainedSortKey(
                functions=functions,
                value=v,
            )

        return dict_sort_key


class TimeoutExceededOnInitial(InvalidInitialExample):
    def __init__(self, runtime: float, timeout: float) -> None:
        self.runtime = runtime
        self.timeout = timeout
        super().__init__(
            f"Initial test call exceeded timeout of {timeout}s. Try raising or disabling timeout."
        )


# Constants for dynamic timeout
DYNAMIC_TIMEOUT_CALIBRATION_TIMEOUT = 300.0  # 5 minutes for first call
DYNAMIC_TIMEOUT_MULTIPLIER = 10
DYNAMIC_TIMEOUT_MAX = 300.0  # 5 minutes maximum
DYNAMIC_TIMEOUT_MIN = 1.0  # 1 second minimum to prevent edge cases


def compute_dynamic_timeout(runtime: float) -> float:
    """Compute dynamic timeout based on measured runtime.

    The timeout is set to 10x the measured runtime, clamped between
    DYNAMIC_TIMEOUT_MIN and DYNAMIC_TIMEOUT_MAX.
    """
    return max(
        DYNAMIC_TIMEOUT_MIN,
        min(runtime * DYNAMIC_TIMEOUT_MULTIPLIER, DYNAMIC_TIMEOUT_MAX),
    )


@define(slots=False)
class ShrinkRayState[TestCase](ABC):
    input_type: Any  # InputType from __main__
    in_place: bool
    test: list[str]
    filename: str
    timeout: float | None
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
    can_format: bool = True
    formatter_command: list[str] | None = None

    first_call_time: float | None = None

    # Lazy imports to break circular dependencies:
    # - shrinkray.process imports from shrinkray.work which imports from here
    # - shrinkray.cli imports from here for state configuration
    # These are cached after first import for performance.
    _interrupt_wait_and_kill: Any = None
    _InputType: Any = None  # InputType enum from shrinkray.cli

    # Stores the output from the last debug run
    _last_debug_output: str = ""

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
        # Lazy import to avoid circular dependency
        if self._interrupt_wait_and_kill is None:
            from shrinkray.process import interrupt_wait_and_kill

            self._interrupt_wait_and_kill = interrupt_wait_and_kill
        if self._InputType is None:
            from shrinkray.cli import InputType

            self._InputType = InputType

        if not os.path.exists(working):
            raise ValueError(f"No such file {working}")
        if self.input_type.enabled(self._InputType.arg):
            command = self.test + [working]
        else:
            command = self.test

        kwargs: dict[str, Any] = {
            "universal_newlines": False,
            "preexec_fn": os.setsid,
            "cwd": cwd,
            "check": False,
        }
        if self.input_type.enabled(self._InputType.stdin) and not os.path.isdir(
            working
        ):
            with open(working, "rb") as i:
                kwargs["stdin"] = i.read()
        else:
            kwargs["stdin"] = b""

        # For debug mode, use simpler approach to capture output
        if debug:
            kwargs["capture_stdout"] = True
            kwargs["capture_stderr"] = True
            start_time = time.time()
            completed = await trio.run_process(command, **kwargs)
            runtime = time.time() - start_time

            # Check for timeout violation (only when timeout is explicitly set)
            if self.timeout is not None and runtime >= self.timeout and self.first_call:
                self.initial_exit_code = completed.returncode
                self.first_call = False
                raise TimeoutExceededOnInitial(
                    timeout=self.timeout,
                    runtime=runtime,
                )

            if self.first_call:
                self.initial_exit_code = completed.returncode
                # Set dynamic timeout if not explicitly specified
                if self.timeout is None:
                    self.timeout = compute_dynamic_timeout(runtime)
            self.first_call = False

            # Store captured output
            output_parts = []
            if completed.stdout:
                output_parts.append(completed.stdout.decode("utf-8", errors="replace"))
            if completed.stderr:
                output_parts.append(completed.stderr.decode("utf-8", errors="replace"))
            self._last_debug_output = "\n".join(output_parts).strip()

            return completed.returncode

        # Check if we should stream output to stderr (volume=debug)
        if self.volume == Volume.debug:
            # Inherit stderr from parent process to stream output in real-time
            kwargs["stderr"] = None  # None means inherit
            kwargs["stdout"] = subprocess.DEVNULL
        else:
            # Non-debug mode: discard all output
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL

        async with trio.open_nursery() as nursery:

            def call_with_kwargs(task_status=trio.TASK_STATUS_IGNORED):  # type: ignore
                return trio.run_process(command, **kwargs, task_status=task_status)

            start_time = time.time()
            sp = await nursery.start(call_with_kwargs)

            try:
                # Determine effective timeout for this call
                if self.first_call:
                    # For first call: use calibration timeout if dynamic, otherwise 10x explicit timeout
                    if self.timeout is None:
                        effective_timeout = DYNAMIC_TIMEOUT_CALIBRATION_TIMEOUT
                    else:
                        effective_timeout = self.timeout * 10
                else:
                    # For subsequent calls, timeout must be set (either explicit or computed)
                    assert self.timeout is not None
                    effective_timeout = self.timeout

                with trio.move_on_after(effective_timeout):
                    await sp.wait()

                runtime = time.time() - start_time

                if sp.returncode is None:
                    # Process didn't terminate before timeout - kill it
                    await self._interrupt_wait_and_kill(sp)

                # Check for timeout violation (only when timeout is explicitly set)
                if (
                    self.timeout is not None
                    and runtime >= self.timeout
                    and self.first_call
                ):
                    raise TimeoutExceededOnInitial(
                        timeout=self.timeout,
                        runtime=runtime,
                    )
            finally:
                if self.first_call:
                    self.initial_exit_code = sp.returncode
                    # Set dynamic timeout if not explicitly specified
                    if self.timeout is None:
                        runtime = time.time() - start_time
                        self.timeout = compute_dynamic_timeout(runtime)
                self.first_call = False

            result: int | None = sp.returncode
            assert result is not None

            return result

    async def run_for_exit_code(self, test_case: TestCase, debug: bool = False) -> int:
        # Lazy import
        if self._InputType is None:
            from shrinkray.cli import InputType

            self._InputType = InputType

        if self.in_place:
            if self.input_type == self._InputType.basename:
                working = self.filename
                await self.write_test_case_to_file(working, test_case)

                return await self.run_script_on_file(
                    working=working,
                    debug=debug,
                    cwd=os.getcwd(),
                )
            else:
                base, ext = os.path.splitext(self.filename)
                working = base + "-" + os.urandom(16).hex() + ext
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
            sort_key=sort_key_for_initial(self.initial),
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
            return await self.run_for_exit_code(test_case) == 0

    @property
    def parallel_tasks_running(self) -> int:
        """Number of parallel tasks currently running."""
        return self.is_interesting_limiter.borrowed_tokens

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

    async def build_error_message(self, e: Exception) -> str:
        """Build a detailed error message for an invalid initial example.

        This is used by the subprocess worker to provide helpful error messages
        without printing directly to stderr or calling sys.exit.
        """
        lines = [
            "Shrink ray cannot proceed because the initial call of the "
            "interestingness test resulted in an uninteresting test case."
        ]

        if isinstance(e, TimeoutExceededOnInitial):
            lines.append(
                f"This is because your initial test case took {e.runtime:.2f}s "
                f"exceeding your timeout setting of {self.timeout}."
            )
            lines.append(f"Try rerunning with --timeout={math.ceil(e.runtime * 2)}.")
        else:
            lines.append("Rerunning the interestingness test for debugging purposes...")
            exit_code = await self.run_for_exit_code(self.initial, debug=True)
            if exit_code != 0:
                lines.append(
                    f"This exited with code {exit_code}, but the script should "
                    "return 0 for interesting test cases."
                )
                # Include the captured output from the debug run
                if self._last_debug_output:
                    lines.append("\nOutput from the interestingness test:")
                    lines.append(self._last_debug_output)
                local_exit_code = await self.run_script_on_file(
                    working=self.filename,
                    debug=False,
                    cwd=os.getcwd(),
                )
                if local_exit_code == 0:
                    lines.append(
                        "\nNote that Shrink Ray runs your script on a copy of the file "
                        "in a temporary directory. Here are the results of running it "
                        "in the current directory..."
                    )
                    other_exit_code = await self.run_script_on_file(
                        working=self.filename,
                        debug=True,
                        cwd=os.getcwd(),
                    )
                    # Include the output from running in current directory
                    if self._last_debug_output:
                        lines.append(self._last_debug_output)
                    if other_exit_code != local_exit_code:
                        lines.append(
                            f"This interestingness test is probably flaky as the first "
                            f"time we reran it locally it exited with {local_exit_code}, "
                            f"but the second time it exited with {other_exit_code}. "
                            "Please make sure your interestingness test is deterministic."
                        )
                    else:
                        lines.append(
                            "This suggests that your script depends on being run from "
                            "the current working directory. Please fix it to be "
                            "directory independent."
                        )
            else:
                assert self.initial_exit_code not in (None, 0)
                lines.append(
                    f"This exited with code 0, but previously the script exited with "
                    f"{self.initial_exit_code}. This suggests your interestingness "
                    "test exhibits nondeterministic behaviour."
                )

        return "\n".join(lines)

    async def report_error(self, e):
        error_message = await self.build_error_message(e)
        print(error_message, file=sys.stderr)
        sys.exit(1)

    def check_trivial_result(self, problem) -> str | None:
        """Check if the result is trivially small and return error message if so.

        Returns None if the result is acceptable, or an error message string
        if the result is trivial and trivial_is_error is True.
        """
        if len(problem.current_test_case) <= 1 and self.trivial_is_error:
            return (
                f"Reduced to a trivial test case of size {len(problem.current_test_case)}\n"
                "This probably wasn't what you intended. If so, please modify your "
                "interestingness test to be more restrictive.\n"
                "If you intended this behaviour, you can run with '--trivial-is-not-error' "
                "to suppress this message."
            )
        return None


@define(slots=False)
class ShrinkRayStateSingleFile(ShrinkRayState[bytes]):
    def new_reducer(self, problem: ReductionProblem[bytes]) -> Reducer[bytes]:
        return ShrinkRay(problem, clang_delta=self.clang_delta_executable)

    def setup_formatter(self):
        from shrinkray.formatting import (
            default_reformat_data,
            determine_formatter_command,
        )

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
            return await self.run_for_exit_code(test_case) == 0

    async def print_exit_message(self, problem):
        formatting_increase = 0
        final_result = problem.current_test_case
        reformatted = await self.attempt_format(final_result)
        if reformatted != final_result:
            # attempt_format only returns a different value if is_interesting was True
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


class ShrinkRayDirectoryState(ShrinkRayState[dict[str, bytes]]):
    def setup_formatter(self): ...

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
        # Formatting not supported for directory reduction
        return None

    async def run_formatter_command(
        self, command: str | list[str], input: dict[str, bytes]
    ) -> subprocess.CompletedProcess:
        # Formatting not supported for directory reduction
        raise NotImplementedError("Directory formatting not supported")

    async def print_exit_message(self, problem):
        print("All done!")
