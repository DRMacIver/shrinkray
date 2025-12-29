"""State management for shrink ray reduction sessions."""

import math
import os
import random
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from tempfile import TemporaryDirectory
from typing import Any

import humanize
import trio
from attrs import define

from shrinkray.history import HistoryManager
from shrinkray.passes.clangdelta import ClangDelta
from shrinkray.problem import (
    BasicReductionProblem,
    InvalidInitialExample,
    ReductionProblem,
    sort_key_for_initial,
)
from shrinkray.reducer import DirectoryShrinkRay, Reducer, ShrinkRay
from shrinkray.work import Volume, WorkContext


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


@define
class OutputCaptureManager:
    """Manages temporary files for test output capture.

    Allocates unique files for each test's stdout/stderr output,
    tracks active and completed tests, and cleans up old files.
    """

    output_dir: str
    max_files: int = 50
    max_age_seconds: float = 60.0
    min_display_seconds: float = 1.0  # Minimum time to show completed output
    grace_period_seconds: float = (
        0.5  # Extra time to wait for new test after min_display
    )

    _sequence: int = 0
    _active_outputs: dict[int, str] = {}
    # Completed outputs: (test_id, file_path, completion_time, return_code)
    _completed_outputs: deque[tuple[int, str, float, int]] = deque()

    def __attrs_post_init__(self) -> None:
        # Initialize mutable defaults
        self._active_outputs = {}
        self._completed_outputs = deque()

    def allocate_output_file(self) -> tuple[int, str]:
        """Allocate a new output file for a test. Returns (test_id, file_path)."""
        test_id = self._sequence
        self._sequence += 1
        file_path = os.path.join(self.output_dir, f"test_{test_id}.log")
        self._active_outputs[test_id] = file_path
        return test_id, file_path

    def mark_completed(self, test_id: int, return_code: int = 0) -> None:
        """Mark a test as completed and move to completed queue."""
        if test_id in self._active_outputs:
            file_path = self._active_outputs.pop(test_id)
            self._completed_outputs.append(
                (test_id, file_path, time.time(), return_code)
            )
            self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove old output files based on count and age limits."""
        now = time.time()
        # Remove files older than max_age_seconds
        while (
            self._completed_outputs
            and now - self._completed_outputs[0][2] > self.max_age_seconds
        ):
            _, file_path, _, _ = self._completed_outputs.popleft()
            self._safe_delete(file_path)
        # Remove excess files beyond max_files
        while len(self._completed_outputs) > self.max_files:
            _, file_path, _, _ = self._completed_outputs.popleft()
            self._safe_delete(file_path)

    @staticmethod
    def _file_has_content(path: str) -> bool:
        """Check if a file exists and has non-zero size."""
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    def get_current_output(self) -> tuple[str | None, int | None, int | None]:
        """Get the current output to display.

        Returns (file_path, test_id, return_code) where:
        - file_path: path to the output file to display
        - test_id: the test ID (for display in header)
        - return_code: the return code (None if test is still running)

        Active tests take priority only if they have produced output.
        Otherwise, shows recently completed test output for min_display_seconds,
        plus an additional grace_period if no new test has started.
        """
        # Active tests take priority only if they have content
        if self._active_outputs:
            max_id = max(self._active_outputs.keys())
            active_path = self._active_outputs[max_id]
            if self._file_has_content(active_path):
                # Active test with output - no return code yet
                return active_path, max_id, None
            # Active test has no output yet - fall through to show previous output

        # Check for recently completed test that should stay visible,
        # or fall back to most recent completed (even if past display window)
        if self._completed_outputs:
            test_id, file_path, _, return_code = self._completed_outputs[-1]
            return file_path, test_id, return_code

        return None, None, None

    def cleanup_all(self) -> None:
        """Clean up all output files (called on shutdown)."""
        for file_path in self._active_outputs.values():
            self._safe_delete(file_path)
        for _, file_path, _, _ in self._completed_outputs:
            self._safe_delete(file_path)
        self._active_outputs.clear()
        self._completed_outputs.clear()

    @staticmethod
    def _safe_delete(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass


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

    # Optional output manager for capturing test output (TUI mode or history)
    output_manager: OutputCaptureManager | None = None

    # History recording (enabled by default)
    history_enabled: bool = True
    history_manager: HistoryManager | None = None

    # Also-interesting exit code (None = disabled)
    # When a test returns this code, it's recorded but not used for reduction
    also_interesting_code: int | None = None

    # Set of test cases to exclude from interestingness (for restart-from-point)
    # These are byte-identical matches of previously reduced values
    excluded_test_cases: set[bytes] | None = None

    # Temp directory for output capture (when not using TUI's output manager)
    _output_tempdir: TemporaryDirectory | None = None

    def __attrs_post_init__(self):
        self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))
        self.setup_formatter()
        self._setup_history()

    @abstractmethod
    def setup_formatter(self): ...

    @property
    def is_directory_mode(self) -> bool:
        """Whether this state manages directory test cases."""
        return False

    def _setup_history(self) -> None:
        """Set up history recording if enabled or also-interesting is configured."""
        # Create history manager if either:
        # 1. Full history is enabled, or
        # 2. also_interesting_code is set (records only also-interesting cases)
        if not self.history_enabled and self.also_interesting_code is None:
            return

        # Create history manager (record_reductions=False if only also-interesting)
        self.history_manager = HistoryManager.create(
            self.test,
            self.filename,
            record_reductions=self.history_enabled,
            is_directory=self.is_directory_mode,
        )

        # Ensure we have an output manager for capturing test output
        if self.output_manager is None:
            self._output_tempdir = TemporaryDirectory()
            self.output_manager = OutputCaptureManager(
                output_dir=self._output_tempdir.name
            )

    def _get_last_captured_output(self) -> bytes | None:
        """Get the output from the most recently completed test.

        Returns the output content if available, None otherwise.
        """
        if self.output_manager is None:
            return None
        output_path, _, _ = self.output_manager.get_current_output()
        if output_path is None:
            return None
        try:
            with open(output_path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def _check_also_interesting(self, exit_code: int, test_case: TestCase) -> None:
        """Check if exit code matches also-interesting and record if so.

        Args:
            exit_code: The exit code from the test
            test_case: The test case that was tested
        """
        if (
            self.also_interesting_code is not None
            and exit_code == self.also_interesting_code
            and self.history_manager is not None
        ):
            test_case_bytes = self._get_test_case_bytes(test_case)
            output = self._get_last_captured_output()
            self.history_manager.record_also_interesting(test_case_bytes, output)

    @abstractmethod
    def new_reducer(self, problem: ReductionProblem[TestCase]) -> Reducer[TestCase]: ...

    @abstractmethod
    def _get_initial_bytes(self) -> bytes:
        """Get the initial test case as bytes for history recording."""
        ...

    @abstractmethod
    def _get_test_case_bytes(self, test_case: TestCase) -> bytes:
        """Convert a test case to bytes for history recording."""
        ...

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

        # Determine output handling
        test_id: int | None = None
        output_file_handle = None
        exit_code: int | None = None  # Track for output manager

        if self.output_manager is not None:
            # Capture output to a file for TUI display
            test_id, output_path = self.output_manager.allocate_output_file()
            output_file_handle = open(output_path, "wb")
            kwargs["stdout"] = output_file_handle.fileno()
            kwargs["stderr"] = subprocess.STDOUT  # Combine stderr into stdout
        elif self.volume == Volume.debug:
            # Inherit stderr from parent process to stream output in real-time
            kwargs["stderr"] = None  # None means inherit
            kwargs["stdout"] = subprocess.DEVNULL
        else:
            # Non-debug mode: discard all output
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL

        try:
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
                exit_code = result

                return result
        finally:
            # Clean up output file handle and mark test as completed
            if output_file_handle is not None:
                output_file_handle.close()
            if test_id is not None and self.output_manager is not None:
                self.output_manager.mark_completed(test_id, exit_code or 0)

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
            return self._cached_reducer
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

        # Initialize history and register callback if enabled
        if self.history_manager is not None:
            self._initialize_history_manager()

            @problem.on_reduce
            async def record_history(test_case: TestCase):
                output = self._get_last_captured_output()
                test_case_bytes = self._get_test_case_bytes(test_case)
                assert self.history_manager is not None
                self.history_manager.record_reduction(test_case_bytes, output)

        self._cached_reducer = self.new_reducer(problem)
        return self._cached_reducer

    @property
    def extra_problem_kwargs(self):
        return {}

    @property
    def problem(self):
        return self.reducer.target

    async def is_interesting(self, test_case: TestCase) -> bool:
        # Check exclusion set first (for restart-from-point feature)
        if self.excluded_test_cases is not None:
            test_case_bytes = self._get_test_case_bytes(test_case)
            if test_case_bytes in self.excluded_test_cases:
                return False

        if self.first_call_time is None:
            self.first_call_time = time.time()
        async with self.is_interesting_limiter:
            exit_code = await self.run_for_exit_code(test_case)
            self._check_also_interesting(exit_code, test_case)
            return exit_code == 0

    def reset_for_restart(self, new_initial: bytes, excluded: set[bytes]) -> None:
        """Reset state for restart from a history point.

        This clears the cached reducer so it will be recreated with the new
        initial value, and sets the exclusion set to reject previously
        reduced values.

        Args:
            new_initial: The new initial test case content
            excluded: Set of test cases to reject as uninteresting
        """
        self.excluded_test_cases = excluded
        # Clear cached reducer so it will be recreated on next access
        try:
            del self._cached_reducer
        except AttributeError:
            pass
        # Update initial (implementation depends on subclass)
        self._set_initial_for_restart(new_initial)

    @abstractmethod
    def _set_initial_for_restart(self, content: bytes) -> None:
        """Set the initial test case for restart. Subclasses implement."""
        ...

    def _initialize_history_manager(self) -> None:
        """Initialize the history manager. Subclasses can override for different modes."""
        assert self.history_manager is not None
        self.history_manager.initialize(
            self._get_initial_bytes(),
            self.test,
            self.filename,
        )

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

    def _get_initial_bytes(self) -> bytes:
        return self.initial

    def _get_test_case_bytes(self, test_case: bytes) -> bytes:
        return test_case

    def _set_initial_for_restart(self, content: bytes) -> None:
        self.initial = content

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
        # Check exclusion set first (for restart-from-point feature)
        if self.excluded_test_cases is not None and test_case in self.excluded_test_cases:
            return False

        async with self.is_interesting_limiter:
            exit_code = await self.run_for_exit_code(test_case)
            self._check_also_interesting(exit_code, test_case)
            return exit_code == 0

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

    @property
    def is_directory_mode(self) -> bool:
        """Whether this state manages directory test cases."""
        return True

    @property
    def extra_problem_kwargs(self) -> dict[str, Any]:
        return {
            "size": lambda tc: sum(len(v) for v in tc.values()),
        }

    def new_reducer(
        self, problem: ReductionProblem[dict[str, bytes]]
    ) -> Reducer[dict[str, bytes]]:
        return DirectoryShrinkRay(
            target=problem, clang_delta=self.clang_delta_executable
        )

    def _get_initial_bytes(self) -> bytes:
        # Serialize directory content for history recording
        return self._serialize_directory(self.initial)

    def _get_test_case_bytes(self, test_case: dict[str, bytes]) -> bytes:
        # Serialize directory content for comparison/exclusion
        return self._serialize_directory(test_case)

    def _set_initial_for_restart(self, content: bytes) -> None:
        # Deserialize and update initial directory content
        self.initial = self._deserialize_directory(content)

    def _initialize_history_manager(self) -> None:
        """Initialize the history manager in directory mode."""
        assert self.history_manager is not None
        self.history_manager.initialize_directory(
            self.initial,
            self.test,
            self.filename,
        )

    @staticmethod
    def _serialize_directory(content: dict[str, bytes]) -> bytes:
        """Serialize directory content to bytes for comparison/storage."""
        import base64
        import json

        serialized = {k: base64.b64encode(v).decode() for k, v in sorted(content.items())}
        return json.dumps(serialized, sort_keys=True).encode()

    @staticmethod
    def _deserialize_directory(data: bytes) -> dict[str, bytes]:
        """Deserialize bytes back to directory content."""
        import base64
        import json

        serialized = json.loads(data.decode())
        return {k: base64.b64decode(v) for k, v in serialized.items()}

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
