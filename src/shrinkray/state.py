"""State management for shrink ray reduction sessions."""

import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import timedelta
from typing import Any

import attrs
import humanize
import trio
from attrs import define

from shrinkray.adaptive_timeout import AdaptiveTimeoutPolicy
from shrinkray.cli import InputType
from shrinkray.formatting import default_reformat_data, determine_formatter_command
from shrinkray.history import (
    HistoryManager,
    deserialize_directory,
    serialize_directory,
)
from shrinkray.passes.cpp import C_FILE_EXTENSIONS
from shrinkray.problem import (
    BasicReductionProblem,
    InterestingnessResult,
    InvalidInitialExample,
    ReductionProblem,
    sort_key_for_initial,
)
from shrinkray.process import (
    child_preexec,
    default_memory_limit,
    interrupt_wait_and_kill,
    kill_process_group,
    peak_child_rss_bytes,
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


class MemoryLimitExceededOnInitial(InvalidInitialExample):
    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(
            f"Initial test call used {humanize.naturalsize(used)} of memory, "
            f"exceeding the limit of {humanize.naturalsize(limit)}. Try raising "
            f"or disabling --memory-limit."
        )


# The first call to the interestingness test is given generous headroom so
# that we can measure its true runtime (and report clearly if it exceeds an
# explicitly configured timeout) rather than killing it prematurely. When no
# timeout was configured this fixed calibration timeout is used; afterwards
# the AdaptiveTimeoutPolicy chooses timeouts from measured runtimes.
DYNAMIC_TIMEOUT_CALIBRATION_TIMEOUT = 300.0  # 5 minutes for first call


@define(frozen=True)
class ScriptRunResult:
    """The result of a single run of the interestingness test script."""

    exit_code: int
    # Whether the run was killed because it hit the timeout, and if so
    # which timeout (in seconds) it was subject to.
    timed_out: bool = False
    timeout_used: float | None = None


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

    # Address-space cap (bytes) for each interestingness-test subprocess,
    # or None to disable. Prevents a runaway test from exhausting host
    # memory. Enforced via RLIMIT_AS where the platform supports it.
    memory_limit: int | None = None

    first_call: bool = True
    initial_exit_code: int | None = None
    can_format: bool = True
    formatter_command: list[str] | None = None

    # Chooses the timeout for each interestingness test run, adapting to
    # measured runtimes. `timeout` (the user-specified value) acts as its
    # maximum. Injectable for testing.
    timeout_policy: AdaptiveTimeoutPolicy = attrs.field(
        default=attrs.Factory(
            lambda self: AdaptiveTimeoutPolicy(user_timeout=self.timeout),
            takes_self=True,
        )
    )

    # Stores the output from the last debug run
    _last_debug_output: str = ""

    # Stores the output from the most recently completed test (for history recording)
    # This is read immediately after the test's output file is closed to avoid
    # race conditions with other parallel tests
    _last_test_output: bytes | None = None

    # Optional output manager for capturing test output (TUI mode or history)
    output_manager: OutputCaptureManager | None = None

    # History recording (enabled by default)
    history_enabled: bool = True
    history_base_dir: str | None = None  # Base directory for .shrinkray folder
    history_manager: HistoryManager | None = None

    # Also-interesting exit code (None = disabled)
    # When a test returns this code, it's recorded but not used for reduction
    also_interesting_code: int | None = None

    # Set of test cases to exclude from interestingness (for restart-from-point)
    # These are byte-identical matches of previously reduced values
    excluded_test_cases: set[bytes] | None = None

    # Temp directory for output capture (when not using TUI's output manager)
    _output_tempdir: tempfile.TemporaryDirectory | None = None

    # Stores output from successful tests, keyed by test case bytes
    # This avoids race conditions when multiple tests run in parallel
    _successful_outputs: dict[bytes, bytes] = {}

    def __attrs_post_init__(self):
        self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))
        self._successful_outputs = {}  # Initialize mutable default
        self.sweep_stale_working_files()
        self.setup_formatter()
        self._setup_history()

    @abstractmethod
    def setup_formatter(self): ...

    def stale_working_file_pattern(self) -> "tuple[str, re.Pattern[str]] | None":
        """Directory and filename regex for the temporary candidate
        files this state writes next to the target during in-place
        reduction, or None when no such files are created.

        During in-place reduction (except in ``basename`` mode, which
        writes the target itself) each test call writes a candidate to a
        sibling file named ``<stem>-<32 hex digits><ext>`` so the test
        can still see the target's neighbours. These are removed as soon
        as the test finishes, but a hard kill (SIGKILL) can leave some
        behind; this pattern lets us recognise and sweep those."""
        if not self.in_place or self.input_type == InputType.basename:
            return None
        abspath = os.path.abspath(self.filename)
        stem, ext = os.path.splitext(os.path.basename(abspath))
        pattern = re.compile(
            re.escape(stem) + r"-[0-9a-f]{32}" + re.escape(ext) + r"\Z"
        )
        return os.path.dirname(abspath), pattern

    def sweep_stale_working_files(self) -> None:
        """Remove any leftover temporary candidate files from a previous
        run that was killed before it could clean up after itself. Only
        files matching this run's own ``<stem>-<hex><ext>`` pattern are
        removed, so unrelated files are never touched."""
        info = self.stale_working_file_pattern()
        if info is None:
            return
        directory, pattern = info
        try:
            names = os.listdir(directory)
        except OSError:
            return
        for name in names:
            if pattern.match(name):
                path = os.path.join(directory, name)
                try:
                    os.unlink(path)
                except OSError:
                    pass

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
            base_dir=self.history_base_dir,
            input_type=self.input_type,
        )

        # Ensure we have an output manager for capturing test output
        if self.output_manager is None:
            self._output_tempdir = tempfile.TemporaryDirectory()
            self.output_manager = OutputCaptureManager(
                output_dir=self._output_tempdir.name
            )

    def _get_last_captured_output(self) -> bytes | None:
        """Get the output from the most recently completed test.

        Returns the output content if available, None otherwise.
        This returns the output that was captured immediately when the test
        completed, avoiding race conditions with other parallel tests.
        """
        return self._last_test_output

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

    def effective_memory_limit(self, first_call: bool) -> int | None:
        """The address-space cap to impose on a test subprocess, in bytes.

        Returns None (no cap) when memory limiting is disabled. The first
        call is given generous headroom (up to physical RAM) rather than
        the configured limit, so it can run to completion and we can
        measure its true peak usage and report clearly if it exceeds the
        limit — mirroring the timeout's generous first-call calibration.
        """
        if self.memory_limit is None or self.memory_limit <= 0:
            return None
        if first_call:
            return max(self.memory_limit, default_memory_limit())
        return self.memory_limit

    def raise_if_initial_over_memory(self) -> None:
        """Raise if the initial test call used more memory than the limit.

        Mirrors the initial-timeout check: the first call is allowed to
        run, its peak child RSS is measured, and if that meets or exceeds
        the configured limit we fail loudly with a helpful message rather
        than letting later calls be silently killed by the limit. This
        works even on platforms where the limit itself is not enforceable
        (the measurement does not depend on enforcement).
        """
        if self.memory_limit is None or self.memory_limit <= 0:
            return
        peak = peak_child_rss_bytes()
        if peak >= self.memory_limit:
            raise MemoryLimitExceededOnInitial(used=peak, limit=self.memory_limit)

    async def run_script_on_file(
        self, working: str, cwd: str, debug: bool = False
    ) -> ScriptRunResult:
        if not os.path.exists(working):
            raise ValueError(f"No such file {working}")
        if self.input_type.enabled(InputType.arg):
            command = self.test + [working]
        else:
            command = self.test

        kwargs: dict[str, Any] = {
            "universal_newlines": False,
            "preexec_fn": child_preexec(self.effective_memory_limit(self.first_call)),
            "cwd": cwd,
            "check": False,
        }
        if self.input_type.enabled(InputType.stdin) and not os.path.isdir(working):
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
                self.first_call = False
                self.raise_if_initial_over_memory()
            else:
                self.first_call = False

            # Store captured output
            output_parts = []
            if completed.stdout:
                output_parts.append(completed.stdout.decode("utf-8", errors="replace"))
            if completed.stderr:
                output_parts.append(completed.stderr.decode("utf-8", errors="replace"))
            self._last_debug_output = "\n".join(output_parts).strip()

            return ScriptRunResult(exit_code=completed.returncode)

        # Determine output handling
        test_id: int | None = None
        output_file_handle = None
        output_path: str | None = None
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

        sp = None
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
                        effective_timeout = self.timeout_policy.current_timeout()

                    with trio.move_on_after(effective_timeout):
                        await sp.wait()

                    runtime = time.time() - start_time
                    timed_out = sp.returncode is None

                    if timed_out:
                        # Process didn't terminate before timeout - kill it
                        await interrupt_wait_and_kill(sp)
                        self.timeout_policy.record_timeout(effective_timeout)
                    else:
                        self.timeout_policy.record_completion(
                            runtime, interesting=sp.returncode == 0
                        )

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

                    if self.first_call:
                        self.raise_if_initial_over_memory()
                finally:
                    if self.first_call:
                        self.initial_exit_code = sp.returncode
                    self.first_call = False

                result: int | None = sp.returncode
                assert result is not None
                exit_code = result

                return ScriptRunResult(
                    exit_code=result,
                    timed_out=timed_out,
                    timeout_used=effective_timeout if timed_out else None,
                )
        finally:
            # Kill entire process group to clean up child processes.
            # The subprocess uses setsid (preexec_fn=os.setsid), so child
            # processes spawned by the interestingness test form a process
            # group. Trio only kills the direct child on cancellation, but
            # shell scripts often fork children that continue running and
            # may write to the temp directory, causing cleanup failures.
            # Always attempt this even if the direct child has exited,
            # because group children may still be alive.
            if sp is not None:
                kill_process_group(sp)
            # Clean up output file handle and capture output immediately
            if output_file_handle is not None:
                output_file_handle.close()
                # Read the output file NOW, before any other test can interfere
                # This avoids race conditions where get_current_output() returns
                # a different test's partial output
                # output_path must be set since it's assigned with output_file_handle
                assert output_path is not None
                try:
                    with open(output_path, "rb") as f:
                        self._last_test_output = f.read()
                except OSError:
                    self._last_test_output = None
            if test_id is not None and self.output_manager is not None:
                if exit_code is not None:
                    recorded_code = exit_code
                else:
                    # The test never produced an exit code (it was cancelled
                    # or timed out) and its process group was just killed.
                    returncode = sp.returncode if sp is not None else None
                    recorded_code = (
                        returncode if returncode is not None else -int(signal.SIGKILL)
                    )
                self.output_manager.mark_completed(test_id, recorded_code)

    async def run_for_result(
        self, test_case: TestCase, debug: bool = False
    ) -> ScriptRunResult:
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
                # Absolute so that cleanup in the finally below can't be
                # defeated by the working directory changing between here
                # and there: a relative path would make os.path.exists
                # resolve against a different directory and silently leak
                # the file into the user's tree.
                base, ext = os.path.splitext(os.path.abspath(self.filename))
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
                            shutil.rmtree(working, ignore_errors=True)
                        else:
                            try:
                                os.unlink(working)
                            except OSError:
                                pass
        else:
            d = tempfile.mkdtemp()
            try:
                working = os.path.join(d, self.base)
                await self.write_test_case_to_file(working, test_case)

                return await self.run_script_on_file(
                    working=working,
                    debug=debug,
                    cwd=d,
                )
            finally:
                shutil.rmtree(d, ignore_errors=True)

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
            is_interesting=self.check_interesting,
            initial=self.initial,
            work=work,
            sort_key=sort_key_for_initial(self.initial),
            unstick=self._attempt_unstick,
            **self.extra_problem_kwargs,
        )

        # Initialize history and register callback if enabled.
        # The history callback must be registered FIRST so it runs before any
        # other on_reduce callback has a chance to yield to the scheduler.
        # record_reduction is fully synchronous: by running it before the
        # file-write callback (which awaits on a lock and file I/O), we ensure
        # the history directory is always consistent with stats.reductions at
        # every scheduling point. Otherwise emit_progress_updates can wake up
        # between the two callbacks and report a reduction that the history
        # manager hasn't written yet, which breaks restart_from_reduction.
        if self.history_manager is not None:
            self._initialize_history_manager()

            @problem.on_reduce
            async def record_history(test_case: TestCase):
                test_case_bytes = self._get_test_case_bytes(test_case)
                # Use output captured at is_interesting time to avoid race conditions
                output = self._successful_outputs.pop(test_case_bytes, None)
                assert self.history_manager is not None
                self.history_manager.record_reduction(test_case_bytes, output)

        # Writing the file back can't be guaranteed atomic, so we put a lock around
        # writing successful reductions back to the original file so we don't
        # write some confused combination of reductions.
        write_lock = trio.Lock()

        @problem.on_reduce
        async def _(test_case: TestCase):
            async with write_lock:
                await self.write_test_case_to_file(self.filename, test_case)

        # Progress resets the adaptive timeout's stall detection and any
        # in-progress upward exploration of the timeout.
        @problem.on_reduce
        async def _(test_case: TestCase):
            self.timeout_policy.note_reduction()

        self._cached_reducer = self.new_reducer(problem)
        return self._cached_reducer

    async def _attempt_unstick(self) -> bool:
        """Unstick hook for the reduction problem: raise the adaptive
        timeout if timeouts may have been hiding possible reductions."""
        await trio.lowlevel.checkpoint()
        return self.timeout_policy.attempt_unstick()

    @property
    def extra_problem_kwargs(self):
        return {}

    @property
    def problem(self):
        return self.reducer.target

    async def check_interesting(self, test_case: TestCase) -> InterestingnessResult:
        """Run the interestingness test on test_case.

        The result carries caching metadata: runs that timed out are only
        valid while the adaptive timeout is no larger than the timeout they
        ran under, so that they are retried if the timeout is later raised.
        """
        # Check exclusion set first (for restart-from-point feature)
        if self.excluded_test_cases is not None:
            test_case_bytes = self._get_test_case_bytes(test_case)
            if test_case_bytes in self.excluded_test_cases:
                return InterestingnessResult(interesting=False)

        async with self.is_interesting_limiter:
            result = await self.run_for_result(test_case)
            self._check_also_interesting(result.exit_code, test_case)
            if result.exit_code == 0:
                # Capture output now while still in the limiter to avoid race conditions
                # where another test starts and overwrites the "current" output
                test_case_bytes = self._get_test_case_bytes(test_case)
                output = self._get_last_captured_output()
                if output is not None:
                    self._successful_outputs[test_case_bytes] = output
                return InterestingnessResult(interesting=True)
            if result.timed_out and result.timeout_used is not None:
                timeout_used = result.timeout_used
                return InterestingnessResult(
                    interesting=False,
                    cache_valid=lambda: self.timeout_policy.cached_timeout_valid(
                        timeout_used
                    ),
                )
            return InterestingnessResult(interesting=False)

    async def is_interesting(self, test_case: TestCase) -> bool:
        return (await self.check_interesting(test_case)).interesting

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
        # Clear stored successful outputs (no longer relevant after restart)
        self._successful_outputs.clear()
        # Forget learned timeout state: the restart point may be much
        # slower than what the timeout had adapted down to.
        self.timeout_policy.reset()
        # Reset initial_exit_code - the new initial is known to be interesting
        # (it came from history) so its exit code was 0
        self.initial_exit_code = 0
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
        elif isinstance(e, MemoryLimitExceededOnInitial):
            lines.append(
                f"This is because your initial test case used "
                f"{humanize.naturalsize(e.used)} of memory, exceeding your "
                f"--memory-limit setting of {humanize.naturalsize(e.limit)}."
            )
            lines.append(
                "Try rerunning with a higher --memory-limit, or --memory-limit=0 "
                "to disable the limit."
            )
        else:
            lines.append("Rerunning the interestingness test for debugging purposes...")
            exit_code = (await self.run_for_result(self.initial, debug=True)).exit_code
            if exit_code != 0:
                lines.append(
                    f"This exited with code {exit_code}, but the script should "
                    "return 0 for interesting test cases."
                )
                # Include the captured output from the debug run
                if self._last_debug_output:
                    lines.append("\nOutput from the interestingness test:")
                    lines.append(self._last_debug_output)
                local_exit_code = (
                    await self.run_script_on_file(
                        working=self.filename,
                        debug=False,
                        cwd=os.getcwd(),
                    )
                ).exit_code
                if local_exit_code == 0:
                    lines.append(
                        "\nNote that Shrink Ray runs your script on a copy of the file "
                        "in a temporary directory. Here are the results of running it "
                        "in the current directory..."
                    )
                    other_exit_code = (
                        await self.run_script_on_file(
                            working=self.filename,
                            debug=True,
                            cwd=os.getcwd(),
                        )
                    ).exit_code
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
        return ShrinkRay(
            problem,
            enable_cpp_passes=os.path.splitext(self.filename)[1] in C_FILE_EXTENSIONS,
        )

    def _get_initial_bytes(self) -> bytes:
        return self.initial

    def _get_test_case_bytes(self, test_case: bytes) -> bytes:
        return test_case

    def _set_initial_for_restart(self, content: bytes) -> None:
        self.initial = content

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
        return DirectoryShrinkRay(target=problem)

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
        return serialize_directory(content)

    @staticmethod
    def _deserialize_directory(data: bytes) -> dict[str, bytes]:
        return deserialize_directory(data)

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
