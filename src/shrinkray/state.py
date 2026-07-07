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
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from typing import IO, Any

import attrs
import humanize
import trio
from attrs import define

from shrinkray.adaptive_timeout import AdaptiveTimeoutPolicy
from shrinkray.cli import InputType
from shrinkray.downloads import DownloadCoordinator, grammar_plan, missing_grammars
from shrinkray.formatting import default_reformat_data, determine_formatter_command
from shrinkray.history import (
    HistoryManager,
    deserialize_directory,
    serialize_directory,
)
from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.cpp import C_FILE_EXTENSIONS
from shrinkray.passes.llm import (
    DEFAULT_MODEL_SPEC,
    LLMConfig,
    parse_model_spec,
    read_oracle_script,
)
from shrinkray.problem import (
    BasicReductionProblem,
    InterestingnessResult,
    InvalidInitialExample,
    ReductionProblem,
    sort_key_for_initial,
)
from shrinkray.process import (
    default_memory_limit,
    interrupt_wait_and_kill,
    kill_process_group,
    memory_limited_command,
    peak_child_rss_bytes,
)
from shrinkray.reducer import DirectoryShrinkRay, Reducer, ShrinkRay
from shrinkray.work import Volume, WorkContext


@contextmanager
def stdin_source(input_type: InputType, working: str) -> Generator[IO[bytes] | int]:
    """The stdin to give an interestingness-test subprocess.

    When the stdin input type is enabled, this is the working file itself,
    opened for reading, rather than its contents fed through a pipe. Piped
    bytes deadlock on OpenBSD when the test case is bigger than the pipe
    buffer and the script exits without reading stdin: OpenBSD's kqueue
    never reports the pipe's write end as writable once the read end is
    closed, so trio's stdin-feeder task blocks forever
    (https://github.com/DRMacIver/shrinkray/issues/56). A real file
    descriptor involves no feeder task at all, and also avoids copying the
    whole test case through a pipe on every call.
    """
    if input_type.enabled(InputType.stdin) and not os.path.isdir(working):
        with open(working, "rb") as f:
            yield f
    else:
        yield subprocess.DEVNULL


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

    # Whether the user explicitly set --memory-limit. When False, memory_limit
    # holds the physical-RAM default, which may be auto-disabled if it blocks
    # the initial test: sanitizer builds (ASan/MSan/TSan) reserve tens of
    # terabytes of virtual address space and abort under any address-space
    # cap, so the default would otherwise fail them out of the box.
    memory_limit_explicit: bool = False

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

    # External reducers (argv lists) to run as reduction passes, and whether to
    # run the built-in Python reducer. Passed through to the reducer.
    external_reducers: list[list[str]] = attrs.Factory(list)
    python_reducer: bool = True

    # Whether the reducer re-reduces from the original input once it
    # reaches a fixpoint (see ShrinkRay.restart_at_fixpoint). On by
    # default; the evaluation harnesses disable it for speed.
    restart_at_fixpoint: bool = True

    # LLM passes: whether they run at all, the model they use (a local
    # .gguf path or a Hugging Face repo:filename), and whether they
    # replace every other pass.
    llm_enabled: bool = False
    llm_model: str = DEFAULT_MODEL_SPEC
    llm_only: bool = False

    # The lazily-built LLM client, shared by every reducer this state
    # creates so the model is only loaded once.
    _llm_client: LlamaCppClient | None = None

    # The lazily-built coordinator for the background downloads this
    # reduction may need. Built on first use; loading of already-cached
    # resources starts then, downloads wait for start_downloads.
    _downloads: DownloadCoordinator | None = None

    # Set of test cases to exclude from interestingness (for restart-from-point)
    # These are byte-identical matches of previously reduced values
    excluded_test_cases: set[bytes] | None = None

    # Temp directory for output capture (when not using TUI's output manager)
    _output_tempdir: tempfile.TemporaryDirectory | None = None

    # Stores output from successful tests, keyed by test case bytes
    # This avoids race conditions when multiple tests run in parallel
    _successful_outputs: dict[bytes, bytes] = {}

    # Sort keys of the test cases whose output is stored above, keyed by the
    # same test case bytes. record_reduction uses these to prune losing
    # candidates' outputs while retaining any candidate that can still be
    # adopted (see _record_reduction_history).
    _successful_output_keys: dict[bytes, Any] = {}

    def __attrs_post_init__(self):
        self.is_interesting_limiter = trio.CapacityLimiter(max(self.parallelism, 1))
        self._successful_outputs = {}  # Initialize mutable default
        self._successful_output_keys = {}  # Initialize mutable default
        self.sweep_stale_working_files()
        self.setup_formatter()
        self._setup_history()

    def _ensure_llm_client(self) -> LlamaCppClient:
        if self._llm_client is None:
            self._llm_client = LlamaCppClient(model=parse_model_spec(self.llm_model))
        return self._llm_client

    @property
    def downloads(self) -> DownloadCoordinator:
        """The coordinator for this reduction's background downloads."""
        if self._downloads is None:
            llm_client = None
            llm_needs_download = False
            llm_description = ""
            if self.llm_enabled:
                llm_client = self._ensure_llm_client()
                llm_needs_download = llm_client.model_needs_download()
                llm_description = f"LLM model {self.llm_model}"
                if self.llm_model == DEFAULT_MODEL_SPEC:
                    llm_description += " (about 2.7GB)"
            self._downloads = DownloadCoordinator(
                llm_client=llm_client,
                llm_needs_download=llm_needs_download,
                llm_description=llm_description,
                grammars=missing_grammars(self._input_filenames()),
            )
            self._downloads.start_immediate()
        return self._downloads

    def pending_downloads(self) -> list[dict[str, str]]:
        """What would be downloaded, for the UI to offer opting out of."""
        return [
            {"id": item_id, "description": description}
            for item_id, description in self.downloads.pending()
        ]

    def start_downloads(self, disabled: list[str]) -> None:
        """Record the user's decision and begin the approved downloads."""
        self.downloads.start(disabled)

    @abstractmethod
    def _input_filenames(self) -> list[str]:
        """The file names being reduced, for grammar detection."""

    def llm_reducer_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs wiring the LLM configuration into a reducer."""
        if not self.llm_enabled:
            return {}
        self._ensure_llm_client()
        return {
            "llm_client": self._llm_client,
            "llm_config": LLMConfig(
                filename=self.base,
                oracle=read_oracle_script(self.test[0]),
                # The output the interestingness test produced for a given
                # test case, captured when it ran. In directory mode the
                # per-file test cases never match these whole-directory
                # keys, so the prompt section is simply omitted there.
                test_output=lambda tc: self._successful_outputs.get(tc),
            ),
            "llm_only": self.llm_only,
        }

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
        entries matching this run's own ``<stem>-<hex><ext>`` pattern are
        removed, so unrelated files are never touched.

        In in-place directory mode each candidate is a *directory*, so
        directories are removed recursively; plain files are unlinked."""
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
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
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

    def reducer_log_dir(self) -> str | None:
        """Directory for external reducer stderr logs, or None to discard them.

        Uses a subdirectory of the run's history directory when history is
        enabled, so reducer logs live alongside the run's other artifacts.
        """
        if self.history_manager is not None:
            return os.path.join(self.history_manager.history_dir, "reducers")
        return None

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

        command = memory_limited_command(
            command, self.effective_memory_limit(self.first_call)
        )
        kwargs: dict[str, Any] = {
            "universal_newlines": False,
            "start_new_session": True,
            "cwd": cwd,
            "check": False,
        }

        # For debug mode, use simpler approach to capture output
        if debug:
            kwargs["capture_stdout"] = True
            kwargs["capture_stderr"] = True
            start_time = time.time()
            with stdin_source(self.input_type, working) as stdin:
                kwargs["stdin"] = stdin
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
                # nursery.start returns once the child has been spawned and
                # inherited the stdin descriptor, so it can be closed then.
                with stdin_source(self.input_type, working) as stdin:
                    kwargs["stdin"] = stdin
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
            # The subprocess uses setsid (start_new_session=True), so child
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

    def _default_memory_limit_may_block_initial(self) -> bool:
        """Whether an unset (default) memory limit is currently in effect.

        The auto-disable only applies to the physical-RAM default the user
        never asked for; an explicitly configured limit is left alone so its
        loud, actionable failure (with a --memory-limit=0 suggestion) stands.
        """
        return (
            not self.memory_limit_explicit
            and self.memory_limit is not None
            and self.memory_limit > 0
        )

    async def _retry_initial_without_memory_limit(
        self, test_case: TestCase
    ) -> ScriptRunResult:
        """Re-run the initial test with no address-space cap.

        Called when the initial test was uninteresting under the default
        memory limit. If it now passes, the cap (not a genuine failure) was
        the culprit — typical of sanitizer builds — so the limit is disabled
        for the rest of the run and the user is warned. If it still fails the
        failure is genuine, so the limit is restored and the normal
        invalid-initial-example path is left to report it.
        """
        saved_limit = self.memory_limit
        self.memory_limit = None
        # Re-run as a fresh calibration call: the capped run's exit code and
        # timeout measurement are discarded so the no-limit run becomes the
        # single logical initial call.
        self.first_call = True
        self.timeout_policy.reset()
        result = await self._run_for_result_once(test_case)
        if result.exit_code == 0:
            print(
                "Warning: the initial interestingness test only passed with the "
                "default --memory-limit disabled, so it has been disabled for the "
                "rest of this run. Sanitizer builds (ASan/MSan/TSan), which reserve "
                "huge amounts of virtual address space, are the usual cause.",
                file=sys.stderr,
                flush=True,
            )
        else:
            self.memory_limit = saved_limit
        return result

    async def run_for_result(
        self, test_case: TestCase, debug: bool = False
    ) -> ScriptRunResult:
        was_first_call = self.first_call
        result = await self._run_for_result_once(test_case, debug=debug)
        # A default memory limit that blocks the very first (calibration) test
        # is retried once without the cap; sanitizer builds abort under any
        # address-space limit, so this is the difference between the flagship
        # "reduce a sanitizer crash" workflow working and failing out of the
        # box. The debug reruns from build_error_message never re-probe (they
        # happen after the initial call, so was_first_call is False).
        if (
            not debug
            and was_first_call
            and result.exit_code != 0
            and self._default_memory_limit_may_block_initial()
        ):
            result = await self._retry_initial_without_memory_limit(test_case)
        return result

    async def _run_for_result_once(
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
                self._record_reduction_history(test_case)

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

    def _record_reduction_history(self, test_case: TestCase) -> None:
        """Record an adopted reduction in history and prune stored outputs.

        The recorded output is the one captured when the test case was found
        interesting (see check_interesting), not a fresh read, so a
        concurrently running test cannot overwrite it.

        Pruning keeps the adopted test case's output (the LLM passes read it
        from _successful_outputs) and the output of any candidate that still
        sorts better than the adopted one: adoption only ever moves to a
        strictly better test case, so a better-sorting stored candidate may
        itself be adopted moments later (which happens under parallelism, when
        several candidates are interesting at once). Everything worse than the
        adopted candidate is a loser whose output nothing can use again, so it
        is dropped to bound memory.
        """
        assert self.history_manager is not None
        test_case_bytes = self._get_test_case_bytes(test_case)
        output = self._successful_outputs.get(test_case_bytes)
        adopted_key = self.problem.sort_key(test_case)
        survivors = {
            tcb
            for tcb, key in self._successful_output_keys.items()
            if key < adopted_key
        }
        survivors.add(test_case_bytes)
        self._successful_outputs = {
            tcb: out
            for tcb, out in self._successful_outputs.items()
            if tcb in survivors
        }
        self._successful_output_keys = {
            tcb: key
            for tcb, key in self._successful_output_keys.items()
            if tcb in survivors
        }
        self.history_manager.record_reduction(test_case_bytes, output)

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
                    self._successful_output_keys[test_case_bytes] = (
                        self.problem.sort_key(test_case)
                    )
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
        self._successful_output_keys.clear()
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
            # An explicitly configured memory limit is never auto-disabled, so
            # point at it here in case it (not the test itself) is the problem:
            # sanitizer builds abort under any address-space cap. The default
            # limit reaches this branch only after a no-limit retry already
            # failed, so the suggestion would be misleading there and is
            # withheld.
            if self.memory_limit_explicit and self.memory_limit:
                lines.append(
                    "\nNote: an address-space limit (--memory-limit) is in effect. "
                    "Some builds — sanitizer builds (ASan/MSan/TSan) in particular — "
                    "reserve huge amounts of virtual address space and abort under "
                    "any such limit. If that may be the cause, rerun with "
                    "--memory-limit=0 to disable it."
                )

        return "\n".join(lines)

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
        treesitter_language, pending_treesitter = grammar_plan(
            self.filename, self.downloads
        )
        return ShrinkRay(
            problem,
            enable_cpp_passes=os.path.splitext(self.filename)[1] in C_FILE_EXTENSIONS,
            treesitter_language=treesitter_language,
            pending_treesitter_language=pending_treesitter,
            downloads=self.downloads,
            external_reducers=self.external_reducers,
            python_reducer=self.python_reducer,
            reducer_log_dir=self.reducer_log_dir(),
            restart_at_fixpoint=self.restart_at_fixpoint,
            **self.llm_reducer_kwargs(),
        )

    def _input_filenames(self) -> list[str]:
        return [self.filename]

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
        # The formatter reads its input from a temp file rather than piped
        # bytes for the same reason as stdin_source: a formatter that exits
        # without draining stdin would deadlock the pipe feeder on OpenBSD.
        with tempfile.TemporaryFile() as stdin:
            stdin.write(input)
            stdin.seek(0)
            return await trio.run_process(
                command,
                stdin=stdin,
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

        trivial_message = self.check_trivial_result(problem)
        if trivial_message is not None:
            print(trivial_message)
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
            target=problem,
            downloads=self.downloads,
            external_reducers=self.external_reducers,
            python_reducer=self.python_reducer,
            reducer_log_dir=self.reducer_log_dir(),
            restart_at_fixpoint=self.restart_at_fixpoint,
            **self.llm_reducer_kwargs(),
        )

    def _input_filenames(self) -> list[str]:
        return list(self.initial)

    def _get_initial_bytes(self) -> bytes:
        # Serialize directory content for history recording
        return serialize_directory(self.initial)

    def _get_test_case_bytes(self, test_case: dict[str, bytes]) -> bytes:
        # Serialize directory content for comparison/exclusion
        return serialize_directory(test_case)

    def _set_initial_for_restart(self, content: bytes) -> None:
        # Deserialize and update initial directory content
        self.initial = deserialize_directory(content)

    def _initialize_history_manager(self) -> None:
        """Initialize the history manager in directory mode."""
        assert self.history_manager is not None
        self.history_manager.initialize_directory(
            self.initial,
            self.test,
            self.filename,
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


def load_state_for_path(
    *,
    filename: str,
    input_type: Any,
    in_place: bool,
    test: list[str],
    timeout: float | None,
    memory_limit: int | None,
    memory_limit_explicit: bool,
    parallelism: int,
    formatter: str,
    trivial_is_error: bool,
    seed: int,
    volume: Volume,
    history_enabled: bool,
    also_interesting_code: int | None,
    external_reducers: list[list[str]],
    python_reducer: bool,
    restart_at_fixpoint: bool,
    llm_enabled: bool,
    llm_model: str,
    llm_only: bool,
) -> ShrinkRayState[Any]:
    """Read `filename` from disk and build the appropriate reduction state.

    A directory becomes a ShrinkRayDirectoryState over every file under it
    (keyed by relative path); a regular file becomes a
    ShrinkRayStateSingleFile. This is the single construction path shared
    by the CLI (basic UI) and the worker subprocess (textual UI), so the
    two cannot drift apart.
    """
    kwargs: dict[str, Any] = {
        "input_type": input_type,
        "in_place": in_place,
        "test": test,
        "timeout": timeout,
        "memory_limit": memory_limit,
        "memory_limit_explicit": memory_limit_explicit,
        "base": os.path.basename(filename),
        "parallelism": parallelism,
        "filename": filename,
        "formatter": formatter,
        "trivial_is_error": trivial_is_error,
        "seed": seed,
        "volume": volume,
        "history_enabled": history_enabled,
        "also_interesting_code": also_interesting_code,
        "external_reducers": external_reducers,
        "python_reducer": python_reducer,
        "restart_at_fixpoint": restart_at_fixpoint,
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
        "llm_only": llm_only,
    }
    if os.path.isdir(filename):
        initial = {}
        for d, _, fs in os.walk(filename):
            for f in fs:
                path = os.path.join(d, f)
                with open(path, "rb") as reader:
                    initial[os.path.relpath(path, filename)] = reader.read()
        return ShrinkRayDirectoryState(initial=initial, **kwargs)
    with open(filename, "rb") as reader:
        return ShrinkRayStateSingleFile(initial=reader.read(), **kwargs)
