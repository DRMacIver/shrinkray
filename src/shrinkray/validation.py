"""Initial validation of interestingness tests before reduction.

This module provides validation that runs in the main process using trio,
before the TUI is launched. It prints commands and temporary directories
to stderr so users can understand what's happening with slow tests, and
preserves temporary directories on failure for debugging.
"""

import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass

import trio

from shrinkray.cli import InputType
from shrinkray.process import run_managed_process
from shrinkray.state import DYNAMIC_TIMEOUT_CALIBRATION_TIMEOUT


class _ValidationTimedOut(Exception):
    """The interestingness test did not finish within its timeout."""


@dataclass
class ValidationResult:
    """Result of initial validation."""

    success: bool
    error_message: str | None = None
    exit_code: int | None = None
    # Temp directories to clean up only on success
    temp_dirs: list[str] | None = None
    # Whether formatter is usable (None if no formatter specified)
    formatter_works: bool | None = None


def _build_command(
    test: list[str],
    working_file: str,
    input_type: InputType,
) -> list[str]:
    """Build the command to run, adding test file path if needed."""
    if input_type.enabled(InputType.arg):
        return test + [working_file]
    return list(test)


def _format_command_for_display(command: list[str], cwd: str) -> str:
    """Format a command for display, with cd on its own line and relative paths.

    Returns a multi-line string with:
    - cd <directory>
    - <command with relative paths for files in cwd>
    """
    # Convert absolute paths within cwd to relative paths for readability
    display_parts = []
    for part in command:
        if part.startswith(cwd + os.sep):
            # Convert to relative path
            display_parts.append(os.path.relpath(part, cwd))
        else:
            display_parts.append(part)

    quoted = " ".join(shlex.quote(part) for part in display_parts)
    return f"cd {shlex.quote(cwd)}\n{quoted}"


async def _run_validation_test(
    test: list[str],
    initial_content: bytes,
    base: str,
    input_type: InputType,
    in_place: bool,
    filename: str,
    retries: int = 0,
    timeout: float | None = None,
    memory_limit: int | None = None,
) -> ValidationResult:
    """Run the interestingness test and check if it passes, re-running a
    failed run up to `retries` times in case the test is nondeterministic.

    Returns ValidationResult with success=True if the test passed (exit code 0),
    or success=False with error details if it failed.
    """
    temp_dirs: list[str] = []
    # If we write candidate content over the user's original file, remember
    # what it held so we can put it back. No backup exists yet at
    # validation time, so failing to restore would lose the original.
    restore_content: bytes | None = None
    restore_path: str | None = None

    try:
        # Determine working directory and file path
        if in_place:
            if input_type == InputType.basename:
                working = filename
                cwd = os.getcwd()
                # Write directly to original file
                with open(working, "rb") as reader:
                    existing_content = reader.read()
                # Even an identical rewrite can be cancelled after truncation,
                # before any backup exists. Always restore the original.
                restore_path = working
                restore_content = existing_content
                async with await trio.open_file(working, "wb") as f:
                    await f.write(initial_content)
            else:
                # Create a temp file in same directory with random suffix
                base_name, ext = os.path.splitext(filename)
                working = base_name + "-" + os.urandom(16).hex() + ext
                cwd = os.getcwd()
                async with await trio.open_file(working, "wb") as f:
                    await f.write(initial_content)
                temp_dirs.append(working)  # Track for cleanup
        else:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix="shrinkray-validate-")
            temp_dirs.append(temp_dir)
            working = os.path.join(temp_dir, base)
            cwd = temp_dir
            async with await trio.open_file(working, "wb") as f:
                await f.write(initial_content)

        # Build command
        command = _build_command(test, working, input_type)

        # Print what we're doing to stderr
        print(
            "\nRunning interestingness test:",
            file=sys.stderr,
            flush=True,
        )
        print(
            _format_command_for_display(command, cwd),
            file=sys.stderr,
            flush=True,
        )
        print(file=sys.stderr, flush=True)

        # Handle stdin if needed
        stdin_data: bytes | None = None
        if input_type.enabled(InputType.stdin) and not os.path.isdir(working):
            with open(working, "rb") as f:
                stdin_data = f.read()

        async def run_subprocess() -> subprocess.CompletedProcess[bytes]:
            try:
                output_fd = sys.stderr.fileno()
            except (io.UnsupportedOperation, OSError):
                output_fd = None
            # Without a configured timeout the initial call gets the same
            # calibration bound the reducer would give it.
            if timeout is None:
                effective_timeout = DYNAMIC_TIMEOUT_CALIBRATION_TIMEOUT
                advice = "Pass --timeout to allow a slower test."
            else:
                effective_timeout = timeout
                advice = "Try raising or disabling --timeout."
            try:
                return await run_managed_process(
                    command,
                    cwd=cwd,
                    input=stdin_data,
                    output_fd=output_fd,
                    timeout=effective_timeout,
                    memory_limit=memory_limit,
                )
            except subprocess.TimeoutExpired:
                raise _ValidationTimedOut(
                    f"Interestingness test timed out after {effective_timeout:g}s "
                    f"on the initial test case. {advice}\n\n"
                    f"To reproduce:\n{_format_command_for_display(command, cwd)}"
                )

        result = await run_subprocess()
        _report_run(result)

        # A nondeterministic test may fail its first run on an interesting
        # test case. Rather than refusing to start, retry a few times: one
        # success means the test case is interesting and the test is
        # nondeterministic, which the reducer then handles.
        failures = 0
        while result.returncode != 0 and failures < retries:
            failures += 1
            print(
                f"The interestingness test has failed {failures} time(s) on "
                "the initial test case; retrying in case it is "
                f"nondeterministic (retry {failures} of {retries}).",
                file=sys.stderr,
                flush=True,
            )
            result = await run_subprocess()
            _report_run(result)
        if failures and result.returncode == 0:
            print(
                "The interestingness test is nondeterministic: it failed on "
                f"{failures} of {failures + 1} runs of the initial test case.",
                file=sys.stderr,
                flush=True,
            )

        if result.returncode != 0:
            return ValidationResult(
                success=False,
                error_message=(
                    f"Interestingness test exited with code {result.returncode}, "
                    f"but should return 0 for interesting test cases.\n\n"
                    f"To reproduce:\n{_format_command_for_display(command, cwd)}"
                ),
                exit_code=result.returncode,
                temp_dirs=temp_dirs,
            )

        return ValidationResult(
            success=True,
            exit_code=0,
            temp_dirs=temp_dirs,
        )

    except _ValidationTimedOut as e:
        return ValidationResult(
            success=False,
            error_message=str(e),
            temp_dirs=temp_dirs,
        )
    except Exception as e:
        traceback.print_exc()
        return ValidationResult(
            success=False,
            error_message=f"Error running interestingness test: {e}",
            temp_dirs=temp_dirs,
        )
    finally:
        if restore_path is not None:
            assert restore_content is not None
            with open(restore_path, "wb") as writer:
                writer.write(restore_content)


async def _run_formatter(
    formatter_command: list[str],
    content: bytes,
) -> subprocess.CompletedProcess[bytes]:
    """Run the formatter command on content, streaming output to stderr."""

    print("\nRunning formatter:", file=sys.stderr, flush=True)
    print(
        " ".join(shlex.quote(part) for part in formatter_command),
        file=sys.stderr,
        flush=True,
    )

    result = await run_managed_process(formatter_command, input=content)

    # Show stderr from formatter if any
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.flush()

    print(
        f"Formatter exit code: {result.returncode}",
        file=sys.stderr,
        flush=True,
    )

    return result


def _report_run(result: subprocess.CompletedProcess[bytes]) -> None:
    """Print a completed interestingness-test run's captured output (in
    fallback mode) and exit code to stderr."""
    if result.stdout:
        sys.stderr.buffer.write(result.stdout)
        sys.stderr.flush()
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.flush()
    print(file=sys.stderr, flush=True)
    print(f"Exit code: {result.returncode}", file=sys.stderr, flush=True)


async def validate_initial_example(
    file_path: str,
    test: list[str],
    input_type: InputType,
    in_place: bool,
    formatter_command: list[str] | None = None,
    retries: int = 0,
    timeout: float | None = None,
    memory_limit: int | None = None,
) -> ValidationResult:
    """Validate that the initial example passes the interestingness test.

    This runs directly in the main process using trio, streaming output
    to stderr so users can see progress for slow tests. Also checks the
    formatter if one is specified.

    Args:
        file_path: Path to the file to reduce
        test: The interestingness test command
        input_type: How to pass input to the test
        in_place: Whether to run in the current directory
        formatter_command: Optional formatter command to validate

    Returns:
        ValidationResult indicating success or failure with details.
        On failure, temp_dirs are preserved for debugging.
    """
    # Read the initial content
    if os.path.isdir(file_path):
        # For directories, we need different handling
        # For now, just validate that it's a valid directory
        return ValidationResult(success=True)

    with open(file_path, "rb") as f:
        initial_content = f.read()

    base = os.path.basename(file_path)

    print("Validating interestingness test...", file=sys.stderr, flush=True)

    result = await _run_validation_test(
        test=test,
        initial_content=initial_content,
        base=base,
        input_type=input_type,
        in_place=in_place,
        filename=file_path,
        retries=retries,
        timeout=timeout,
        memory_limit=memory_limit,
    )

    if not result.success:
        # On failure, keep temp directories and tell user
        if result.temp_dirs:
            print(
                "\nTemporary files preserved for debugging:",
                file=sys.stderr,
                flush=True,
            )
            for path in result.temp_dirs:
                print(f"  {path}", file=sys.stderr, flush=True)
        return result

    # Clean up temp directories from initial test
    if result.temp_dirs:
        for path in result.temp_dirs:
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass  # Best effort cleanup

    print("Initial validation passed.", file=sys.stderr, flush=True)

    # Now check formatter if specified
    formatter_works: bool | None = None
    if formatter_command is not None:
        formatter_result = await _run_formatter(formatter_command, initial_content)

        if formatter_result.returncode != 0:
            # The formatter is only a cosmetic aid, so a formatter that
            # crashes on the initial test case should not abort the whole
            # reduction. Warn and carry on without it, mirroring the
            # in-reduction behaviour (attempt_format disables a formatter
            # that fails or changes interestingness).
            print(
                "\nFormatter exited unexpectedly on the initial test case; "
                "continuing without formatting. Pass --formatter=none to "
                "silence this, or choose a working formatter.",
                file=sys.stderr,
                flush=True,
            )
            print(
                formatter_result.stderr.decode("utf-8", errors="replace").strip(),
                file=sys.stderr,
                flush=True,
            )
            return ValidationResult(
                success=True,
                exit_code=0,
                formatter_works=False,
            )

        reformatted = formatter_result.stdout

        # If formatter changed the content, verify it's still interesting
        if reformatted != initial_content:
            print(
                "\nChecking if formatted version is still interesting...",
                file=sys.stderr,
                flush=True,
            )
            formatted_result = await _run_validation_test(
                test=test,
                initial_content=reformatted,
                base=base,
                input_type=input_type,
                in_place=in_place,
                filename=file_path,
                timeout=timeout,
                memory_limit=memory_limit,
            )

            # Clean up temp dirs from formatted test
            if formatted_result.temp_dirs:
                for path in formatted_result.temp_dirs:
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.exists(path):
                            os.unlink(path)
                    except Exception:
                        pass

            if not formatted_result.success:
                return ValidationResult(
                    success=False,
                    error_message=(
                        "Formatting initial test case made it uninteresting. "
                        "If this is expected, please run with --formatter=none.\n\n"
                        f"Formatter stderr:\n{formatter_result.stderr.decode('utf-8', errors='replace').strip()}"
                    ),
                    exit_code=formatted_result.exit_code,
                )

            print("Formatted version is also interesting.", file=sys.stderr, flush=True)

        formatter_works = True

    return ValidationResult(
        success=True,
        exit_code=0,
        formatter_works=formatter_works,
    )


def run_validation(
    file_path: str,
    test: list[str],
    input_type: InputType,
    in_place: bool,
    formatter_command: list[str] | None = None,
    retries: int = 0,
    timeout: float | None = None,
    memory_limit: int | None = None,
) -> ValidationResult:
    """Run initial validation synchronously using trio.run().

    This is the main entry point for validation from the CLI/TUI.
    It runs validation directly in the main process before any asyncio
    event loop is started.
    """

    async def _run() -> ValidationResult:
        return await validate_initial_example(
            file_path,
            test,
            input_type,
            in_place,
            formatter_command,
            retries=retries,
            timeout=timeout,
            memory_limit=memory_limit,
        )

    return trio.run(_run)
