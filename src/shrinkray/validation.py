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
) -> ValidationResult:
    """Run the interestingness test once and check if it passes.

    Returns ValidationResult with success=True if the test passed (exit code 0),
    or success=False with error details if it failed.
    """
    temp_dirs: list[str] = []

    try:
        # Determine working directory and file path
        if in_place:
            if input_type == InputType.basename:
                working = filename
                cwd = os.getcwd()
                # Write directly to original file
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

        # Run subprocess with real-time output streaming
        # We use subprocess.run in a thread because trio.run_process doesn't
        # properly support file descriptor inheritance for streaming output.
        def run_subprocess() -> subprocess.CompletedProcess[bytes]:
            # Try to stream output directly to stderr if possible
            # This allows real-time output visibility for slow tests
            try:
                stderr_fd = sys.stderr.fileno()
                return subprocess.run(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL if stdin_data is None else None,
                    stdout=stderr_fd,
                    stderr=stderr_fd,
                    input=stdin_data,
                    check=False,
                )
            except (io.UnsupportedOperation, OSError):
                # Falls back to capturing if stderr doesn't have a real file
                # descriptor (e.g., when running under pytest with capture)
                return subprocess.run(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL if stdin_data is None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    input=stdin_data,
                    check=False,
                )

        result = await trio.to_thread.run_sync(run_subprocess)

        # If we captured output (fallback mode), print it now
        if result.stdout:
            sys.stderr.buffer.write(result.stdout)
            sys.stderr.flush()
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.flush()

        print(file=sys.stderr, flush=True)
        print(
            f"Exit code: {result.returncode}",
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

    except Exception as e:
        traceback.print_exc()
        return ValidationResult(
            success=False,
            error_message=f"Error running interestingness test: {e}",
            temp_dirs=temp_dirs,
        )


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

    def run_subprocess() -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            formatter_command,
            input=content,
            capture_output=True,
            check=False,
        )

    result = await trio.to_thread.run_sync(run_subprocess)

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


async def validate_initial_example(
    file_path: str,
    test: list[str],
    input_type: InputType,
    in_place: bool,
    formatter_command: list[str] | None = None,
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
            return ValidationResult(
                success=False,
                error_message=(
                    "Formatter exited unexpectedly on initial test case. "
                    "If this is expected, please run with --formatter=none.\n\n"
                    f"Formatter stderr:\n{formatter_result.stderr.decode('utf-8', errors='replace').strip()}"
                ),
                exit_code=formatter_result.returncode,
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
        )

    return trio.run(_run)
