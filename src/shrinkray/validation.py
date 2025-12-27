"""Initial validation of interestingness tests before reduction.

This module provides validation that runs in the main process using trio,
before the TUI is launched. It prints commands and temporary directories
to stderr so users can understand what's happening with slow tests, and
preserves temporary directories on failure for debugging.
"""

import os
import shlex
import shutil
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


def _build_command(
    test: list[str],
    working_file: str,
    input_type: InputType,
) -> list[str]:
    """Build the command to run, adding test file path if needed."""
    if input_type.enabled(InputType.arg):
        return test + [working_file]
    return list(test)


def _format_command(command: list[str], cwd: str) -> str:
    """Format a command for display, showing cd and the command."""
    # Quote command parts for display
    quoted = " ".join(shlex.quote(part) for part in command)
    return f"cd {shlex.quote(cwd)} && {quoted}"


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
            f"\nRunning interestingness test from: {cwd}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"Command: {_format_command(command, cwd)}",
            file=sys.stderr,
            flush=True,
        )
        print(file=sys.stderr, flush=True)

        # Build subprocess kwargs - stream output to stderr
        kwargs: dict = {
            "cwd": cwd,
            "check": False,
            # Stream stdout/stderr to parent's stderr for visibility
            "stdout": sys.stderr.fileno(),
            "stderr": sys.stderr.fileno(),
        }

        # Handle stdin if needed
        if input_type.enabled(InputType.stdin) and not os.path.isdir(working):
            with open(working, "rb") as f:
                kwargs["stdin"] = f.read()
        else:
            kwargs["stdin"] = b""

        # Run the process
        result = await trio.run_process(command, **kwargs)

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
                    f"but should return 0 for interesting test cases.\n"
                    f"Test was run from: {cwd}\n"
                    f"Command: {_format_command(command, cwd)}"
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


async def validate_initial_example(
    file_path: str,
    test: list[str],
    input_type: InputType,
    in_place: bool,
) -> ValidationResult:
    """Validate that the initial example passes the interestingness test.

    This runs directly in the main process using trio, streaming output
    to stderr so users can see progress for slow tests.

    Args:
        file_path: Path to the file to reduce
        test: The interestingness test command
        input_type: How to pass input to the test
        in_place: Whether to run in the current directory

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

    if result.success:
        print(
            "Initial validation passed.",
            file=sys.stderr,
            flush=True,
        )
        # Clean up temp directories on success
        if result.temp_dirs:
            for path in result.temp_dirs:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass  # Best effort cleanup
    else:
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


def run_validation(
    file_path: str,
    test: list[str],
    input_type: InputType,
    in_place: bool,
) -> ValidationResult:
    """Run initial validation synchronously using trio.run().

    This is the main entry point for validation from the CLI/TUI.
    It runs validation directly in the main process before any asyncio
    event loop is started.
    """
    return trio.run(
        validate_initial_example,
        file_path,
        test,
        input_type,
        in_place,
    )
