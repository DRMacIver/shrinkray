"""Tests for the validation module."""

import io
import os
import shutil
import stat
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from shrinkray.cli import InputType
from shrinkray.validation import (
    ValidationResult,
    _build_command,
    _format_command_for_display,
    run_validation,
    validate_initial_example,
)


# =============================================================================
# ValidationResult tests
# =============================================================================


def test_validation_result_success():
    """Test ValidationResult with success=True."""
    result = ValidationResult(success=True, exit_code=0)
    assert result.success
    assert result.exit_code == 0
    assert result.error_message is None
    assert result.temp_dirs is None


def test_validation_result_failure():
    """Test ValidationResult with failure."""
    result = ValidationResult(
        success=False,
        error_message="Test failed",
        exit_code=1,
        temp_dirs=["/tmp/test-dir"],
    )
    assert not result.success
    assert result.error_message == "Test failed"
    assert result.exit_code == 1
    assert result.temp_dirs == ["/tmp/test-dir"]


# =============================================================================
# _build_command tests
# =============================================================================


@pytest.mark.parametrize(
    "input_type,expected_includes_file",
    [
        pytest.param(InputType.all, True, id="all-includes-arg"),
        pytest.param(InputType.arg, True, id="arg-includes-file"),
        pytest.param(InputType.stdin, False, id="stdin-no-file"),
        pytest.param(InputType.basename, False, id="basename-no-file"),
    ],
)
def test_build_command_input_types(input_type, expected_includes_file):
    """Test _build_command builds correct command for each input type."""
    test = ["./test.sh"]
    working_file = "/tmp/test.txt"

    result = _build_command(test, working_file, input_type)

    if expected_includes_file:
        assert result == ["./test.sh", "/tmp/test.txt"]
    else:
        assert result == ["./test.sh"]


def test_build_command_with_existing_args():
    """Test _build_command preserves existing test arguments."""
    test = ["./test.sh", "--verbose", "--flag"]
    working_file = "/tmp/test.txt"

    result = _build_command(test, working_file, InputType.arg)

    assert result == ["./test.sh", "--verbose", "--flag", "/tmp/test.txt"]


# =============================================================================
# _format_command_for_display tests
# =============================================================================


def test_format_command_simple():
    """Test _format_command_for_display formats a simple command."""
    result = _format_command_for_display(["./test.sh", "file.txt"], "/tmp/dir")
    lines = result.split("\n")
    assert lines[0] == "cd /tmp/dir"
    assert lines[1] == "./test.sh file.txt"


def test_format_command_with_spaces():
    """Test _format_command_for_display quotes paths with spaces."""
    result = _format_command_for_display(
        ["./test.sh", "file with spaces.txt"], "/tmp/my dir"
    )
    lines = result.split("\n")
    assert lines[0] == "cd '/tmp/my dir'"
    assert "'file with spaces.txt'" in lines[1]


def test_format_command_with_special_chars():
    """Test _format_command_for_display quotes special characters."""
    result = _format_command_for_display(["./test.sh", "file$var.txt"], "/tmp")
    # Should quote the filename to prevent shell expansion
    assert "'file$var.txt'" in result


def test_format_command_converts_paths_in_cwd_to_relative():
    """Test _format_command_for_display converts paths within cwd to relative."""
    result = _format_command_for_display(
        ["./test.sh", "/tmp/workdir/subdir/file.txt"], "/tmp/workdir"
    )
    lines = result.split("\n")
    assert lines[0] == "cd /tmp/workdir"
    # The path /tmp/workdir/subdir/file.txt should become subdir/file.txt
    assert "subdir/file.txt" in lines[1]
    assert "/tmp/workdir/subdir/file.txt" not in lines[1]


def test_format_command_keeps_paths_outside_cwd_absolute():
    """Test _format_command_for_display keeps paths outside cwd absolute."""
    result = _format_command_for_display(
        ["/usr/bin/test.sh", "/other/path/file.txt"], "/tmp/workdir"
    )
    lines = result.split("\n")
    # Paths outside cwd should remain absolute
    assert "/usr/bin/test.sh" in lines[1]
    assert "/other/path/file.txt" in lines[1]


# =============================================================================
# validate_initial_example tests
# =============================================================================


async def test_validate_initial_example_success():
    """Test successful validation with exit code 0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        assert result.success
        assert result.exit_code == 0
        # Temp dirs should be cleaned up on success
        if result.temp_dirs:
            for d in result.temp_dirs:
                assert not os.path.exists(d)


async def test_validate_initial_example_failure():
    """Test validation failure with exit code 1."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that exits 1
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        assert not result.success
        assert result.exit_code == 1
        assert result.error_message is not None
        assert "exit" in result.error_message.lower()
        # Temp dirs should be preserved on failure
        assert result.temp_dirs is not None
        for d in result.temp_dirs:
            assert os.path.exists(d)
            # Clean up for test hygiene
            shutil.rmtree(d, ignore_errors=True)


async def test_validate_initial_example_directory():
    """Test validation with directory input returns success immediately."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a subdirectory as the "input"
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir)

        result = await validate_initial_example(
            file_path=input_dir,
            test=["./test.sh"],
            input_type=InputType.all,
            in_place=False,
        )

        # Directories currently bypass validation
        assert result.success


async def test_validate_initial_example_stdin_input():
    """Test validation with stdin input type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that reads stdin and exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\ncat > /dev/null\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.stdin,
            in_place=False,
        )

        assert result.success


async def test_validate_initial_example_in_place():
    """Test validation with in_place=True uses current directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file in the temp dir
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=True,
        )

        assert result.success


async def test_validate_initial_example_in_place_basename():
    """Test validation with in_place=True and basename input type."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file in the temp dir
        test_file = os.path.join(tmp_dir, "test.txt")
        original_content = b"test content"
        with open(test_file, "wb") as f:
            f.write(original_content)

        # Create test script that exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.basename,
            in_place=True,
        )

        assert result.success

        # Verify the file was written to directly (no temp dirs)
        assert result.temp_dirs is None or len(result.temp_dirs) == 0


async def test_validate_initial_example_in_place_cleans_temp_file():
    """Test validation with in_place=True cleans up temp file on success."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file in the temp dir
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=True,
        )

        assert result.success
        # temp_dirs should have contained the temp file path(s)
        # After successful validation, the cleanup code should have removed them
        # result.temp_dirs still contains the paths, but the files should be gone
        assert result.temp_dirs is not None
        assert len(result.temp_dirs) > 0
        # All temp files should have been cleaned up (exercises line 218-219)
        for path in result.temp_dirs:
            assert not os.path.exists(path), f"Temp file should be cleaned up: {path}"


async def test_validate_initial_example_in_place_failure_preserves_temp():
    """Test in_place validation failure preserves temp file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that fails
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=True,
        )

        assert not result.success
        # Temp file should be preserved on failure
        assert result.temp_dirs is not None
        for path in result.temp_dirs:
            assert os.path.exists(path)
            # Clean up for test hygiene
            os.unlink(path)


# =============================================================================
# run_validation tests (synchronous entry point)
# =============================================================================


def test_run_validation_success():
    """Test run_validation with successful test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = run_validation(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        assert result.success


def test_run_validation_failure():
    """Test run_validation with failing test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that fails
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 42\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = run_validation(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        assert not result.success
        assert result.exit_code == 42
        assert result.error_message is not None

        # Clean up temp dirs
        if result.temp_dirs:
            for d in result.temp_dirs:
                shutil.rmtree(d, ignore_errors=True)


def test_run_validation_captures_output(capsys):
    """Test that run_validation outputs to stderr."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        run_validation(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        captured = capsys.readouterr()
        assert "Validating" in captured.err
        assert "Running interestingness test:" in captured.err


def test_run_validation_shows_temp_dir_on_failure(capsys):
    """Test that failed validation shows temp directory in output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that fails
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = run_validation(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
        )

        captured = capsys.readouterr()
        assert "preserved for debugging" in captured.err

        # Clean up temp dirs
        if result.temp_dirs:
            for d in result.temp_dirs:
                shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# Edge case / defensive code tests
# =============================================================================


async def test_cleanup_handles_already_deleted_file():
    """Test cleanup handles case where temp file was already deleted.

    This covers the branch where os.path.exists() returns False in cleanup.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that deletes the temp file then exits 0
        script = os.path.join(tmp_dir, "test.sh")
        # The script will receive the temp file path as an argument and delete it
        with open(script, "w") as f:
            f.write("#!/bin/bash\nrm -f \"$1\" 2>/dev/null; exit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Run with in_place=True + arg so it creates a temp file that gets passed to script
        # The script will delete it before returning, so cleanup finds it missing
        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=True,
        )

        # Validation should still succeed even though temp file was pre-deleted
        assert result.success


async def test_cleanup_handles_exception():
    """Test cleanup handles exceptions gracefully.

    This covers the exception handler in cleanup code.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Mock os.unlink to raise an exception
        with patch("shrinkray.validation.os.unlink", side_effect=PermissionError("Cannot delete")):
            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.arg,
                in_place=True,
            )

        # Validation should still succeed despite cleanup exception
        assert result.success
        # Temp file might still exist due to failed cleanup - clean up for test hygiene
        if result.temp_dirs:
            for path in result.temp_dirs:
                try:
                    os.unlink(path)
                except Exception:
                    pass


async def test_validation_with_captured_stderr():
    """Test validation when stderr doesn't have a real file descriptor.

    This exercises the fallback path where we capture and then print output.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that prints output and exits 0
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\necho 'test output'\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Mock sys.stderr.fileno to raise UnsupportedOperation
        # This simulates running under pytest with output capture
        original_stderr = sys.stderr
        mock_stderr = MagicMock()
        mock_stderr.fileno.side_effect = io.UnsupportedOperation("fileno")
        mock_stderr.buffer = original_stderr.buffer
        mock_stderr.flush = original_stderr.flush

        with patch("shrinkray.validation.sys.stderr", mock_stderr):
            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.all,
                in_place=False,
            )

        assert result.success


async def test_validation_with_captured_output_failure():
    """Test validation failure when output is captured.

    This exercises the output capture path with a failing test.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that prints output and exits 1
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\necho 'error output' >&2\nexit 1\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Mock sys.stderr.fileno to raise UnsupportedOperation
        original_stderr = sys.stderr
        mock_stderr = MagicMock()
        mock_stderr.fileno.side_effect = io.UnsupportedOperation("fileno")
        mock_stderr.buffer = original_stderr.buffer
        mock_stderr.flush = original_stderr.flush

        with patch("shrinkray.validation.sys.stderr", mock_stderr):
            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.all,
                in_place=False,
            )

        assert not result.success
        # Clean up temp dirs
        if result.temp_dirs:
            for d in result.temp_dirs:
                shutil.rmtree(d, ignore_errors=True)


async def test_validation_handles_run_process_exception():
    """Test validation handles exceptions during process execution.

    This covers the exception handler in _run_validation_test.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Use a non-existent command to trigger an exception
        result = await validate_initial_example(
            file_path=test_file,
            test=["/nonexistent/command/that/does/not/exist"],
            input_type=InputType.all,
            in_place=False,
        )

        assert not result.success
        assert result.error_message is not None
        assert "Error running interestingness test" in result.error_message

        # Clean up temp dirs
        if result.temp_dirs:
            for d in result.temp_dirs:
                shutil.rmtree(d, ignore_errors=True)


async def test_failure_with_no_temp_dirs():
    """Test validation failure when no temp directories were created.

    This covers the branch where temp_dirs is empty on failure.
    This can happen when in_place=True with basename input type and the test fails.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that fails
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Run with in_place=True + basename so no temp files are created
        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.basename,
            in_place=True,
        )

        # Validation should fail, but with no temp dirs to preserve
        assert not result.success
        assert result.temp_dirs is None or len(result.temp_dirs) == 0
