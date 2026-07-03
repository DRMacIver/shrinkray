"""Tests for the validation module."""

import io
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
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


async def test_formatter_failure_does_not_clobber_original_file_in_place_basename():
    """Regression test: with --in-place --input-type=basename, checking
    whether the formatted version is still interesting wrote the
    formatter's output directly over the user's original file and never
    restored it. No backup exists yet at validation time, so the user's
    original content was silently lost."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, "test.txt")
        original_content = b"test content"
        with open(test_file, "wb") as f:
            f.write(original_content)

        # Test passes only for the exact original content.
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write(f'#!/bin/bash\ngrep -qx "test content" {shlex.quote(test_file)}\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.basename,
            in_place=True,
            # Uppercases the content, making it uninteresting.
            formatter_command=["tr", "a-z", "A-Z"],
        )

        assert not result.success
        assert result.error_message is not None
        assert "uninteresting" in result.error_message
        with open(test_file, "rb") as f:
            assert f.read() == original_content


async def test_formatter_success_restores_original_file_in_place_basename():
    """Validating the formatted version must leave the user's file as it
    was even when the formatted version is still interesting: reduction
    starts from the file's content, and the .bak backup has not been
    created yet."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, "test.txt")
        original_content = b"test content"
        with open(test_file, "wb") as f:
            f.write(original_content)

        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.basename,
            in_place=True,
            formatter_command=["tr", "a-z", "A-Z"],
        )

        assert result.success
        with open(test_file, "rb") as f:
            assert f.read() == original_content


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

        # List files before validation
        files_before = set(os.listdir(tmp_dir))

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=True,
        )

        assert result.success

        # Any temp files created during validation should have been cleaned up
        files_after = set(os.listdir(tmp_dir))
        # Only the original test file and script should remain
        assert files_after == files_before


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
            f.write('#!/bin/bash\nrm -f "$1" 2>/dev/null; exit 0\n')
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
        with patch(
            "shrinkray.validation.os.unlink",
            side_effect=PermissionError("Cannot delete"),
        ):
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


# =============================================================================
# Integration test for real-time output streaming
# =============================================================================


@pytest.mark.slow
def test_validation_output_streams_immediately():
    """Test that validation output appears immediately, not buffered until completion.

    This is an integration test that runs the actual shrinkray CLI. The test
    script prints a marker and then blocks indefinitely, so if we observe the
    marker while the process is still running, the output must have been
    streamed rather than buffered until the test completed. This avoids any
    wall-clock threshold, which proved flaky under parallel test load.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"test content")

        # Create test script that prints immediately, then blocks until we
        # kill it.
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write(
                "#!/bin/bash\necho 'MARKER_OUTPUT_APPEARED' >&2\nsleep 1000\nexit 0\n"
            )
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Run the actual shrinkray command in its own process group, so we
        # can kill the blocked test script (which shares our stderr pipe
        # and shrinkray's process group) along with it.
        proc = subprocess.Popen(
            [sys.executable, "-m", "shrinkray", script, test_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        marker_seen = threading.Event()
        output_lines: list[bytes] = []

        def read_stderr():
            assert proc.stderr is not None
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                output_lines.append(line)
                if b"MARKER_OUTPUT_APPEARED" in line:
                    marker_seen.set()

        # Start reading in a daemon thread: it exits once the pipe closes,
        # but under heavy load that can take a moment and pytest must not
        # wait on it.
        reader_thread = threading.Thread(target=read_stderr, daemon=True)
        reader_thread.start()

        try:
            appeared = marker_seen.wait(timeout=60)
            still_running = proc.poll() is None
        finally:
            # Kill the whole process group - we don't need the full
            # reduction, and the test script must die too or the reader
            # thread would block on the shared pipe until it exits.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
            reader_thread.join(timeout=5)

        assert appeared, (
            "Never saw MARKER_OUTPUT_APPEARED in output. Got lines: "
            f"{[line.decode('utf-8', errors='replace') for line in output_lines]}"
        )
        # The test script never exits on its own, so seeing the marker while
        # the process is alive proves the output was streamed.
        assert still_running, (
            "Marker only appeared after the process exited, so validation "
            "output was buffered rather than streamed."
        )


# =============================================================================
# Formatter validation tests
# =============================================================================


async def test_validate_with_formatter_success():
    """Test validation with a formatter that succeeds."""
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

        # Create formatter that outputs to stderr and passes content through
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\necho 'FORMATTER_STDERR' >&2\ncat\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
            formatter_command=[formatter],
        )

        assert result.success
        assert result.formatter_works is True


async def test_validate_with_formatter_failure():
    """Test validation with a formatter that fails."""
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

        # Create formatter that fails
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\necho 'FORMATTER_ERROR' >&2\nexit 1\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
            formatter_command=[formatter],
        )

        assert not result.success
        assert result.error_message is not None
        assert "Formatter exited unexpectedly" in result.error_message
        assert result.exit_code == 1


async def test_validate_with_formatter_makes_content_uninteresting():
    """Test validation when formatter changes content to something uninteresting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file with specific content
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"hello")

        # Create test script that only accepts "hello", not "goodbye"
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write('#!/bin/bash\ngrep -q hello "$1"\n')
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Create formatter that changes content to "goodbye"
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\necho 'goodbye'\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=False,
            formatter_command=[formatter],
        )

        assert not result.success
        assert result.error_message is not None
        assert (
            "Formatting initial test case made it uninteresting" in result.error_message
        )


async def test_validate_with_formatter_preserves_interesting_content():
    """Test validation when formatter changes content but result is still interesting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"hello world")

        # Create test script that accepts any content
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Create formatter that changes content (uppercase)
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\ntr '[:lower:]' '[:upper:]'\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,
            in_place=False,
            formatter_command=[formatter],
        )

        assert result.success
        assert result.formatter_works is True


def test_run_validation_with_formatter(capsys):
    """Test run_validation with formatter via sync wrapper."""
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

        # Create formatter that outputs stderr
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\necho 'FORMATTER_STDERR' >&2\ncat\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        result = run_validation(
            file_path=test_file,
            test=[script],
            input_type=InputType.all,
            in_place=False,
            formatter_command=[formatter],
        )

        assert result.success

        captured = capsys.readouterr()
        # Formatter stderr should be visible
        assert "FORMATTER_STDERR" in captured.err
        assert "Running formatter:" in captured.err


async def test_validate_with_formatter_in_place_cleans_temp_file():
    """Test that in_place formatter validation cleans up temp files.

    This exercises the file cleanup path (os.unlink) in the formatted result cleanup.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"hello world")

        # Create test script that accepts any content
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Create formatter that changes content (adds newline)
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            # Change content so it's different from original
            f.write("#!/bin/bash\ncat; echo ''\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        # List files before validation
        files_before = set(os.listdir(tmp_dir))

        result = await validate_initial_example(
            file_path=test_file,
            test=[script],
            input_type=InputType.arg,  # arg, not basename, so temp file is created
            in_place=True,
            formatter_command=[formatter],
        )

        assert result.success
        assert result.formatter_works is True

        # Check that temp files were cleaned up
        files_after = set(os.listdir(tmp_dir))
        # Only the original test file and script should remain
        assert files_after == files_before


async def test_validate_with_formatter_basename_no_temp_dirs():
    """Test formatter validation with in_place=True and basename mode.

    This exercises the path where formatted_result.temp_dirs is empty/None.
    """
    # Need to run from the directory containing the test file for basename mode
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Create test file
            test_file = os.path.join(tmp_dir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"hello world")

            # Create test script that accepts any content
            script = os.path.join(tmp_dir, "test.sh")
            with open(script, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

            # Create formatter that changes content
            formatter = os.path.join(tmp_dir, "formatter.sh")
            with open(formatter, "w") as f:
                f.write("#!/bin/bash\ncat; echo ''\n")
            os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.basename,  # basename mode = no temp files
                in_place=True,
                formatter_command=[formatter],
            )

            assert result.success
            assert result.formatter_works is True
        finally:
            os.chdir(original_cwd)


async def test_validate_with_formatter_cleanup_handles_exception():
    """Test that cleanup of formatted test temp dirs handles exceptions gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"hello world")

        # Create test script that accepts any content
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Create formatter that changes content
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\ncat; echo ''\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        # Mock shutil.rmtree to raise an exception during cleanup
        def failing_rmtree(path: str) -> None:
            # Only fail for temp dirs created during formatted test
            if "shrinkray-" in path:
                raise PermissionError("Cannot remove directory")
            # The validation code doesn't use kwargs, so we can ignore them
            shutil.rmtree(path)

        with patch("shrinkray.validation.shutil.rmtree", side_effect=failing_rmtree):
            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.arg,  # Creates temp files
                in_place=False,  # Creates temp directories
                formatter_command=[formatter],
            )

        # Should succeed even though cleanup failed with exception
        assert result.success
        assert result.formatter_works is True


async def test_validate_with_formatter_cleanup_nonexistent_path():
    """Test that cleanup handles paths that no longer exist (neither dir nor file)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test file
        test_file = os.path.join(tmp_dir, "test.txt")
        with open(test_file, "wb") as f:
            f.write(b"hello world")

        # Create test script that accepts any content
        script = os.path.join(tmp_dir, "test.sh")
        with open(script, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)

        # Create formatter that changes content
        formatter = os.path.join(tmp_dir, "formatter.sh")
        with open(formatter, "w") as f:
            f.write("#!/bin/bash\ncat; echo ''\n")
        os.chmod(formatter, os.stat(formatter).st_mode | stat.S_IEXEC)

        # Mock os.path.isdir and os.path.exists to simulate a path that doesn't exist
        # This covers the branch where the path is neither a directory nor a file
        original_isdir = os.path.isdir
        original_exists = os.path.exists

        def mock_isdir(path: str) -> bool:
            if "shrinkray-" in path and "validate" in path:
                return False  # Pretend the shrinkray temp dir doesn't exist as a dir
            return original_isdir(path)

        def mock_exists(path: str) -> bool:
            if "shrinkray-" in path and "validate" in path:
                return False  # Pretend the shrinkray temp path doesn't exist at all
            return original_exists(path)

        with (
            patch("shrinkray.validation.os.path.isdir", side_effect=mock_isdir),
            patch("shrinkray.validation.os.path.exists", side_effect=mock_exists),
        ):
            result = await validate_initial_example(
                file_path=test_file,
                test=[script],
                input_type=InputType.arg,  # Creates temp files
                in_place=False,  # Creates temp directories
                formatter_command=[formatter],
            )

        # Should succeed - cleanup of nonexistent path is a no-op
        assert result.success
        assert result.formatter_works is True
