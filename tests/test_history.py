"""Tests for shrinkray.history module."""

from __future__ import annotations

import base64
import json
import os
import stat
import tempfile

import pytest

from shrinkray.history import HistoryManager, sanitize_for_filename


# === sanitize_for_filename tests ===


@pytest.mark.parametrize(
    "input_str,expected",
    [
        pytest.param("simple", "simple", id="simple_name"),
        pytest.param("test.sh", "test.sh", id="with_extension"),
        pytest.param("my-test", "my-test", id="with_dash"),
        pytest.param("my_test", "my_test", id="with_underscore"),
        pytest.param("test/path", "test_path", id="with_slash"),
        pytest.param("test:name", "test_name", id="with_colon"),
        pytest.param("test name", "test_name", id="with_space"),
        pytest.param("test__name", "test_name", id="multiple_underscores"),
        pytest.param("___test___", "test", id="leading_trailing_underscores"),
        pytest.param(
            "a" * 100, "a" * 50, id="long_name_truncated"
        ),  # Truncated to 50 chars
        pytest.param("", "", id="empty_string"),
        pytest.param("$@#!", "", id="all_special_chars"),
    ],
)
def test_sanitize_for_filename(input_str: str, expected: str) -> None:
    assert sanitize_for_filename(input_str) == expected


# === HistoryManager.create tests ===


def test_create_generates_unique_run_id() -> None:
    """Each call to create() should generate a unique run ID."""
    manager1 = HistoryManager.create(["./test.sh"], "buggy.c")
    manager2 = HistoryManager.create(["./test.sh"], "buggy.c")
    assert manager1.run_id != manager2.run_id


def test_create_run_id_format() -> None:
    """Run ID should have format: test-filename-datetime-hex."""
    manager = HistoryManager.create(["./check.sh"], "example.py")
    parts = manager.run_id.split("-")
    # Expected: check.sh-example.py-YYYYMMDD-HHMMSS-hexhexhexhex
    # After split by "-": [check.sh, example.py, YYYYMMDD, HHMMSS, hex]
    assert len(parts) >= 5
    assert "check.sh" in parts[0]
    assert "example.py" in parts[1]
    # Date part should be 8 digits
    assert len(parts[2]) == 8
    assert parts[2].isdigit()
    # Time part should be 6 digits
    assert len(parts[3]) == 6
    assert parts[3].isdigit()
    # Random hex should be 8 hex chars
    assert len(parts[4]) == 8


def test_create_sets_target_basename() -> None:
    manager = HistoryManager.create(["./test.sh"], "/path/to/buggy.c")
    assert manager.target_basename == "buggy.c"


def test_create_sets_history_dir() -> None:
    manager = HistoryManager.create(["./test.sh"], "buggy.c")
    assert manager.history_dir.endswith(manager.run_id)
    assert ".shrinkray" in manager.history_dir


# === HistoryManager.initialize tests ===


def test_initialize_creates_directory_structure() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original content", ["./test.sh"], "buggy.c")

            assert os.path.isdir(manager.history_dir)
            assert os.path.isdir(os.path.join(manager.history_dir, "initial"))
            assert os.path.isdir(os.path.join(manager.history_dir, "reductions"))
        finally:
            os.chdir(original_cwd)


def test_initialize_copies_target_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            content = b"int main() { return 0; }"
            manager.initialize(content, ["./test.sh"], "buggy.c")

            target_path = os.path.join(manager.history_dir, "initial", "buggy.c")
            assert os.path.isfile(target_path)
            with open(target_path, "rb") as f:
                assert f.read() == content
        finally:
            os.chdir(original_cwd)


def test_initialize_copies_local_test_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Create a test script
            test_script = os.path.join(tmpdir, "check.sh")
            test_content = b"#!/bin/bash\nexit 0\n"
            with open(test_script, "wb") as f:
                f.write(test_content)

            manager = HistoryManager.create([test_script], "buggy.c")
            manager.initialize(b"content", [test_script], "buggy.c")

            copied_test = os.path.join(manager.history_dir, "initial", "check.sh")
            assert os.path.isfile(copied_test)
            with open(copied_test, "rb") as f:
                assert f.read() == test_content
        finally:
            os.chdir(original_cwd)


def test_initialize_creates_wrapper_script() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Create a test script
            test_script = os.path.join(tmpdir, "check.sh")
            with open(test_script, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")

            manager = HistoryManager.create([test_script], "buggy.c")
            manager.initialize(b"content", [test_script], "buggy.c")

            wrapper = os.path.join(manager.history_dir, "initial", "run.sh")
            assert os.path.isfile(wrapper)
            # Check it's executable
            assert os.stat(wrapper).st_mode & stat.S_IXUSR
        finally:
            os.chdir(original_cwd)


def test_initialize_wrapper_script_content_with_local_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Create a test script
            test_script = os.path.join(tmpdir, "check.sh")
            with open(test_script, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")

            manager = HistoryManager.create([test_script, "--flag"], "buggy.c")
            manager.initialize(b"content", [test_script, "--flag"], "buggy.c")

            wrapper = os.path.join(manager.history_dir, "initial", "run.sh")
            with open(wrapper) as f:
                content = f.read()

            # Should reference the copied test via $(dirname "$0")
            assert '"$(dirname "$0")/check.sh"' in content
            # Should include the flag argument
            assert "--flag" in content
            # Should have proper default for TARGET
            assert 'TARGET="${1:-"$DIR/buggy.c"}"' in content
        finally:
            os.chdir(original_cwd)


def test_initialize_wrapper_script_content_without_local_test() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Use a non-existent path (simulating a system command)
            manager = HistoryManager.create(["python", "-c", "pass"], "buggy.py")
            manager.initialize(b"content", ["python", "-c", "pass"], "buggy.py")

            wrapper = os.path.join(manager.history_dir, "initial", "run.sh")
            with open(wrapper) as f:
                content = f.read()

            # Should use the command directly (not $(dirname "$0"))
            assert "python" in content
            assert "$(dirname" not in content or "check.sh" not in content
        finally:
            os.chdir(original_cwd)


def test_initialize_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"content", ["./test.sh"], "buggy.c")
            manager.initialize(
                b"different content", ["./test.sh"], "buggy.c"
            )  # Should not overwrite

            target_path = os.path.join(manager.history_dir, "initial", "buggy.c")
            with open(target_path, "rb") as f:
                assert f.read() == b"content"  # Original content preserved
        finally:
            os.chdir(original_cwd)


# === HistoryManager.record_reduction tests ===


def test_record_reduction_creates_numbered_directories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduction 1")
            manager.record_reduction(b"reduction 2")
            manager.record_reduction(b"reduction 3")

            reductions_dir = os.path.join(manager.history_dir, "reductions")
            assert os.path.isdir(os.path.join(reductions_dir, "0001"))
            assert os.path.isdir(os.path.join(reductions_dir, "0002"))
            assert os.path.isdir(os.path.join(reductions_dir, "0003"))
        finally:
            os.chdir(original_cwd)


def test_record_reduction_writes_reduced_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduced content")

            reduced_file = os.path.join(
                manager.history_dir, "reductions", "0001", "buggy.c"
            )
            with open(reduced_file, "rb") as f:
                assert f.read() == b"reduced content"
        finally:
            os.chdir(original_cwd)


def test_record_reduction_writes_output_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduced", output=b"test output here")

            output_file = os.path.join(
                manager.history_dir, "reductions", "0001", "buggy.c.out"
            )
            with open(output_file, "rb") as f:
                assert f.read() == b"test output here"
        finally:
            os.chdir(original_cwd)


def test_record_reduction_without_output() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduced", output=None)

            output_file = os.path.join(
                manager.history_dir, "reductions", "0001", "buggy.c.out"
            )
            assert not os.path.exists(output_file)
        finally:
            os.chdir(original_cwd)


def test_record_reduction_counter_increments() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            assert manager.reduction_counter == 0
            manager.record_reduction(b"r1")
            assert manager.reduction_counter == 1
            manager.record_reduction(b"r2")
            assert manager.reduction_counter == 2
        finally:
            os.chdir(original_cwd)


# === record_also_interesting tests ===


def test_record_also_interesting_creates_numbered_directories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_also_interesting(b"case 1")
            manager.record_also_interesting(b"case 2")
            manager.record_also_interesting(b"case 3")

            also_interesting_dir = os.path.join(
                manager.history_dir, "also-interesting"
            )
            assert os.path.isdir(os.path.join(also_interesting_dir, "0001"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0002"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0003"))
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_writes_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_also_interesting(b"interesting content")

            saved_file = os.path.join(
                manager.history_dir, "also-interesting", "0001", "buggy.c"
            )
            with open(saved_file, "rb") as f:
                assert f.read() == b"interesting content"
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_writes_output_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_also_interesting(b"case", output=b"test output here")

            output_file = os.path.join(
                manager.history_dir, "also-interesting", "0001", "buggy.c.out"
            )
            with open(output_file, "rb") as f:
                assert f.read() == b"test output here"
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_without_output() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_also_interesting(b"case", output=None)

            output_file = os.path.join(
                manager.history_dir, "also-interesting", "0001", "buggy.c.out"
            )
            assert not os.path.exists(output_file)
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_counter_increments() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            assert manager.also_interesting_counter == 0
            manager.record_also_interesting(b"c1")
            assert manager.also_interesting_counter == 1
            manager.record_also_interesting(b"c2")
            assert manager.also_interesting_counter == 2
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_and_reduction_independent_counters() -> None:
    """Test that also-interesting and reduction counters are independent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_also_interesting(b"a1")
            manager.record_reduction(b"r2")
            manager.record_also_interesting(b"a2")

            assert manager.reduction_counter == 2
            assert manager.also_interesting_counter == 2

            # Both directories should have their respective subdirectories
            reductions_dir = os.path.join(manager.history_dir, "reductions")
            also_interesting_dir = os.path.join(
                manager.history_dir, "also-interesting"
            )
            assert os.path.isdir(os.path.join(reductions_dir, "0001"))
            assert os.path.isdir(os.path.join(reductions_dir, "0002"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0001"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0002"))
        finally:
            os.chdir(original_cwd)


# === record_reductions=False tests ===


def test_create_with_record_reductions_false() -> None:
    """Test that create() accepts record_reductions parameter."""
    manager = HistoryManager.create(
        ["./test.sh"], "buggy.c", record_reductions=False
    )
    assert manager.record_reductions is False


def test_initialize_with_record_reductions_false_skips_reductions_dir() -> None:
    """Test that initialize() doesn't create reductions/ when record_reductions=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(
                ["./test.sh"], "buggy.c", record_reductions=False
            )
            manager.initialize(b"original content", ["./test.sh"], "buggy.c")

            # initial/ should be created
            assert os.path.isdir(os.path.join(manager.history_dir, "initial"))
            # reductions/ should NOT be created
            assert not os.path.exists(os.path.join(manager.history_dir, "reductions"))
        finally:
            os.chdir(original_cwd)


def test_record_reduction_with_record_reductions_false_is_noop() -> None:
    """Test that record_reduction() does nothing when record_reductions=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(
                ["./test.sh"], "buggy.c", record_reductions=False
            )
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            # Call record_reduction multiple times
            manager.record_reduction(b"reduction 1")
            manager.record_reduction(b"reduction 2")

            # Counter should NOT increment
            assert manager.reduction_counter == 0

            # No reductions directory or files should exist
            reductions_dir = os.path.join(manager.history_dir, "reductions")
            assert not os.path.exists(reductions_dir)
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_works_with_record_reductions_false() -> None:
    """Test that record_also_interesting() works even when record_reductions=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(
                ["./test.sh"], "buggy.c", record_reductions=False
            )
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            # record_also_interesting should still work
            manager.record_also_interesting(b"also interesting case")

            assert manager.also_interesting_counter == 1

            # File should exist
            also_interesting_dir = os.path.join(
                manager.history_dir, "also-interesting", "0001"
            )
            assert os.path.isdir(also_interesting_dir)
            with open(os.path.join(also_interesting_dir, "buggy.c"), "rb") as f:
                assert f.read() == b"also interesting case"
        finally:
            os.chdir(original_cwd)


# === get_reduction_content tests ===


def test_get_reduction_content_returns_correct_content() -> None:
    """Test that get_reduction_content returns the file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduction 1")
            manager.record_reduction(b"reduction 2")
            manager.record_reduction(b"reduction 3")

            assert manager.get_reduction_content(1) == b"reduction 1"
            assert manager.get_reduction_content(2) == b"reduction 2"
            assert manager.get_reduction_content(3) == b"reduction 3"
        finally:
            os.chdir(original_cwd)


def test_get_reduction_content_raises_for_nonexistent_reduction() -> None:
    """Test that get_reduction_content raises FileNotFoundError for invalid number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduction 1")

            with pytest.raises(FileNotFoundError):
                manager.get_reduction_content(999)
        finally:
            os.chdir(original_cwd)


# === restart_from_reduction tests ===


def test_restart_from_reduction_returns_content_and_exclusion_set() -> None:
    """Test that restart_from_reduction returns restart content and exclusion set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"reduction 1")
            manager.record_reduction(b"reduction 2")
            manager.record_reduction(b"reduction 3")
            manager.record_reduction(b"reduction 4")

            # Restart from reduction 2
            restart_content, excluded = manager.restart_from_reduction(2)

            # Should return content of reduction 2
            assert restart_content == b"reduction 2"

            # Excluded set should contain reductions 3 and 4
            assert excluded == {b"reduction 3", b"reduction 4"}
        finally:
            os.chdir(original_cwd)


def test_restart_from_reduction_moves_entries_to_also_interesting() -> None:
    """Test that restart_from_reduction moves later reductions to also-interesting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")
            manager.record_reduction(b"r3")
            manager.record_reduction(b"r4")

            # Restart from reduction 2
            manager.restart_from_reduction(2)

            # Reductions 0001 and 0002 should remain
            reductions_dir = os.path.join(manager.history_dir, "reductions")
            assert os.path.isdir(os.path.join(reductions_dir, "0001"))
            assert os.path.isdir(os.path.join(reductions_dir, "0002"))
            # Reductions 0003 and 0004 should be gone
            assert not os.path.exists(os.path.join(reductions_dir, "0003"))
            assert not os.path.exists(os.path.join(reductions_dir, "0004"))

            # They should be in also-interesting now
            also_interesting_dir = os.path.join(manager.history_dir, "also-interesting")
            assert os.path.isdir(os.path.join(also_interesting_dir, "0001"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0002"))

            # Verify content is correct
            with open(
                os.path.join(also_interesting_dir, "0001", "buggy.c"), "rb"
            ) as f:
                assert f.read() == b"r3"
            with open(
                os.path.join(also_interesting_dir, "0002", "buggy.c"), "rb"
            ) as f:
                assert f.read() == b"r4"
        finally:
            os.chdir(original_cwd)


def test_restart_from_reduction_updates_counters() -> None:
    """Test that restart_from_reduction updates both counters correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")
            manager.record_reduction(b"r3")
            manager.record_reduction(b"r4")

            assert manager.reduction_counter == 4
            assert manager.also_interesting_counter == 0

            # Restart from reduction 2
            manager.restart_from_reduction(2)

            # Reduction counter should be reset to 2
            assert manager.reduction_counter == 2
            # Also-interesting counter should be 2 (moved 2 entries)
            assert manager.also_interesting_counter == 2
        finally:
            os.chdir(original_cwd)


def test_restart_from_reduction_preserves_restart_entry() -> None:
    """Test that restart_from_reduction doesn't move the restart entry itself."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")
            manager.record_reduction(b"r3")

            # Restart from reduction 2
            manager.restart_from_reduction(2)

            # Reduction 2 should still be in reductions
            reduction_file = os.path.join(
                manager.history_dir, "reductions", "0002", "buggy.c"
            )
            assert os.path.isfile(reduction_file)
            with open(reduction_file, "rb") as f:
                assert f.read() == b"r2"
        finally:
            os.chdir(original_cwd)


def test_restart_from_reduction_handles_existing_also_interesting() -> None:
    """Test restart when there are already also-interesting entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            # Add some also-interesting entries first
            manager.record_also_interesting(b"ai1")
            manager.record_also_interesting(b"ai2")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")
            manager.record_reduction(b"r3")

            assert manager.also_interesting_counter == 2

            # Restart from reduction 1
            manager.restart_from_reduction(1)

            # Should have 4 also-interesting entries now (2 original + 2 moved)
            assert manager.also_interesting_counter == 4

            # New entries should be numbered 0003 and 0004
            also_interesting_dir = os.path.join(manager.history_dir, "also-interesting")
            assert os.path.isdir(os.path.join(also_interesting_dir, "0003"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0004"))
        finally:
            os.chdir(original_cwd)


def test_restart_from_reduction_raises_for_nonexistent_reduction() -> None:
    """Test that restart_from_reduction raises FileNotFoundError for invalid number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")

            with pytest.raises(FileNotFoundError):
                manager.restart_from_reduction(999)
        finally:
            os.chdir(original_cwd)


def test_restart_from_last_reduction_returns_empty_exclusion_set() -> None:
    """Test restart from the last reduction returns empty exclusion set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")
            manager.record_reduction(b"r3")

            # Restart from the last reduction
            restart_content, excluded = manager.restart_from_reduction(3)

            assert restart_content == b"r3"
            assert excluded == set()  # Nothing to exclude
        finally:
            os.chdir(original_cwd)


# === Directory mode tests ===


def test_initialize_directory_creates_structure() -> None:
    """Test that initialize_directory creates the correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            # Create a test script
            test_script = os.path.join(tmpdir, "test.sh")
            with open(test_script, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(test_script, 0o755)

            manager = HistoryManager.create(
                [test_script], "target_dir", is_directory=True
            )
            initial_content = {"file1.txt": b"hello", "subdir/file2.txt": b"world"}
            manager.initialize_directory(initial_content, [test_script], "target_dir")

            assert manager.initialized
            assert manager.is_directory

            # Check directory structure
            initial_dir = os.path.join(manager.history_dir, "initial", "target_dir")
            assert os.path.isdir(initial_dir)
            assert os.path.isfile(os.path.join(initial_dir, "file1.txt"))
            assert os.path.isfile(os.path.join(initial_dir, "subdir", "file2.txt"))

            with open(os.path.join(initial_dir, "file1.txt"), "rb") as f:
                assert f.read() == b"hello"
        finally:
            os.chdir(original_cwd)


def test_initialize_directory_skips_if_already_initialized() -> None:
    """Test that initialize_directory is idempotent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )
            initial1 = {"file.txt": b"first"}
            initial2 = {"file.txt": b"second"}

            manager.initialize_directory(initial1, ["./test.sh"], "target_dir")
            manager.initialize_directory(initial2, ["./test.sh"], "target_dir")

            # Should still have first content
            initial_dir = os.path.join(manager.history_dir, "initial", "target_dir")
            with open(os.path.join(initial_dir, "file.txt"), "rb") as f:
                assert f.read() == b"first"
        finally:
            os.chdir(original_cwd)


def test_record_reduction_in_directory_mode() -> None:
    """Test record_reduction works in directory mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )
            manager.initialize_directory(
                {"file.txt": b"original"}, ["./test.sh"], "target_dir"
            )

            # Serialize directory content as the caller would
            reduced = {"file.txt": b"reduced"}
            serialized = json.dumps(
                {k: base64.b64encode(v).decode() for k, v in sorted(reduced.items())},
                sort_keys=True,
            ).encode()

            manager.record_reduction(serialized)

            assert manager.reduction_counter == 1

            # Check directory was written
            reduction_dir = os.path.join(
                manager.history_dir, "reductions", "0001", "target_dir"
            )
            assert os.path.isdir(reduction_dir)
            with open(os.path.join(reduction_dir, "file.txt"), "rb") as f:
                assert f.read() == b"reduced"
        finally:
            os.chdir(original_cwd)


def test_record_also_interesting_in_directory_mode() -> None:
    """Test record_also_interesting works in directory mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )
            manager.initialize_directory(
                {"file.txt": b"original"}, ["./test.sh"], "target_dir"
            )

            # Serialize directory content
            also_int = {"file.txt": b"also-interesting"}
            serialized = json.dumps(
                {k: base64.b64encode(v).decode() for k, v in sorted(also_int.items())},
                sort_keys=True,
            ).encode()

            manager.record_also_interesting(serialized)

            assert manager.also_interesting_counter == 1

            # Check directory was written
            also_int_dir = os.path.join(
                manager.history_dir, "also-interesting", "0001", "target_dir"
            )
            assert os.path.isdir(also_int_dir)
            with open(os.path.join(also_int_dir, "file.txt"), "rb") as f:
                assert f.read() == b"also-interesting"
        finally:
            os.chdir(original_cwd)


def test_read_directory_content() -> None:
    """Test _read_directory_content reads directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )

            # Create directory structure
            target = os.path.join(tmpdir, "test_target")
            os.makedirs(os.path.join(target, "subdir"))
            with open(os.path.join(target, "file1.txt"), "wb") as f:
                f.write(b"content1")
            with open(os.path.join(target, "subdir", "file2.txt"), "wb") as f:
                f.write(b"content2")

            content = manager._read_directory_content(target)

            assert content == {
                "file1.txt": b"content1",
                "subdir/file2.txt": b"content2",
            }
        finally:
            os.chdir(original_cwd)


def test_deserialize_directory() -> None:
    """Test _deserialize_directory parses serialized content."""
    original = {"file.txt": b"hello\x00world", "other.bin": b"\xff\xfe"}
    serialized = json.dumps(
        {k: base64.b64encode(v).decode() for k, v in sorted(original.items())},
        sort_keys=True,
    ).encode()

    result = HistoryManager._deserialize_directory(serialized)
    assert result == original


def test_restart_from_reduction_directory_mode() -> None:
    """Test restart_from_reduction works in directory mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )
            manager.initialize_directory(
                {"file.txt": b"original"}, ["./test.sh"], "target_dir"
            )

            # Serialize directory content
            def serialize(content):
                return json.dumps(
                    {
                        k: base64.b64encode(v).decode()
                        for k, v in sorted(content.items())
                    },
                    sort_keys=True,
                ).encode()

            # Record some reductions
            r1_content = {"file.txt": b"r1"}
            r2_content = {"file.txt": b"r2"}
            r3_content = {"file.txt": b"r3"}
            manager.record_reduction(serialize(r1_content))
            manager.record_reduction(serialize(r2_content))
            manager.record_reduction(serialize(r3_content))

            # Restart from reduction 1
            restart_content, excluded = manager.restart_from_reduction(1)

            # restart_content should be serialized directory content
            assert restart_content == serialize(r1_content)
            assert len(excluded) == 2  # r2 and r3 should be excluded
            assert serialize(r2_content) in excluded
            assert serialize(r3_content) in excluded

            # Reductions 2 and 3 should be moved to also-interesting
            also_interesting_dir = os.path.join(manager.history_dir, "also-interesting")
            assert os.path.isdir(os.path.join(also_interesting_dir, "0001"))
            assert os.path.isdir(os.path.join(also_interesting_dir, "0002"))
        finally:
            os.chdir(original_cwd)


def test_restart_skips_non_numeric_entries() -> None:
    """Test that restart_from_reduction skips non-numeric directory entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            manager = HistoryManager.create(["./test.sh"], "buggy.c")
            manager.initialize(b"original", ["./test.sh"], "buggy.c")

            manager.record_reduction(b"r1")
            manager.record_reduction(b"r2")

            # Create a non-numeric directory in reductions
            reductions_dir = os.path.join(manager.history_dir, "reductions")
            os.makedirs(os.path.join(reductions_dir, "readme"))
            with open(os.path.join(reductions_dir, "readme", "notes.txt"), "w") as f:
                f.write("These are reduction notes")

            # Restart should work despite non-numeric entry
            restart_content, excluded = manager.restart_from_reduction(1)

            assert restart_content == b"r1"
            assert len(excluded) == 1  # Only r2
        finally:
            os.chdir(original_cwd)


def test_write_directory_content_with_nested_paths() -> None:
    """Test _write_directory_content handles nested paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True
            )

            target = os.path.join(tmpdir, "test_target")
            content = {
                "a/b/c/deep.txt": b"deep content",
                "file.txt": b"root content",
            }

            manager._write_directory_content(target, content)

            assert os.path.isfile(os.path.join(target, "file.txt"))
            assert os.path.isfile(os.path.join(target, "a", "b", "c", "deep.txt"))

            with open(os.path.join(target, "a", "b", "c", "deep.txt"), "rb") as f:
                assert f.read() == b"deep content"
        finally:
            os.chdir(original_cwd)


def test_initialize_directory_without_record_reductions() -> None:
    """Test initialize_directory with record_reductions=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            manager = HistoryManager.create(
                ["./test.sh"], "target_dir", is_directory=True, record_reductions=False
            )
            initial_content = {"file.txt": b"content"}
            manager.initialize_directory(initial_content, ["./test.sh"], "target_dir")

            assert manager.initialized
            # reductions directory should NOT be created
            assert not os.path.exists(
                os.path.join(manager.history_dir, "reductions")
            )
            # But initial directory should still exist
            assert os.path.isdir(os.path.join(manager.history_dir, "initial"))
        finally:
            os.chdir(original_cwd)
