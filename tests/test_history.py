"""Tests for shrinkray.history module."""

from __future__ import annotations

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
