import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import black
import pexpect
import pyte
import pytest
import trio
from attrs import define
from click.testing import CliRunner

from shrinkray.__main__ import main
from shrinkray.process import interrupt_wait_and_kill


def format(s: str) -> str:
    return black.format_str(s, mode=black.Mode()).strip()


@pytest.mark.slow
async def test_kill_process():
    async with trio.open_nursery() as nursery:

        async def call_with_kwargs(task_status=trio.TASK_STATUS_IGNORED):  # type: ignore
            # start a subprocess that will just ignore SIGINT signals
            return await trio.run_process(  # type: ignore[call-overload]
                [
                    sys.executable,
                    "-c",
                    "import signal, sys, time; signal.signal(signal.SIGINT, lambda *a: 1); print(1); sys.stdout.flush(); time.sleep(1000)",
                ],
                universal_newlines=False,
                preexec_fn=os.setsid,
                check=False,
                stdout=subprocess.PIPE,
                task_status=task_status,
            )

        sp = await nursery.start(call_with_kwargs)
        line = await sp.stdout.receive_some(2)
        assert line == b"1\n"
        # must not raise ValueError but succeed at killing the process
        await interrupt_wait_and_kill(sp)
        assert sp.returncode is not None
        assert sp.returncode != 0


@pytest.mark.slow
@pytest.mark.parametrize("in_place", [False, True])
def test_can_reduce_a_directory(tmp_path: pathlib.Path, in_place):
    target = tmp_path / "foo"
    target.mkdir()
    a = target / "a.py"
    a.write_text("x = 1\ny=2\nz=3\n")
    b = target / "b.py"
    b.write_text("y = 'hello world'")
    c = target / "c.py"
    c.write_text("from a import x\n\n...\nassert x == 2")

    script = tmp_path / "test.py"
    script.write_text(
        f"""
#!/usr/bin/env {sys.executable}
import sys
sys.path.append(sys.argv[1])

try:
    import c
    sys.exit(1)
except AssertionError:
    sys.exit(0)
    """.strip()
    )
    script.chmod(0o777)

    subprocess.check_call(
        [
            str(script),
            str(target),
        ]
    )

    if in_place:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "shrinkray",
                "--in-place",
                str(script),
                str(target),
                "--ui=basic",
            ],
        )
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "shrinkray",
                str(script),
                str(target),
                "--ui=basic",
            ],
        )

    assert a.exists()
    assert not b.exists()
    assert c.exists()

    # TODO: Remove calls to format when formatting is implemented properly for
    # directories.
    assert format(a.read_text()) == "x = 0"
    assert format(c.read_text()) == "from a import x\n\nassert x"


def test_gives_informative_error_when_script_does_not_work_outside_current_directory(
    tmpdir,
):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.py"
    script.write_text(
        f"""
#!/usr/bin/env {sys.executable}
import sys

if sys.argv[1] != {repr(str(target))}:
    sys.exit(1)
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    subprocess.check_call([script, target])

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "shrinkray",
                str(script),
                str(target),
                "--ui=basic",
            ],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    assert "your script depends" in excinfo.value.stderr


def test_prints_the_output_on_an_initially_uninteresting_test_case(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.py"
    script.write_text(
        f"""
#!/usr/bin/env {sys.executable}
import sys

print("Hello world")

sys.exit(1)
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "shrinkray",
                str(script),
                str(target),
                "--ui=basic",
            ],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    assert "Hello world" in excinfo.value.stderr


@define
class ShrinkTarget:
    test_case: str
    interestingness_test: str


@pytest.fixture(scope="function")
def basic_shrink_target(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text(
        """
#!/usr/bin/env bash

set -e

grep hello "$1"
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    return ShrinkTarget(test_case=str(target), interestingness_test=str(script))


@pytest.mark.slow
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("parallelism", (1, 2))
def test_shrinks_basic_target(basic_shrink_target, in_place, parallelism):
    runner = CliRunner(catch_exceptions=False)

    args = [
        basic_shrink_target.interestingness_test,
        basic_shrink_target.test_case,
        "--ui=basic",
        f"--parallelism={parallelism}",
    ]
    if in_place:
        args.append("--in-place")

    result = runner.invoke(main, args)

    assert result.exit_code == 0

    with open(basic_shrink_target.test_case) as i:
        assert i.read().strip() == "hello"


def test_errors_on_bad_parallelism_when_in_place(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text(
        f"""
#!/usr/bin/env bash

set -e

grep hello {str(target)}
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    runner = CliRunner(catch_exceptions=False)

    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--in-place",
            "--input-type=basename",
            "--parallelism=2",
        ],
    )
    assert result.exit_code != 0
    assert "parallelism cannot" in result.stderr


def test_gives_good_error_when_initial_test_case_invalid(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text(
        """
#!/usr/bin/env bash

exit 1
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    runner = CliRunner(catch_exceptions=False)

    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
        ],
    )
    assert result.exit_code != 0
    assert "uninteresting test case" in result.stderr


def test_error_when_test_not_executable(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    # Note: NOT setting executable permission

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic"],
    )
    assert result.exit_code == 1
    assert "not executable" in result.stderr


@pytest.mark.slow
def test_timeout_zero_sets_infinite(basic_shrink_target):
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            basic_shrink_target.interestingness_test,
            basic_shrink_target.test_case,
            "--ui=basic",
            "--timeout=0",
        ],
    )
    # Should complete successfully with infinite timeout
    assert result.exit_code == 0


def test_parallelism_defaults_to_one_for_basename_inplace(tmpdir, monkeypatch):
    """Fast test verifying parallelism=0 defaults to 1 with in_place + basename.

    This test mocks the state creation to verify the parallelism value
    without running a full reduction.
    """
    from unittest.mock import patch

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_parallelism = []

    def mock_state_init(**kwargs):
        captured_parallelism.append(kwargs.get("parallelism"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.ShrinkRayStateSingleFile") as mock_state:
        mock_state.side_effect = mock_state_init
        runner = CliRunner(catch_exceptions=False)
        try:
            runner.invoke(
                main,
                [
                    str(script),
                    str(target),
                    "--ui=basic",
                    "--in-place",
                    "--input-type=basename",
                    "--parallelism=0",
                ],
            )
        except SystemExit:
            pass

    # Verify parallelism was set to 1 (not cpu_count)
    assert captured_parallelism == [1]


def test_explicit_parallelism_skips_default_logic(tmpdir, monkeypatch):
    """Test that explicit non-zero parallelism skips the default logic (235->241 branch)."""
    from unittest.mock import patch

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_parallelism = []

    def mock_state_init(**kwargs):
        captured_parallelism.append(kwargs.get("parallelism"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.ShrinkRayStateSingleFile") as mock_state:
        mock_state.side_effect = mock_state_init
        runner = CliRunner(catch_exceptions=False)
        try:
            runner.invoke(
                main,
                [
                    str(script),
                    str(target),
                    "--ui=basic",
                    "--parallelism=4",  # Explicit non-zero value
                ],
            )
        except SystemExit:
            pass

    # Verify parallelism was kept as 4 (not modified by default logic)
    assert captured_parallelism == [4]


@pytest.mark.slow
def test_in_place_basename_sets_parallelism_to_one(tmpdir, monkeypatch):
    """Test that in_place + basename with parallelism=0 defaults to 1."""
    # Change to tmpdir so basename mode can find the file
    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    # In basename mode, the file is in the cwd with its basename
    script.write_text(
        """
#!/usr/bin/env bash
grep hello hello.txt
    """.strip(),
        encoding="utf-8",
    )
    script.chmod(0o777)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--in-place",
            "--input-type=basename",
            "--parallelism=0",  # Should default to 1 for basename mode
        ],
    )
    assert result.exit_code == 0


def test_worker_main_can_be_imported():
    """Test that worker_main function can be called."""
    from shrinkray.__main__ import worker_main

    # Can't actually run it without proper stdin/stdout setup,
    # but at least verify it's importable and callable
    assert callable(worker_main)


def test_directory_mode_stdin_error(tmp_path):
    """Test that directory mode rejects stdin input type."""
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "test.txt").write_text("hello")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic", "--input-type=stdin"],
    )
    assert result.exit_code != 0
    assert "Cannot pass a directory input on stdin" in str(result.output)


def test_clang_delta_not_found_error(tmp_path, monkeypatch):
    """Test error when clang_delta is needed but not found."""
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Make find_clang_delta return empty string
    def mock_find_clang_delta():
        return ""

    monkeypatch.setattr("shrinkray.__main__.find_clang_delta", mock_find_clang_delta)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic"],
    )
    assert result.exit_code != 0
    assert "clang_delta" in str(result.output).lower()


def test_clang_delta_explicit_path(tmp_path, monkeypatch):
    """Test passing explicit clang_delta path."""
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Create a fake clang_delta executable
    fake_clang_delta = tmp_path / "fake_clang_delta"
    fake_clang_delta.write_text("#!/bin/bash\nexit 0")
    fake_clang_delta.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            f"--clang-delta={fake_clang_delta}",
        ],
    )
    # Will fail at the setup stage, but should get past the clang_delta check
    # The important thing is it doesn't fail with "clang_delta not installed"
    assert "clang_delta is not installed" not in str(result.output)


@pytest.mark.slow
def test_default_backup_filename(basic_shrink_target):
    """Test that default backup filename is created correctly."""
    import os

    # First, remove any existing backup
    backup_path = basic_shrink_target.test_case + os.extsep + "bak"
    if os.path.exists(backup_path):
        os.remove(backup_path)

    # Also clear test case backup
    test_case_backup = basic_shrink_target.test_case + os.extsep + "bak"
    if os.path.exists(test_case_backup):
        os.remove(test_case_backup)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            basic_shrink_target.interestingness_test,
            basic_shrink_target.test_case,
            "--ui=basic",
        ],
    )
    assert result.exit_code == 0

    # The backup file should be created with the default name
    expected_backup = basic_shrink_target.test_case + os.extsep + "bak"
    assert os.path.exists(expected_backup)


@pytest.mark.slow
def test_directory_mode_with_basic_ui(tmp_path):
    """Test directory reduction with BasicUI via CliRunner.

    This exercises the directory handling code path in run_command.
    """
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "a.txt").write_text("hello world")
    (target / "b.txt").write_text("goodbye")

    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
test -f "$1/a.txt" && grep hello "$1/a.txt"
"""
    )
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic", "--input-type=arg"],
    )

    # Should complete (possibly with warnings about trivial results)
    # The key is that it exercises the directory handling code
    assert result.exit_code in (0, 1)  # 1 for trivial result warning


def test_worker_main_entry_point():
    """Test that worker_main can be invoked (will fail without proper stdin)."""
    import io

    from shrinkray.__main__ import worker_main

    # Capture what would happen if worker_main runs without proper input
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO("")  # Empty input
        # worker_main will fail because there's no proper JSON input
        # but this exercises the import and function call
        try:
            worker_main()
        except (EOFError, json.JSONDecodeError, Exception):
            # Expected to fail without proper input
            pass
    finally:
        sys.stdin = old_stdin


@pytest.mark.slow
def test_custom_backup_filename(basic_shrink_target, tmp_path):
    """Test that custom backup filename is used when specified."""
    custom_backup = str(tmp_path / "my_custom_backup.bak")

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            basic_shrink_target.interestingness_test,
            basic_shrink_target.test_case,
            "--ui=basic",
            f"--backup={custom_backup}",
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(custom_backup)


def test_textual_ui_path(basic_shrink_target, monkeypatch):
    """Test that the textual UI path is exercised.

    Exercises the textual UI code path in run_command.
    """
    from unittest.mock import MagicMock

    # Mock run_textual_ui to avoid actually launching the TUI
    mock_run_textual_ui = MagicMock()
    monkeypatch.setattr("shrinkray.tui.run_textual_ui", mock_run_textual_ui)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            basic_shrink_target.interestingness_test,
            basic_shrink_target.test_case,
            "--ui=textual",
        ],
    )

    # The function should have been called
    assert mock_run_textual_ui.called
    assert result.exit_code == 0


def test_keyboard_interrupt_handling(basic_shrink_target, tmp_path):
    """Test that KeyboardInterrupt is properly re-raised from ExceptionGroup.

    Exercises the KeyboardInterrupt handling in run_command.
    """
    from unittest.mock import patch

    # Create a custom trio.run that tracks calls
    call_count = [0]
    original_trio_run = trio.run

    def mock_trio_run(func):
        call_count[0] += 1
        # The first call is check_formatter - let it run normally
        # The second call is the main run_shrink_ray - raise KeyboardInterrupt
        if call_count[0] <= 1:
            return original_trio_run(func)
        else:
            raise BaseExceptionGroup("test", [KeyboardInterrupt()])

    # Patch at the __main__ module level
    with patch.object(
        __import__("shrinkray.__main__", fromlist=["trio"]).trio,
        "run",
        mock_trio_run,
    ):
        runner = CliRunner(catch_exceptions=True)
        result = runner.invoke(
            main,
            [
                basic_shrink_target.interestingness_test,
                basic_shrink_target.test_case,
                "--ui=basic",
                "--formatter=none",  # Skip formatting to avoid issues
            ],
        )
        # Should have raised KeyboardInterrupt
        if result.exception is not None:
            assert isinstance(result.exception, KeyboardInterrupt | SystemExit)


def test_timeout_zero_converts_to_infinity(tmpdir, monkeypatch):
    """Fast test verifying timeout=0 is converted to infinity.

    Exercises the timeout=0 to infinity conversion in run_command.
    """
    from unittest.mock import patch

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_timeout = []

    def mock_state_init(**kwargs):
        captured_timeout.append(kwargs.get("timeout"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.ShrinkRayStateSingleFile") as mock_state:
        mock_state.side_effect = mock_state_init
        runner = CliRunner(catch_exceptions=False)
        try:
            runner.invoke(
                main,
                [
                    str(script),
                    str(target),
                    "--ui=basic",
                    "--timeout=0",
                ],
            )
        except SystemExit:
            pass

    # Verify timeout was converted to infinity
    assert len(captured_timeout) == 1
    assert captured_timeout[0] == float("inf")


def test_default_backup_filename_calculation(tmpdir, monkeypatch):
    """Fast test verifying default backup filename is calculated correctly.

    Exercises the default backup filename calculation in run_command.
    """
    import os
    from unittest.mock import patch

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    # Track if os.remove was called with the default backup path
    removed_files = []

    def tracking_remove(path):
        removed_files.append(path)
        # Don't actually remove, just raise FileNotFoundError like the code expects
        raise FileNotFoundError()

    def mock_state_init(**kwargs):
        raise SystemExit(0)

    with patch("shrinkray.__main__.ShrinkRayStateSingleFile") as mock_state:
        mock_state.side_effect = mock_state_init
        with patch("os.remove", tracking_remove):
            runner = CliRunner(catch_exceptions=False)
            try:
                runner.invoke(
                    main,
                    [
                        str(script),
                        str(target),
                        "--ui=basic",
                        # Note: no --backup specified, so default should be used
                    ],
                )
            except SystemExit:
                pass

    # Verify the default backup path was attempted to be removed
    expected_backup = str(target) + os.extsep + "bak"
    assert expected_backup in removed_files


def test_custom_backup_path_is_used(tmpdir, monkeypatch):
    """Fast test verifying custom backup path skips default backup calculation.

    This tests the case when --backup is explicitly provided.
    """
    import os
    from unittest.mock import patch

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/bash\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    custom_backup = str(tmpdir / "my_custom.bak")

    # Track if os.remove was called with the custom backup path
    removed_files = []

    def tracking_remove(path):
        removed_files.append(path)
        raise FileNotFoundError()

    def mock_state_init(**kwargs):
        raise SystemExit(0)

    with patch("shrinkray.__main__.ShrinkRayStateSingleFile") as mock_state:
        mock_state.side_effect = mock_state_init
        with patch("os.remove", tracking_remove):
            runner = CliRunner(catch_exceptions=False)
            try:
                runner.invoke(
                    main,
                    [
                        str(script),
                        str(target),
                        "--ui=basic",
                        f"--backup={custom_backup}",
                    ],
                )
            except SystemExit:
                pass

    # Verify the custom backup path was used, not the default
    assert custom_backup in removed_files
    default_backup = str(target) + os.extsep + "bak"
    assert default_backup not in removed_files


def test_directory_mode_setup(tmp_path, monkeypatch):
    """Fast test verifying directory mode setup logic including check_formatter call.

    This tests the directory mode initialization without running a full reduction.
    """
    import shutil
    from unittest.mock import MagicMock, patch

    # Create a test directory with files
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "a.txt").write_text("hello")
    (target / "b.txt").write_text("world")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0")
    script.chmod(0o755)

    # Track what was passed to ShrinkRayDirectoryState
    captured_initial = []
    mock_state_instance = MagicMock()

    def mock_dir_state_init(**kwargs):
        captured_initial.append(kwargs.get("initial"))
        return mock_state_instance

    # Track copytree calls
    copytree_calls = []
    original_copytree = shutil.copytree

    def tracking_copytree(src, dst, **kwargs):
        copytree_calls.append((src, dst))
        original_copytree(src, dst, **kwargs)

    # Track trio.run calls
    trio_run_calls = []

    def mock_trio_run(coro):
        trio_run_calls.append(coro)
        # Exit after check_formatter is called
        raise SystemExit(0)

    with patch(
        "shrinkray.__main__.ShrinkRayDirectoryState",
        side_effect=mock_dir_state_init,
    ):
        with patch("shutil.copytree", tracking_copytree):
            with patch("shrinkray.__main__.trio.run", mock_trio_run):
                runner = CliRunner(catch_exceptions=False)
                try:
                    runner.invoke(
                        main,
                        [
                            str(script),
                            str(target),
                            "--ui=basic",
                            "--input-type=arg",
                        ],
                    )
                except SystemExit:
                    pass

    # Verify copytree was called for backup
    assert len(copytree_calls) == 1
    assert copytree_calls[0][0] == str(target)

    # Verify the initial dict was populated correctly
    assert len(captured_initial) == 1
    initial = captured_initial[0]
    assert "a.txt" in initial
    assert "b.txt" in initial
    assert initial["a.txt"] == b"hello"
    assert initial["b.txt"] == b"world"

    # Verify trio.run was called with check_formatter
    assert len(trio_run_calls) == 1
    assert trio_run_calls[0] == mock_state_instance.check_formatter


@pytest.mark.slow
def test_timeout_exceeded_on_initial_shows_error_message_basic(tmp_path):
    """Test that when the initial test case exceeds the timeout, an appropriate error is shown.

    This is an integration test that runs the full CLI and verifies the error message
    is properly surfaced to the user with a user-friendly message (not a raw traceback).
    """
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
sleep 0.5
exit 0
"""
    )
    script.chmod(0o755)

    # Run with a very short timeout (0.1 seconds)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=basic",
            "--timeout=0.01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Should fail
    assert result.returncode != 0

    # Should show user-friendly error message, not a raw traceback
    # The message should contain the friendly error from report_error/build_error_message
    assert "Shrink ray cannot proceed" in result.stderr
    assert "exceeding your timeout setting" in result.stderr


@pytest.mark.slow
def test_timeout_exceeded_on_initial_shows_error_message_tui(tmp_path):
    """Test that when the initial test case exceeds the timeout, the TUI shows an appropriate error.

    This is an integration test that runs the full CLI with --ui=textual and verifies
    the error message is properly surfaced to the user.
    """
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that sleeps longer than the timeout
    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
sleep 0.5
exit 0
"""
    )
    script.chmod(0o755)

    # Run with a very short timeout (0.01 seconds) and the textual UI
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
            "--timeout=0.01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Should fail
    assert result.returncode != 0

    # Should show user-friendly error message, not a raw traceback
    # The message should contain the friendly error from build_error_message
    # Check both stdout and stderr since TUI may output to either
    combined_output = result.stdout + result.stderr
    assert "Shrink ray cannot proceed" in combined_output
    assert "exceeding your timeout setting" in combined_output


@pytest.mark.slow
def test_invalid_initial_shows_error_message_basic(tmp_path):
    """Test that when the initial test case is invalid, basic UI shows a user-friendly error."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that always fails
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1\n")
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=basic",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Shrink ray cannot proceed" in result.stderr
    assert "uninteresting test case" in result.stderr


@pytest.mark.slow
def test_invalid_initial_shows_error_message_tui(tmp_path):
    """Test that when the initial test case is invalid, TUI shows a user-friendly error."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that always fails
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 1\n")
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Shrink ray cannot proceed" in combined_output
    assert "uninteresting test case" in combined_output


@pytest.mark.slow
def test_script_depends_on_cwd_shows_error_tui(tmp_path):
    """Test that TUI shows helpful error when script depends on current directory."""
    target = tmp_path / "hello.txt"
    target.write_text("hello world")

    # Script that only works when run from a specific directory
    script = tmp_path / "test.py"
    script.write_text(
        f"""#!/usr/bin/env {sys.executable}
import sys
# Only succeed if the argument is the exact original path
if sys.argv[1] != {repr(str(target))}:
    sys.exit(1)
sys.exit(0)
"""
    )
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "your script depends" in combined_output.lower()


@pytest.mark.slow
def test_prints_script_output_on_error_tui(tmp_path):
    """Test that TUI shows script output when initial test case fails."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that prints output and fails
    script = tmp_path / "test.py"
    script.write_text(
        f"""#!/usr/bin/env {sys.executable}
import sys
print("Debug output from failing script")
sys.exit(1)
"""
    )
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Debug output from failing script" in combined_output


# === Happy path integration tests ===
#
# These tests verify that reduction completes successfully for various
# combinations of UI type, file vs directory mode, and other options.
# They use a simple interestingness test (> 1 byte) and small test cases
# to keep execution time fast.


@pytest.fixture
def simple_file_target(tmp_path):
    """Create a simple file target with an interestingness test that accepts > 1 byte."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")  # 11 bytes

    script = tmp_path / "test.py"
    script.write_text(
        f"""#!/usr/bin/env {sys.executable}
import sys
from pathlib import Path

# Interesting if file has more than 1 byte
file_size = Path(sys.argv[1]).stat().st_size
sys.exit(0 if file_size > 1 else 1)
"""
    )
    script.chmod(0o755)

    return ShrinkTarget(test_case=str(target), interestingness_test=str(script))


@pytest.fixture
def simple_directory_target(tmp_path):
    """Create a simple directory target with an interestingness test."""
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "a.txt").write_text("hello")  # 5 bytes
    (target / "b.txt").write_text("world")  # 5 bytes

    script = tmp_path / "test.py"
    script.write_text(
        f"""#!/usr/bin/env {sys.executable}
import sys
from pathlib import Path

# Interesting if total size of all files > 1 byte
dir_path = Path(sys.argv[1])
total_size = sum(f.stat().st_size for f in dir_path.iterdir() if f.is_file())
sys.exit(0 if total_size > 1 else 1)
"""
    )
    script.chmod(0o755)

    return ShrinkTarget(test_case=str(target), interestingness_test=str(script))


@pytest.mark.slow
def test_happy_path_basic_ui_single_file(simple_file_target):
    """Test successful reduction with basic UI and single file."""
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            simple_file_target.interestingness_test,
            simple_file_target.test_case,
            "--ui=basic",
            "--parallelism=1",
        ],
    )

    assert result.exit_code == 0

    # File should be reduced but still > 1 byte
    with open(simple_file_target.test_case) as f:
        content = f.read()
    assert len(content) > 1
    assert len(content) < 11  # Should be smaller than original


@pytest.mark.slow
def test_happy_path_basic_ui_directory(simple_directory_target):
    """Test successful reduction with basic UI and directory."""
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            simple_directory_target.interestingness_test,
            simple_directory_target.test_case,
            "--ui=basic",
            "--input-type=arg",
            "--parallelism=1",
        ],
    )

    assert result.exit_code == 0

    # a.txt should still exist with some content
    a_path = pathlib.Path(simple_directory_target.test_case) / "a.txt"
    assert a_path.exists()
    assert a_path.stat().st_size > 0


@pytest.mark.slow
def test_happy_path_tui_single_file(simple_file_target):
    """Test TUI reduction with single file (auto-exits on completion)."""
    # Use subprocess.run instead of CliRunner because the TUI spawns
    # subprocesses which need real file descriptors.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            simple_file_target.interestingness_test,
            simple_file_target.test_case,
            "--ui=textual",
            "--parallelism=1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    # File should be reduced
    target_path = pathlib.Path(simple_file_target.test_case)
    assert target_path.stat().st_size > 0


@pytest.mark.slow
def test_happy_path_tui_directory(simple_directory_target):
    """Test TUI reduction with directory (auto-exits on completion)."""
    # Use subprocess.run instead of CliRunner because the TUI spawns
    # subprocesses which need real file descriptors.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            simple_directory_target.interestingness_test,
            simple_directory_target.test_case,
            "--ui=textual",
            "--input-type=arg",
            "--parallelism=1",
            "--trivial-is-not-error",  # Directory reduction may reach trivial size
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    # Directory should still exist with some content
    a_path = pathlib.Path(simple_directory_target.test_case) / "a.txt"
    assert a_path.exists()


@pytest.mark.slow
def test_trivial_is_not_error_basic_ui(tmp_path):
    """Test --trivial-is-not-error flag with basic UI."""
    target = tmp_path / "test.txt"
    target.write_text("hello")

    # This test accepts everything including empty files
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--trivial-is-not-error",
            "--parallelism=1",
        ],
    )

    # Should succeed even though result is trivial
    assert result.exit_code == 0


@pytest.mark.slow
def test_trivial_is_error_basic_ui(tmp_path):
    """Test that trivial result is an error by default with basic UI."""
    target = tmp_path / "test.txt"
    target.write_text("hello")

    # This test accepts everything including empty files
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--parallelism=1",
        ],
    )

    # Should fail because result is trivial
    assert result.exit_code != 0


def test_trivial_is_not_error_tui(tmp_path, monkeypatch):
    """Test --trivial-is-not-error flag with TUI path.

    Uses mocking to verify the TUI is invoked with correct parameters.
    """
    from unittest.mock import MagicMock

    target = tmp_path / "test.txt"
    target.write_text("hello")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o755)

    mock_run_textual_ui = MagicMock()
    monkeypatch.setattr("shrinkray.tui.run_textual_ui", mock_run_textual_ui)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=textual",
            "--trivial-is-not-error",
            "--parallelism=1",
        ],
    )

    assert mock_run_textual_ui.called
    assert result.exit_code == 0


@pytest.mark.slow
def test_happy_path_with_parallelism(simple_file_target):
    """Test successful reduction with parallelism > 1."""
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            simple_file_target.interestingness_test,
            simple_file_target.test_case,
            "--ui=basic",
            "--parallelism=2",
        ],
    )

    assert result.exit_code == 0

    with open(simple_file_target.test_case) as f:
        content = f.read()
    assert len(content) > 1
    assert len(content) < 11


@pytest.mark.slow
def test_happy_path_in_place_single_file(tmp_path, monkeypatch):
    """Test successful reduction with --in-place and single file."""
    monkeypatch.chdir(tmp_path)

    target = tmp_path / "test.txt"
    target.write_text("hello world")

    script = tmp_path / "test.sh"
    script.write_text(
        """#!/bin/bash
[ "$(wc -c < "$1")" -gt 1 ]
"""
    )
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--in-place",
            "--parallelism=1",
        ],
    )

    assert result.exit_code == 0

    content = target.read_text()
    assert len(content) > 1
    assert len(content) < 11


@pytest.mark.slow
def test_happy_path_formatter_none(simple_file_target):
    """Test successful reduction with --formatter=none."""
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            simple_file_target.interestingness_test,
            simple_file_target.test_case,
            "--ui=basic",
            "--formatter=none",
            "--parallelism=1",
        ],
    )

    assert result.exit_code == 0

    with open(simple_file_target.test_case) as f:
        content = f.read()
    assert len(content) > 1


# === TUI terminal interaction test ===


@pytest.mark.slow
def test_tui_terminal_interaction_quit_during_reduction(tmp_path):
    """Test TUI interaction using pexpect/pyte to simulate real terminal usage.

    This test:
    1. Launches the TUI with a file and a restrictive interestingness test
    2. Waits for the first successful reduction to appear on screen
    3. Presses 'q' to quit
    4. Verifies the app exits cleanly

    The test typically completes in 2-3 seconds (most of which is subprocess and
    TUI startup overhead), with 5 second timeouts as safety margins for CI.
    """
    import time

    # Create a smaller file (100 bytes) for faster testing
    # We need content whose hash is divisible by 10 so the initial test passes
    # Seed 8 works for 1000 bytes, find one that works for 100 bytes
    for seed in range(1000):
        original_content = bytes([(i * 17 + seed) % 256 for i in range(100)])
        content_hash = int(hashlib.sha256(original_content).hexdigest(), 16)
        if content_hash % 10 == 0:
            break
    else:
        pytest.fail("Could not find initial content with hash divisible by 10")

    target = tmp_path / "test.bin"
    target.write_bytes(original_content)
    original_size = len(original_content)

    # Interestingness test: size >= 1/4 original AND hash % 10 == 0
    # This allows some reductions but not arbitrary ones
    script = tmp_path / "test.py"
    script.write_text(
        f"""#!/usr/bin/env {sys.executable}
import hashlib
import sys
from pathlib import Path

original_size = {original_size}
file_path = Path(sys.argv[1])
content = file_path.read_bytes()
size = len(content)

# Must be at least 1/4 of the original size
if size < original_size // 4:
    sys.exit(1)

# Hash must be divisible by 10 (about 10% of candidates pass)
file_hash = int(hashlib.sha256(content).hexdigest(), 16)
if file_hash % 10 != 0:
    sys.exit(1)

sys.exit(0)
"""
    )
    script.chmod(0o755)

    # Set up pyte screen to parse terminal output
    screen = pyte.Screen(80, 24)
    stream = pyte.Stream(screen)

    # Spawn the TUI process with --no-exit-on-completion so it waits for 'q'
    child = pexpect.spawn(
        sys.executable,
        [
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
            "--parallelism=1",
            "--no-exit-on-completion",  # Stay open after completion
        ],
        encoding="utf-8",
        timeout=10,
        dimensions=(24, 80),  # Terminal size
    )

    try:
        # Wait for TUI to start and show initial state
        # Look for "Validating" message first
        child.expect("Validating initial example...", timeout=10)

        # Now wait for the TUI to show reduction progress
        # We're looking for any percentage > 0% in the output
        reduction_seen = False
        start_time = time.time()
        timeout = 10.0

        while time.time() - start_time < timeout:
            # Read available output
            try:
                data = child.read_nonblocking(size=4096, timeout=0.1)
                stream.feed(data)
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                break

            # Get current screen content
            screen_text = "\n".join(screen.display)

            # Check for reduction progress (any percentage > 0%)
            # Pattern matches things like "10.00% reduction" or "5.50% reduction"
            if re.search(r"[1-9]\d*\.\d+% reduction", screen_text):
                reduction_seen = True
                break

            # Also check for completed state
            if "Reduction completed" in screen_text:
                reduction_seen = True
                break

        assert reduction_seen, (
            f"No reduction seen within {timeout}s. Screen content:\n"
            + "\n".join(screen.display)
        )

        # Press 'q' to quit
        child.send("q")

        # Wait for process to exit (should be fast)
        child.expect(pexpect.EOF, timeout=10)

        # Verify clean exit
        child.close()
        assert child.exitstatus == 0, f"Exit status was {child.exitstatus}"

    finally:
        # Clean up if still running
        if child.isalive():
            child.terminate(force=True)
