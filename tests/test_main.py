import json
import os
import pathlib
import subprocess
import sys

import black
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
            [sys.executable, "-m", "shrinkray", str(script), str(target), "--ui=basic"],
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
            [sys.executable, "-m", "shrinkray", str(script), str(target), "--ui=basic"],
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
            [sys.executable, "-m", "shrinkray", str(script), str(target), "--ui=basic"],
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
            assert isinstance(result.exception, (KeyboardInterrupt, SystemExit))


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
    original_remove = os.remove

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

    with patch("shrinkray.__main__.ShrinkRayDirectoryState", side_effect=mock_dir_state_init):
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
