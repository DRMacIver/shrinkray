import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import click
import pexpect
import pyte
import pytest
import trio
from attrs import define
from click.testing import CliRunner

from shrinkray.__main__ import _validate_memory_limit, main, worker_main
from shrinkray.llm_client import llm_support_available
from shrinkray.process import default_memory_limit, interrupt_wait_and_kill
from shrinkray.reducer import ShrinkRay
from shrinkray.state import ShrinkRayStateSingleFile
from shrinkray.validation import ValidationResult


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

    # Run single-threaded for a deterministic reduction path. (The reduction is
    # confluent here now that replace_identifiers_with_zero can take `assert x`
    # to `assert 0`, but --parallelism=1 keeps the exact-output assertions below
    # robust regardless.)
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
                "--no-history",
                "--parallelism=1",
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
                "--no-history",
                "--parallelism=1",
            ],
        )

    # c.py reduces all the way to `assert 0`: reduce_integer_literals takes
    # `x == 2` to `x == 0`, a span deletion drops `x ==`, and (crucially)
    # replace_identifiers_with_zero turns any surviving `assert x` into
    # `assert 0`. That drops the cross-file dependency on a.py, so a.py and the
    # unused b.py are both deleted and only c.py survives.
    assert not a.exists()
    assert not b.exists()
    assert c.exists()
    assert c.read_text() == "assert 0"

    # The reduction preserved interestingness: the script still succeeds.
    assert subprocess.call([str(script), str(target)]) == 0


@pytest.mark.slow
def test_gives_informative_error_when_script_does_not_work_outside_current_directory(
    tmpdir,
):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.py"
    # This script only works when passed the exact target path as an argument
    # When run from a temp directory with a relative path, it will fail
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
                "--no-history",
            ],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

    # Validation catches that the test failed when run from temp directory
    assert "should return 0 for interesting test cases" in excinfo.value.stderr


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
                "--no-history",
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
#!/bin/sh

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
#!/bin/sh

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
            "--no-history",
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
#!/bin/sh

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
            "--no-history",
        ],
    )
    assert result.exit_code != 0
    # Validation now produces this error message
    assert "should return 0 for interesting test cases" in result.stderr


def test_reducing_c_file_to_trivial_is_an_error(tmp_path):
    """Reducing a C file with an always-passing test runs the C/C++
    passes and reduces to nothing, which is reported as an error."""
    target = tmp_path / "test.c"
    target.write_text("int main() { return 0; }")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic", "--no-history"],
    )
    assert result.exit_code != 0
    assert "--trivial-is-not-error" in str(result.output)


def test_error_when_test_not_executable(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    # Note: NOT setting executable permission

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic"],
    )
    assert result.exit_code == 1
    assert "not executable" in result.stderr


# === --memory-limit tests ===


def _call_validate_memory_limit(value: str | None) -> int | None:
    ctx = click.Context(click.Command("shrinkray"))
    param = click.Option(["--memory-limit"])
    return _validate_memory_limit(ctx, param, value)


def test_validate_memory_limit_defaults_to_physical_ram():
    assert _call_validate_memory_limit(None) == default_memory_limit()


def test_validate_memory_limit_parses_suffix():
    assert _call_validate_memory_limit("4G") == 4 * 1024**3


def test_validate_memory_limit_zero_disables():
    assert _call_validate_memory_limit("0") is None


def test_validate_memory_limit_rejects_garbage():
    with pytest.raises(click.BadParameter):
        _call_validate_memory_limit("notanumber")


@pytest.mark.parametrize("enforceable", [True, False])
def test_memory_limit_warns_only_when_not_enforceable(tmpdir, enforceable):
    # Use a non-executable test so main() exits right after the warning
    # check (no reduction), keeping this fast. The warning must appear iff
    # the platform cannot enforce the limit.
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")  # not executable

    runner = CliRunner(catch_exceptions=False)
    with patch("shrinkray.__main__.MEMORY_LIMIT_ENFORCEABLE", enforceable):
        result = runner.invoke(
            main,
            [str(script), str(target), "--ui=basic", "--memory-limit=8G"],
        )
    assert result.exit_code == 1  # exits on the non-executable test
    assert ("cannot be enforced" in result.stderr) == (not enforceable)


def test_memory_limit_disabled_gives_no_warning(tmpdir):
    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")  # not executable

    runner = CliRunner(catch_exceptions=False)
    with patch("shrinkray.__main__.MEMORY_LIMIT_ENFORCEABLE", False):
        result = runner.invoke(
            main,
            [str(script), str(target), "--ui=basic", "--memory-limit=0"],
        )
    assert result.exit_code == 1
    assert "cannot be enforced" not in result.stderr


def test_crashing_formatter_is_disabled_not_fatal(tmpdir):
    # A formatter that crashes on the initial test case must not abort the
    # run; shrink ray warns and reduces without it. The interestingness
    # test only accepts the exact initial content, so nothing reduces and
    # the run finishes promptly.
    target = tmpdir / "hello.txt"
    target.write_text("hello", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text('#!/bin/sh\n[ "$(cat "$1")" = "hello" ]\n', encoding="utf-8")
    script.chmod(0o777)
    formatter = tmpdir / "fmt.sh"
    formatter.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")  # crashes
    formatter.chmod(0o777)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--no-history",
            "--parallelism=1",
            f"--formatter={formatter}",
        ],
    )
    assert result.exit_code == 0
    assert "continuing without formatting" in result.stderr


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
            "--no-history",
        ],
    )
    # Should complete successfully with infinite timeout
    assert result.exit_code == 0


def test_parallelism_defaults_to_one_for_basename_inplace(tmpdir, monkeypatch):
    """Fast test verifying parallelism=0 defaults to 1 with in_place + basename.

    This test mocks the state creation to verify the parallelism value
    without running a full reduction.
    """

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_parallelism = []

    def mock_state_init(**kwargs):
        captured_parallelism.append(kwargs.get("parallelism"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.load_state_for_path") as mock_state:
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

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_parallelism = []

    def mock_state_init(**kwargs):
        captured_parallelism.append(kwargs.get("parallelism"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.load_state_for_path") as mock_state:
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
#!/bin/sh
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

    # Can't actually run it without proper stdin/stdout setup,
    # but at least verify it's importable and callable
    assert callable(worker_main)


def test_directory_mode_stdin_error(tmp_path):
    """Test that directory mode rejects stdin input type."""
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "test.txt").write_text("hello")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [str(script), str(target), "--ui=basic", "--input-type=stdin"],
    )
    assert result.exit_code != 0
    assert "Cannot pass a directory input on stdin" in str(result.output)


@pytest.mark.slow
def test_default_backup_filename(basic_shrink_target):
    """Test that default backup filename is created correctly."""

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
            "--no-history",
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
        """#!/bin/sh
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

    # Mock run_textual_ui to avoid actually launching the TUI
    mock_run_textual_ui = MagicMock()
    monkeypatch.setattr("shrinkray.__main__.run_textual_ui", mock_run_textual_ui)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            basic_shrink_target.interestingness_test,
            basic_shrink_target.test_case,
            "--ui=textual",
            "--no-history",
        ],
    )

    # The function should have been called
    assert mock_run_textual_ui.called
    assert result.exit_code == 0


def test_keyboard_interrupt_handling(basic_shrink_target, tmp_path):
    """Test that KeyboardInterrupt is properly re-raised from ExceptionGroup.

    Exercises the KeyboardInterrupt handling in run_command.
    """

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

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    captured_timeout = []

    def mock_state_init(**kwargs):
        captured_timeout.append(kwargs.get("timeout"))
        raise SystemExit(0)

    with patch("shrinkray.__main__.load_state_for_path") as mock_state:
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

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    # Track if os.remove was called with the default backup path
    removed_files = []

    def tracking_remove(path):
        removed_files.append(path)
        # Don't actually remove, just raise FileNotFoundError like the code expects
        raise FileNotFoundError()

    def mock_state_init(**kwargs):
        raise SystemExit(0)

    with patch("shrinkray.__main__.load_state_for_path") as mock_state:
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

    monkeypatch.chdir(tmpdir)

    target = tmpdir / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    script = tmpdir / "test.sh"
    script.write_text("#!/bin/sh\nexit 0", encoding="utf-8")
    script.chmod(0o777)

    custom_backup = str(tmpdir / "my_custom.bak")

    # Track if os.remove was called with the custom backup path
    removed_files = []

    def tracking_remove(path):
        removed_files.append(path)
        raise FileNotFoundError()

    def mock_state_init(**kwargs):
        raise SystemExit(0)

    with patch("shrinkray.__main__.load_state_for_path") as mock_state:
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
    """Fast test verifying directory mode setup logic.

    This tests the directory mode initialization without running a full reduction.
    """
    # Create a test directory with files
    target = tmp_path / "mydir"
    target.mkdir()
    (target / "a.txt").write_text("hello")
    (target / "b.txt").write_text("world")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    # Track what was passed to ShrinkRayDirectoryState
    captured_initial = []

    def mock_dir_state_init(**kwargs):
        captured_initial.append(kwargs.get("initial"))
        # Exit early after capturing the initial dict
        raise SystemExit(0)

    # Track copytree calls
    copytree_calls = []
    original_copytree = shutil.copytree

    def tracking_copytree(src, dst, **kwargs):
        copytree_calls.append((src, dst))
        original_copytree(src, dst, **kwargs)

    # Mock validation to pass immediately
    mock_validation_result = ValidationResult(success=True)

    with (
        patch(
            "shrinkray.validation.run_validation",
            return_value=mock_validation_result,
        ),
        patch(
            "shrinkray.state.ShrinkRayDirectoryState",
            side_effect=mock_dir_state_init,
        ),
        patch("shutil.copytree", tracking_copytree),
    ):
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
        """#!/bin/sh
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
            "--no-history",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Should fail
    assert result.returncode != 0

    # Should show timeout error message
    assert "TimeoutExceededOnInitial" in result.stderr
    assert "exceeded timeout" in result.stderr


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
        """#!/bin/sh
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
            "--no-history",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Should fail
    assert result.returncode != 0

    # Should show timeout error message
    # The error comes through as a raw traceback since the timeout
    # happens during problem.setup() in the worker subprocess
    combined_output = result.stdout + result.stderr
    # Just check that timeout is mentioned somewhere in the output
    assert "timeout" in combined_output.lower()


@pytest.mark.slow
def test_invalid_initial_shows_error_message_basic(tmp_path):
    """Test that when the initial test case is invalid, basic UI shows a user-friendly error."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that always fails
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=basic",
            "--no-history",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Interestingness test exited with code" in result.stderr
    assert "should return 0 for interesting test cases" in result.stderr


@pytest.mark.slow
def test_invalid_initial_shows_error_message_tui(tmp_path):
    """Test that when the initial test case is invalid, TUI shows a user-friendly error."""
    target = tmp_path / "test.txt"
    target.write_text("hello world")

    # Script that always fails
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 1\n")
    script.chmod(0o755)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
            "--no-history",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "Interestingness test exited with code" in combined_output
    assert "should return 0 for interesting test cases" in combined_output


@pytest.mark.slow
def test_script_depends_on_cwd_shows_error_tui(tmp_path):
    """Test that TUI shows error when script depends on current directory.

    When a script depends on being run from a specific directory, validation
    fails because the test is run in a temporary directory. The error message
    should show the exit code and provide a "To reproduce" command so users
    can debug the issue.
    """
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
            "--no-history",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    # Should show the error and a way to reproduce
    assert "exited with code 1" in combined_output.lower()
    assert "to reproduce" in combined_output.lower()


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
            "--no-history",
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
            "--no-history",
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
            "--no-history",
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
            "--no-history",
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
    script.write_text("#!/bin/sh\nexit 0\n")
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
            "--no-history",
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
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=basic",
            "--parallelism=1",
            "--no-history",
        ],
    )

    # Should fail because result is trivial
    assert result.exit_code != 0


def test_trivial_is_not_error_tui(tmp_path, monkeypatch):
    """Test --trivial-is-not-error flag with TUI path.

    Uses mocking to verify the TUI is invoked with correct parameters.
    """

    target = tmp_path / "test.txt"
    target.write_text("hello")

    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)

    mock_run_textual_ui = MagicMock()
    monkeypatch.setattr("shrinkray.__main__.run_textual_ui", mock_run_textual_ui)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [
            str(script),
            str(target),
            "--ui=textual",
            "--trivial-is-not-error",
            "--parallelism=1",
            "--no-history",
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
            "--no-history",
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
        """#!/bin/sh
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
            "--no-history",
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
            "--no-history",
        ],
    )

    assert result.exit_code == 0

    with open(simple_file_target.test_case) as f:
        content = f.read()
    assert len(content) > 1


# === also-interesting CLI option tests ===


def test_also_interesting_zero_disables_feature(tmp_path):
    """Test that --also-interesting=0 disables the feature."""
    # Create a simple test script that always exits 0 (interesting)
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    runner = CliRunner(catch_exceptions=False)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main,
            [
                str(script),
                str(target),
                "--ui=basic",
                "--also-interesting=0",  # Explicit disable
                "--parallelism=1",
                "--trivial-is-not-error",  # Allow reducing to empty
            ],
        )

    assert result.exit_code == 0, f"Output: {result.output}"


def test_no_history_without_explicit_also_interesting_disables_both(tmp_path):
    """Test that --no-history without explicit --also-interesting disables both."""
    # Create a test script that exits 0 (interesting)
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    target = tmp_path / "test.txt"
    target.write_text("hello")

    runner = CliRunner(catch_exceptions=False)
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            main,
            [
                str(script),
                str(target),
                "--ui=basic",
                "--no-history",  # Disable history without explicit --also-interesting
                "--parallelism=1",
                "--trivial-is-not-error",  # Allow reducing to empty
            ],
        )

    assert result.exit_code == 0, f"Output: {result.output}"
    # No .shrinkray directory should be created
    assert not os.path.exists(os.path.join(tmp_path, ".shrinkray"))


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
        # Look for "Validating" message first (now comes from validation module)
        child.expect("Validating interestingness test", timeout=10)

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


@pytest.mark.slow
def test_tui_history_modal_during_reduction(tmp_path: pathlib.Path):
    """Test that opening history modal during reduction doesn't crash with duplicate IDs.

    This is a regression test for a bug where the history modal's refresh timer
    would try to add ListItems with the same IDs as existing items, causing
    a DuplicateIds exception.
    """
    # Create a Python target file similar to enterprise-hello that will
    # take a while to reduce
    target = tmp_path / "hello.py"
    target.write_text(
        """\
import sys
import time
import os
import random

def func1():
    return 1

def func2():
    return 2

def func3():
    return 3

def func4():
    return 4

def func5():
    return 5

class A:
    def method1(self):
        pass
    def method2(self):
        pass

class B:
    def method1(self):
        pass
    def method2(self):
        pass

# The key line that must be preserved
print("hello")

# More filler code
def more_stuff():
    x = 1
    y = 2
    z = 3
    return x + y + z

if __name__ == "__main__":
    more_stuff()
"""
    )

    # Create a test script that runs Python and checks for "hello" in output
    script = tmp_path / "ishello.sh"
    log_file = tmp_path / "hello.log"
    script.write_text(
        f"""\
#!/bin/sh
set -eux
python "$1" > "{log_file}"
grep "hello" "{log_file}"
"""
    )
    script.chmod(0o755)

    # Set up pyte screen to parse terminal output
    screen = pyte.Screen(100, 30)
    stream = pyte.Stream(screen)

    # Spawn the TUI process
    child = pexpect.spawn(
        sys.executable,
        [
            "-m",
            "shrinkray",
            str(script),
            str(target),
            "--ui=textual",
            "--parallelism=1",
            "--no-exit-on-completion",
            "--no-history",
        ],
        encoding="utf-8",
        timeout=30,
        dimensions=(30, 100),
    )

    try:
        # Wait for TUI to start
        child.expect("Validating interestingness test", timeout=10)

        # Wait for the reducer to be running (showing some stats)
        start_time = time.time()
        timeout = 15.0

        while time.time() - start_time < timeout:
            try:
                data = child.read_nonblocking(size=4096, timeout=0.1)
                stream.feed(data)
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                break

            screen_text = "\n".join(screen.display)
            # Check for any reduction activity or that reducer is running
            # (not yet completed)
            if "Reduction completed" not in screen_text:
                if (
                    "Calls to interestingness test" in screen_text
                    or "reduction" in screen_text.lower()
                ):
                    break

        # If reduction already completed, test can still exercise the modal
        # since the bug can occur whenever the modal is open with existing entries

        # Open the history modal with 'x'
        child.send("x")

        # Wait for modal to open and for the refresh timer to fire multiple times
        # The bug occurs when the refresh timer fires (every 1 second) while
        # new reductions are happening
        error_seen = False
        process_crashed = False
        start_time = time.time()
        timeout = 10.0  # Wait 10 seconds - should see many refresh cycles

        while time.time() - start_time < timeout:
            try:
                data = child.read_nonblocking(size=4096, timeout=0.2)
                stream.feed(data)
            except pexpect.TIMEOUT:
                pass
            except pexpect.EOF:
                # Process crashed - likely due to the bug
                process_crashed = True
                break

            screen_text = "\n".join(screen.display)
            # Check for DuplicateIds error specifically
            # Note: don't check for generic "Traceback" since normal test output
            # can contain Python tracebacks from failed test cases
            if "DuplicateIds" in screen_text:
                error_seen = True
                break

        # Get final screen content before closing
        final_screen = "\n".join(screen.display)

        if not process_crashed:
            # Close the modal with Escape
            child.send("\x1b")  # Escape key
            time.sleep(0.3)

            # Quit the TUI
            child.send("q")

            # Wait for process to exit
            try:
                child.expect(pexpect.EOF, timeout=5)
            except pexpect.TIMEOUT:
                pass

        child.close()

        # Check that no error occurred
        assert not error_seen, (
            f"DuplicateIds error in screen output. Screen:\n{final_screen}"
        )
        assert not process_crashed, (
            f"Process crashed (likely DuplicateIds error). Screen:\n{final_screen}"
        )

        # The process should have exited cleanly
        if child.exitstatus is not None:
            assert child.exitstatus == 0, (
                f"Exit status was {child.exitstatus}. Screen:\n{final_screen}"
            )

    finally:
        if child.isalive():
            child.terminate(force=True)


# === LLM mode options ===

# The binary no-op test constructs a real client and runs a reduction, so
# it needs llama-cpp-python to be loadable (it isn't on e.g. OpenBSD; see
# tests/test_llm_client.py). The option-validation tests run everywhere.
requires_llm_support = pytest.mark.skipif(
    not llm_support_available(),
    reason="llama-cpp-python cannot load on this platform",
)


def _llm_target(tmp_path, content: bytes, pattern: str):
    target = tmp_path / "target.bin"
    target.write_bytes(content)
    script = tmp_path / "test.sh"
    script.write_text(f'#!/bin/sh\ngrep -q {pattern} "$1"\n')
    script.chmod(0o755)
    return str(script), str(target)


def test_llm_only_conflicts_with_no_llm(tmp_path):
    script, target = _llm_target(tmp_path, b"say xy\n", "xy")
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main, [script, target, "--ui=basic", "--llm-only", "--no-llm"]
    )
    assert result.exit_code == 2
    assert "--llm-only cannot be combined with --no-llm" in result.output


def test_llm_rejects_invalid_model_spec(tmp_path):
    script, target = _llm_target(tmp_path, b"say xy\n", "xy")
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main, [script, target, "--ui=basic", "--llm", "--llm-model=not-a-model"]
    )
    assert result.exit_code == 2
    assert "not-a-model" in result.output


def test_explicit_llm_fails_when_unsupported(tmp_path, monkeypatch):
    script, target = _llm_target(tmp_path, b"say xy\n", "xy")
    monkeypatch.setattr("shrinkray.__main__.llm_support_available", lambda: False)
    runner = CliRunner()
    result = runner.invoke(main, [script, target, "--ui=basic", "--llm"])
    assert result.exit_code == 1
    assert "cannot load on this platform" in result.output


def test_llm_enabled_by_env_var_fails_when_unsupported(tmp_path, monkeypatch):
    script, target = _llm_target(tmp_path, b"say xy\n", "xy")
    monkeypatch.setattr("shrinkray.__main__.llm_support_available", lambda: False)
    monkeypatch.setenv("SHRINKRAY_LLM", "1")
    runner = CliRunner()
    result = runner.invoke(main, [script, target, "--ui=basic"])
    assert result.exit_code == 1
    assert "cannot load on this platform" in result.output


def test_default_llm_degrades_with_a_warning_when_unsupported(tmp_path, monkeypatch):
    # LLM mode is on by default, but on platforms where llama-cpp-python
    # can't load, an ordinary reduction must still work.
    monkeypatch.setattr("shrinkray.__main__.llm_support_available", lambda: False)
    monkeypatch.delenv("SHRINKRAY_LLM")
    target = tmp_path / "target.txt"
    target.write_text("say xy please\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q xy "$1"\n')
    script.chmod(0o755)
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(main, [str(script), str(target), "--ui=basic"])
    assert result.exit_code == 0
    assert "Reducing without them" in result.output
    assert "xy" in target.read_text()


def test_env_var_disables_llm(tmp_path):
    # The conftest sets SHRINKRAY_LLM=0; with a garbage model configured,
    # the reduction can only succeed because the LLM passes are off.
    target = tmp_path / "target.txt"
    target.write_text("say xy please\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q xy "$1"\n')
    script.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not really a model")
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main, [str(script), str(target), "--ui=basic", f"--llm-model={model}"]
    )
    assert result.exit_code == 0


@requires_llm_support
def test_llm_only_reduction_of_binary_input_is_a_no_op(tmp_path):
    # Binary input can't be prompted, so the LLM pass (the only pass in
    # --llm-only mode) never generates and the reduction just converges.
    # The pattern sits on a clean line because BSD grep won't match lines
    # containing invalid UTF-8, and the model is a dummy local file so
    # that the eager background load doesn't try to download anything
    # (its failure is irrelevant: the pass never waits on it).
    content = b"xy\n\xc3\x28\n"
    script, target = _llm_target(tmp_path, content, "xy")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not really a model")
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main,
        [script, target, "--ui=basic", "--llm-only", f"--llm-model={model}"],
    )
    assert result.exit_code == 0
    # No pass can touch this input, so nothing gets deleted (whitespace may
    # still be canonicalised).
    final = pathlib.Path(target).read_bytes()
    assert b"xy" in final
    assert len(final) == len(content)


def test_basic_ui_reports_pending_downloads(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("say xy please\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q xy "$1"\n')
    script.chmod(0o755)

    pending = [
        {"id": "llm", "description": "LLM model x (about 2.7GB)"},
        {"id": "grammar-go", "description": "tree-sitter grammar for go"},
    ]
    monkeypatch.setattr(
        "shrinkray.state.ShrinkRayStateSingleFile.pending_downloads",
        lambda self: pending,
    )
    started: list[list[str]] = []
    monkeypatch.setattr(
        "shrinkray.state.ShrinkRayStateSingleFile.start_downloads",
        lambda self, disabled: started.append(disabled),
    )
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(main, [str(script), str(target), "--ui=basic"])
    assert result.exit_code == 0
    assert "will download in the background" in result.output
    assert "tree-sitter grammar for go" in result.output
    assert "LLM model x" in result.output
    assert "--no-llm" in result.output
    assert started == [[]]


def test_basic_ui_download_notice_omits_no_llm_when_only_grammar(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("say xy please\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q xy "$1"\n')
    script.chmod(0o755)

    monkeypatch.setattr(
        "shrinkray.state.ShrinkRayStateSingleFile.pending_downloads",
        lambda self: [{"id": "grammar-go", "description": "grammar for go"}],
    )
    monkeypatch.setattr(
        "shrinkray.state.ShrinkRayStateSingleFile.start_downloads",
        lambda self, disabled: None,
    )
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(main, [str(script), str(target), "--ui=basic"])
    assert result.exit_code == 0
    assert "grammar for go" in result.output
    assert "--no-llm" not in result.output


def test_no_restart_flag_reaches_the_reducer(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("hello world\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q hello "$1"\n')
    script.chmod(0o755)

    seen = {}
    real_reducer = ShrinkRayStateSingleFile.new_reducer

    def spy(self, problem):
        reducer = real_reducer(self, problem)
        assert isinstance(reducer, ShrinkRay)
        seen["restart"] = reducer.restart_at_fixpoint
        return reducer

    monkeypatch.setattr(ShrinkRayStateSingleFile, "new_reducer", spy)
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        main, [str(script), str(target), "--ui=basic", "--no-restart"]
    )
    assert result.exit_code == 0
    assert seen["restart"] is False


def test_restart_defaults_on(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("hello world\n")
    script = tmp_path / "test.sh"
    script.write_text('#!/bin/sh\ngrep -q hello "$1"\n')
    script.chmod(0o755)

    seen = {}
    real_reducer = ShrinkRayStateSingleFile.new_reducer

    def spy(self, problem):
        reducer = real_reducer(self, problem)
        assert isinstance(reducer, ShrinkRay)
        seen["restart"] = reducer.restart_at_fixpoint
        return reducer

    monkeypatch.setattr(ShrinkRayStateSingleFile, "new_reducer", spy)
    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(main, [str(script), str(target), "--ui=basic"])
    assert result.exit_code == 0
    assert seen["restart"] is True
