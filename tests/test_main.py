import os
import subprocess
import sys
import pathlib
import pytest

import trio
import black
from shrinkray.__main__ import interrupt_wait_and_kill, main
from click.testing import CliRunner
from attrs import define


def format(s):
    return black.format_str(s, mode=black.Mode()).strip()


async def test_kill_process():
    async with trio.open_nursery() as nursery:
        kwargs = dict(
            universal_newlines=False,
            preexec_fn=os.setsid,
            check=False,
            stdout=subprocess.PIPE,
        )

        def call_with_kwargs(task_status=trio.TASK_STATUS_IGNORED):  # type: ignore
            # start a subprocess that will just ignore SIGINT signals
            return trio.run_process(
                [
                    sys.executable,
                    "-c",
                    "import signal, sys, time; signal.signal(signal.SIGINT, lambda *a: 1); print(1); sys.stdout.flush(); time.sleep(1000)",
                ],
                **kwargs,
                task_status=task_status,
            )

        sp = await nursery.start(call_with_kwargs)
        line = await sp.stdout.receive_some(2)
        assert line == b"1\n"
        # must not raise ValueError but succeed at killing the process
        await interrupt_wait_and_kill(sp)
        assert sp.returncode is not None
        assert sp.returncode != 0


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
            universal_newlines=True,
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
            universal_newlines=True,
        )

    assert "Hello world" in excinfo.value.stdout


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
