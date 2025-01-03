import os
import subprocess
import sys
import pathlib
import pytest

import trio
import black
from shrinkray.__main__ import interrupt_wait_and_kill


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


def test_can_reduce_a_directory(tmp_path: pathlib.Path):
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
        """
#!/usr/bin/env python
import sys
sys.path.append(sys.argv[1])

try:
    import c
    sys.exit(1)
except AssertionError:
    sys.exit(0)
    """.strip()
    )
    script.chmod(0x777)

    subprocess.check_call(
        [
            str(script),
            str(target),
        ]
    )

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


def test_gives_informative_error_when_script_does_not_work_outside_current_directory(tmpdir):
    target = (tmpdir / "hello.txt")
    target.write_text("hello world", encoding='utf-8')
    script = tmpdir / "test.py"
    script.write_text(
        f"""
#!/usr/bin/env python
import sys

if sys.argv[1] != {repr(str(target))}:
    sys.exit(1)
    """.strip(), encoding='utf-8'
    )
    script.chmod(0x777)

    subprocess.check_call([script, target])

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(
            [sys.executable, "-m", "shrinkray", str(script), str(target), "--ui=basic"],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    assert 'your script depends' in excinfo.value.stderr


def test_prints_the_output_on_an_initially_uninteresting_test_case(tmpdir):
    target = (tmpdir / "hello.txt")
    target.write_text("hello world", encoding='utf-8')
    script = tmpdir / "test.py"
    script.write_text(
        f"""
#!/usr/bin/env python
import sys

print("Hello world")

sys.exit(1)
    """.strip(), encoding='utf-8'
    )
    script.chmod(0x777)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        subprocess.run(
            [sys.executable, "-m", "shrinkray", str(script), str(target), "--ui=basic"],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

    assert 'Hello world' in excinfo.value.stdout
