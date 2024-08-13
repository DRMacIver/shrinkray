import os
import subprocess
import sys
import pathlib

import trio

from shrinkray.__main__ import interrupt_wait_and_kill


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
    (target / "b.py").write_text("y = 'hello world'")
    c = target / "c.py"
    c.write_text("from a import x\n\n...\nassert x == 2")

    script = tmp_path / "test.py"
    script.write_text(
        """
#!/usr/bin/env python
import sys
sys.path.append('.')

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
            sys.executable,
            "-m",
            "shrinkray",
            str(script),
            str(target),
        ]
    )

    assert sorted(target.glob("*")) == ["a.py", "c.py"]

    assert a.read_text().strip() == "x = 1"
    assert c.read_text().strip() == "from a import x\nassert x == 2"
