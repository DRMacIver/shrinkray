import os
import subprocess
import sys

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
