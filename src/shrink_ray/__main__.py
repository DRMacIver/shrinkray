import hashlib
import os
import random
import shlex
import signal
import subprocess
import sys
import time
import traceback
from shutil import which
from tempfile import TemporaryDirectory

import click

from anyreduce.sat.reduction import SATShrinker
from anyreduce.sat.dimacscnf import clauses_to_dimacs
from anyreduce.sat.dimacscnf import dimacs_to_clauses


def validate_command(ctx, param, value):
    parts = shlex.split(value)
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter("%s: command not found" % (command,))
        command = os.path.abspath(what)
    return [command] + parts[1:]


def signal_group(sp, signal):
    gid = os.getpgid(sp.pid)
    assert gid != os.getgid()
    os.killpg(gid, signal)


def interrupt_wait_and_kill(sp, timeout=0.1):
    if sp.returncode is None:
        try:
            # In case the subprocess forked. Python might hang if you don't close
            # all pipes.
            for pipe in [sp.stdout, sp.stderr, sp.stdin]:
                if pipe:
                    pipe.close()
            signal_group(sp, signal.SIGINT)
            for _ in range(10):
                if sp.poll() is not None:
                    return
                time.sleep(timeout)
            signal_group(sp, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover
            # This is incredibly hard to trigger reliably, because it only happens
            # if the process exits at exactly the wrong time.
            pass
        sp.wait(timeout=timeout)


@click.command(
    help="""
anyreduce.sat takes a file in simplified DIMACS CNF format and a test command and
attempts to produce a minimal example of the file such that the test command
returns 0.
""".strip()
)
@click.version_option()
@click.option(
    "--debug/--no-debug",
    default=False,
    is_flag=True,
    help=("Emit (extremely verbose) debug output while shrinking"),
)
@click.option(
    "--backup",
    default="",
    help=(
        "Name of the backup file to create. Defaults to adding .bak to the "
        "name of the source file"
    ),
)
@click.option(
    "--timeout",
    default=1,
    type=click.FLOAT,
    help=(
        "Time out subprocesses after this many seconds. If set to <= 0 then "
        "no timeout will be used. Any commands that time out will be treated "
        "as failing the test"
    ),
)
@click.option(
    "--parallelism",
    default=os.cpu_count(),
    type=click.INT,
    help="Number of tests to run in parallel.",
)
@click.option(
    "--input-type",
    default="all",
    type=click.Choice(["all", "basename", "arg", "stdin"]),
    help=""""
How to pass input to the test function. Options are:

1. basename writes it to a file of the same basename as the original, in the current working directory where the test is run.
2. arg passes it in a file whose name is provided as an argument to the test.
3. stdin passes its contents on stdin.
4. all (the default) does all of the above.
    """.strip(),
)
@click.argument("test", callback=validate_command)
@click.argument(
    "filename",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False, allow_dash=False),
)
def main(
    input_type,
    debug,
    backup,
    filename,
    test,
    timeout,
    parallelism,
):
    if debug:
        # This is a debugging option so that when the reducer seems to be taking
        # a long time you can Ctrl-\ to find out what it's up to. I have no idea
        # how to test it in a way that shows up in coverage.
        def dump_trace(signum, frame):  # pragma: no cover
            traceback.print_stack()

        signal.signal(signal.SIGQUIT, dump_trace)

    if not backup:
        backup = filename + os.extsep + "bak"

    try:
        os.remove(backup)
    except FileNotFoundError:
        pass

    base = os.path.basename(filename)
    first_call = True

    async def test_clauses(clauses):
        nonlocal first_call
        if not clauses or not all(clauses):
            assert not first_call
            return False
        cnf = clauses_to_dimacs(clauses)
        with TemporaryDirectory() as d:
            working = os.path.join(d, base)
            with open(working, "w") as o:
                o.write(cnf)

            if input_type in ("all", "arg"):
                command = test + [working]
            else:
                command = test

            kwargs = dict(
                universal_newlines=True,
                preexec_fn=os.setsid,
                cwd=d,
            )
            if input_type in ("all", "stdin"):
                kwargs["stdin"] = subprocess.PIPE
                input_string = cnf
            else:
                kwargs["stdin"] = subprocess.DEVNULL
                input_string = ""

            if not debug:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL

            sp = subprocess.Popen(command, **kwargs)

            try:
                sp.communicate(input_string, timeout=timeout)
            except subprocess.TimeoutExpired:
                if first_call:
                    raise ValueError(
                        f"Initial test call exceeded timeout of {timeout}s. Try raising or disabling timeout."
                    )
                return False
            finally:
                first_call = False
                interrupt_wait_and_kill(sp)
            return sp.returncode == 0

    if timeout <= 0:
        timeout = None

    with open(filename, "r") as o:
        initial = o.read()

    with open(backup, "w") as o:
        o.write(initial)

    shrinker = SATShrinker(
        dimacs_to_clauses(initial),
        test_clauses,
        debug=debug,
        parallelism=parallelism,
    )

    @shrinker.on_reduce
    def _(clauses):
        with open(filename, "w") as o:
            o.write(clauses_to_dimacs(clauses))

    shrinker.reduce()


if __name__ == "__main__":
    main(prog_name="sat-reduce")  # pragma: no cover
