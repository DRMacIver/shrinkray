"""Main entry point for shrink ray."""

import os
import shutil
import signal
import sys
import traceback
from typing import Any

import click
import trio

from shrinkray.cli import (
    EnumChoice,
    InputType,
    UIType,
    validate_command,
    validate_ui,
)
from shrinkray.passes.clangdelta import (
    C_FILE_EXTENSIONS,
    ClangDelta,
    find_clang_delta,
)
from shrinkray.problem import InvalidInitialExample
from shrinkray.state import (
    ShrinkRayDirectoryState,
    ShrinkRayState,
    ShrinkRayStateSingleFile,
)
from shrinkray.ui import BasicUI, ShrinkRayUI
from shrinkray.work import Volume


async def run_shrink_ray(
    state: ShrinkRayState[Any],
    ui: ShrinkRayUI[Any],
) -> None:
    """Run the shrink ray reduction process."""
    async with trio.open_nursery() as nursery:
        problem = state.problem
        try:
            await problem.setup()
        except InvalidInitialExample as e:
            await state.report_error(e)

        reducer = state.reducer

        @nursery.start_soon
        async def _() -> None:
            await reducer.run()
            nursery.cancel_scope.cancel()

        ui.install_into_nursery(nursery)

        await ui.run(nursery)

    await state.print_exit_message(problem)


@click.command(
    help="""
""".strip()
)
@click.version_option()
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
    "--seed",
    default=0,
    type=click.INT,
    help=("Random seed to use for any non-deterministic reductions."),
)
@click.option(
    "--volume",
    default="normal",
    type=EnumChoice(Volume),
    help="Level of output to provide.",
)
@click.option(
    "--in-place/--not-in-place",
    default=False,
    help="""
If `--in-place` is passed, shrinkray will run in the current working directory instead of
creating a temporary subdirectory. Note that this requires you to either run with no
parallelism or be very careful about files created in your interestingness
test not conflicting with each other.
""",
)
@click.option(
    "--input-type",
    default="all",
    type=EnumChoice(InputType),
    help="""
How to pass input to the test function. Options are:

1. `basename` writes it to a file of the same basename as the original, in the current working directory where the test is run.

2. `arg` passes it in a file whose name is provided as an argument to the test.

3. `stdin` passes its contents on stdin.

4. `all` (the default) does all of the above.

If --in-place is specified, all will not include basename by default, only arg and stdin.
If you want basename with --in-place you may pass it explicitly, but note that this is incompatible
with any parallelism.
    """.strip(),
)
@click.option(
    "--parallelism",
    type=click.INT,
    help="Number of tests to run in parallel. If set to 0 will default to either 1 or number of cpus depending on other options.",
    default=0,
)
@click.option(
    "--ui",
    "ui_type",
    type=EnumChoice(UIType),
    help="""
UI mode to use. Options are:

* 'textual' (default): Modern terminal UI using the textual library.
* 'basic': Simple text output, suitable for scripts or non-interactive use.

When not specified, defaults to 'textual' for interactive terminals, 'basic' otherwise.
    """.strip(),
    callback=validate_ui,
)
@click.option(
    "--theme",
    type=click.Choice(["auto", "dark", "light"]),
    default="auto",
    help="""
Theme mode for the textual UI. Options are:

* 'auto' (default): Detect terminal's color scheme automatically.
* 'dark': Use dark theme.
* 'light': Use light theme.
    """.strip(),
)
@click.option(
    "--formatter",
    default="default",
    help="""
Path to a formatter for Shrink Ray to use. This is mostly used for display purposes,
and to format the final test case.

A formatter should accept input on stdin and write to stdout, and exit with a status
code of 0. If the formatter exits with a non-zero status code its output will be
ignored.

Special values for this:

* 'none' turns off formatting.
* 'default' causes Shrink Ray to use its default behaviour, which is to look for
  formatters it knows about on PATH and use one of those if found, otherwise to
  use a very simple language-agnostic formatter.
""",
)
@click.option(
    "--trivial-is-error/--trivial-is-not-error",
    default=True,
    help="""
It's easy to write interestingness tests which accept too much, and one common way this
happens is if they accept empty or otherwise trivial files. By default Shrink Ray will
print an error message at the end of reduction and exit with non-zero status in this case.
This behaviour can be disabled by passing --trivial-is-not-error.
""",
)
@click.option(
    "--no-clang-delta",
    is_flag=True,
    default=False,
    help="Pass this if you do not want to use clang delta for C/C++ transformations.",
)
@click.option(
    "--clang-delta",
    default="",
    help="Path to your clang_delta executable.",
)
@click.argument("test", callback=validate_command)
@click.argument(
    "filename",
    type=click.Path(exists=True, resolve_path=False, dir_okay=True, allow_dash=False),
)
def main(
    input_type: InputType,
    backup: str,
    filename: str,
    test: list[str],
    timeout: float,
    in_place: bool,
    parallelism: int,
    seed: int,
    volume: Volume,
    formatter: str,
    no_clang_delta: bool,
    clang_delta: str,
    trivial_is_error: bool,
    ui_type: UIType,
    theme: str,
) -> None:
    if timeout <= 0:
        timeout = float("inf")

    if not os.access(test[0], os.X_OK):
        print(
            f"Interestingness test {os.path.relpath(test[0])} is not executable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if in_place and input_type == InputType.basename and parallelism > 1:
        raise click.BadParameter(
            f"parallelism cannot be greater than 1 when --in-place and --input-type=basename (got {parallelism})"
        )

    if parallelism == 0:
        if in_place and input_type == InputType.basename:
            parallelism = 1
        else:
            parallelism = os.cpu_count() or 1

    clang_delta_executable: ClangDelta | None = None
    if os.path.splitext(filename)[1] in C_FILE_EXTENSIONS and not no_clang_delta:
        if not clang_delta:
            clang_delta = find_clang_delta()
        if not clang_delta:
            raise click.UsageError(
                "Attempting to reduce a C or C++ file, but clang_delta is not installed. "
                "Please run with --no-clang-delta, or install creduce on your system. "
                "If creduce is already installed and you wish to use clang_delta, please "
                "pass its path with the --clang-delta argument."
            )

        clang_delta_executable = ClangDelta(clang_delta)

    # This is a debugging option so that when the reducer seems to be taking
    # a long time you can Ctrl-\ to find out what it's up to. I have no idea
    # how to test it in a way that shows up in coverage.
    def dump_trace(signum: int, frame: Any) -> None:  # pragma: no cover
        traceback.print_stack()

    signal.signal(signal.SIGQUIT, dump_trace)

    if not backup:
        backup = filename + os.extsep + "bak"

    state_kwargs: dict[str, Any] = dict(
        input_type=input_type,
        in_place=in_place,
        test=test,
        timeout=timeout,
        base=os.path.basename(filename),
        parallelism=parallelism,
        filename=filename,
        formatter=formatter,
        trivial_is_error=trivial_is_error,
        seed=seed,
        volume=volume,
        clang_delta_executable=clang_delta_executable,
    )

    state: ShrinkRayState[Any]
    ui: ShrinkRayUI[Any]

    if os.path.isdir(filename):
        if input_type == InputType.stdin:
            raise click.UsageError("Cannot pass a directory input on stdin.")

        shutil.rmtree(backup, ignore_errors=True)
        shutil.copytree(filename, backup)

        files = [os.path.join(d, f) for d, _, fs in os.walk(filename) for f in fs]

        initial = {}
        for f in files:
            with open(f, "rb") as i:
                initial[os.path.relpath(f, filename)] = i.read()

        state = ShrinkRayDirectoryState(initial=initial, **state_kwargs)

        trio.run(state.check_formatter)

    else:
        try:
            os.remove(backup)
        except FileNotFoundError:
            pass

        with open(filename, "rb") as reader:
            initial = reader.read()

        with open(backup, "wb") as writer:
            writer.write(initial)

        state = ShrinkRayStateSingleFile(initial=initial, **state_kwargs)

        trio.run(state.check_formatter)

    if ui_type == UIType.textual:
        from shrinkray.tui import run_textual_ui

        run_textual_ui(
            file_path=filename,
            test=test,
            parallelism=parallelism,
            timeout=timeout,
            seed=seed,
            input_type=input_type.name,
            in_place=in_place,
            formatter=formatter,
            volume=volume.name,
            no_clang_delta=no_clang_delta,
            clang_delta=clang_delta,
            trivial_is_error=trivial_is_error,
            theme=theme,  # type: ignore[arg-type]
        )
        return

    # At this point, ui_type must be UIType.basic since textual returned above
    assert ui_type == UIType.basic
    ui = BasicUI(state)

    try:
        trio.run(
            lambda: run_shrink_ray(
                state=state,
                ui=ui,
            )
        )
    # If you try to sys.exit from within an exception handler, trio will instead
    # put it in an exception group. I wish to register the complaint that this is
    # incredibly fucking stupid, but anyway this is a workaround for it.
    except* SystemExit as eg:
        raise eg.exceptions[0]
    except* KeyboardInterrupt as eg:
        raise eg.exceptions[0]


def worker_main() -> None:
    """Entry point for the worker subprocess."""
    from shrinkray.subprocess.worker import main as worker_entry

    worker_entry()


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="shrinkray")
