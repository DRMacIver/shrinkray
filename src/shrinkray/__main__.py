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
    validate_commands,
    validate_ui,
)
from shrinkray.formatting import determine_formatter_command
from shrinkray.passes.llm import (
    DEFAULT_MODEL_SPEC,
    llm_support_available,
    parse_model_spec,
)
from shrinkray.process import (
    MEMORY_LIMIT_ENFORCEABLE,
    default_memory_limit,
    parse_memory_limit,
)
from shrinkray.state import (
    ShrinkRayDirectoryState,
    ShrinkRayState,
    ShrinkRayStateSingleFile,
)
from shrinkray.tui import run_textual_ui
from shrinkray.ui import BasicUI, ShrinkRayUI
from shrinkray.validation import run_validation
from shrinkray.work import Volume


def _validate_memory_limit(
    ctx: "click.Context", param: "click.Parameter", value: str | None
) -> int | None:
    """Click callback: parse --memory-limit, defaulting to physical RAM."""
    if value is None:
        return default_memory_limit()
    try:
        return parse_memory_limit(value)
    except ValueError as e:
        raise click.BadParameter(str(e))


async def run_shrink_ray(
    state: ShrinkRayState[Any],
    ui: ShrinkRayUI[Any],
) -> None:
    """Run the shrink ray reduction process."""
    async with trio.open_nursery() as nursery:
        problem = state.problem
        # Validation runs before run_shrink_ray is called, so setup() should
        # always succeed. If it doesn't, there's a bug and we want it to propagate.
        await problem.setup()

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
    default=None,
    type=click.FLOAT,
    help=(
        "Maximum time in seconds to allow the interestingness test to run. "
        "Shrink Ray adapts the actual timeout to measured test runtimes over "
        "the course of the run, never exceeding this value (or 5 minutes if "
        "not specified), and temporarily raises it again when reduction "
        "stalls with tests timing out. If set to <= 0 the adaptive timeout "
        "has no upper bound. Any commands that time out will be treated as "
        "failing the test"
    ),
)
@click.option(
    "--memory-limit",
    default=None,
    callback=_validate_memory_limit,
    help=(
        "Cap the address space of each interestingness-test subprocess so a "
        "runaway test cannot exhaust host memory. Accepts a byte count or a "
        "K/M/G/T suffix (e.g. '4G'). Set to 0 to disable. Defaults to the "
        "machine's physical RAM. Enforced via RLIMIT_AS, which is not honoured "
        "on macOS (there it only warns if the initial test exceeds it)."
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
    "--exit-on-completion/--no-exit-on-completion",
    default=True,
    help="Exit automatically when reduction completes (TUI only). Default: exit on completion.",
)
@click.option(
    "--history/--no-history",
    default=True,
    help="""
Record reduction history to a .shrinkray directory. Each run creates a unique
subdirectory containing the initial test case and all successful reductions.
This is useful for debugging and analyzing the reduction process.
Enabled by default; use --no-history to disable.
""".strip(),
)
@click.option(
    "--also-interesting",
    type=int,
    default=101,
    help="""
Exit code indicating a test case is interesting enough to record but should not
be used for reduction. When the test script returns this code, the test case is
saved to the also-interesting/ directory within the history folder.
If --no-history is passed, also-interesting recording is disabled unless
--also-interesting is explicitly specified (in which case only also-interesting
cases are recorded, not reductions). Set to 0 to disable. Default: 101.
""".strip(),
)
@click.option(
    "--reduce-with",
    "reduce_with",
    multiple=True,
    callback=validate_commands,
    help="""
An external reducer to run as a reduction pass, specified as a command. May be
passed more than once to run several reducers.

An external reducer is a program that shrink ray drives over its stdin/stdout
using a small JSON protocol (its stderr is logged to the .shrinkray directory).
See the documentation for the protocol.
""".strip(),
)
@click.option(
    "--python-reducer/--no-python-reducer",
    default=True,
    help="""
Run shrink ray's built-in libcst-based Python reducer (as an external reducer)
when the input looks like Python. Enabled by default; use --no-python-reducer
to disable it.
""".strip(),
)
@click.option(
    "--llm/--no-llm",
    "llm",
    default=False,
    help="""
Enable reduction passes that ask a language model, running locally in-process,
to propose smaller test cases. Requires the llm extra (install
'shrinkray[llm]'), and the first use downloads the default model (about 2.7GB)
from Hugging Face. Disabled by default.
""".strip(),
)
@click.option(
    "--llm-model",
    default=DEFAULT_MODEL_SPEC,
    show_default=True,
    help="""
The model the LLM passes use: either a path to a local .gguf file, or a
Hugging Face repo:filename reference naming a GGUF file to download.
""".strip(),
)
@click.option(
    "--llm-only",
    is_flag=True,
    default=False,
    help="""
Run only the LLM passes, disabling all of shrink ray's other reduction passes.
Implies --llm.
""".strip(),
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
    timeout: float | None,
    memory_limit: int | None,
    in_place: bool,
    parallelism: int,
    seed: int,
    volume: Volume,
    formatter: str,
    trivial_is_error: bool,
    exit_on_completion: bool,
    ui_type: UIType,
    theme: str,
    history: bool,
    also_interesting: int,
    reduce_with: list[list[str]],
    python_reducer: bool,
    llm: bool,
    llm_model: str,
    llm_only: bool,
) -> None:
    if timeout is not None and timeout <= 0:
        timeout = float("inf")

    if memory_limit is not None and not MEMORY_LIMIT_ENFORCEABLE:
        print(
            "Warning: --memory-limit cannot be enforced on this platform "
            "(macOS does not honour RLIMIT_AS); shrink ray will still warn if "
            "the initial test exceeds it, but later calls will not be capped.",
            file=sys.stderr,
        )

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

    # This is a debugging option so that when the reducer seems to be taking
    # a long time you can Ctrl-\ to find out what it's up to. I have no idea
    # how to test it in a way that shows up in coverage.
    def dump_trace(signum: int, frame: Any) -> None:  # pragma: no cover
        traceback.print_stack()

    signal.signal(signal.SIGQUIT, dump_trace)

    if not backup:
        backup = filename + os.extsep + "bak"

    # Run initial validation before any state setup
    # This validates the interestingness test and formatter with proper output streaming
    formatter_command = None
    if not os.path.isdir(filename) and formatter.lower() != "none":
        formatter_command = determine_formatter_command(formatter, filename)

    validation_result = run_validation(
        file_path=filename,
        test=test,
        input_type=input_type,
        in_place=in_place,
        formatter_command=formatter_command,
    )

    if not validation_result.success:
        print(f"\nError: {validation_result.error_message}", file=sys.stderr)
        sys.exit(1)

    if validation_result.formatter_works is False:
        # The formatter misbehaved on the initial test case; reduce without
        # it rather than aborting (validation already warned the user).
        formatter = "none"

    print("\nStarting reduction...", file=sys.stderr, flush=True)

    # Determine if --also-interesting was explicitly passed
    # If --no-history and --also-interesting not explicit, disable also-interesting
    ctx = click.get_current_context()

    if (
        llm_only
        and not llm
        and ctx.get_parameter_source("llm")
        == click.core.ParameterSource.COMMANDLINE
    ):
        raise click.UsageError("--llm-only cannot be combined with --no-llm.")
    llm_enabled = llm or llm_only
    if llm_enabled:
        if not llm_support_available():
            print(
                "The LLM passes need llama-cpp-python, which is not installed. "
                "Install shrink ray's llm extra (e.g. `uv tool install "
                "'shrinkray[llm]'`) or run without --llm.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            parse_model_spec(llm_model)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--llm-model")
    also_interesting_explicit = (
        ctx.get_parameter_source("also_interesting")
        == click.core.ParameterSource.COMMANDLINE
    )
    if also_interesting == 0:
        also_interesting_code: int | None = None
    elif not history and not also_interesting_explicit:
        # --no-history without explicit --also-interesting: disable both
        also_interesting_code = None
    else:
        also_interesting_code = also_interesting

    state_kwargs: dict[str, Any] = {
        "input_type": input_type,
        "in_place": in_place,
        "test": test,
        "timeout": timeout,
        "memory_limit": memory_limit,
        "base": os.path.basename(filename),
        "parallelism": parallelism,
        "filename": filename,
        "formatter": formatter,
        "trivial_is_error": trivial_is_error,
        "seed": seed,
        "volume": volume,
        "history_enabled": history,
        "also_interesting_code": also_interesting_code,
        "external_reducers": reduce_with,
        "python_reducer": python_reducer,
        "llm_enabled": llm_enabled,
        "llm_model": llm_model,
        "llm_only": llm_only,
    }

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

    if ui_type == UIType.textual:
        run_textual_ui(
            file_path=filename,
            test=test,
            parallelism=parallelism,
            timeout=timeout,
            memory_limit=memory_limit,
            seed=seed,
            input_type=input_type.name,
            in_place=in_place,
            formatter=formatter,
            volume=volume.name,
            trivial_is_error=trivial_is_error,
            exit_on_completion=exit_on_completion,
            theme=theme,  # type: ignore[arg-type]
            history_enabled=history,
            also_interesting_code=also_interesting_code,
            external_reducers=reduce_with,
            python_reducer=python_reducer,
            llm_enabled=llm_enabled,
            llm_model=llm_model,
            llm_only=llm_only,
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
    # Lazy import to avoid loading worker module in main process (fast CLI startup)
    from shrinkray.subprocess.worker import (  # noqa: I001, no-import-in-function
        main as worker_entry,
    )

    worker_entry()


if __name__ == "__main__":  # pragma: no cover
    main(prog_name="shrinkray")
