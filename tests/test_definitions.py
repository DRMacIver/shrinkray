"""Tests for pass definitions and Format class."""

from random import Random

from shrinkray.passes.definitions import Format, ParseError, compose
from shrinkray.problem import BasicReductionProblem
from shrinkray.work import Volume, WorkContext


class SimpleFormat(Format[str, int]):
    """A simple test format that parses strings to ints."""

    def parse(self, input: str) -> int:
        try:
            return int(input)
        except ValueError as e:
            raise ParseError(*e.args)

    def dumps(self, input: int) -> str:
        return str(input)


def test_format_default_name_uses_repr():
    """Test that Format.name returns repr(self) by default."""
    fmt = SimpleFormat()
    # Default name should be repr of the object
    assert "SimpleFormat" in fmt.name


def test_format_is_valid_for_parseable():
    """Test that Format.is_valid returns True for parseable input."""
    fmt = SimpleFormat()
    assert fmt.is_valid("42") is True


def test_format_is_valid_for_unparseable():
    """Test that Format.is_valid returns False for unparseable input."""
    fmt = SimpleFormat()
    assert fmt.is_valid("not a number") is False


async def test_compose_returns_early_on_parse_error():
    """Test that compose returns early when parsing fails after problem changes."""

    # Create a format that parses strings starting with "VALID:" successfully
    class SelectiveFormat(Format[str, str]):
        def parse(self, input: str) -> str:
            if input.startswith("VALID:"):
                return input[6:]
            raise ParseError("must start with VALID:")

        def dumps(self, input: str) -> str:
            return "VALID:" + input

    fmt = SelectiveFormat()

    # Track whether the pass was actually called
    pass_called = False

    async def tracking_pass(problem):
        nonlocal pass_called
        pass_called = True

    composed = compose(fmt, tracking_pass)

    # Start with a valid string so the View can be created
    current_value = "VALID:test"

    async def is_interesting(s: str) -> bool:
        return len(s) > 0

    work = WorkContext(random=Random(0), volume=Volume.quiet, parallelism=1)
    problem = BasicReductionProblem(
        initial=current_value, is_interesting=is_interesting, work=work
    )

    # First, create and cache the View by calling problem.view()
    # This succeeds because the current test case is still valid
    _ = problem.view(fmt)

    # Now change the problem's current test case to something that won't parse
    # We need to bypass normal update mechanisms to simulate a concurrent change
    problem._BasicReductionProblem__current = "INVALID"  # type: ignore[attr-defined]

    # Run the composed pass - it should get the cached View
    # Then when accessing view.current_test_case, it sees the problem changed
    # and tries to re-parse, which fails and triggers the except clause
    await composed(problem)

    # The inner pass should not have been called because re-parsing failed
    assert not pass_called
