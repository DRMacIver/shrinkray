"""Tests for pass definitions and Format class."""

from shrinkray.passes.definitions import Format, ParseError


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
