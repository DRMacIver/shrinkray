"""CLI utilities and types for shrink ray."""

import os
import shlex
import sys
from enum import Enum, IntEnum, auto
from shutil import which
from typing import Any

import click


def resolve_command(value: str) -> list[str]:
    """Parse a command string into argv, resolving the executable on PATH.

    Raises click.BadParameter if the command is empty, unparseable, or the
    executable cannot be found.
    """
    try:
        parts = shlex.split(value)
    except ValueError as e:
        raise click.BadParameter(f"Could not parse command: {e}")
    if not parts:
        raise click.BadParameter("Command cannot be empty.")
    command = parts[0]

    if os.path.exists(command):
        command = os.path.abspath(command)
    else:
        what = which(command)
        if what is None:
            raise click.BadParameter(f"{command}: command not found")
        command = os.path.abspath(what)
    return [command] + parts[1:]


def validate_command(ctx: Any, param: Any, value: str) -> list[str]:
    """Validate and resolve a single command string (click callback)."""
    return resolve_command(value)


def validate_commands(ctx: Any, param: Any, value: tuple[str, ...]) -> list[list[str]]:
    """Validate and resolve a tuple of command strings (click callback).

    Used for repeatable options like --reduce-with.
    """
    return [resolve_command(v) for v in value]


class EnumChoice[EnumType: Enum](click.Choice):
    """A click Choice that works with Enums."""

    def __init__(self, enum: type[EnumType]) -> None:
        self.enum = enum
        choices = [str(e.name) for e in enum]
        self.__values = {e.name: e for e in enum}
        super().__init__(choices)

    def convert(self, value: str | EnumType, param: Any, ctx: Any) -> EnumType:
        # click may call convert() with an already-converted value (e.g.
        # when processing a default a second time), so enum members must
        # pass through unchanged.
        if isinstance(value, self.enum):
            return value
        # Let click.Choice reject invalid values with a proper usage error.
        return self.__values[super().convert(value, param, ctx)]


class InputType(IntEnum):
    """How input is passed to the test function."""

    all = 0
    stdin = 1
    arg = 2
    basename = 3

    def enabled(self, value: InputType) -> bool:
        if self == InputType.all:
            return True
        return self == value


class UIType(Enum):
    """Type of UI to use."""

    basic = auto()
    textual = auto()


def validate_ui(ctx, param, value) -> UIType:
    """Validate and determine UI type."""
    if value is None:
        if sys.stdin.isatty() and sys.stdout.isatty():
            return UIType.textual
        else:
            return UIType.basic
    else:
        return value
