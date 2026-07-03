"""CLI utilities and types for shrink ray."""

import os
import shlex
import sys
from enum import Enum, IntEnum, auto
from shutil import which
from typing import Any

import click


def validate_command(ctx: Any, param: Any, value: str) -> list[str]:
    """Validate and resolve a command string."""
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


class EnumChoice[EnumType: Enum](click.Choice):
    """A click Choice that works with Enums."""

    def __init__(self, enum: type[EnumType]) -> None:
        self.enum = enum
        choices = [str(e.name) for e in enum]
        self.__values = {e.name: e for e in enum}
        super().__init__(choices)

    def convert(self, value: str, param: Any, ctx: Any) -> EnumType:
        # Let click.Choice reject invalid values with a proper usage error.
        return self.__values[super().convert(value, param, ctx)]


class InputType(IntEnum):
    """How input is passed to the test function."""

    all = 0
    stdin = 1
    arg = 2
    basename = 3

    def enabled(self, value: "InputType") -> bool:
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
