"""Tests for CLI utilities."""

import os
import sys
from unittest.mock import patch

import click
import pytest

from shrinkray.cli import (
    EnumChoice,
    InputType,
    UIType,
    validate_command,
    validate_commands,
    validate_ui,
)


# === validate_command tests ===


def test_validate_command_existing_file(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho hello")
    script.chmod(0o755)

    result = validate_command(None, None, str(script))
    assert result == [str(script.resolve())]


def test_validate_command_with_args(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text("#!/bin/bash\necho hello")
    script.chmod(0o755)

    result = validate_command(None, None, f"{script} arg1 arg2")
    assert result == [str(script.resolve()), "arg1", "arg2"]


def test_validate_command_resolves_on_path():
    # 'ls' should be on PATH on any Unix system
    result = validate_command(None, None, "ls")
    assert os.path.isabs(result[0])
    assert os.path.basename(result[0]) == "ls"


def test_validate_command_resolves_on_path_with_args():
    result = validate_command(None, None, "ls -la")
    assert os.path.isabs(result[0])
    assert result[1:] == ["-la"]


def test_validate_command_raises_for_nonexistent():
    with pytest.raises(click.BadParameter, match="command not found"):
        validate_command(None, None, "nonexistent_command_xyz123")


def test_validate_command_raises_for_empty_command():
    # Regression test: an empty command produced an IndexError traceback
    # instead of a clean usage error.
    with pytest.raises(click.BadParameter, match="empty"):
        validate_command(None, None, "")


def test_validate_command_raises_for_unparseable_command():
    # Regression test: an unclosed quote raised ValueError from shlex.
    with pytest.raises(click.BadParameter):
        validate_command(None, None, "'unclosed quote")


# === validate_commands tests (repeatable --reduce-with) ===


def test_validate_commands_empty_tuple():
    assert validate_commands(None, None, ()) == []


def test_validate_commands_resolves_each():
    result = validate_commands(None, None, ("ls -la", "ls"))
    assert len(result) == 2
    assert os.path.basename(result[0][0]) == "ls"
    assert result[0][1:] == ["-la"]
    assert os.path.basename(result[1][0]) == "ls"


def test_validate_commands_raises_for_nonexistent():
    with pytest.raises(click.BadParameter, match="command not found"):
        validate_commands(None, None, ("ls", "nonexistent_command_xyz123"))


# === EnumChoice tests ===


def test_enum_choice_creates_choices():
    choice = EnumChoice(InputType)
    assert "all" in choice.choices
    assert "stdin" in choice.choices
    assert "arg" in choice.choices
    assert "basename" in choice.choices


def test_enum_choice_converts_string():
    choice = EnumChoice(InputType)
    result = choice.convert("stdin", None, None)
    assert result == InputType.stdin


def test_enum_choice_converts_all_values():
    choice = EnumChoice(UIType)
    assert choice.convert("basic", None, None) == UIType.basic
    assert choice.convert("textual", None, None) == UIType.textual


def test_enum_choice_accepts_already_converted_value():
    # Regression test: click may call convert() with an already-converted
    # value (it does this for option defaults in some versions), which
    # produced a bad-parameter error instead of passing the enum through.
    choice = EnumChoice(InputType)
    result = choice.convert(InputType.stdin, None, None)
    assert result is InputType.stdin


def test_enum_choice_rejects_invalid_value():
    # Regression test: an invalid choice produced a raw KeyError traceback
    # instead of click's "invalid choice" usage error.
    choice = EnumChoice(InputType)
    with pytest.raises(click.UsageError):
        choice.convert("bogus", None, None)


# === InputType tests ===


def test_input_type_all_enables_everything():
    assert InputType.all.enabled(InputType.stdin) is True
    assert InputType.all.enabled(InputType.arg) is True
    assert InputType.all.enabled(InputType.basename) is True
    assert InputType.all.enabled(InputType.all) is True


def test_input_type_specific_only_enables_itself():
    assert InputType.stdin.enabled(InputType.stdin) is True
    assert InputType.stdin.enabled(InputType.arg) is False
    assert InputType.stdin.enabled(InputType.basename) is False

    assert InputType.arg.enabled(InputType.arg) is True
    assert InputType.arg.enabled(InputType.stdin) is False

    assert InputType.basename.enabled(InputType.basename) is True
    assert InputType.basename.enabled(InputType.arg) is False


# === validate_ui tests ===


def test_validate_ui_returns_value_when_provided():
    assert validate_ui(None, None, UIType.basic) == UIType.basic
    assert validate_ui(None, None, UIType.textual) == UIType.textual


def test_validate_ui_returns_textual_for_tty():
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(sys.stdout, "isatty", return_value=True),
    ):
        result = validate_ui(None, None, None)
        assert result == UIType.textual


def test_validate_ui_returns_basic_for_non_tty_stdin():
    with (
        patch.object(sys.stdin, "isatty", return_value=False),
        patch.object(sys.stdout, "isatty", return_value=True),
    ):
        result = validate_ui(None, None, None)
        assert result == UIType.basic


def test_validate_ui_returns_basic_for_non_tty_stdout():
    with (
        patch.object(sys.stdin, "isatty", return_value=True),
        patch.object(sys.stdout, "isatty", return_value=False),
    ):
        result = validate_ui(None, None, None)
        assert result == UIType.basic


def test_validate_ui_returns_basic_for_non_tty_both():
    with (
        patch.object(sys.stdin, "isatty", return_value=False),
        patch.object(sys.stdout, "isatty", return_value=False),
    ):
        result = validate_ui(None, None, None)
        assert result == UIType.basic
