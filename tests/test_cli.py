"""Tests for CLI utilities."""

import os
import sys
from unittest.mock import patch

import click
import pytest

from shrinkray.cli import EnumChoice, InputType, UIType, validate_command, validate_ui


class TestValidateCommand:
    """Tests for validate_command function."""

    def test_validates_existing_file(self, tmp_path):
        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho hello")
        script.chmod(0o755)

        result = validate_command(None, None, str(script))
        assert result == [str(script.resolve())]

    def test_validates_command_with_args(self, tmp_path):
        script = tmp_path / "test.sh"
        script.write_text("#!/bin/bash\necho hello")
        script.chmod(0o755)

        result = validate_command(None, None, f"{script} arg1 arg2")
        assert result == [str(script.resolve()), "arg1", "arg2"]

    def test_resolves_command_on_path(self):
        # 'ls' should be on PATH on any Unix system
        result = validate_command(None, None, "ls")
        assert os.path.isabs(result[0])
        assert os.path.basename(result[0]) == "ls"

    def test_resolves_command_on_path_with_args(self):
        result = validate_command(None, None, "ls -la")
        assert os.path.isabs(result[0])
        assert result[1:] == ["-la"]

    def test_raises_for_nonexistent_command(self):
        with pytest.raises(click.BadParameter, match="command not found"):
            validate_command(None, None, "nonexistent_command_xyz123")


class TestEnumChoice:
    """Tests for EnumChoice class."""

    def test_creates_choices_from_enum(self):
        choice = EnumChoice(InputType)
        assert "all" in choice.choices
        assert "stdin" in choice.choices
        assert "arg" in choice.choices
        assert "basename" in choice.choices

    def test_converts_string_to_enum(self):
        choice = EnumChoice(InputType)
        result = choice.convert("stdin", None, None)
        assert result == InputType.stdin

    def test_converts_all_values(self):
        choice = EnumChoice(UIType)
        assert choice.convert("basic", None, None) == UIType.basic
        assert choice.convert("textual", None, None) == UIType.textual


class TestInputType:
    """Tests for InputType enum."""

    def test_all_enables_everything(self):
        assert InputType.all.enabled(InputType.stdin) is True
        assert InputType.all.enabled(InputType.arg) is True
        assert InputType.all.enabled(InputType.basename) is True
        assert InputType.all.enabled(InputType.all) is True

    def test_specific_type_only_enables_itself(self):
        assert InputType.stdin.enabled(InputType.stdin) is True
        assert InputType.stdin.enabled(InputType.arg) is False
        assert InputType.stdin.enabled(InputType.basename) is False

        assert InputType.arg.enabled(InputType.arg) is True
        assert InputType.arg.enabled(InputType.stdin) is False

        assert InputType.basename.enabled(InputType.basename) is True
        assert InputType.basename.enabled(InputType.arg) is False


class TestValidateUI:
    """Tests for validate_ui function."""

    def test_returns_value_when_provided(self):
        assert validate_ui(None, None, UIType.basic) == UIType.basic
        assert validate_ui(None, None, UIType.textual) == UIType.textual

    def test_returns_textual_for_tty(self):
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = validate_ui(None, None, None)
                assert result == UIType.textual

    def test_returns_basic_for_non_tty_stdin(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            with patch.object(sys.stdout, "isatty", return_value=True):
                result = validate_ui(None, None, None)
                assert result == UIType.basic

    def test_returns_basic_for_non_tty_stdout(self):
        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdout, "isatty", return_value=False):
                result = validate_ui(None, None, None)
                assert result == UIType.basic

    def test_returns_basic_for_non_tty_both(self):
        with patch.object(sys.stdin, "isatty", return_value=False):
            with patch.object(sys.stdout, "isatty", return_value=False):
                result = validate_ui(None, None, None)
                assert result == UIType.basic
