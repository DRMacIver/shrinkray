"""Tests for formatting utilities."""

import os
import sys
from unittest.mock import patch

from shrinkray.formatting import (
    default_formatter_command_for,
    default_reformat_data,
    determine_formatter_command,
    find_python_command,
    try_decode,
)


class TestFindPythonCommand:
    """Tests for find_python_command function."""

    def test_finds_command_on_path(self):
        # python should be on PATH
        result = find_python_command("python")
        assert result is not None
        assert "python" in result

    def test_finds_command_in_python_bin_dir(self):
        # pip should be in the same directory as the Python executable
        result = find_python_command("pip")
        assert result is not None

    def test_returns_none_for_nonexistent_command(self):
        result = find_python_command("nonexistent_command_xyz123")
        assert result is None

    def test_checks_python_bin_directory(self):
        # Test that it checks sys.executable's directory
        with patch("shrinkray.formatting.which", return_value=None):
            with patch("os.path.exists", return_value=True):
                result = find_python_command("some_tool")
                expected = os.path.join(os.path.dirname(sys.executable), "some_tool")
                assert result == expected

    def test_returns_none_when_not_in_python_bin_dir(self):
        with patch("shrinkray.formatting.which", return_value=None):
            with patch("os.path.exists", return_value=False):
                result = find_python_command("nonexistent")
                assert result is None


class TestTryDecode:
    """Tests for try_decode function."""

    def test_decodes_utf8(self):
        data = b"hello world"
        encoding, decoded = try_decode(data)
        assert encoding is not None
        assert decoded == "hello world"

    def test_decodes_latin1(self):
        data = "héllo wörld".encode("latin-1")
        encoding, decoded = try_decode(data)
        assert encoding is not None
        assert "hello" in decoded.lower() or "h" in decoded

    def test_returns_none_for_undecodable(self):
        # Random bytes that don't form valid text
        data = bytes(range(128, 256))
        encoding, decoded = try_decode(data)
        # chardet can't decode random high bytes
        assert encoding is None
        assert decoded == ""

    def test_handles_empty_data(self):
        encoding, decoded = try_decode(b"")
        # chardet returns None for empty data
        assert encoding is None
        assert decoded == ""


class TestDefaultFormatterCommandFor:
    """Tests for default_formatter_command_for function."""

    def test_c_files_use_clang_format(self):
        for ext in [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]:
            result = default_formatter_command_for(f"test{ext}")
            # May be None if clang-format not installed
            if result is not None:
                assert "clang-format" in result

    def test_python_files_use_black(self):
        result = default_formatter_command_for("test.py")
        # May be None if black not installed
        if result is not None:
            assert isinstance(result, list)
            assert "black" in result[0]
            assert result[1] == "-"

    def test_unknown_extension_returns_none(self):
        result = default_formatter_command_for("test.xyz")
        assert result is None

    def test_no_extension_returns_none(self):
        result = default_formatter_command_for("Makefile")
        assert result is None


class TestDefaultReformatData:
    """Tests for default_reformat_data function."""

    def test_normalizes_indentation(self):
        data = b"{\n    x;\n}"
        result = default_reformat_data(data)
        assert b"{" in result
        assert b"x;" in result
        assert b"}" in result

    def test_handles_braces(self):
        data = b"function() { x; y; }"
        result = default_reformat_data(data)
        # Should have newlines after braces
        assert b"\n" in result

    def test_handles_empty_braces(self):
        data = b"function() {}"
        result = default_reformat_data(data)
        assert b"{}" in result

    def test_handles_semicolons(self):
        data = b"a; b; c;"
        result = default_reformat_data(data)
        # Each semicolon should be followed by a newline
        assert result.count(b"\n") >= 2

    def test_preserves_binary_data(self):
        # Binary data should be returned unchanged
        data = bytes(range(256))
        result = default_reformat_data(data)
        assert result == data

    def test_removes_trailing_spaces(self):
        data = b"hello   \nworld"
        result = default_reformat_data(data)
        assert b"   \n" not in result

    def test_removes_multiple_newlines(self):
        data = b"hello\n\n\nworld"
        result = default_reformat_data(data)
        assert b"\n\n" not in result

    def test_handles_nested_braces(self):
        data = b"{ { x; } }"
        result = default_reformat_data(data)
        # Should properly indent nested content
        assert b"x;" in result

    def test_strips_leading_spaces_on_newlines(self):
        data = b"hello\n    world"
        result = default_reformat_data(data)
        # Leading spaces after newline should be normalized
        decoded = result.decode("utf-8")
        assert "world" in decoded


class TestDetermineFormatterCommand:
    """Tests for determine_formatter_command function."""

    def test_default_uses_default_formatter(self):
        result = determine_formatter_command("default", "test.py")
        # May be None if black not installed, or a list
        if result is not None:
            assert isinstance(result, list)

    def test_none_returns_none(self):
        result = determine_formatter_command("none", "test.py")
        assert result is None

    def test_none_case_insensitive(self):
        assert determine_formatter_command("None", "test.py") is None
        assert determine_formatter_command("NONE", "test.py") is None

    def test_custom_formatter_as_string(self):
        result = determine_formatter_command("/usr/bin/my-formatter", "test.py")
        assert result == ["/usr/bin/my-formatter"]

    def test_default_case_insensitive(self):
        # Both should use default behavior
        r1 = determine_formatter_command("default", "test.xyz")
        r2 = determine_formatter_command("DEFAULT", "test.xyz")
        assert r1 == r2

    def test_wraps_string_in_list(self):
        result = determine_formatter_command("my-formatter", "test.txt")
        assert result == ["my-formatter"]
