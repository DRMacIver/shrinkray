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


# === find_python_command tests ===


def test_find_python_command_on_path():
    # python should be on PATH
    result = find_python_command("python")
    assert result is not None
    assert "python" in result


def test_find_python_command_in_python_bin_dir():
    # pip should be in the same directory as the Python executable
    result = find_python_command("pip")
    assert result is not None


def test_find_python_command_returns_none_for_nonexistent():
    result = find_python_command("nonexistent_command_xyz123")
    assert result is None


def test_find_python_command_checks_python_bin_directory():
    # Test that it checks sys.executable's directory
    with patch("shrinkray.formatting.which", return_value=None):
        with patch("os.path.exists", return_value=True):
            result = find_python_command("some_tool")
            expected = os.path.join(os.path.dirname(sys.executable), "some_tool")
            assert result == expected


def test_find_python_command_returns_none_when_not_in_bin_dir():
    with patch("shrinkray.formatting.which", return_value=None):
        with patch("os.path.exists", return_value=False):
            result = find_python_command("nonexistent")
            assert result is None


# === try_decode tests ===


def test_try_decode_utf8():
    data = b"hello world"
    encoding, decoded = try_decode(data)
    assert encoding is not None
    assert decoded == "hello world"


def test_try_decode_latin1():
    data = "héllo wörld".encode("latin-1")
    encoding, decoded = try_decode(data)
    assert encoding is not None
    assert "hello" in decoded.lower() or "h" in decoded


def test_try_decode_returns_none_for_undecodable():
    # Random bytes that don't form valid text
    data = bytes(range(128, 256))
    encoding, decoded = try_decode(data)
    # chardet can't decode random high bytes
    assert encoding is None
    assert decoded == ""


def test_try_decode_handles_empty_data():
    encoding, decoded = try_decode(b"")
    # chardet returns None for empty data
    assert encoding is None
    assert decoded == ""


# === default_formatter_command_for tests ===


def test_default_formatter_c_files_use_clang_format():
    for ext in [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"]:
        result = default_formatter_command_for(f"test{ext}")
        # May be None if clang-format not installed
        if result is not None:
            assert "clang-format" in result


def test_default_formatter_python_files_use_black():
    result = default_formatter_command_for("test.py")
    # May be None if black not installed
    if result is not None:
        assert isinstance(result, list)
        assert "black" in result[0]
        assert result[1] == "-"


def test_default_formatter_unknown_extension_returns_none():
    result = default_formatter_command_for("test.xyz")
    assert result is None


def test_default_formatter_no_extension_returns_none():
    result = default_formatter_command_for("Makefile")
    assert result is None


# === default_reformat_data tests ===


def test_default_reformat_normalizes_indentation():
    data = b"{\n    x;\n}"
    result = default_reformat_data(data)
    assert b"{" in result
    assert b"x;" in result
    assert b"}" in result


def test_default_reformat_handles_braces():
    data = b"function() { x; y; }"
    result = default_reformat_data(data)
    # Should have newlines after braces
    assert b"\n" in result


def test_default_reformat_handles_empty_braces():
    data = b"function() {}"
    result = default_reformat_data(data)
    assert b"{}" in result


def test_default_reformat_handles_semicolons():
    data = b"a; b; c;"
    result = default_reformat_data(data)
    # Each semicolon should be followed by a newline
    assert result.count(b"\n") >= 2


def test_default_reformat_preserves_binary_data():
    # Binary data should be returned unchanged
    data = bytes(range(256))
    result = default_reformat_data(data)
    assert result == data


def test_default_reformat_removes_trailing_spaces():
    data = b"hello   \nworld"
    result = default_reformat_data(data)
    assert b"   \n" not in result


def test_default_reformat_removes_multiple_newlines():
    data = b"hello\n\n\nworld"
    result = default_reformat_data(data)
    assert b"\n\n" not in result


def test_default_reformat_handles_nested_braces():
    data = b"{ { x; } }"
    result = default_reformat_data(data)
    # Should properly indent nested content
    assert b"x;" in result


def test_default_reformat_strips_leading_spaces_on_newlines():
    data = b"hello\n    world"
    result = default_reformat_data(data)
    # Leading spaces after newline should be normalized
    decoded = result.decode("utf-8")
    assert "world" in decoded


# === determine_formatter_command tests ===


def test_determine_formatter_default_uses_default():
    result = determine_formatter_command("default", "test.py")
    # May be None if black not installed, or a list
    if result is not None:
        assert isinstance(result, list)


def test_determine_formatter_none_returns_none():
    result = determine_formatter_command("none", "test.py")
    assert result is None


def test_determine_formatter_none_case_insensitive():
    assert determine_formatter_command("None", "test.py") is None
    assert determine_formatter_command("NONE", "test.py") is None


def test_determine_formatter_custom_as_string():
    result = determine_formatter_command("/usr/bin/my-formatter", "test.py")
    assert result == ["/usr/bin/my-formatter"]


def test_determine_formatter_default_case_insensitive():
    # Both should use default behavior
    r1 = determine_formatter_command("default", "test.xyz")
    r2 = determine_formatter_command("DEFAULT", "test.xyz")
    assert r1 == r2


def test_determine_formatter_wraps_string_in_list():
    result = determine_formatter_command("my-formatter", "test.txt")
    assert result == ["my-formatter"]
