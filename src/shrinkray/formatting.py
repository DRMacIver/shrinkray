"""Formatting utilities for shrink ray."""

import os
import sys
from shutil import which

import chardet

from shrinkray.reformat import basic_format


def find_python_command(name: str) -> str | None:
    """Find a Python command, checking both PATH and the current Python's bin directory."""
    first_attempt = which(name)
    if first_attempt is not None:
        return first_attempt
    second_attempt = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.exists(second_attempt):
        return second_attempt
    return None


def try_decode(data: bytes) -> tuple[str | None, str]:
    """Try to decode bytes using detected encoding."""
    for guess in chardet.detect_all(data):
        try:
            enc = guess["encoding"]
            if enc is not None:
                return enc, data.decode(enc)
        except UnicodeDecodeError:
            pass
    return None, ""


def default_formatter_command_for(filename: str) -> list[str] | str | None:
    """Get the default formatter command for a file based on its extension."""
    *_, ext = os.path.splitext(filename)

    if ext in (".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"):
        return which("clang-format")

    if ext == ".py":
        black = find_python_command("black")
        if black is not None:
            return [black, "-"]

    return None


def default_reformat_data(data: bytes) -> bytes:
    """Apply a simple language-agnostic reformatting to data.

    Delegates to the reflow formatter (``reformat.basic_format``), which
    canonicalises whitespace per language family (Python indentation, HTML/XML
    tags, and a brace/`;` default). This is the same canonicalisation the sort
    key orders by, so reformatting the final result only ever makes the layout
    match what the reducer was already preferring. Binary (undecodable) data is
    returned unchanged.
    """
    encoding, decoded = try_decode(data)
    if encoding is None:
        return data
    return basic_format(decoded).encode(encoding)


def determine_formatter_command(formatter: str, filename: str) -> list[str] | None:
    """Determine the formatter command to use based on settings and filename."""
    if formatter.lower() == "default":
        formatter_command = default_formatter_command_for(filename)
    elif formatter.lower() != "none":
        formatter_command = formatter
    else:
        formatter_command = None
    if isinstance(formatter_command, str):
        formatter_command = [formatter_command]
    return formatter_command
