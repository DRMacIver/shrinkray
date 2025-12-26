"""Display utilities for shrink ray."""

import shutil
from collections.abc import Iterable

from binaryornot.check import is_binary_string  # type: ignore[import-not-found]


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size, with sensible fallbacks.

    Returns:
        (columns, lines) tuple. Defaults to (80, 24) if terminal size
        cannot be determined.
    """
    size = shutil.get_terminal_size(fallback=(80, 24))
    return (size.columns, size.lines)


def to_lines(test_case: bytes) -> list[str]:
    """Convert a test case to displayable lines."""
    result = []
    for line in test_case.split(b"\n"):
        if is_binary_string(line):
            result.append(line.hex())
        else:
            try:
                result.append(line.decode("utf-8"))
            except UnicodeDecodeError:
                result.append(line.hex())
    return result


def to_blocks(test_case: bytes, block_size: int | None = None) -> list[str]:
    """Convert a test case to hex blocks for display.

    Args:
        test_case: The bytes to convert
        block_size: Number of bytes per block. If None, automatically
                    calculated from terminal width (each byte becomes 2 hex chars).
    """
    if block_size is None:
        columns, _ = get_terminal_size()
        # Each byte becomes 2 hex chars, leave some margin
        block_size = max(1, (columns - 4) // 2)
    return [
        test_case[i : i + block_size].hex()
        for i in range(0, len(test_case), block_size)
    ]


def format_diff(diff: Iterable[str], max_lines: int | None = None) -> str:
    """Format a diff for display, truncating if too long.

    Args:
        diff: Iterable of diff lines
        max_lines: Maximum number of lines to include. If None, uses
                   terminal height multiplied by a factor to allow scrolling
                   through substantial context.
    """
    if max_lines is None:
        _, lines = get_terminal_size()
        # Allow multiple screenfuls of context for scrolling
        max_lines = max(lines * 20, 100)
    results = []
    start_writing = False
    for line in diff:
        if not start_writing and line.startswith("@@"):
            start_writing = True
        if start_writing:
            results.append(line)
            if len(results) > max_lines:
                results.append("...")
                break
    return "\n".join(results)
