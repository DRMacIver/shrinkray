"""Display utilities for shrink ray."""

from collections.abc import Iterable

from binaryornot.check import is_binary_string  # type: ignore[import-not-found]


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


def to_blocks(test_case: bytes, block_size: int = 80) -> list[str]:
    """Convert a test case to hex blocks for display.

    Args:
        test_case: The bytes to convert
        block_size: Number of bytes per block (default 80 for ~160 hex chars,
                    fitting most terminal widths)
    """
    return [
        test_case[i : i + block_size].hex()
        for i in range(0, len(test_case), block_size)
    ]


def format_diff(diff: Iterable[str], max_lines: int = 500) -> str:
    """Format a diff for display, truncating if too long.

    Args:
        diff: Iterable of diff lines
        max_lines: Maximum number of lines to include (default 500, enough
                   to show substantial context without overwhelming output)
    """
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
