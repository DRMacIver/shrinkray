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


def to_blocks(test_case: bytes) -> list[str]:
    """Convert a test case to hex blocks for display."""
    return [test_case[i : i + 80].hex() for i in range(0, len(test_case), 80)]


def format_diff(diff: Iterable[str]) -> str:
    """Format a diff for display, truncating if too long."""
    results = []
    start_writing = False
    for line in diff:
        if not start_writing and line.startswith("@@"):
            start_writing = True
        if start_writing:
            results.append(line)
            if len(results) > 500:
                results.append("...")
                break
    return "\n".join(results)
