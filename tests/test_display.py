"""Tests for display utilities."""

from shrinkray.display import format_diff, to_blocks, to_lines


# === to_lines tests ===


def test_to_lines_simple_text():
    result = to_lines(b"hello\nworld")
    assert result == ["hello", "world"]


def test_to_lines_empty_input():
    result = to_lines(b"")
    assert result == [""]


def test_to_lines_single_line():
    result = to_lines(b"hello")
    assert result == ["hello"]


def test_to_lines_binary_line_shows_hex():
    # Binary content should be displayed as hex
    # Note: split on \n means the newline is not included in the line
    result = to_lines(b"\x00\x01\x02\n")
    assert result[0] == "000102"


def test_to_lines_mixed_text_and_binary():
    result = to_lines(b"hello\n\x00\x01\x02\nworld")
    assert result[0] == "hello"
    assert result[1] == "000102"
    assert result[2] == "world"


def test_to_lines_unicode_text():
    result = to_lines("héllo wörld".encode())
    assert result == ["héllo wörld"]


def test_to_lines_invalid_utf8_shows_hex():
    # Invalid UTF-8 bytes should be shown as hex
    result = to_lines(b"\xff\xfe")
    assert result == ["fffe"]


# === to_blocks tests ===


def test_to_blocks_empty_input():
    result = to_blocks(b"")
    assert result == []


def test_to_blocks_small_input():
    result = to_blocks(b"hello")
    assert result == ["68656c6c6f"]


def test_to_blocks_exactly_80_bytes():
    data = b"A" * 80
    result = to_blocks(data, block_size=80)
    assert len(result) == 1
    assert result[0] == "41" * 80


def test_to_blocks_splits_at_80_bytes():
    data = b"A" * 160
    result = to_blocks(data, block_size=80)
    assert len(result) == 2
    assert result[0] == "41" * 80
    assert result[1] == "41" * 80


def test_to_blocks_partial_last_block():
    data = b"A" * 100
    result = to_blocks(data, block_size=80)
    assert len(result) == 2
    assert result[0] == "41" * 80
    assert result[1] == "41" * 20


# === format_diff tests ===


def test_format_diff_empty():
    result = format_diff([])
    assert result == ""


def test_format_diff_skips_header_lines():
    diff_lines = [
        "--- a/file.txt",
        "+++ b/file.txt",
        "@@ -1,3 +1,2 @@",
        " line1",
        "-line2",
        " line3",
    ]
    result = format_diff(diff_lines)
    assert result == "@@ -1,3 +1,2 @@\n line1\n-line2\n line3"


def test_format_diff_starts_at_first_hunk():
    diff_lines = [
        "diff --git a/file.txt b/file.txt",
        "index abc123..def456 100644",
        "--- a/file.txt",
        "+++ b/file.txt",
        "@@ -1,3 +1,2 @@",
        " context",
    ]
    result = format_diff(diff_lines)
    assert result.startswith("@@ -1,3 +1,2 @@")


def test_format_diff_truncates_long_diff():
    diff_lines = ["@@ -1,1 +1,1 @@"] + [f"line{i}" for i in range(600)]
    result = format_diff(diff_lines, max_lines=500)
    lines = result.split("\n")
    # format_diff stops after 501 lines and appends "..."
    assert len(lines) == 502
    assert lines[-1] == "..."


def test_format_diff_no_hunk_header_returns_empty():
    diff_lines = [
        "--- a/file.txt",
        "+++ b/file.txt",
        "no hunk header here",
    ]
    result = format_diff(diff_lines)
    assert result == ""
