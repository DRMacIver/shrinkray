"""Tests for display utilities."""

from shrinkray.display import format_diff, to_blocks, to_lines


class TestToLines:
    """Tests for to_lines function."""

    def test_simple_text(self):
        result = to_lines(b"hello\nworld")
        assert result == ["hello", "world"]

    def test_empty_input(self):
        result = to_lines(b"")
        assert result == [""]

    def test_single_line(self):
        result = to_lines(b"hello")
        assert result == ["hello"]

    def test_binary_line_shows_hex(self):
        # Binary content should be displayed as hex
        # Note: split on \n means the newline is not included in the line
        result = to_lines(b"\x00\x01\x02\n")
        assert result[0] == "000102"

    def test_mixed_text_and_binary(self):
        result = to_lines(b"hello\n\x00\x01\x02\nworld")
        assert result[0] == "hello"
        assert result[1] == "000102"
        assert result[2] == "world"

    def test_unicode_text(self):
        result = to_lines("héllo wörld".encode())
        assert result == ["héllo wörld"]

    def test_invalid_utf8_shows_hex(self):
        # Invalid UTF-8 bytes should be shown as hex
        result = to_lines(b"\xff\xfe")
        assert result == ["fffe"]


class TestToBlocks:
    """Tests for to_blocks function."""

    def test_empty_input(self):
        result = to_blocks(b"")
        assert result == []

    def test_small_input(self):
        result = to_blocks(b"hello")
        assert result == ["68656c6c6f"]

    def test_exactly_80_bytes(self):
        data = b"A" * 80
        result = to_blocks(data)
        assert len(result) == 1
        assert result[0] == "41" * 80

    def test_splits_at_80_bytes(self):
        data = b"A" * 160
        result = to_blocks(data)
        assert len(result) == 2
        assert result[0] == "41" * 80
        assert result[1] == "41" * 80

    def test_partial_last_block(self):
        data = b"A" * 100
        result = to_blocks(data)
        assert len(result) == 2
        assert result[0] == "41" * 80
        assert result[1] == "41" * 20


class TestFormatDiff:
    """Tests for format_diff function."""

    def test_empty_diff(self):
        result = format_diff([])
        assert result == ""

    def test_skips_header_lines(self):
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

    def test_starts_at_first_hunk(self):
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

    def test_truncates_long_diff(self):
        diff_lines = ["@@ -1,1 +1,1 @@"] + [f"line{i}" for i in range(600)]
        result = format_diff(diff_lines)
        lines = result.split("\n")
        assert len(lines) == 502  # 501 lines + "..."
        assert lines[-1] == "..."

    def test_no_hunk_header_returns_empty(self):
        diff_lines = [
            "--- a/file.txt",
            "+++ b/file.txt",
            "no hunk header here",
        ]
        result = format_diff(diff_lines)
        assert result == ""
