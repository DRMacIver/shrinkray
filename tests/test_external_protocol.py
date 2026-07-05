import base64
import json

import pytest
import trio
from hypothesis import given
from hypothesis import strategies as st

from shrinkray.reducers.protocol import (
    Feedback,
    Idle,
    LineReader,
    Query,
    ReduceRequest,
    decode_feedback,
    decode_query,
    encode_feedback,
    encode_idle,
    encode_query,
    encode_reduce,
    parse_from_reducer,
    parse_to_reducer,
)


# === Query encode/decode ===


@given(st.binary())
def test_query_round_trip(content: bytes) -> None:
    line = encode_query(content)
    assert line.endswith(b"\n")
    assert decode_query(line) == content


def test_query_line_is_single_line() -> None:
    line = encode_query(b"a\nb\nc")
    # The newline in the content must not leak into the transport framing.
    assert line.count(b"\n") == 1


def test_decode_query_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        decode_query(b"[1, 2, 3]")


def test_decode_query_rejects_missing_content() -> None:
    with pytest.raises(ValueError):
        decode_query(json.dumps({"nope": 1}))


def test_decode_query_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        decode_query(b"not json")


# === Feedback encode/decode ===


@given(st.binary(), st.booleans())
def test_feedback_round_trip(content: bytes, interesting: bool) -> None:
    line = encode_feedback(content, interesting)
    assert line.endswith(b"\n")
    assert decode_feedback(line) == (content, interesting)


def test_feedback_line_is_single_line() -> None:
    line = encode_feedback(b"x\ny", True)
    assert line.count(b"\n") == 1


def test_encode_feedback_coerces_truthy_to_bool() -> None:
    line = encode_feedback(b"", 1)  # type: ignore[arg-type]
    obj = json.loads(line)
    assert obj["interesting"] is True


def test_decode_feedback_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        decode_feedback(b'"a string"')


def test_decode_feedback_rejects_missing_content() -> None:
    with pytest.raises(ValueError):
        decode_feedback(json.dumps({"interesting": True}))


def test_decode_feedback_rejects_missing_interesting() -> None:
    content = base64.b64encode(b"x").decode()
    with pytest.raises(ValueError):
        decode_feedback(json.dumps({"content": content}))


# === reduce / idle + message classification ===


@given(st.binary())
def test_parse_to_reducer_classifies_reduce(content: bytes) -> None:
    assert parse_to_reducer(encode_reduce(content)) == ReduceRequest(content)


@given(st.binary(), st.booleans())
def test_parse_to_reducer_classifies_feedback(content: bytes, ok: bool) -> None:
    assert parse_to_reducer(encode_feedback(content, ok)) == Feedback(content, ok)


def test_parse_to_reducer_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_to_reducer(json.dumps({"nope": 1}))


def test_parse_to_reducer_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_to_reducer(b"[1, 2]")


@given(st.binary())
def test_parse_from_reducer_classifies_query(content: bytes) -> None:
    assert parse_from_reducer(encode_query(content)) == Query(content)


def test_parse_from_reducer_classifies_idle() -> None:
    assert parse_from_reducer(encode_idle()) == Idle()


def test_parse_from_reducer_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        parse_from_reducer(json.dumps({"nope": 1}))


def test_parse_from_reducer_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        parse_from_reducer(b"42")


# === LineReader ===


class ChunkStream:
    """A fake receive stream that yields a fixed list of chunks then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        await trio.lowlevel.checkpoint()
        if self._chunks:
            return self._chunks.pop(0)
        return b""


async def test_line_reader_splits_lines() -> None:
    reader = LineReader(ChunkStream([b"one\ntwo\nthree\n"]))
    assert await reader.readline() == b"one"
    assert await reader.readline() == b"two"
    assert await reader.readline() == b"three"
    assert await reader.readline() is None


async def test_line_reader_reassembles_partial_chunks() -> None:
    reader = LineReader(ChunkStream([b"he", b"ll", b"o\nwor", b"ld\n"]))
    assert await reader.readline() == b"hello"
    assert await reader.readline() == b"world"
    assert await reader.readline() is None


async def test_line_reader_returns_trailing_line_without_newline() -> None:
    reader = LineReader(ChunkStream([b"a\nb"]))
    assert await reader.readline() == b"a"
    assert await reader.readline() == b"b"
    assert await reader.readline() is None


async def test_line_reader_immediate_eof() -> None:
    reader = LineReader(ChunkStream([]))
    assert await reader.readline() is None


async def test_line_reader_handles_blank_lines() -> None:
    reader = LineReader(ChunkStream([b"\n\n"]))
    assert await reader.readline() == b""
    assert await reader.readline() == b""
    assert await reader.readline() is None
