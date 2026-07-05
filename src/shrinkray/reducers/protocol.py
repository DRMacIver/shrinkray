"""Wire protocol for external reducers.

An external reducer is a subprocess that communicates with shrink ray over
stdin/stdout using newline-delimited JSON messages. Test-case content is
base64-encoded so that arbitrary bytes survive the (textual) JSON transport.

Messages shrink ray sends to the reducer:

- **feedback** ``{"content": <base64>, "interesting": <bool>}`` — either the
  result of the interestingness test for a query the reducer emitted, or, when
  the content matches no outstanding query, the current test case for the
  reducer to work on. The reducer correlates a reply with a query by its
  ``content``; a feedback message it did not ask for (interesting, unmatched) is
  a fresh current test case to reduce to a fixpoint and then report ``idle`` on.
  The first message the reducer receives is one of these, carrying the initial
  test case.

Messages the reducer sends to shrink ray:

- **query** ``{"content": <base64>}`` — a candidate it wants evaluated.
- **idle** ``{"idle": true}`` — "I have reached a fixpoint for the current test
  case; I will do nothing until I am given a new one".

The reducer stays alive between test cases so that expensive startup (such as
importing libcst) happens once. A reducer that instead exits at a fixpoint
(closing its stdout) is also supported: shrink ray relaunches it next time.
"""

import base64
import json
from dataclasses import dataclass
from typing import Protocol


def encode_query(content: bytes) -> bytes:
    """Encode a candidate query (reducer -> shrink ray) as a protocol line."""
    obj = {"content": base64.b64encode(content).decode("ascii")}
    return (json.dumps(obj) + "\n").encode("utf-8")


def decode_query(line: bytes | str) -> bytes:
    """Decode a query line into its content bytes.

    Raises ValueError (or a subclass such as json.JSONDecodeError) if the line
    is not a well-formed query.
    """
    obj = json.loads(line)
    if not isinstance(obj, dict) or "content" not in obj:
        raise ValueError("query message must be an object with a 'content' field")
    return base64.b64decode(obj["content"])


def encode_feedback(content: bytes, interesting: bool) -> bytes:
    """Encode a feedback message (shrink ray -> reducer) as a protocol line."""
    obj = {
        "content": base64.b64encode(content).decode("ascii"),
        "interesting": bool(interesting),
    }
    return (json.dumps(obj) + "\n").encode("utf-8")


def decode_feedback(line: bytes | str) -> tuple[bytes, bool]:
    """Decode a feedback line into ``(content, interesting)``.

    Raises ValueError (or a subclass such as json.JSONDecodeError) if the line
    is not a well-formed feedback message.
    """
    obj = json.loads(line)
    if not isinstance(obj, dict) or "content" not in obj or "interesting" not in obj:
        raise ValueError(
            "feedback message must be an object with 'content' and 'interesting'"
        )
    return base64.b64decode(obj["content"]), bool(obj["interesting"])


def encode_idle() -> bytes:
    """Encode an idle notification (reducer -> shrink ray) as a protocol line."""
    return (json.dumps({"idle": True}) + "\n").encode("utf-8")


# === Reducer -> shrink ray message classification ===
#
# This direction carries two message kinds, so the reader classifies a line into
# a small tagged object rather than assuming which kind it is.


@dataclass(frozen=True)
class Query:
    """The reducer asks whether ``content`` is interesting."""

    content: bytes


@dataclass(frozen=True)
class Idle:
    """The reducer reports it has reached a fixpoint."""


def parse_from_reducer(line: bytes | str) -> Query | Idle:
    """Classify a message the reducer sends to shrink ray.

    Raises ValueError if the line is not a recognised message.
    """
    obj = json.loads(line)
    if isinstance(obj, dict):
        if obj.get("idle") is True:
            return Idle()
        if "content" in obj:
            return Query(base64.b64decode(obj["content"]))
    raise ValueError("unrecognised message from reducer")


class ReceiveStream(Protocol):
    """The subset of a trio receive stream that :class:`LineReader` needs."""

    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray: ...


class LineReader:
    """Buffered newline-delimited reader over an async byte stream.

    Wraps any object with an awaitable ``receive_some`` method (such as a trio
    ``ReceiveStream``) and yields whole lines, buffering partial reads. A final
    line without a trailing newline is returned before EOF.
    """

    def __init__(self, stream: ReceiveStream, max_chunk: int = 65536) -> None:
        self._stream = stream
        self._buffer = bytearray()
        self._max_chunk = max_chunk
        self._eof = False

    async def readline(self) -> bytes | None:
        """Return the next line (without its trailing newline), or None at EOF."""
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            if self._eof:
                if self._buffer:
                    line = bytes(self._buffer)
                    self._buffer.clear()
                    return line
                return None
            chunk = await self._stream.receive_some(self._max_chunk)
            if chunk:
                self._buffer.extend(chunk)
            else:
                self._eof = True
