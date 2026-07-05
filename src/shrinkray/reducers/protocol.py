"""Wire protocol for external reducers.

An external reducer is a subprocess that communicates with shrink ray over
stdin/stdout using newline-delimited JSON messages. Test-case content is
base64-encoded so that arbitrary bytes survive the (textual) JSON transport.

There are two message types:

- **query** (reducer -> shrink ray): ``{"content": <base64>}``

  The reducer emits a candidate test case it wants evaluated.

- **feedback** (shrink ray -> reducer): ``{"content": <base64>, "interesting": <bool>}``

  Shrink ray replies to each query with the result of its interestingness
  test. The reducer is also sent one feedback message on launch (the initial
  test case, with ``interesting`` true), and may be sent further unsolicited
  feedback messages when the current test case changes underneath it (for
  example because another pass reduced it, or because it was reformatted).

The reducer correlates a feedback message with a query by its ``content``:
feedback whose content matches an outstanding query is that query's result;
feedback whose content does not is an update to the current test case.
"""

import base64
import json
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
