"""Line-oriented JSON protocol for subprocess communication."""

import base64
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Request:
    """A command request from the UI to the worker subprocess."""

    id: str
    command: str
    params: dict = field(default_factory=dict)


@dataclass
class Response:
    """A response from the worker subprocess to the UI."""

    id: str
    result: Any = None
    error: str | None = None


@dataclass
class ProgressUpdate:
    """An unsolicited progress update from the worker subprocess."""

    status: str
    size: int
    original_size: int
    calls: int
    reductions: int


def serialize(msg: Request | Response | ProgressUpdate) -> str:
    """Serialize a message to a JSON line (without newline)."""
    if isinstance(msg, Request):
        data = {
            "id": msg.id,
            "command": msg.command,
            "params": msg.params,
        }
    elif isinstance(msg, Response):
        data = {
            "id": msg.id,
            "result": msg.result,
            "error": msg.error,
        }
    elif isinstance(msg, ProgressUpdate):
        data = {
            "type": "progress",
            "data": {
                "status": msg.status,
                "size": msg.size,
                "original_size": msg.original_size,
                "calls": msg.calls,
                "reductions": msg.reductions,
            },
        }
    else:
        raise TypeError(f"Cannot serialize {type(msg)}")
    return json.dumps(data, separators=(",", ":"))


def deserialize(line: str) -> Request | Response | ProgressUpdate:
    """Deserialize a JSON line to a message object."""
    data = json.loads(line)

    # Check for progress update (has "type" field)
    if data.get("type") == "progress":
        d = data["data"]
        return ProgressUpdate(
            status=d["status"],
            size=d["size"],
            original_size=d["original_size"],
            calls=d["calls"],
            reductions=d["reductions"],
        )

    # Check for response (has "result" or "error" field)
    if "result" in data or "error" in data:
        return Response(
            id=data["id"],
            result=data.get("result"),
            error=data.get("error"),
        )

    # Otherwise it's a request
    return Request(
        id=data["id"],
        command=data["command"],
        params=data.get("params", {}),
    )


def encode_bytes(data: bytes) -> str:
    """Encode bytes to base64 string for JSON transport."""
    return base64.b64encode(data).decode("ascii")


def decode_bytes(data: str) -> bytes:
    """Decode base64 string back to bytes."""
    return base64.b64decode(data.encode("ascii"))
