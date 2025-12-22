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

    # Current reducer pass/pump status
    status: str
    # Size information
    size: int
    original_size: int
    # Call statistics
    calls: int
    reductions: int
    interesting_calls: int = 0
    wasted_calls: int = 0
    # Runtime in seconds
    runtime: float = 0.0
    # Parallelism stats
    parallel_workers: int = 0
    average_parallelism: float = 0.0
    effective_parallelism: float = 0.0
    # Time since last reduction
    time_since_last_reduction: float = 0.0
    # Content preview (truncated for large files)
    content_preview: str = ""
    # Whether content is hex mode
    hex_mode: bool = False


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
                "interesting_calls": msg.interesting_calls,
                "wasted_calls": msg.wasted_calls,
                "runtime": msg.runtime,
                "parallel_workers": msg.parallel_workers,
                "average_parallelism": msg.average_parallelism,
                "effective_parallelism": msg.effective_parallelism,
                "time_since_last_reduction": msg.time_since_last_reduction,
                "content_preview": msg.content_preview,
                "hex_mode": msg.hex_mode,
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
            interesting_calls=d.get("interesting_calls", 0),
            wasted_calls=d.get("wasted_calls", 0),
            runtime=d.get("runtime", 0.0),
            parallel_workers=d.get("parallel_workers", 0),
            average_parallelism=d.get("average_parallelism", 0.0),
            effective_parallelism=d.get("effective_parallelism", 0.0),
            time_since_last_reduction=d.get("time_since_last_reduction", 0.0),
            content_preview=d.get("content_preview", ""),
            hex_mode=d.get("hex_mode", False),
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
