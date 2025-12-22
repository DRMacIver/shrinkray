"""Subprocess communication for separating reducer from UI."""

from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    encode_bytes,
    decode_bytes,
    serialize,
)
from shrinkray.subprocess.client import SubprocessClient

__all__ = [
    "Request",
    "Response",
    "ProgressUpdate",
    "serialize",
    "deserialize",
    "encode_bytes",
    "decode_bytes",
    "SubprocessClient",
]
