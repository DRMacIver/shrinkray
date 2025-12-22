"""Subprocess communication for separating reducer from UI."""

from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    decode_bytes,
    deserialize,
    encode_bytes,
    serialize,
)


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
