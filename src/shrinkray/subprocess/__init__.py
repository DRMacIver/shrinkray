"""Subprocess communication for separating reducer from UI."""

from shrinkray.subprocess.client import SubprocessClient
from shrinkray.subprocess.protocol import (
    ProgressUpdate,
    Request,
    Response,
    deserialize,
    serialize,
)


__all__ = [
    "Request",
    "Response",
    "ProgressUpdate",
    "serialize",
    "deserialize",
    "SubprocessClient",
]
