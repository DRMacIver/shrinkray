# Reduced from a real-world debug-adapter module (ipykernel's debugger).
# Triggers a ruff 0.0.277 Rust panic ("range start index 2 out of range
# for slice of length 1" in rules/isort/block.rs) when consecutive
# imports inside a try block carry `# isort: skip` trailing comments.
# astral-sh/ruff issue #5621, fixed by PR #5623 in 0.0.278.
"""Debug adapter bridging the kernel to an optional native profiler."""

import json
import os
import sys
import typing as t
from queue import Empty, Queue

try:
    # This import is required to have the next ones working...
    from profiler.server import api  # noqa

    from _profiler_bundle import frame_utils  # isort: skip
    from _profiler_bundle.suspended_frames import (  # isort: skip
        FramesTracker,
        SuspendedFramesManager,
    )

    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False

ROUTING_ID_LENGTH = 8
_MAX_PENDING = 128


class MessageRouter:
    """Buffers adapter messages and routes them to waiting clients."""

    def __init__(self, event_callback: t.Optional[t.Callable] = None):
        self._pending: Queue = Queue(maxsize=_MAX_PENDING)
        self._event_callback = event_callback
        self._sequence = 0

    def put(self, raw: bytes) -> None:
        message = json.loads(raw.decode("utf-8"))
        self._sequence += 1
        message["seq"] = self._sequence
        if message.get("type") == "event" and self._event_callback:
            self._event_callback(message)
            return
        self._pending.put(message)

    def get(self, timeout: float = 1.0) -> t.Optional[dict]:
        try:
            return self._pending.get(timeout=timeout)
        except Empty:
            return None


class ProfilerSession:
    """Wraps the optional native profiler behind a uniform interface."""

    def __init__(self, router: MessageRouter):
        self.router = router
        self._tracker = None
        self._manager = None

    @property
    def available(self) -> bool:
        return _PROFILER_AVAILABLE

    def start(self) -> None:
        if not _PROFILER_AVAILABLE:
            raise RuntimeError("native profiler is not installed")
        self._manager = SuspendedFramesManager()
        self._tracker = FramesTracker(self._manager)
        api.attach(pid=os.getpid(), log_dir=None)

    def snapshot(self) -> dict:
        if self._tracker is None:
            return {"frames": [], "python": sys.version}
        frames = frame_utils.flatten(self._tracker.current_frames())
        return {"frames": frames, "python": sys.version}


def make_session(callback: t.Optional[t.Callable] = None) -> ProfilerSession:
    """Build a ProfilerSession with a fresh router."""
    return ProfilerSession(MessageRouter(event_callback=callback))
