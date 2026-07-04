# Reduced from a real-world event-dispatch module. Triggers a mypy 0.942
# INTERNAL ERROR (AssertionError in checkpattern.construct_sequence_child)
# when a match statement's subject is a Union of a fixed-length tuple type
# and a non-tuple type. python/mypy issue #12533, fixed by PR #13514.
"""Dispatch wire-format events received from the message broker."""

from dataclasses import dataclass, field
from typing import Callable, Union


@dataclass
class Envelope:
    """A decoded message envelope, before payload dispatch."""

    topic: str
    payload: Union[str, tuple[str]]
    headers: dict[str, str] = field(default_factory=dict)

    def routing_key(self) -> str:
        return self.topic.split(".", 1)[0]


class HandlerRegistry:
    """Maps topic prefixes to payload handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[str], None]] = {}
        self._fallbacks: list[Callable[[str], None]] = []

    def register(self, prefix: str, handler: Callable[[str], None]) -> None:
        if prefix in self._handlers:
            raise ValueError(f"duplicate handler for {prefix!r}")
        self._handlers[prefix] = handler

    def add_fallback(self, handler: Callable[[str], None]) -> None:
        self._fallbacks.append(handler)

    def resolve(self, topic: str) -> Callable[[str], None]:
        prefix = topic.split(".", 1)[0]
        try:
            return self._handlers[prefix]
        except KeyError:
            if self._fallbacks:
                return self._fallbacks[-1]
            raise


def decode_payload(raw: bytes) -> Union[str, tuple[str]]:
    """Decode a raw frame into either a bare string or a 1-tuple.

    Single-frame messages decode to ``str``; enveloped legacy messages
    decode to a one-element tuple so callers can tell them apart.
    """
    text = raw.decode("utf-8", "replace")
    if text.startswith("!"):
        return (text[1:],)
    return text


def dispatch(registry: HandlerRegistry, envelope: Envelope) -> None:
    """Route one envelope's payload to the registered handler."""
    handler = registry.resolve(envelope.topic)
    e = envelope.payload
    match e:
        case (a,) if isinstance(a, str):
            handler(a)
        case str() as text:
            handler(text)


def drain(registry: HandlerRegistry, envelopes: list[Envelope]) -> int:
    """Dispatch every envelope, returning the number processed."""
    count = 0
    for envelope in envelopes:
        dispatch(registry, envelope)
        count += 1
    return count
