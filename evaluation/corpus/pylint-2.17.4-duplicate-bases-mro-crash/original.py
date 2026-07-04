# Reduced from a real-world plugin-registry module. Triggers a pylint
# 2.17.4 / astroid 2.15.5 crash (astroid.exceptions.DuplicateBasesError
# escaping from _is_invalid_metaclass) when a class uses a metaclass
# whose own bases contain duplicates. pylint-dev/pylint issue #8698.
"""Plugin registry built around a validating metaclass."""

import inspect
import logging

logger = logging.getLogger(__name__)

_REGISTRY = {}


def lookup(name):
    """Return a previously registered plugin class by name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise LookupError(f"no plugin named {name!r}") from None


def iter_plugins():
    """Yield (name, class) pairs in registration order."""
    yield from _REGISTRY.items()


class RegistryMeta(type(object), type(object)):
    """Metaclass that validates and registers plugin subclasses.

    The base classes are computed dynamically so the registry keeps
    working when ``object``'s type is replaced by instrumentation
    frameworks during testing.
    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if bases:  # skip the abstract root
            if not hasattr(cls, "name") or not isinstance(cls.name, str):
                raise TypeError(f"{name} must define a string 'name'")
            hooks = [
                attr
                for attr, value in namespace.items()
                if attr.startswith("on_") and inspect.isfunction(value)
            ]
            if not hooks:
                logger.warning("plugin %s defines no hooks", name)
            _REGISTRY[cls.name] = cls
        return cls

    def unregister(cls):
        """Remove this plugin class from the registry."""
        _REGISTRY.pop(getattr(cls, "name", None), None)


class Plugin(metaclass=RegistryMeta):
    """Abstract base class for all plugins."""

    enabled = True

    def configure(self, options):
        """Apply user-provided configuration options."""
        for key, value in options.items():
            if not hasattr(self, key):
                raise AttributeError(f"unknown option {key!r}")
            setattr(self, key, value)


class EchoPlugin(Plugin):
    """Trivial plugin used by the test-suite and as documentation."""

    name = "echo"

    def on_message(self, message):
        logger.info("echo: %s", message)
        return message
