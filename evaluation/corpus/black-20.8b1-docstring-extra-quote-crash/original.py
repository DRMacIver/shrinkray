# Reduced from a real-world subprocess helper module. Triggers a black
# 20.8b1 "INTERNAL ERROR: Black produced different code on the second
# pass of the formatter" when a docstring opens with four single quotes
# and contains a further non-empty line. psf/black issue #1926.
"""Utilities for running shell commands with logging and retries."""

import logging
import shlex
import subprocess
import time

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
BACKOFF_BASE = 0.5


class CommandError(RuntimeError):
    '''Raised when a command fails after exhausting all retries.'''

    def __init__(self, cmd, status, output):
        super().__init__("%r failed with status %d" % (cmd, status))
        self.cmd = cmd
        self.status = status
        self.output = output

    def summary(self):
        '''Return a one-line summary suitable for log files.'''
        first = self.output.splitlines()[0] if self.output else ""
        return "%s (status %d): %s" % (self.cmd, self.status, first)


def _quote(args):
    '''Render an argument vector as a copy-pasteable shell string.'''
    return " ".join(shlex.quote(a) for a in args)


class Runner:
    '''Runs commands and remembers recent invocations for debugging.'''

    def __init__(self, shell="/bin/sh", env=None):
        self.shell = shell
        self.env = dict(env or {})
        self.history = []

    def run(self, cmd, timeout=DEFAULT_TIMEOUT):
        ''''Run a single command, returning its captured output.
        Transient failures are retried with exponential backoff before
        giving up and raising CommandError to the caller.
        '''
        last = None
        for attempt in range(MAX_RETRIES):
            started = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    shell=isinstance(cmd, str),
                    capture_output=True,
                    timeout=timeout,
                    env=self.env or None,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning("timeout running %s: %s", _quote(cmd), exc)
                last = CommandError(cmd, -1, "")
                time.sleep(BACKOFF_BASE * 2 ** attempt)
                continue
            elapsed = time.monotonic() - started
            self.history.append((cmd, proc.returncode, elapsed))
            if proc.returncode == 0:
                return proc.stdout.decode("utf-8", "replace")
            last = CommandError(
                cmd, proc.returncode, proc.stderr.decode("utf-8", "replace")
            )
            logger.info("attempt %d failed: %s", attempt + 1, last.summary())
            time.sleep(BACKOFF_BASE * 2 ** attempt)
        raise last

    def recent_failures(self):
        '''Yield history entries whose exit status was non-zero.'''
        for cmd, status, elapsed in self.history:
            if status != 0:
                yield cmd, status, elapsed


def run_checked(cmd, **kwargs):
    '''Module-level convenience wrapper around a shared Runner.'''
    return Runner().run(cmd, **kwargs)
