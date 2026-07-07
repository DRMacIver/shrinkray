"""Coordination of the background downloads a reduction wants at startup.

Two kinds of resources may need fetching when a reduction starts: the LLM
model (multi-gigabyte) and tree-sitter grammars for the input's language
(small shared libraries). Neither should be downloaded silently, so the
state computes what is missing and the UI presents it — the TUI as a
modal over the already-running reduction, the basic UI as a printed
notice. Nothing pending downloads until start() records the user's
decision; the reduction proceeds on the classical passes meanwhile, and
passes that depend on a download join in when it completes (the LLM
passes via LLMClient.wait_until_ready, the tree-sitter passes via the
reducer's pending-grammar mechanism).
"""

import sys
import threading
from collections.abc import Iterable

import tree_sitter_language_pack
import trio
from attrs import define, field

from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.treesitter import (
    language_for_filename,
    loadable_language_for_filename,
)


@define
class GrammarDownload:
    """One missing tree-sitter grammar, downloaded on a background thread."""

    language: str
    disabled: bool = field(default=False, init=False)
    _ok: bool = field(default=False, init=False)
    _resolved: threading.Event = field(factory=threading.Event, init=False)

    def start(self) -> None:
        def run() -> None:
            try:
                tree_sitter_language_pack.download([self.language])
                # Downloading is not enough: verify the grammar actually
                # loads on this platform before passes rely on it.
                tree_sitter_language_pack.get_language(self.language)
                self._ok = True
            except Exception as e:
                print(
                    "WARNING: could not download the tree-sitter grammar "
                    f"{self.language!r} ({type(e).__name__}); reducing "
                    "without its passes.",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                self._resolved.set()

        threading.Thread(target=run, daemon=True).start()

    def resolve_disabled(self) -> None:
        self.disabled = True
        self._resolved.set()


@define
class DownloadCoordinator:
    """Knows what needs downloading and runs the approved downloads."""

    # The LLM client (when LLM mode is on), whether its model is missing
    # from the local cache, and how to describe that download to the user.
    llm_client: LlamaCppClient | None
    llm_needs_download: bool
    llm_description: str

    # The missing grammars, keyed by language name.
    grammars: dict[str, GrammarDownload]

    def pending(self) -> list[tuple[str, str]]:
        """(item id, human description) pairs to offer the user."""
        items: list[tuple[str, str]] = []
        if self.llm_client is not None and self.llm_needs_download:
            items.append(("llm", self.llm_description))
        for language in sorted(self.grammars):
            items.append((f"grammar-{language}", f"tree-sitter grammar for {language}"))
        return items

    def start_immediate(self) -> None:
        """Begin work that needs no consent: loading an already-cached model."""
        if self.llm_client is not None and not self.llm_needs_download:
            self.llm_client.start_loading()

    def start(self, disabled: list[str]) -> None:
        """Record the user's decision and begin the approved downloads."""
        if self.llm_client is not None and self.llm_needs_download:
            if "llm" in disabled:
                self.llm_client.disable()
            else:
                self.llm_client.start_loading()
        for language, grammar in self.grammars.items():
            if f"grammar-{language}" in disabled:
                grammar.resolve_disabled()
            else:
                grammar.start()

    def knows_grammar(self, language: str) -> bool:
        """Whether this grammar's availability is this coordinator's to answer."""
        return language in self.grammars

    def grammar_available(self, language: str) -> bool:
        """The grammar arrived, loads, and wasn't disabled."""
        grammar = self.grammars.get(language)
        return grammar is not None and grammar._ok and not grammar.disabled

    def grammar_pending(self, language: str) -> bool:
        """The grammar might still become available."""
        grammar = self.grammars.get(language)
        return grammar is not None and not grammar._resolved.is_set()

    async def wait_for_grammar(self, language: str) -> bool:
        """Wait for the grammar's outcome; True when it became available.

        Also waits out the user's decision: until start() runs (the TUI
        modal is still open) nothing is resolved.
        """
        grammar = self.grammars.get(language)
        if grammar is None:
            await trio.lowlevel.checkpoint()
            return False
        await trio.to_thread.run_sync(grammar._resolved.wait, abandon_on_cancel=True)
        return self.grammar_available(language)


def missing_grammars(filenames: Iterable[str]) -> dict[str, GrammarDownload]:
    """The grammars the given files need that aren't downloaded yet."""
    downloaded = set(tree_sitter_language_pack.downloaded_languages())
    result: dict[str, GrammarDownload] = {}
    for filename in filenames:
        language = language_for_filename(filename)
        if language is not None and language not in downloaded:
            result[language] = GrammarDownload(language=language)
    return result


def grammar_plan(
    filename: str, downloads: DownloadCoordinator | None
) -> tuple[str | None, str | None]:
    """How a reducer should get tree-sitter passes for this file.

    Returns (usable-now language, pending language): the first can have
    its passes registered immediately, the second is downloading in the
    background (or awaiting the user's decision) and joins later.
    """
    if downloads is None:
        return loadable_language_for_filename(filename), None
    language = language_for_filename(filename)
    if language is None:
        return None, None
    if downloads.knows_grammar(language):
        return None, language
    return loadable_language_for_filename(filename), None
