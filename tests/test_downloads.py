"""Tests for the startup download coordinator.

Grammar downloads are exercised against a stubbed
tree_sitter_language_pack download function (the real one hits the
network); the LLM side uses the real client with loading gated.
"""

from typing import Any

import pytest
import trio

import shrinkray.downloads
from shrinkray.downloads import DownloadCoordinator, GrammarDownload
from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.llm import LocalModel
from shrinkray.problem import BasicReductionProblem
from shrinkray.reducer import ShrinkRay
from shrinkray.subprocess.protocol import Request
from shrinkray.subprocess.worker import ReducerWorker
from shrinkray.work import WorkContext


class MemoryOutputStream:
    def __init__(self) -> None:
        self.data = b""

    async def send(self, data: bytes) -> None:
        self.data += data


@pytest.fixture
def grammar_pack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the language pack: record downloads, control success."""
    state: dict[str, Any] = {"downloaded": [], "fail": set(), "unloadable": set()}

    def download(names: list[str]) -> int:
        for name in names:
            if name in state["fail"]:
                raise RuntimeError(f"download of {name} failed")
        state["downloaded"].extend(names)
        return len(names)

    def get_language(name: str) -> object:
        if name in state["unloadable"]:
            raise shrinkray.downloads.tree_sitter_language_pack.exceptions.Error(
                "broken"
            )
        return object()

    monkeypatch.setattr(
        shrinkray.downloads.tree_sitter_language_pack, "download", download
    )
    monkeypatch.setattr(
        shrinkray.downloads.tree_sitter_language_pack, "get_language", get_language
    )
    return state


def make_coordinator(
    languages: list[str], llm_client: LlamaCppClient | None = None
) -> DownloadCoordinator:
    return DownloadCoordinator(
        llm_client=llm_client,
        llm_needs_download=llm_client is not None,
        llm_description="LLM model example/repo (model.gguf)",
        grammars={lang: GrammarDownload(language=lang) for lang in languages},
    )


async def test_grammar_download_lifecycle(grammar_pack: dict[str, Any]):
    coordinator = make_coordinator(["go"])
    assert coordinator.pending() == [
        ("grammar-go", "tree-sitter grammar for go"),
    ]
    assert not coordinator.grammar_available("go")
    assert coordinator.grammar_pending("go")

    coordinator.start(disabled=[])
    assert await coordinator.wait_for_grammar("go")
    assert coordinator.grammar_available("go")
    assert not coordinator.grammar_pending("go")
    assert grammar_pack["downloaded"] == ["go"]


async def test_disabled_grammar_never_downloads(grammar_pack: dict[str, Any]):
    coordinator = make_coordinator(["go"])
    coordinator.start(disabled=["grammar-go"])
    assert not await coordinator.wait_for_grammar("go")
    assert not coordinator.grammar_available("go")
    assert not coordinator.grammar_pending("go")
    assert grammar_pack["downloaded"] == []


async def test_failed_download_reports_unavailable(
    grammar_pack: dict[str, Any], capsys: pytest.CaptureFixture[str]
):
    grammar_pack["fail"].add("go")
    coordinator = make_coordinator(["go"])
    coordinator.start(disabled=[])
    assert not await coordinator.wait_for_grammar("go")
    assert not coordinator.grammar_available("go")
    assert not coordinator.grammar_pending("go")
    assert "could not download" in capsys.readouterr().err


async def test_unloadable_grammar_reports_unavailable(
    grammar_pack: dict[str, Any], capsys: pytest.CaptureFixture[str]
):
    grammar_pack["unloadable"].add("go")
    coordinator = make_coordinator(["go"])
    coordinator.start(disabled=[])
    assert not await coordinator.wait_for_grammar("go")
    assert "could not download" in capsys.readouterr().err


async def test_wait_blocks_until_start_decision(grammar_pack: dict[str, Any]):
    # Before start() is called (e.g. while the TUI modal is open), waiting
    # must neither trigger the download nor return.
    coordinator = make_coordinator(["go"])
    started = False

    async def waiter() -> None:
        nonlocal started
        assert await coordinator.wait_for_grammar("go")
        started = True

    async with trio.open_nursery() as nursery:
        nursery.start_soon(waiter)
        await trio.sleep(0.05)
        assert not started
        assert grammar_pack["downloaded"] == []
        coordinator.start(disabled=[])
    assert started


async def test_unknown_grammar_is_never_pending(grammar_pack: dict[str, Any]):
    coordinator = make_coordinator([])
    coordinator.start(disabled=[])
    assert not coordinator.grammar_pending("go")
    assert not coordinator.grammar_available("go")
    assert not await coordinator.wait_for_grammar("go")


async def test_llm_item_gates_model_loading(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real model")
    client = LlamaCppClient(model=LocalModel(path=str(model)))
    coordinator = make_coordinator([], llm_client=client)
    assert coordinator.pending() == [
        ("llm", "LLM model example/repo (model.gguf)"),
    ]
    # Loading has not been started by coordinator construction.
    assert client._load_thread is None
    coordinator.start(disabled=["llm"])
    assert client.is_disabled()
    assert client._load_thread is None
    # Disabled clients release waiters immediately.
    await client.wait_until_ready()


async def test_llm_download_starts_when_enabled(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real model")
    client = LlamaCppClient(model=LocalModel(path=str(model)))
    coordinator = make_coordinator([], llm_client=client)
    coordinator.start(disabled=[])
    assert client._load_thread is not None


def test_cached_llm_model_starts_loading_immediately(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real model")
    client = LlamaCppClient(model=LocalModel(path=str(model)))
    coordinator = DownloadCoordinator(
        llm_client=client,
        llm_needs_download=False,
        llm_description="",
        grammars={},
    )
    assert coordinator.pending() == []
    coordinator.start_immediate()
    assert client._load_thread is not None


# === Reducer integration: passes joining after a grammar download ===


def stub_download_only(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    downloaded: list[str] = []

    def download(names: list[str]) -> int:
        downloaded.extend(names)
        return len(names)

    monkeypatch.setattr(
        shrinkray.downloads.tree_sitter_language_pack, "download", download
    )
    return downloaded


PYTHON_INPUT = b"""\
import os
import sys

def keep():
    return "MARKER"

def unused():
    return os.path.join(sys.argv[0], "x")

keep()
"""


def make_reducer(coordinator: DownloadCoordinator | None) -> ShrinkRay:
    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"MARKER" in x

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=PYTHON_INPUT,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    return ShrinkRay(
        target=problem,
        python_reducer=False,
        pending_treesitter_language="python" if coordinator else None,
        downloads=coordinator,
    )


def treesitter_pass_names(reducer: ShrinkRay) -> list[str]:
    return [p.__name__ for p in reducer.great_passes if "treesitter" in p.__name__]


async def test_treesitter_passes_join_after_download(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_download_only(monkeypatch)
    coordinator = make_coordinator(["python"])
    coordinator.start(disabled=[])
    reducer = make_reducer(coordinator)
    assert treesitter_pass_names(reducer) == []
    await reducer.run()
    assert treesitter_pass_names(reducer) != []
    assert reducer.pending_treesitter_language is None
    # Late registration also flows into restarts.
    assert reducer.treesitter_language == "python"


async def test_disabled_grammar_never_adds_passes(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_download_only(monkeypatch)
    coordinator = make_coordinator(["python"])
    coordinator.start(disabled=["grammar-python"])
    reducer = make_reducer(coordinator)
    await reducer.run()
    assert treesitter_pass_names(reducer) == []
    assert reducer.pending_treesitter_language is None


async def test_reduction_waits_for_the_download_decision(
    monkeypatch: pytest.MonkeyPatch,
):
    # With the modal still open (start() not called), the reduction runs
    # everything else, then waits at its fixpoint for the outcome rather
    # than finishing without the grammar passes.
    stub_download_only(monkeypatch)
    coordinator = make_coordinator(["python"])
    reducer = make_reducer(coordinator)
    finished = False

    async def run_reduction() -> None:
        nonlocal finished
        await reducer.run()
        finished = True

    async with trio.open_nursery() as nursery:
        nursery.start_soon(run_reduction)
        await trio.sleep(0.3)
        assert not finished
        coordinator.start(disabled=[])
    assert finished
    assert treesitter_pass_names(reducer) != []


async def test_reducer_defers_llm_loading_to_the_coordinator(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not a real model")
    client = LlamaCppClient(model=LocalModel(path=str(model)))
    client.disable()  # resolve immediately so the run terminates

    async def is_interesting(x: bytes) -> bool:
        await trio.lowlevel.checkpoint()
        return b"MARKER" in x

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=b"a MARKER b\n",
        is_interesting=is_interesting,
        work=WorkContext(parallelism=1),
    )
    coordinator = DownloadCoordinator(
        llm_client=client,
        llm_needs_download=True,
        llm_description="model",
        grammars={},
    )
    reducer = ShrinkRay(
        target=problem,
        python_reducer=False,
        llm_client=client,
        downloads=coordinator,
    )
    await reducer.run()
    # The reducer did not start the load itself: that's the coordinator's
    # call once the user has decided.
    assert client._load_thread is None


# === grammar_plan / missing_grammars helpers ===


def test_grammar_plan_without_coordinator_uses_loadable(monkeypatch):
    monkeypatch.setattr(
        shrinkray.downloads, "loadable_language_for_filename", lambda f: "python"
    )
    assert shrinkray.downloads.grammar_plan("x.py", None) == ("python", None)


def test_grammar_plan_unknown_extension_is_neither():
    coordinator = make_coordinator([])
    assert shrinkray.downloads.grammar_plan("x.unknownext", coordinator) == (
        None,
        None,
    )


def test_grammar_plan_pending_when_coordinator_knows_grammar(monkeypatch):
    monkeypatch.setattr(shrinkray.downloads, "language_for_filename", lambda f: "go")
    coordinator = make_coordinator(["go"])
    assert shrinkray.downloads.grammar_plan("x.go", coordinator) == (None, "go")


def test_grammar_plan_falls_back_to_loadable_when_not_pending(monkeypatch):
    # The coordinator has no pending download for this already-present
    # grammar, so it's usable now.
    monkeypatch.setattr(
        shrinkray.downloads, "language_for_filename", lambda f: "python"
    )
    monkeypatch.setattr(
        shrinkray.downloads, "loadable_language_for_filename", lambda f: "python"
    )
    coordinator = make_coordinator([])
    assert shrinkray.downloads.grammar_plan("x.py", coordinator) == ("python", None)


def test_missing_grammars_finds_uncached_language(monkeypatch):
    monkeypatch.setattr(
        shrinkray.downloads.tree_sitter_language_pack,
        "downloaded_languages",
        lambda: ["python"],
    )
    monkeypatch.setattr(
        shrinkray.downloads,
        "language_for_filename",
        lambda f: "go" if f.endswith(".go") else None,
    )
    result = shrinkray.downloads.missing_grammars(["a.go", "b.py", "c.txt"])
    assert set(result) == {"go"}


# === Worker command dispatch ===


async def test_worker_dispatches_start_downloads_command(tmp_path):
    target = tmp_path / "t.txt"
    target.write_text("hello")
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    worker = ReducerWorker(output_stream=MemoryOutputStream())
    await worker._handle_start(
        "s",
        {
            "file_path": str(target),
            "test": [str(script)],
            "formatter": "none",
            "volume": "quiet",
            "skip_validation": True,
        },
    )
    response = await worker.handle_command(
        Request(id="d", command="start_downloads", params={"disabled": []})
    )
    assert response.result == {"status": "downloads_started"}


async def test_worker_start_with_pending_downloads_does_not_auto_start(
    tmp_path, monkeypatch
):
    target = tmp_path / "t.txt"
    target.write_text("hello")
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nexit 0")
    script.chmod(0o755)

    worker = ReducerWorker(output_stream=MemoryOutputStream())

    started: list[list[str]] = []
    pending = [{"id": "grammar-go", "description": "tree-sitter grammar for go"}]

    # Patch the state's download surface once it exists by wrapping _start.
    real_start = ReducerWorker._start_reduction

    async def wrapped(self, params):
        await real_start(self, params)
        monkeypatch.setattr(self.state, "pending_downloads", lambda: pending)
        monkeypatch.setattr(self.state, "start_downloads", lambda d: started.append(d))

    monkeypatch.setattr(ReducerWorker, "_start_reduction", wrapped)

    response = await worker._handle_start(
        "s",
        {
            "file_path": str(target),
            "test": [str(script)],
            "formatter": "none",
            "volume": "quiet",
            "skip_validation": True,
        },
    )
    assert response.result is not None
    assert response.result["pending_downloads"] == pending
    # With something pending, downloads wait for the explicit decision.
    assert started == []
