"""Tests for the in-process llama-cpp LLM client.

These run against the real llama-cpp-python library with a real (tiny)
model: ggml-org's stories260K, a ~1MB GGUF that generates in well under a
second on CPU. Behavioural edge cases (serialization, cancellation,
degenerate responses) wrap the real loaded model with spies rather than
replacing the library.
"""

import importlib
import sys
import threading
import time
from typing import Any

import pytest
import trio
from huggingface_hub import hf_hub_download

import shrinkray.llm_client
from shrinkray.llm_client import LlamaCppClient
from shrinkray.passes.llm import HuggingFaceModel, LocalModel


TINY_REPO = "ggml-org/models"
TINY_FILE = "tinyllamas/stories260K.gguf"

# llama-cpp-python builds on OpenBSD but its shared-library loader refuses
# the platform at import time, so the real-model tests can only run where
# the library actually loads. The graceful-degradation tests below run
# everywhere (they are exactly what such platforms get at runtime).
requires_llama_cpp = pytest.mark.skipif(
    not shrinkray.llm_client.llm_support_available(),
    reason="llama-cpp-python cannot load on this platform",
)


@pytest.fixture(scope="module")
def tiny_model_path() -> str:
    return hf_hub_download(TINY_REPO, TINY_FILE)


def tiny_client(path: str) -> LlamaCppClient:
    # CPU-only with few threads: for a model this small, GPU offload and
    # wide threading cost far more in setup than they save.
    return LlamaCppClient(
        model=LocalModel(path=path), n_ctx=512, n_gpu_layers=0, n_threads=2
    )


@requires_llama_cpp
async def test_completes_deterministically_with_a_real_model(
    tiny_model_path: str,
):
    client = tiny_client(tiny_model_path)
    first = await client.complete(
        "Tell me a story.", max_tokens=8, seed=42, temperature=0.0
    )
    loaded = client._llama
    second = await client.complete(
        "Tell me a story.", max_tokens=8, seed=42, temperature=0.0
    )
    assert first == second
    assert first.strip()
    # The model was loaded lazily, once, and reused.
    assert loaded is not None
    assert client._llama is loaded


@requires_llama_cpp
async def test_resolves_hugging_face_models_through_the_hub(
    tiny_model_path: str,
):
    client = LlamaCppClient(
        model=HuggingFaceModel(repo_id=TINY_REPO, filename=TINY_FILE),
        n_ctx=512,
        n_gpu_layers=0,
        n_threads=2,
    )
    result = await client.complete(
        "Tell me a story.", max_tokens=4, seed=1, temperature=0.0
    )
    assert isinstance(result, str)
    assert result


@requires_llama_cpp
async def test_concurrent_calls_are_serialized(
    tiny_model_path: str, monkeypatch: pytest.MonkeyPatch
):
    client = tiny_client(tiny_model_path)
    # Load eagerly so we can wrap the real generation with a reentry check.
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    llama = client._llama
    assert llama is not None

    real = llama.create_chat_completion
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            return real(*args, **kwargs)
        finally:
            with counter_lock:
                active -= 1

    async def one(seed: int) -> None:
        await client.complete("hi", max_tokens=1, seed=seed, temperature=0.0)

    monkeypatch.setattr(llama, "create_chat_completion", counted)
    async with trio.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(one, i)
    assert max_active == 1


@requires_llama_cpp
async def test_cancellation_abandons_the_running_generation(
    tiny_model_path: str,
):
    client = tiny_client(tiny_model_path)
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    llama = client._llama
    assert llama is not None

    real = llama.create_chat_completion
    release = threading.Event()
    blocked_calls = 0

    def blocking(*args: Any, **kwargs: Any) -> Any:
        nonlocal blocked_calls
        blocked_calls += 1
        release.wait()
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(llama, "create_chat_completion", blocking)
        try:
            with trio.move_on_after(0.5) as scope:
                await client.complete("slow", max_tokens=1, seed=1, temperature=0.0)
            assert scope.cancelled_caught
        finally:
            release.set()

    # The abandoned generation finished behind the lock; the client is
    # still usable afterwards.
    result = await client.complete("after", max_tokens=1, seed=2, temperature=0.0)
    assert isinstance(result, str)
    assert blocked_calls == 1


@requires_llama_cpp
async def test_missing_content_becomes_empty_string(
    tiny_model_path: str, monkeypatch: pytest.MonkeyPatch
):
    client = tiny_client(tiny_model_path)
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    llama = client._llama
    assert llama is not None

    def no_content(*args: Any, **kwargs: Any) -> Any:
        return iter([{"choices": [{"delta": {"role": "assistant"}}]}])

    monkeypatch.setattr(llama, "create_chat_completion", no_content)
    result = await client.complete("hi", max_tokens=1, seed=1, temperature=0.0)
    assert result == ""


async def test_missing_llama_cpp_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(shrinkray.llm_client, "llama_cpp", None)
    client = LlamaCppClient(model=LocalModel(path="/m.gguf"))
    with pytest.raises(ImportError, match=r"shrinkray\[llm\]"):
        await client.complete("hi", max_tokens=1, seed=1, temperature=0.0)


class _BlockOptionalImports:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname in ("llama_cpp", "huggingface_hub"):
            raise ImportError(f"{fullname} blocked for test")
        return None


def test_module_imports_without_the_llm_extra():
    blocker = _BlockOptionalImports()
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in ("llama_cpp", "huggingface_hub")
    }
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.reload(shrinkray.llm_client)
        assert module.llama_cpp is None
        assert module.huggingface_hub is None
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        importlib.reload(shrinkray.llm_client)


@requires_llama_cpp
async def test_background_loading_loads_once_and_serves(
    tiny_model_path: str, monkeypatch: pytest.MonkeyPatch
):
    assert shrinkray.llm_client.llama_cpp is not None
    real_llama = shrinkray.llm_client.llama_cpp.Llama
    constructions = 0

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal constructions
        constructions += 1
        return real_llama(*args, **kwargs)

    monkeypatch.setattr(shrinkray.llm_client.llama_cpp, "Llama", counting)

    client = tiny_client(tiny_model_path)
    client.start_loading()
    client.start_loading()  # idempotent
    await client.wait_until_ready()
    assert client._llama is not None
    result = await client.complete("hi", max_tokens=1, seed=1, temperature=0.0)
    assert isinstance(result, str)
    assert constructions == 1

    # Once loaded, waiting again returns immediately.
    await client.wait_until_ready()


@requires_llama_cpp
async def test_wait_until_ready_surfaces_load_failure(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"this is not a gguf file")
    client = LlamaCppClient(
        model=LocalModel(path=str(bad)), n_gpu_layers=0, n_threads=2
    )
    # wait_until_ready starts the load itself if nobody else has.
    with pytest.raises(ValueError):
        await client.wait_until_ready()


@requires_llama_cpp
def test_llm_support_reflects_import_state(monkeypatch: pytest.MonkeyPatch):
    assert shrinkray.llm_client.llm_support_available()
    monkeypatch.setattr(shrinkray.llm_client, "llama_cpp", None)
    assert not shrinkray.llm_client.llm_support_available()


def endless_stream(counter: "list[int]", pace: float):
    def fake(*args: Any, **kwargs: Any) -> Any:
        def chunks():
            while True:
                counter[0] += 1
                yield {"choices": [{"delta": {"content": "x"}}]}
                time.sleep(pace)

        return chunks()

    return fake


@requires_llama_cpp
async def test_cancelled_generation_stops_consuming_the_stream(
    tiny_model_path: str,
):
    client = tiny_client(tiny_model_path)
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    llama = client._llama
    assert llama is not None

    consumed = [0]
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            llama, "create_chat_completion", endless_stream(consumed, 0.005)
        )
        with trio.move_on_after(0.2) as scope:
            await client.complete(
                "spin forever", max_tokens=64, seed=1, temperature=0.0
            )
        assert scope.cancelled_caught

    # The abandoned generation notices the abort at a token boundary and
    # releases the model, so a real completion goes through afterwards
    # rather than queueing behind an endless stream.
    result = await client.complete("after", max_tokens=1, seed=2, temperature=0.0)
    assert isinstance(result, str)


@requires_llama_cpp
async def test_exit_hook_is_registered_and_joins_generations(
    tiny_model_path: str, monkeypatch: pytest.MonkeyPatch
):
    registered: list[Any] = []
    monkeypatch.setattr(
        shrinkray.llm_client.atexit, "register", registered.append
    )
    client = tiny_client(tiny_model_path)
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    assert registered == [client._join_at_exit]
    llama = client._llama
    assert llama is not None

    # With a generation in flight, the exit hook tells it to stop and
    # waits for it to leave llama.cpp before returning.
    consumed = [0]
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            llama, "create_chat_completion", endless_stream(consumed, 0.005)
        )
        with trio.move_on_after(0.2):
            await client.complete(
                "spin forever", max_tokens=64, seed=1, temperature=0.0
            )
        await trio.to_thread.run_sync(client._join_at_exit)
    consumed_after_join = consumed[0]
    assert not client._thread_lock.locked()
    assert consumed[0] <= consumed_after_join + 1


@requires_llama_cpp
async def test_exit_hook_gives_up_on_a_stuck_generation(
    tiny_model_path: str,
):
    client = tiny_client(tiny_model_path)
    await client.complete("warm up", max_tokens=1, seed=0, temperature=0.0)
    llama = client._llama
    assert llama is not None

    release = threading.Event()

    def stuck(*args: Any, **kwargs: Any) -> Any:
        def chunks():
            release.wait()
            yield {"choices": [{"delta": {"content": "x"}}]}

        return chunks()

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(llama, "create_chat_completion", stuck)
        try:
            with trio.move_on_after(0.2):
                await client.complete(
                    "stuck", max_tokens=1, seed=1, temperature=0.0
                )
            # The generation ignores the abort (it's blocked inside the
            # model), so the exit hook times out rather than hanging.
            await trio.to_thread.run_sync(
                lambda: client._join_at_exit(timeout=0.05)
            )
        finally:
            release.set()
