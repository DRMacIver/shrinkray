"""In-process LLM inference for the LLM passes, via llama-cpp-python.

llama-cpp-python is an optional dependency (the ``llm`` extra): it's a
C++ build that many installs won't want, so this module degrades to a
clear error when it's missing. The model itself is loaded on the first
completion request, downloading it from Hugging Face first if necessary.

Inference is blocking and CPU-heavy, so it runs in a worker thread. The
model is not safe for concurrent generation, so calls are serialized on a
threading.Lock taken inside the worker thread: a completion abandoned by
cancellation keeps holding the lock until it actually finishes, and later
calls queue behind it rather than corrupting the model state.
"""

import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

import trio
from attrs import define, field

from shrinkray.passes.llm import HuggingFaceModel, LLMClient, LocalModel


if TYPE_CHECKING:
    from llama_cpp import Llama

try:
    import huggingface_hub
    import llama_cpp
except ImportError:
    huggingface_hub = None
    llama_cpp = None


@define
class LlamaCppClient(LLMClient):
    """Runs completions on a local GGUF model through llama-cpp-python."""

    model: LocalModel | HuggingFaceModel

    # Context window for the loaded model. The pass-level input size limit
    # (LLMConfig.max_input_bytes) is chosen so prompt plus completion fit.
    n_ctx: int = 16384

    # Layers to offload to a GPU: -1 offloads everything when the
    # installed llama-cpp-python was built with GPU support, and
    # harmlessly stays on CPU otherwise.
    n_gpu_layers: int = -1

    # Threads for CPU inference; None lets llama.cpp pick.
    n_threads: int | None = None

    _llama: "Llama | None" = field(default=None, init=False)
    _thread_lock: threading.Lock = field(factory=threading.Lock, init=False)
    _load_thread: threading.Thread | None = field(default=None, init=False)
    _load_error: Exception | None = field(default=None, init=False)
    _ready: threading.Event = field(factory=threading.Event, init=False)

    def start_loading(self) -> None:
        """Download and load the model on a background thread.

        Idempotent. Called at the start of a reduction so the (possibly
        multi-gigabyte) download overlaps with the cheap passes; the LLM
        pass then calls wait_until_ready before its first generation.
        """
        if self._load_thread is not None or self._llama is not None:
            return

        def load() -> None:
            try:
                with self._thread_lock:
                    self._ensure_loaded()
            except Exception as e:
                self._load_error = e
            finally:
                self._ready.set()

        self._load_thread = threading.Thread(target=load, daemon=True)
        self._load_thread.start()

    async def wait_until_ready(self) -> None:
        if self._llama is not None:
            await trio.lowlevel.checkpoint()
            return
        self.start_loading()
        await trio.to_thread.run_sync(self._ready.wait, abandon_on_cancel=True)
        if self._load_error is not None:
            raise self._load_error

    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        def run_blocking() -> str:
            return self._complete_blocking(
                prompt, max_tokens=max_tokens, seed=seed, temperature=temperature
            )

        # abandon_on_cancel so that skipping the pass doesn't have to wait
        # out an in-flight generation; the lock in _complete_blocking keeps
        # the abandoned thread from overlapping with the next call.
        return await trio.to_thread.run_sync(run_blocking, abandon_on_cancel=True)

    def _complete_blocking(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        with self._thread_lock:
            llama = self._ensure_loaded()
            response = llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            # Streaming is never requested, so the response is a mapping,
            # not an iterator of chunks.
            assert not isinstance(response, Iterator)
            content = response["choices"][0]["message"].get("content")
            return content or ""

    def _ensure_loaded(self) -> "Llama":
        if self._llama is None:
            if llama_cpp is None or huggingface_hub is None:
                raise ImportError(
                    "The LLM passes need llama-cpp-python, which is not "
                    "installed. Install shrink ray's llm extra "
                    "(e.g. `uv tool install 'shrinkray[llm]'`) or run "
                    "without --llm."
                )
            if isinstance(self.model, HuggingFaceModel):
                path = huggingface_hub.hf_hub_download(
                    self.model.repo_id, self.model.filename
                )
            else:
                path = self.model.path
            self._llama = llama_cpp.Llama(
                model_path=path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=False,
            )
        return self._llama
