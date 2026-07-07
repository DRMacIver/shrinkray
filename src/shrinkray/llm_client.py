"""In-process LLM inference for the LLM passes, via llama-cpp-python.

llama-cpp-python is a required dependency, but one that can fail to load
(it has no support for some platforms shrink ray runs on), so this module
degrades to reporting LLM support as unavailable rather than breaking
imports. The model itself is loaded on the first completion request,
downloading it from Hugging Face first if necessary.

Inference is blocking and CPU-heavy, so it runs in a worker thread. The
model is not safe for concurrent generation, so calls are serialized on a
threading.Lock taken inside the worker thread. Generations are consumed
as a token stream so that a completion abandoned by cancellation can be
told to stop at the next token, rather than running to its token budget
while holding the lock; an atexit hook stops and joins any in-flight
generation, because exiting while llama.cpp is mid-generation crashes in
its C++/Metal finalizers.
"""

import atexit
import sys
import threading
from collections import OrderedDict
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
except (ImportError, RuntimeError):
    # llama-cpp-python raises RuntimeError (not ImportError) at import
    # time on platforms its shared-library loader doesn't recognise
    # (e.g. OpenBSD, where the package builds but refuses to load).
    huggingface_hub = None
    llama_cpp = None


# Cap on remembered generations. Each entry keeps a full prompt (up to
# LLMConfig.max_input_bytes plus the oracle/output text) and its whole
# completion, so entries run to tens of kilobytes; every llm_rewrite
# round draws a fresh seed, making each generation a new key, so without
# a bound the cache would grow for the entire run (hundreds of megabytes
# on long reductions). A few hundred entries keeps the restart-at-fixpoint
# replay mostly served from cache while bounding memory to a few tens of
# megabytes; past the cap the least-recently-used entry is evicted.
DEFAULT_GENERATION_CACHE_SIZE = 256


def llm_support_available() -> bool:
    """Whether the LLM passes' dependencies are usable.

    Deliberately based on the import above rather than on whether the
    packages are installed: llama-cpp-python can be installed but
    unloadable on platforms it doesn't support.
    """
    return llama_cpp is not None and huggingface_hub is not None


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

    # Maximum number of generations kept in the LRU cache below.
    generation_cache_size: int = DEFAULT_GENERATION_CACHE_SIZE

    _llama: "Llama | None" = field(default=None, init=False)
    _thread_lock: threading.Lock = field(factory=threading.Lock, init=False)
    _load_thread: threading.Thread | None = field(default=None, init=False)
    _load_error: Exception | None = field(default=None, init=False)
    _ready: threading.Event = field(factory=threading.Event, init=False)
    _shutting_down: threading.Event = field(factory=threading.Event, init=False)
    _disabled: bool = field(default=False, init=False)

    # Completed generations, keyed by everything that determines them
    # (generation is deterministic for a fixed seed). The reducer's
    # restart-at-fixpoint phase replays the whole run's candidates from
    # the interestingness cache; without this cache the replay would
    # re-run every generation for real, which for --llm-only means
    # silently re-doing the entire reduction's model work at fixpoint. An
    # OrderedDict gives an LRU: reads move an entry to the end and inserts
    # past generation_cache_size evict the least-recently-used entry (see
    # _cached_generation / _remember_generation).
    _generation_cache: "OrderedDict[tuple[str, int, int, float], str]" = field(
        factory=OrderedDict, init=False
    )

    def start_loading(self) -> None:
        """Download and load the model on a background thread.

        Idempotent. Called at the start of a reduction so the (possibly
        multi-gigabyte) download overlaps with the cheap passes; the LLM
        pass then calls wait_until_ready before its first generation.
        """
        if self._disabled or self._load_thread is not None or self._llama is not None:
            return

        def load() -> None:
            try:
                with self._thread_lock:
                    self._ensure_loaded()
            except Exception as e:
                # A genuine failure to download or load the model (offline
                # or flaky network, an unsupported or corrupt file, out of
                # memory) must not crash the reduction, which is running on
                # the classical passes: disable the client so
                # wait_until_ready returns and the LLM passes skip
                # themselves, and warn the user once, mirroring how a
                # failed grammar download is reported. `except Exception`
                # is deliberately narrow: KeyboardInterrupt and SystemExit
                # (never delivered to this background thread anyway, since
                # signals go to the main thread) and trio.Cancelled (which
                # cannot arise here: this is a plain thread, not a trio
                # task) derive from BaseException and so propagate rather
                # than silently disabling the client.
                # The exception (with its traceback) is kept on
                # _load_error for inspection; the warning names only its
                # type, mirroring how a failed grammar download is reported.
                self._load_error = e
                self._disabled = True
                print(
                    "WARNING: could not load the LLM model "
                    f"({type(e).__name__}); reducing without the LLM passes.",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                self._ready.set()

        self._load_thread = threading.Thread(target=load, daemon=True)
        self._load_thread.start()

    async def wait_until_ready(self) -> None:
        """Wait for the model, however long its download/load takes.

        Deliberately does not start the load itself: whether (and when)
        loading starts is the reducer's or the download coordinator's
        decision, and when the model needs downloading the user gets a
        say first.
        """
        if self._llama is not None or self._disabled:
            await trio.lowlevel.checkpoint()
            return
        await trio.to_thread.run_sync(self._ready.wait, abandon_on_cancel=True)
        # A load that failed has already disabled the client and warned
        # the user (see start_loading); waiters simply return so the LLM
        # passes see is_disabled() and skip themselves.

    def model_needs_download(self) -> bool:
        """Whether the first use would download the model from the Hub."""
        if not isinstance(self.model, HuggingFaceModel) or huggingface_hub is None:
            return False
        cached = huggingface_hub.try_to_load_from_cache(
            self.model.repo_id, self.model.filename
        )
        return not isinstance(cached, str)

    def is_disabled(self) -> bool:
        return self._disabled

    def disable(self) -> None:
        """Decline the model: never download it, and release any waiters."""
        self._disabled = True
        self._ready.set()

    def _cached_generation(self, key: tuple[str, int, int, float]) -> str | None:
        """The cached completion for these parameters, marked most-recently
        used, or None if it isn't cached."""
        try:
            value = self._generation_cache[key]
        except KeyError:
            return None
        self._generation_cache.move_to_end(key)
        return value

    def _remember_generation(
        self, key: tuple[str, int, int, float], value: str
    ) -> None:
        """Cache a completion as most-recently used, evicting the
        least-recently used entries once over generation_cache_size."""
        self._generation_cache[key] = value
        self._generation_cache.move_to_end(key)
        while len(self._generation_cache) > self.generation_cache_size:
            self._generation_cache.popitem(last=False)

    async def complete(
        self, prompt: str, *, max_tokens: int, seed: int, temperature: float
    ) -> str:
        cached = self._cached_generation((prompt, max_tokens, seed, temperature))
        if cached is not None:
            await trio.lowlevel.checkpoint()
            return cached
        abort = threading.Event()

        def run_blocking() -> str:
            return self._complete_blocking(
                prompt,
                max_tokens=max_tokens,
                seed=seed,
                temperature=temperature,
                abort=abort,
            )

        # abandon_on_cancel so that skipping the pass doesn't have to wait
        # out an in-flight generation; the lock in _complete_blocking keeps
        # the abandoned thread from overlapping with the next call.
        try:
            return await trio.to_thread.run_sync(run_blocking, abandon_on_cancel=True)
        except trio.Cancelled:
            # Tell the abandoned generation to stop at the next token so
            # it releases the model promptly instead of running out its
            # whole token budget.
            abort.set()
            raise

    def _complete_blocking(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
        temperature: float,
        abort: threading.Event,
    ) -> str:
        key = (prompt, max_tokens, seed, temperature)
        with self._thread_lock:
            # This request may have queued behind another generation while
            # holding no lock. Now that it has the lock, don't pay for a
            # full model load and generation if it was cancelled in the
            # meantime, or during interpreter shutdown.
            if abort.is_set() or self._shutting_down.is_set():
                return ""
            # Another request may have generated exactly this while we
            # waited for the lock; serve its result rather than
            # regenerating it.
            cached = self._cached_generation(key)
            if cached is not None:
                return cached
            llama = self._ensure_loaded()
            try:
                response = llama.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    stream=True,
                )
                # Streaming is requested, so the response is an iterator of
                # chunks, not a single mapping.
                assert isinstance(response, Iterator)
                parts: list[str] = []
                for chunk in response:
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        parts.append(delta)
                    if abort.is_set() or self._shutting_down.is_set():
                        break
            except ValueError:
                # llama.cpp raises ValueError when the prompt plus the
                # completion budget exceeds the context window. The pass's
                # byte-based size limit is only a heuristic, so token-dense
                # input (emoji, rare scripts, dense punctuation) can clear
                # it yet overflow here. A failed generation must never
                # crash the reduction, so treat an overflow as an empty
                # (unhelpful) answer rather than propagating.
                return ""
            result = "".join(parts)
            # Only completed generations are cached: an aborted one is a
            # truncated answer that must not satisfy a later request.
            if not (abort.is_set() or self._shutting_down.is_set()):
                self._remember_generation(key, result)
            return result

    def _join_at_exit(self, timeout: float = 30.0) -> None:
        """Stop and wait out any in-flight generation before interpreter exit.

        Exiting while llama.cpp is mid-generation crashes in its C++/Metal
        finalizers. In-flight generations notice _shutting_down at the next
        token; acquiring the lock then guarantees nothing is inside
        llama.cpp when teardown proceeds. The timeout bounds shutdown if a
        generation is somehow stuck inside the model.
        """
        self._shutting_down.set()
        if self._thread_lock.acquire(timeout=timeout):
            self._thread_lock.release()

    def _ensure_loaded(self) -> "Llama":
        if self._llama is None:
            if llama_cpp is None or huggingface_hub is None:
                raise ImportError(
                    "llama-cpp-python is not installed or cannot load on "
                    "this platform, so the LLM passes are unavailable. Run "
                    "with --no-llm to silence this."
                )
            if isinstance(self.model, HuggingFaceModel):
                cached = huggingface_hub.try_to_load_from_cache(
                    self.model.repo_id, self.model.filename
                )
                if not isinstance(cached, str):
                    print(
                        f"Downloading the LLM model {self.model.repo_id} "
                        f"({self.model.filename}) from Hugging Face. This "
                        "happens once and may take a while; reduction "
                        "continues meanwhile, and the LLM passes join in "
                        "when it's ready. Pass --no-llm to disable them.",
                        file=sys.stderr,
                        flush=True,
                    )
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
            atexit.register(self._join_at_exit)
        return self._llama
