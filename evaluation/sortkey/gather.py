"""Gather valid test cases via random reductions.

For each seed we run shrink ray several times with an interestingness test
that (a) requires the candidate to parse under the language's strict front end
and (b) otherwise accepts a uniform-at-random fraction ``p`` of candidates,
keyed deterministically by ``Random(hash(salt + candidate))`` so the predicate
is stable (and cache-friendly). Every *parsing* candidate the reducer visits is
recorded, regardless of whether it was accepted -- so a single run sweeps out a
whole neighbourhood of valid programs, not just the reduction path.

Running with several salts and several values of ``p`` explores different
regions of the valid-program space. The union of everything seen, grouped by
seed, is written to ``corpus/<lang>/<seed>.jsonl`` (one base64 candidate per
line, deduplicated, the seed itself first).

    python evaluation/sortkey/gather.py python          # one language
    python evaluation/sortkey/gather.py python json      # several
    python evaluation/sortkey/gather.py                  # all languages
"""

from __future__ import annotations

import base64
import hashlib
import random
import sys
import time
from pathlib import Path

import trio

sys.path.insert(0, str(Path(__file__).parent))
from langs import LANGUAGES, Language, load_seeds  # noqa: E402

from shrinkray.problem import BasicReductionProblem, sort_key_for_initial  # noqa: E402
from shrinkray.reducer import ShrinkRay  # noqa: E402
from shrinkray.work import WorkContext  # noqa: E402

# A variety of acceptance probabilities and salts. Small p reduces hard (few
# accepted, so the reducer drives deep); large p keeps most candidates (so we
# sweep a wide, shallow neighbourhood). Different salts pick different subsets.
P_VALUES = [0.1, 0.5, 0.9]
SALTS = [b"alpha", b"beta"]

# Per-language execution knobs. C/C++ parse by shelling out to clang, which is
# slow but releases the GIL, so we run those in threads with real parallelism.
# The pure-Python front ends run single-threaded (fast + reproducible).
LANG_CONFIG: dict[str, dict[str, object]] = {
    "c": {"threaded": True, "parallelism": 8, "time_cap": 75.0},
    "cpp": {"threaded": True, "parallelism": 8, "time_cap": 75.0},
    "python": {"threaded": False, "parallelism": 1, "time_cap": 45.0},
    "sql": {"threaded": False, "parallelism": 1, "time_cap": 45.0},
    "json": {"threaded": False, "parallelism": 1, "time_cap": 45.0},
    "xml": {"threaded": False, "parallelism": 1, "time_cap": 45.0},
    "html": {"threaded": False, "parallelism": 1, "time_cap": 45.0},
}

# Stop recording (and cancel the run) once a seed has this many distinct valid
# candidates, to bound memory and time.
MAX_CANDIDATES = 15000


def accepts(salt: bytes, data: bytes, p: float) -> bool:
    return random.Random(hashlib.sha1(salt + data).digest()).random() <= p


def corpus_dir() -> Path:
    return Path(__file__).parent / "corpus"


async def _run_once(
    lang: Language,
    initial: bytes,
    salt: bytes,
    p: float,
    seen: dict[bytes, bytes],
    threaded: bool,
    parallelism: int,
    time_cap: float,
) -> None:
    """One random-reduction run, recording every parsing candidate into ``seen``."""

    def record_if_parses(data: bytes) -> bool:
        if data in seen:
            return True  # already known to parse
        ok = lang.parse(data)
        if ok:
            seen[data] = data
        return ok

    async def is_interesting(data: bytes) -> bool:
        if not data:
            return False
        if threaded:
            # Only the (blocking) parse goes to a thread; recording stays on the
            # trio task so the shared dict is never touched concurrently.
            if data in seen:
                parses = True
            else:
                parses = await trio.to_thread.run_sync(lang.parse, data)
                if parses:
                    seen[data] = data
        else:
            await trio.lowlevel.checkpoint()
            parses = record_if_parses(data)
        if not parses:
            return False
        if len(seen) >= MAX_CANDIDATES:
            raise _Enough()
        return data == initial or accepts(salt, data, p)

    problem: BasicReductionProblem[bytes] = BasicReductionProblem(
        initial=initial,
        is_interesting=is_interesting,
        work=WorkContext(parallelism=parallelism),
        sort_key=sort_key_for_initial(initial),
    )
    reducer = ShrinkRay(target=problem)
    try:
        with trio.move_on_after(time_cap):
            await reducer.run()
    except* _Enough:
        pass


class _Enough(Exception):
    """Raised to stop a run once MAX_CANDIDATES is reached."""


def gather_language(name: str) -> None:
    lang = LANGUAGES[name]
    cfg = LANG_CONFIG[name]
    seeds = load_seeds(name)
    out_dir = corpus_dir() / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed_name, initial in seeds:
        seen: dict[bytes, bytes] = {initial: initial}
        start = time.monotonic()
        for salt in SALTS:
            for p in P_VALUES:
                if len(seen) >= MAX_CANDIDATES:
                    break
                trio.run(
                    _run_once,
                    lang,
                    initial,
                    salt,
                    p,
                    seen,
                    cfg["threaded"],
                    cfg["parallelism"],
                    cfg["time_cap"],
                )
        elapsed = time.monotonic() - start
        stem = seed_name.rsplit(".", 1)[0]
        out_path = out_dir / f"{stem}.jsonl"
        # Seed first, then the rest in a stable (size, bytes) order.
        ordered = [initial] + sorted(
            (d for d in seen if d != initial), key=lambda d: (len(d), d)
        )
        with out_path.open("w") as f:
            for d in ordered:
                f.write(base64.b64encode(d).decode() + "\n")
        print(
            f"{name}/{stem}: {len(ordered)} valid candidates "
            f"in {elapsed:.0f}s -> {out_path.relative_to(Path.cwd())}"
        )


def main() -> int:
    names = sys.argv[1:] or list(LANGUAGES)
    for name in names:
        if name not in LANGUAGES:
            print(f"unknown language: {name}")
            return 2
        gather_language(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
