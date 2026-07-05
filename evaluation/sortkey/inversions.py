"""Find sort-key / formatter disagreements in a gathered corpus.

For a language's gathered candidates we look for pairs (x, y) where shrink ray's
current sort key prefers x over y, but after formatting the preference flips:

    sort_key(x) < sort_key(y)   but   sort_key(format(x)) > sort_key(format(y))

Each such pair is a place where the sort key and the formatter disagree. This is
a starting point for investigation, not proof the sort key is wrong: a formatter
is only one proxy for human preference, and the sort key may well be reasonable.
These pairs are the raw material for deciding how a new sort key should behave.

We report, per seed:
  * how often the sort key and the formatter disagree (inversion rate), and
  * a curated set of the most striking example pairs, with a diagnosis of which
    sort-key criterion is responsible for the flip.

Example pairs are written to ``corpus/_inversions/<lang>.json`` (gitignored) and
the most striking ones are printed.

    python evaluation/sortkey/inversions.py python
    python evaluation/sortkey/inversions.py            # all languages
"""

from __future__ import annotations

import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from gather import corpus_dir  # noqa: E402
from langs import LANGUAGES, Language  # noqa: E402

from shrinkray.problem import (  # noqa: E402
    NATURAL_ORDERING_FUNCTIONS,
    sort_key_for_initial,
)

# Names of the natural-ordering criteria, in priority order, for diagnosis.
CRITERIA = [
    "length",
    "avg_sq_line_length",
    "line_count",
    "line_length_list",
    "char_order",
]

TOP_K = 12  # example pairs to keep/print per seed


def components(data: bytes) -> list[object]:
    """The natural-ordering criteria vector for a candidate (for diagnosis)."""
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        s = data.decode("latin-1")
    return [f(s) for f in NATURAL_ORDERING_FUNCTIONS]


def first_diff_criterion(a: list[object], b: list[object]) -> str:
    for name, av, bv in zip(CRITERIA, a, b, strict=True):
        if av != bv:
            return name
    return "equal"


@dataclass
class Candidate:
    raw: bytes
    formatted: bytes
    key: Any  # sort_key(raw)
    fkey: Any  # sort_key(formatted)


@dataclass
class Example:
    x: bytes
    y: bytes
    fx: bytes
    fy: bytes
    criterion_raw: str  # which criterion makes x < y
    criterion_fmt: str  # which criterion makes fx > fy


def load_candidates(path: Path) -> list[bytes]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(base64.b64decode(line))
    return out


def build_candidates(
    lang: Language, raws: list[bytes], threaded: bool
) -> list[Candidate]:
    key = sort_key_for_initial(raws[0])
    if threaded:
        with ThreadPoolExecutor(max_workers=8) as pool:
            formatted = list(pool.map(lang.format, raws))
    else:
        formatted = [lang.format(r) for r in raws]
    result = []
    for raw, fmt in zip(raws, formatted, strict=True):
        if fmt is None:
            continue
        result.append(Candidate(raw=raw, formatted=fmt, key=key(raw), fkey=key(fmt)))
    return result


def _merge_count(values: list[Any]) -> int:
    """Count pairs i < j with values[i] > values[j] (strict), in O(n log n)."""
    if len(values) <= 1:
        return 0
    mid = len(values) // 2
    left, right = values[:mid], values[mid:]
    inv = _merge_count(left) + _merge_count(right)
    merged: list[Any] = []
    i = j = 0
    while i < len(left) and j < len(right):
        # Take from the left while it is <= the right element: those form no
        # inversion. When the left element is strictly greater, every remaining
        # left element is greater than this right element.
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv += len(left) - i
    merged.extend(left[i:])
    merged.extend(right[j:])
    values[:] = merged
    return inv


def count_inversions(cands: list[Candidate]) -> tuple[int, int]:
    """Return (inversion_count, comparable_pairs) over all pairs.

    An inversion is a pair with key(x) < key(y) but fkey(x) > fkey(y). Distinct
    strings always have distinct natural-ordering keys (the char-order criterion
    is injective), so every pair is comparable and comparable_pairs = C(n, 2);
    the inversion count is then the number of strict inversions in the sequence
    of formatted keys taken in raw-key order.
    """
    ordered = sorted(cands, key=lambda c: c.key)
    fkeys: list[Any] = [c.fkey for c in ordered]
    inv = _merge_count(fkeys)
    n = len(ordered)
    pairs = n * (n - 1) // 2
    return inv, pairs


_WHITESPACE = frozenset(b" \t\r\n\f\v")


def nontrivial(data: bytes) -> bool:
    """Reject degenerate candidates for tuning purposes.

    Valid-but-trivial programs (``\\n``, a lone ``*``) are real but useless for
    tuning human preference, and they dominate the "smallest" examples. We also
    drop candidates containing control characters (e.g. a comment full of
    ``\\x10`` bytes): they parse but are artifacts, not code a human would write.
    Require a handful of non-whitespace characters and clean text.
    """
    if sum(1 for b in data if b not in _WHITESPACE) < 3:
        return False
    return not any(b < 0x20 and b not in _WHITESPACE for b in data)


def adjacent_inversions(cands: list[Candidate]) -> list[Example]:
    """Every adjacent-in-sort-key pair whose order flips once formatted.

    Adjacent pairs are the tightest examples: x is *just* below y by sort key,
    yet formatting makes x the larger of the two. The full list also gives a
    representative sample for tallying which criterion causes flips.
    """
    ordered = sorted(cands, key=lambda c: c.key)
    out: list[Example] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a.key < b.key and a.fkey > b.fkey and nontrivial(a.raw) and nontrivial(b.raw):
            out.append(
                Example(
                    x=a.raw,
                    y=b.raw,
                    fx=a.formatted,
                    fy=b.formatted,
                    criterion_raw=first_diff_criterion(
                        components(a.raw), components(b.raw)
                    ),
                    criterion_fmt=first_diff_criterion(
                        components(a.formatted), components(b.formatted)
                    ),
                )
            )
    return out


def criterion_tally(examples: list[Example]) -> dict[str, dict[str, int]]:
    raw: dict[str, int] = {}
    fmt: dict[str, int] = {}
    for ex in examples:
        raw[ex.criterion_raw] = raw.get(ex.criterion_raw, 0) + 1
        fmt[ex.criterion_fmt] = fmt.get(ex.criterion_fmt, 0) + 1
    return {"raw": raw, "fmt": fmt}


def top_examples(examples: list[Example]) -> list[Example]:
    """A diverse set of up to TOP_K adjacent inversions for tuning.

    We keep both the highest-severity flips (formatting inflates x the most, in
    absolute bytes) and the smallest flips (tightest, most readable pairs), so
    the tuning set spans big-impact and easy-to-inspect cases.
    """
    half = TOP_K // 2
    by_severity = sorted(examples, key=lambda e: len(e.fx) - len(e.fy), reverse=True)
    by_size = sorted(examples, key=lambda e: len(e.fx) + len(e.fy))
    picked: list[Example] = []
    seen: set[tuple[bytes, bytes]] = set()
    for ex in by_severity[: TOP_K - half] + by_size:
        marker = (ex.x, ex.y)
        if marker in seen:
            continue
        seen.add(marker)
        picked.append(ex)
        if len(picked) >= TOP_K:
            break
    return picked


def analyse_language(name: str) -> None:
    lang = LANGUAGES[name]
    threaded = name in ("c", "cpp")
    lang_dir = corpus_dir() / name
    if not lang_dir.exists():
        print(f"{name}: no corpus (run gather.py {name} first)")
        return
    out: dict[str, object] = {"language": name, "formatter": lang.formatter, "seeds": {}}
    seeds_out = out["seeds"]
    assert isinstance(seeds_out, dict)
    print(f"\n=== {name}  (formatter: {lang.formatter}) ===")
    for jsonl in sorted(lang_dir.glob("*.jsonl")):
        raws = load_candidates(jsonl)
        cands = build_candidates(lang, raws, threaded)
        formattable = len(cands)
        inv, pairs = count_inversions(cands)
        adjacent = adjacent_inversions(cands)
        tally = criterion_tally(adjacent)
        examples = top_examples(adjacent)
        rate = (inv / pairs * 100) if pairs else 0.0
        print(
            f"  {jsonl.stem:32} candidates={len(raws):>5} "
            f"formattable={formattable:>5} inversions={inv:>6}/{pairs:<7} "
            f"({rate:4.1f}%) flip-criterion={tally['fmt']}"
        )
        seeds_out[jsonl.stem] = {
            "candidates": len(raws),
            "formattable": formattable,
            "inversions": inv,
            "comparable_pairs": pairs,
            "inversion_rate_pct": round(rate, 3),
            "adjacent_inversions": len(adjacent),
            "flip_criterion_tally": tally,
            "examples": [
                {
                    "x": base64.b64encode(ex.x).decode(),
                    "y": base64.b64encode(ex.y).decode(),
                    "fx": base64.b64encode(ex.fx).decode(),
                    "fy": base64.b64encode(ex.fy).decode(),
                    "criterion_raw": ex.criterion_raw,
                    "criterion_fmt": ex.criterion_fmt,
                }
                for ex in examples
            ],
        }
    out_dir = corpus_dir() / "_inversions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))


def main() -> int:
    names = sys.argv[1:] or list(LANGUAGES)
    for name in names:
        if name not in LANGUAGES:
            print(f"unknown language: {name}")
            return 2
        analyse_language(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
