"""Score a sort order against the labelled ordering corpus.

For each pair in `ordering_pairs.json`, the labelled `simpler` member should be
ranked strictly smaller by a good shrink order. This reports, per pair and in
aggregate (overall and by kind), whether a given key agrees.

A sort order is expressed as an ordered *chain* of named criteria (see
`CRITERIA` / `KEYS`); the key is the tuple of those criteria, compared
lexicographically — matching how `LazyChainedSortKey` works. This makes it easy
to test candidate orders and see exactly which criterion decides each pair.

    python evaluation/sortkey/ordering_eval.py                 # current key
    python evaluation/sortkey/ordering_eval.py proposed        # a candidate
    python evaluation/sortkey/ordering_eval.py current proposed  # compare two

`evaluate(key)` also accepts any bare `str -> comparable` function.
"""

from __future__ import annotations

import string
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shrinkray.problem import natural_string_lex

_WS = set(string.whitespace)


def _lines(s: str) -> list[str]:
    return s.split("\n")


# Named criteria a sort order can be built from (all: str -> comparable).
CRITERIA: dict[str, Callable[[str], Any]] = {
    # total byte length (the current primary criterion)
    "byte_len": len,
    # length ignoring all whitespace
    "nonws_len": lambda s: sum(1 for c in s if c not in _WS),
    # average squared line length: penalises a few very long lines
    "avg_sq_line": lambda s: sum(len(l) ** 2 for l in _lines(s)) / len(_lines(s)) ** 2,
    "line_count": lambda s: len(s.splitlines()),
    "line_len_list": lambda s: list(map(len, s.splitlines())),
    # number of blank (whitespace-only) lines: junk the sort key should avoid
    "blank_lines": lambda s: sum(1 for l in s.splitlines() if not l.strip()),
    # natural character order (whitespace < digits < lower < upper); final tiebreak
    "char_order": natural_string_lex,
}

# Candidate sort orders, as ordered chains of criteria.
KEYS: dict[str, list[str]] = {
    # shrink ray's current natural ordering
    "current": ["byte_len", "avg_sq_line", "line_count", "line_len_list", "char_order"],
    # proposal: ignore whitespace in the primary length; only bring in total byte
    # length just before the final character-order tiebreaker.
    "proposed": [
        "nonws_len",
        "avg_sq_line",
        "line_count",
        "line_len_list",
        "byte_len",
        "char_order",
    ],
    # proposed + a blank-line penalty before the structure criterion, so blank
    # lines never win a tie (fixes the avg_sq blank-line hurts).
    "blank_aware": [
        "nonws_len",
        "blank_lines",
        "avg_sq_line",
        "line_count",
        "line_len_list",
        "byte_len",
        "char_order",
    ],
    # content-first, then prefer the most compact byte layout (no structure
    # criterion). Illustrates the tiny-compact vs readable-code tension.
    "compact": ["nonws_len", "byte_len", "char_order"],
}


def build_key(chain: list[str]) -> Callable[[str], Any]:
    funcs = [CRITERIA[c] for c in chain]
    return lambda s: tuple(f(s) for f in funcs)


def deciding_criterion(chain: list[str], a: str, b: str) -> str:
    for c in chain:
        if CRITERIA[c](a) != CRITERIA[c](b):
            return c
    return "equal"


def load_pairs() -> list[dict]:
    return __import__("json").loads(
        (Path(__file__).parent / "ordering_pairs.json").read_text()
    )


def evaluate(key: Callable[[str], Any], chain: list[str] | None = None) -> list[dict]:
    results = []
    for p in load_pairs():
        simpler = p[p["simpler"]]
        other = p["a" if p["simpler"] == "b" else "b"]
        results.append(
            {
                **p,
                "correct": key(simpler) < key(other),
                "byte_delta": len(simpler) - len(other),
                "decides": deciding_criterion(chain, p["a"], p["b"]) if chain else "",
            }
        )
    return results


def _score(name: str) -> list[dict]:
    return evaluate(build_key(KEYS[name]), KEYS[name])


def main() -> int:
    names = sys.argv[1:] or ["current"]
    for n in names:
        if n not in KEYS:
            print(f"unknown key '{n}'; known: {', '.join(KEYS)}")
            return 2

    if len(names) == 2:  # comparison mode
        a, b = names
        ra = {r["id"]: r for r in _score(a)}
        rb = {r["id"]: r for r in _score(b)}
        print(f"comparing '{a}' vs '{b}':\n")
        print(f"{'id':40} {'kind':11} {a[:8]:>8} {b[:8]:>8}  change")
        print("-" * 80)
        for pid in ra:
            ca, cb = ra[pid]["correct"], rb[pid]["correct"]
            ch = "fixed" if (cb and not ca) else "broke" if (ca and not cb) else ""
            print(
                f"{pid:40} {ra[pid]['kind']:11} "
                f"{'OK' if ca else 'XX':>8} {'OK' if cb else 'XX':>8}  {ch}"
            )
        print("-" * 80)
        print(
            f"{a}: {sum(r['correct'] for r in ra.values())}/{len(ra)}   "
            f"{b}: {sum(r['correct'] for r in rb.values())}/{len(rb)}"
        )
        return 0

    (name,) = names
    results = _score(name)
    by_kind: dict[str, list[bool]] = {}
    print(f"sort order: {name} = {' > '.join(KEYS[name])}\n")
    print(f"{'id':40} {'kind':11} {'conf':7} {'ok?':4} {'Δbytes':>7} {'decides':>13}")
    print("-" * 88)
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])
        print(
            f"{r['id']:40} {r['kind']:11} {r['confidence']:7} "
            f"{'OK' if r['correct'] else 'XX':4} {r['byte_delta']:>+7} {r['decides']:>13}"
        )
    total = len(results)
    right = sum(r["correct"] for r in results)
    print("-" * 88)
    print(f"overall: {right}/{total}")
    for kind, oks in sorted(by_kind.items()):
        print(f"  {kind:11} {sum(oks)}/{len(oks)}")
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\ndisagrees ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['id']:40} [{r['confidence']}] — {r['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
