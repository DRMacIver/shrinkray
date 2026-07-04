"""Score a sort order against the labelled ordering corpus.

For each pair in `ordering_pairs.json`, the labelled `simpler` member should be
ranked strictly smaller by a good shrink order. This reports, per pair and in
aggregate (overall and by kind), whether a given key agrees — defaulting to
shrink ray's current natural-ordering key.

    python evaluation/sortkey/ordering_eval.py          # score the current key

To score a candidate key, import `evaluate` and pass a `str -> comparable`
function; `accuracy` / the printed WRONG list are the regression signal to
improve without regressing the cases already handled.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from shrinkray.problem import NATURAL_ORDERING_FUNCTIONS, sort_key_for_initial

CRITERIA = ["length", "avg_sq_line", "line_count", "line_len_list", "char_order"]


def current_key(s: str) -> Any:
    """Shrink ray's current text sort key (natural ordering)."""
    return sort_key_for_initial(s.encode("utf-8"))(s.encode("utf-8"))


def deciding_criterion(a: str, b: str) -> str:
    for name, f in zip(CRITERIA, NATURAL_ORDERING_FUNCTIONS, strict=True):
        if f(a) != f(b):
            return name
    return "equal"


def load_pairs() -> list[dict]:
    path = Path(__file__).parent / "ordering_pairs.json"
    return json.loads(path.read_text())


def evaluate(key: Callable[[str], Any] = current_key) -> list[dict]:
    """Return per-pair results: correct = key ranks the labelled-simpler one lower."""
    results = []
    for p in load_pairs():
        simpler = p[p["simpler"]]
        other = p["a" if p["simpler"] == "b" else "b"]
        correct = key(simpler) < key(other)
        results.append(
            {
                **p,
                "correct": correct,
                "byte_delta": len(simpler) - len(other),  # <0 means simpler is shorter
                "decides": deciding_criterion(p["a"], p["b"]),
            }
        )
    return results


def main() -> int:
    results = evaluate()
    by_kind: dict[str, list[bool]] = {}
    print(f"{'id':40} {'kind':11} {'conf':7} {'ok?':4} {'Δbytes':>7} {'decides':>12}")
    print("-" * 86)
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])
        mark = "OK" if r["correct"] else "XX"
        print(
            f"{r['id']:40} {r['kind']:11} {r['confidence']:7} {mark:4} "
            f"{r['byte_delta']:>+7} {r['decides']:>12}"
        )
    total = len(results)
    right = sum(r["correct"] for r in results)
    print("-" * 86)
    print(f"overall: {right}/{total} match the labelled-simpler judgement")
    print("by kind:")
    for kind, oks in sorted(by_kind.items()):
        print(f"  {kind:11} {sum(oks)}/{len(oks)}")
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nwhere the current order disagrees ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['id']:40} — {r['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
