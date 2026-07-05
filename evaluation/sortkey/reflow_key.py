"""Score the PRODUCTION reflow sort key against the labelled ordering corpus.

This imports the real key from shrinkray (src), so the number reported is the
one the shipped reducer uses. See problem.reflow_sort_key / reformat.basic_format.

    python evaluation/sortkey/reflow_key.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ordering_eval import evaluate, load_pairs  # noqa: E402

from shrinkray.problem import reflow_sort_key  # noqa: E402
from shrinkray.reformat import basic_format  # noqa: E402


def report(name, key) -> None:
    results = evaluate(key)
    by_kind: dict[str, list[bool]] = {}
    wrong = []
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])
        if not r["correct"]:
            wrong.append(r)
    right = sum(r["correct"] for r in results)
    print(f"\n### {name}: {right}/{len(results)}")
    for kind, oks in sorted(by_kind.items()):
        print(f"  {kind:11} {sum(oks)}/{len(oks)}")
    for r in wrong:
        print(f"    XX {r['id']:38} [{r['confidence']}]")


if __name__ == "__main__":
    report("production reflow_sort_key", reflow_sort_key)
    # sanity: layout-only pairs canonicalise to the same string
    same = sum(
        1
        for p in load_pairs()
        if p["kind"] == "formatting" and basic_format(p["a"]) == basic_format(p["b"])
    )
    print(f"\nformatting pairs that canonicalise equal: {same}")
