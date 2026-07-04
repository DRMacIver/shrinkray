"""Test a reflow-based sort key with a closeness-to-canonical tie-break.

reflow_key(x) = (
    natural_key(basic_format(x)),   # layout-immune: canonical content+structure
    distance(x, basic_format(x)),   # prefer raw already CLOSE to its canonical form
    natural_key(x),                 # deterministic final tiebreak
)

The second term is the user's point: candidates that canonicalise to the same
string are NOT equal (the reflow may be lossy/invalid), so break the tie toward
the one that needs the least reformatting -- i.e. the cleanest valid raw form.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "evaluation/sortkey")
from basic_format import basic_format  # noqa: E402
from ordering_eval import CRITERIA, KEYS, evaluate, load_pairs  # noqa: E402

NAT = KEYS["current"]


def nk(s: str):
    return tuple(CRITERIA[c](s) for c in NAT)


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


def reflow_tie(x: str):
    c = basic_format(x)
    # (canonical content+structure, closeness, structural closeness in line count
    #  toward the readable canonical, deterministic tail).
    return (
        nk(c),
        lev(x, c),
        abs(len(x.splitlines()) - len(c.splitlines())),
        nk(x),
    )


def reflow_notie(x: str):
    # no tie-break: canonicalised-equal candidates tie (for comparison)
    return (nk(basic_format(x)), nk(x))  # nk(x) still there so it's not a pure tie


def report(name, key):
    results = evaluate(key)
    by_kind: dict[str, list[bool]] = {}
    wrong = []
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["correct"])
        if not r["correct"]:
            wrong.append(r)
    total = len(results)
    right = sum(r["correct"] for r in results)
    print(f"\n### {name}: {right}/{total}")
    for kind, oks in sorted(by_kind.items()):
        print(f"  {kind:11} {sum(oks)}/{len(oks)}")
    for r in wrong:
        print(f"    XX {r['id']:38} [{r['confidence']}]")


if __name__ == "__main__":
    report("reflow + closeness tiebreak", reflow_tie)
    report("reflow, size-only tiebreak", reflow_notie)
    # sanity: show a couple of formatting pairs canonicalise equal and the
    # tiebreak picks the labelled-simpler one.
    print("\n--- formatting pairs: does closeness pick the labelled-simpler? ---")
    for p in load_pairs():
        if p["kind"] != "formatting":
            continue
        a, b = p["a"], p["b"]
        ca, cb = basic_format(a), basic_format(b)
        if ca == cb:
            s = p["simpler"]
            simpler, other = p[s], p["a" if s == "b" else "b"]
            pick = "simpler" if lev(simpler, ca) < lev(other, ca) else (
                "other" if lev(other, ca) < lev(simpler, ca) else "tie")
            print(f"  {p['id']:38} canon-equal, closeness picks: {pick}")
