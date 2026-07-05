"""Test the "deletion + reformat is always a shrink" property.

For each formatted instance ``f`` in a corpus, delete a single byte or a single
line; if the result still parses, format it to ``f2``. The property under test:

    sort_key(f2) < sort_key(f)      (deleting + reformatting shrinks)

Because distinct strings have distinct natural-ordering keys, the only way to
get ``sort_key(f2) == sort_key(f)`` is ``f2 == f`` -- the formatter fully
absorbed the deletion (a benign no-op). So a genuine **violation** is a valid
deletion with ``sort_key(f2) > sort_key(f)``: deleting and reformatting yields
something the sort key considers *larger*.

We only consider formatted instances that are formatter fixed points
(``format(f) == f``) so the baseline is meaningful.

Output: ``corpus/_deletion_shrink/<lang>.json`` records, per seed, the counts
(split into byte vs line deletions) AND *every* violation, grouped by the
instance it came from (``violation_instances`` = ``[{f, violations:[{kind,
index, f2}]}]``). These are meant to be dug through when tuning the sort order.

    python evaluation/sortkey/deletion_shrink.py python json
    python evaluation/sortkey/deletion_shrink.py            # all languages
    SR_DEL_KINDS=line python evaluation/sortkey/deletion_shrink.py xml  # isolate
"""

from __future__ import annotations

import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gather import corpus_dir  # noqa: E402
from inversions import load_candidates  # noqa: E402
from langs import LANGUAGES, PYTHON_RUFF, Language  # noqa: E402

from shrinkray.problem import sort_key_for_initial  # noqa: E402

# Extra (non-corpus) languages usable here for comparison, plus which languages
# format by shelling out to a subprocess (so parse/format should be threaded)
# and which corpus directory to read candidates from.
EXTRA_LANGUAGES = {"python-ruff": PYTHON_RUFF}
THREADED = {"c", "cpp", "python-ruff"}
CORPUS_ALIAS = {"python-ruff": "python"}  # reuse the black-gathered python corpus


def get_language(name: str) -> Language | None:
    if name in LANGUAGES:
        return LANGUAGES[name]
    return EXTRA_LANGUAGES.get(name)

# Bounds to keep the sweep tractable. Byte deletion on a length-L instance is L
# candidates, so we skip byte-deletion on very long instances (line deletion is
# always cheap and still runs on them).
MAX_BYTE_DELETION_LEN = 1200

# Per-language cap on how many formatted instances to probe. Clang-based
# languages parse each deletion by shelling out, so we sample instances there;
# the in-process languages probe every instance.
MAX_INSTANCES = {"c": 30, "cpp": 30}


def sample_instances(instances: list[bytes], cap: int | None) -> list[bytes]:
    """Take an evenly-spread (by size) sample of at most ``cap`` instances."""
    if cap is None or len(instances) <= cap:
        return instances
    ordered = sorted(instances, key=len)
    step = len(ordered) / cap
    return [ordered[int(i * step)] for i in range(cap)]


# Which deletion kinds to test; override with SR_DEL_KINDS=line (or =byte) to
# isolate one kind. Default tests both.
KINDS = frozenset(os.environ.get("SR_DEL_KINDS", "byte,line").split(","))


def deletions(f: bytes) -> list[tuple[str, int, bytes]]:
    """All single-byte and single-line deletions of ``f`` (deduplicated)."""
    out: list[tuple[str, int, bytes]] = []
    seen: set[bytes] = {f}
    if "byte" in KINDS and len(f) <= MAX_BYTE_DELETION_LEN:
        for i in range(len(f)):
            g = f[:i] + f[i + 1 :]
            if g not in seen:
                seen.add(g)
                out.append(("byte", i, g))
    if "line" in KINDS:
        lines = f.split(b"\n")
        for j in range(len(lines)):
            g = b"\n".join(lines[:j] + lines[j + 1 :])
            if g not in seen:
                seen.add(g)
                out.append(("line", j, g))
    return out


def formatted_fixed_points(lang: Language, raws: list[bytes], threaded: bool) -> list[bytes]:
    """Distinct formatted instances that are formatter fixed points."""
    fmt = _format_many(lang, raws, threaded)
    firsts = [f for f in fmt if f is not None]
    refmt = _format_many(lang, firsts, threaded)
    out: list[bytes] = []
    seen: set[bytes] = set()
    for f, f2 in zip(firsts, refmt, strict=True):
        if f2 == f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _format_many(lang: Language, items: list[bytes], threaded: bool) -> list[bytes | None]:
    if threaded:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lang.format, items))
    return [lang.format(i) for i in items]


def _parse_many(lang: Language, items: list[bytes], threaded: bool) -> list[bool]:
    if threaded:
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lang.parse, items))
    return [lang.parse(i) for i in items]


@dataclass
class InstanceResult:
    f: bytes
    valid: int
    noops: int
    # Every violation, as (kind, index, f2). f is the shared instance.
    violations: list[tuple[str, int, bytes]]


def check_instance(lang: Language, f: bytes, key, threaded: bool) -> InstanceResult:
    kf = key(f)
    cand = deletions(f)
    # Only reformat deletions that parse.
    parsed = _parse_many(lang, [g for _, _, g in cand], threaded)
    valid = [c for c, ok in zip(cand, parsed, strict=True) if ok]
    formatted = _format_many(lang, [g for _, _, g in valid], threaded)
    noops = 0
    violations: list[tuple[str, int, bytes]] = []
    for (kind, idx, _g), f2 in zip(valid, formatted, strict=True):
        if f2 is None:
            continue
        if f2 == f:
            noops += 1
        elif key(f2) > kf:
            violations.append((kind, idx, f2))
    return InstanceResult(f=f, valid=len(valid), noops=noops, violations=violations)


def analyse_language(name: str) -> None:
    lang = get_language(name)
    if lang is None:
        print(f"unknown language: {name}")
        return
    threaded = name in THREADED
    lang_dir = corpus_dir() / CORPUS_ALIAS.get(name, name)
    if not lang_dir.exists():
        print(f"{name}: no corpus (run gather.py {CORPUS_ALIAS.get(name, name)} first)")
        return
    print(f"\n=== {name}  (formatter: {lang.formatter}) ===")
    result: dict[str, object] = {"language": name, "seeds": {}}
    seeds_out = result["seeds"]
    assert isinstance(seeds_out, dict)
    for jsonl in sorted(lang_dir.glob("*.jsonl")):
        raws = load_candidates(jsonl)
        key = sort_key_for_initial(raws[0])
        instances = sample_instances(
            formatted_fixed_points(lang, raws, threaded), MAX_INSTANCES.get(name)
        )
        tot_valid = tot_noop = vb = vl = 0
        # Every violation, grouped by the instance it came from (so f is stored
        # once per instance, not once per violation).
        violation_instances: list[dict[str, object]] = []
        for f in instances:
            r = check_instance(lang, f, key, threaded)
            tot_valid += r.valid
            tot_noop += r.noops
            vb += sum(1 for kind, _, _ in r.violations if kind == "byte")
            vl += sum(1 for kind, _, _ in r.violations if kind == "line")
            if r.violations:
                violation_instances.append(
                    {
                        "f": base64.b64encode(r.f).decode(),
                        "violations": [
                            {
                                "kind": kind,
                                "index": idx,
                                "f2": base64.b64encode(f2).decode(),
                            }
                            for kind, idx, f2 in r.violations
                        ],
                    }
                )
        print(
            f"  {jsonl.stem:32} instances={len(instances):>4} "
            f"valid_deletions={tot_valid:>6} no_ops={tot_noop:>6} "
            f"violations(byte={vb:>5} line={vl:>4})"
        )
        seeds_out[jsonl.stem] = {
            "instances": len(instances),
            "valid_deletions": tot_valid,
            "no_ops": tot_noop,
            "violations_byte": vb,
            "violations_line": vl,
            "violation_instances": violation_instances,
        }
    out_dir = corpus_dir() / "_deletion_shrink"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if KINDS == frozenset({"byte", "line"}) else f"__{'_'.join(sorted(KINDS))}"
    (out_dir / f"{name}{suffix}.json").write_text(json.dumps(result, indent=2))


def main() -> int:
    names = sys.argv[1:] or list(LANGUAGES)
    for name in names:
        if get_language(name) is None:
            print(f"unknown language: {name}")
            return 2
        analyse_language(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
