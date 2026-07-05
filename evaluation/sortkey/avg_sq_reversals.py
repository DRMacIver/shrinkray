"""Find formatted-corpus pairs whose sort order flips if avg_sq_line is removed.

Current chain:  byte_len, avg_sq_line, line_count, line_len_list, char_order
Without avg_sq: byte_len,              line_count, line_len_list, char_order

Removing avg_sq only changes an order when byte_len ties (it is the next
criterion), so we bucket formatted instances by byte length and, within a
bucket, find pairs where the two chains give opposite orders (and line_count
disagrees, i.e. a genuine avg_sq-vs-line_count conflict). Degenerate candidates
(control chars, near-empty) are filtered out.
"""

from __future__ import annotations

import base64
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "evaluation/sortkey")
from langs import LANGUAGES  # noqa: E402
from shrinkray.problem import natural_string_lex  # noqa: E402

WS = set(" \t\n\r\x0b\x0c")


def avg_sq(s):
    lines = s.split("\n")
    return sum(len(l) ** 2 for l in lines) / len(lines) ** 2


def line_count(s):
    return len(s.splitlines())


def line_len_list(s):
    return list(map(len, s.splitlines()))


def key_with(s):
    return (len(s), avg_sq(s), line_count(s), line_len_list(s), natural_string_lex(s))


def key_without(s):
    return (len(s), line_count(s), line_len_list(s), natural_string_lex(s))


def norm(s):
    return re.sub(r"[0-9]+", "0", re.sub(r"[A-Za-z_][A-Za-z0-9_]*", "N", s))


def formatted_instances(lang_name):
    lang = LANGUAGES[lang_name]
    raws = []
    for jf in sorted((Path("evaluation/sortkey/corpus") / lang_name).glob("*.jsonl")):
        for line in jf.read_text().splitlines():
            if line.strip():
                raws.append(base64.b64decode(line))
    threaded = lang_name in ("c", "cpp")
    if threaded:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fmts = list(pool.map(lang.format, raws))
    else:
        fmts = [lang.format(r) for r in raws]
    out = set()
    for f in fmts:
        if f is None:
            continue
        try:
            s = f.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if sum(1 for c in s if c not in WS) < 5:
            continue
        if any(ord(c) < 32 and c not in WS for c in s):
            continue
        out.add(s)
    return sorted(out)


def main():
    names = sys.argv[1:] or list(LANGUAGES)
    for name in names:
        insts = formatted_instances(name)
        buckets = defaultdict(list)
        for s in insts:
            buckets[len(s)].append(s)
        reversals = []
        for _blen, group in buckets.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    x, y = group[i], group[j]
                    if (key_with(x) < key_with(y)) != (key_without(x) < key_without(y)):
                        pref, drop = (x, y) if key_with(x) < key_with(y) else (y, x)
                        if line_count(pref) != line_count(drop):
                            reversals.append((pref, drop))
        more = sum(1 for p, d in reversals if line_count(p) > line_count(d))
        print(
            f"\n=== {name}: {len(insts)} formatted instances, "
            f"{len(reversals)} avg_sq-decided reversals "
            f"(avg_sq prefers MORE lines in {more}/{len(reversals)}) ==="
        )
        seen = set()
        curated = []
        for pref, drop in sorted(reversals, key=lambda pd: len(pd[0])):
            sig = (norm(pref), norm(drop))
            if sig in seen:
                continue
            seen.add(sig)
            curated.append((pref, drop))
        print(f"    ({len(curated)} distinct structural signatures)")
        for pref, drop in curated[:4]:
            print(
                f"  --- len={len(pref)}  avg_sq {avg_sq(pref):.1f}(pref) vs "
                f"{avg_sq(drop):.1f}  |  lines {line_count(pref)}(pref) vs "
                f"{line_count(drop)} ---"
            )
            print("  current key (with avg_sq) PREFERS:  " + repr(pref))
            print("  without avg_sq would prefer:        " + repr(drop))


if __name__ == "__main__":
    main()
