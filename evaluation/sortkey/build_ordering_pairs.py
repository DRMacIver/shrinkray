"""Build the labelled ordering corpus `ordering_pairs.json`.

Each pair is two similar test cases (`a`, `b`) plus a human judgement of which
one a *good* shrink order should treat as simpler (i.e. rank strictly smaller,
so the reducer would output it). It is the target the sort key should match.

Fields per pair:
  id         - stable identifier
  lang       - c | cpp | python | sql | json | xml | html
  kind       - formatting | content | cosmetic | quirk (see below)
  a, b       - the two test cases
  simpler    - "a" or "b": which one a good shrink order should prefer
  confidence - high | medium | low (how sure the judgement is)
  rationale  - one line explaining the call

Kinds:
  formatting - a and b are the SAME program modulo whitespace. For real code we
               prefer the readable (formatted) form; for tiny data/markup the
               compact form is fine. Tests that the order does not fight the
               formatter (this is where the current key does worst).
  content    - a genuine difference in program content (fewer statements, args,
               columns, elements). The one with less content is simpler even if
               a formatter makes it longer (the magic-trailing-comma cases).
  cosmetic   - equivalent modulo a local, natural-ordering choice (identifier
               length/case, digit choice, redundant whitespace).
  quirk      - one member is a formatter/parser CORRUPTION (escaped markup,
               injected empty element, mangled parse, merged #defines). The
               clean member is simpler; these guard against accepting garbage.

The real C/C++ formatting pairs are taken from the committed
`evaluation/corpus/*/shrinkray_reduced.{c,cpp}` outputs and their clang-format
rendering (comments stripped, since FixNamespaceComments adds `} // namespace`).

    python evaluation/sortkey/build_ordering_pairs.py
"""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path


def strip_c_comments(s: str) -> str:
    out: list[str] = []
    i, n, st = 0, len(s), None
    while i < n:
        c = s[i]
        nx = s[i + 1] if i + 1 < n else ""
        if st is None:
            if c == "/" and nx == "/":
                st = "line"
                i += 2
                continue
            if c == "/" and nx == "*":
                st = "blk"
                i += 2
                continue
            if c == '"':
                st = "str"
            elif c == "'":
                st = "chr"
            out.append(c)
        elif st in ("str", "chr"):
            out.append(c)
            if c == "\\":
                out.append(nx)
                i += 2
                continue
            if (st == "str" and c == '"') or (st == "chr" and c == "'"):
                st = None
        elif st == "line":
            if c == "\n":
                out.append(c)
                st = None
        elif st == "blk":
            if c == "*" and nx == "/":
                st = None
                i += 2
                continue
        i += 1
    text = "".join(out)
    keep = []
    for orig, new in zip(s.split("\n"), text.split("\n"), strict=False):
        r = new.rstrip()
        if r == "" and orig.strip() != "":
            continue  # a comment-only line collapsed to nothing
        keep.append(r)
    return "\n".join(keep)


def real_cpp_formatting_pairs() -> list[dict]:
    pairs = []
    files = sorted(
        glob.glob("evaluation/corpus/*/shrinkray_reduced.c")
        + glob.glob("evaluation/corpus/*/shrinkray_reduced.cpp")
    )
    for path in files:
        raw = Path(path).read_bytes()
        proc = subprocess.run(["clang-format"], input=raw, capture_output=True)
        if proc.returncode != 0:
            continue
        formatted = strip_c_comments(proc.stdout.decode())
        entry = Path(path).parent.name
        lang = "c" if path.endswith(".c") else "cpp"
        pairs.append(
            {
                "id": f"{lang}-format-{entry}",
                "lang": lang,
                "kind": "formatting",
                "a": raw.decode(),  # cramped (what shrink ray produced)
                "b": formatted,  # readable
                "simpler": "b",
                "confidence": "high",
                "rationale": "same program; the readable multi-line form is the "
                "desired reduced output, not the dense one-liner",
            }
        )
    return pairs


# --- hand-authored pairs ---------------------------------------------------

HAND_PAIRS: list[dict] = [
    # formatting: readable code preferred
    {
        "id": "python-semicolons-vs-lines",
        "lang": "python",
        "kind": "formatting",
        "a": "def f(x):\n    a = x + 1; b = a * 2; return b\n",
        "b": "def f(x):\n    a = x + 1\n    b = a * 2\n    return b\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "same code; one statement per line is the readable form",
    },
    # formatting: for tiny data/markup, compact is fine (order should NOT expand)
    {
        "id": "json-tiny-compact",
        "lang": "json",
        "kind": "formatting",
        "a": '{"a": 1, "b": 2}\n',
        "b": '{\n  "a": 1,\n  "b": 2\n}\n',
        "simpler": "a",
        "confidence": "medium",
        "rationale": "a tiny object is at least as clear compact; expansion adds "
        "nothing",
    },
    {
        "id": "html-tiny-compact",
        "lang": "html",
        "kind": "formatting",
        "a": "<p>hi</p>\n",
        "b": "<p>\n hi\n</p>\n",
        "simpler": "a",
        "confidence": "medium",
        "rationale": "a trivial element is clearer on one line",
    },
    # content: fewer elements is simpler (clean, order should already agree)
    {
        "id": "python-fewer-params-clean",
        "lang": "python",
        "kind": "content",
        "a": "def f(a, b, c):\n    pass\n",
        "b": "def f(a, b):\n    pass\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "two parameters is simpler than three",
    },
    {
        "id": "python-fewer-statements",
        "lang": "python",
        "kind": "content",
        "a": "x = 1\ny = 2\nz = 3\n",
        "b": "x = 1\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "one statement is simpler than three",
    },
    {
        "id": "c-remove-unused-var",
        "lang": "c",
        "kind": "content",
        "a": "int f() {\n  int x = 0;\n  int y = 1;\n  return x;\n}\n",
        "b": "int f() {\n  int x = 0;\n  return x;\n}\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "the unused variable is gone",
    },
    {
        "id": "sql-fewer-columns",
        "lang": "sql",
        "kind": "content",
        "a": "SELECT a, b, c FROM t;\n",
        "b": "SELECT a FROM t;\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "one selected column is simpler than three",
    },
    {
        "id": "html-fewer-list-items",
        "lang": "html",
        "kind": "content",
        "a": "<ul>\n <li>a</li>\n <li>b</li>\n</ul>\n",
        "b": "<ul>\n <li>a</li>\n</ul>\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "one list item is simpler than two",
    },
    # content but formatter inflates it: fewer elements yet LONGER (magic comma)
    {
        "id": "python-fewer-params-magic-comma",
        "lang": "python",
        "kind": "content",
        "a": "def f(a, b, c):\n    pass\n",
        "b": "def f(\n    a,\n    b,\n):\n    pass\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "b has two parameters (fewer content) vs three; black's "
        "magic trailing comma only makes it longer, not more complex",
    },
    {
        "id": "python-fewer-list-magic-comma",
        "lang": "python",
        "kind": "content",
        "a": "xs = [1, 2, 3]\n",
        "b": "xs = [\n    1,\n    2,\n]\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "b has two elements vs three; the exploded layout is "
        "incidental, the content is simpler",
    },
    # cosmetic: natural-ordering tiebreaks (order should already agree)
    {
        "id": "cosmetic-ident-length",
        "lang": "c",
        "kind": "cosmetic",
        "a": "int counter;\n",
        "b": "int c;\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "shorter identifier",
    },
    {
        "id": "cosmetic-ident-case",
        "lang": "c",
        "kind": "cosmetic",
        "a": "int A;\n",
        "b": "int a;\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "lowercase reads as simpler than uppercase (natural order)",
    },
    {
        "id": "cosmetic-digit-choice",
        "lang": "python",
        "kind": "cosmetic",
        "a": "x = 9\n",
        "b": "x = 0\n",
        "simpler": "b",
        "confidence": "medium",
        "rationale": "0 is the canonical smallest literal",
    },
    {
        "id": "cosmetic-number-magnitude",
        "lang": "python",
        "kind": "cosmetic",
        "a": "x = 100000\n",
        "b": "x = 0\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "a smaller literal is simpler",
    },
    {
        "id": "cosmetic-whitespace-noise",
        "lang": "python",
        "kind": "cosmetic",
        "a": "x  =  1\n",
        "b": "x = 1\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "no redundant internal spaces",
    },
    # quirk: prefer the clean member over a formatter/parser corruption
    {
        "id": "quirk-html-empty-element",
        "lang": "html",
        "kind": "quirk",
        "a": "<ul>\n <li>\n  x\n </li>\n <li>\n </li>\n</ul>\n",
        "b": "<ul>\n <li>\n  x\n </li>\n</ul>\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "the injected empty <li> is noise",
    },
    {
        "id": "quirk-xml-stray-escape",
        "lang": "xml",
        "kind": "quirk",
        "a": "<p>ok&gt;</p>\n",
        "b": "<p>ok</p>\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "the stray escaped '>' is corruption, not content",
    },
    {
        "id": "quirk-sql-mangled-alias",
        "lang": "sql",
        "kind": "quirk",
        "a": "SELECT total AS FROMorders;\n",
        "b": "SELECT total FROM orders;\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "a is a mangled parse (FROM absorbed into an alias); b is a "
        "real query",
    },
    {
        "id": "quirk-cpp-define-merge",
        "lang": "cpp",
        "kind": "quirk",
        "a": "#define A 1 #define B 2\n",
        "b": "#define A 1\n#define B 2\n",
        "simpler": "b",
        "confidence": "high",
        "rationale": "a merges two macros onto one line (A expands to '1 #define "
        "B 2'); b is two correct macros",
    },
    # borderline: the inline-comment tension surfaced in the deletion audit
    {
        "id": "python-inline-comment",
        "lang": "python",
        "kind": "formatting",
        "a": "x = 1\n# note\n",
        "b": "x = 1  # note\n",
        "simpler": "b",
        "confidence": "low",
        "rationale": "same code+comment; the inline form has one fewer line, but "
        "this is marginal",
    },
    # avg_sq_line ablation: same program, EQUAL byte length, so avg_sq_line is the
    # deciding criterion. It always prefers the version with more (shorter) lines.
    # "helps" = the extra line-break is at a statement/element boundary (good, so
    # avg_sq is right); "hurts" = it splits mid-construct or adds a blank line
    # (bad, so avg_sq is wrong). A good order should get BOTH right; the current
    # key gets helps right and hurts wrong. See avg_sq_reversals.py.
    {
        "id": "avgsq-helps-python-two-statements",
        "lang": "python",
        "kind": "formatting",
        "a": "x = 1;y = 2\n",
        "b": "x = 1\ny = 2\n",
        "simpler": "b",
        "confidence": "medium",
        "rationale": "one statement per line reads better than joining with ';' "
        "(avg_sq_line correctly prefers the split)",
    },
    {
        "id": "avgsq-helps-c-two-decls",
        "lang": "c",
        "kind": "formatting",
        "a": "int a; int b;\n",
        "b": "int a;\nint b;\n",
        "simpler": "b",
        "confidence": "medium",
        "rationale": "two declarations on their own lines (avg_sq_line right)",
    },
    {
        "id": "avgsq-helps-sql-two-statements",
        "lang": "sql",
        "kind": "formatting",
        "a": "SELECT 1; SELECT 2;\n",
        "b": "SELECT 1;\nSELECT 2;\n",
        "simpler": "b",
        "confidence": "medium",
        "rationale": "one statement per line (avg_sq_line right)",
    },
    {
        "id": "avgsq-helps-html-two-elements",
        "lang": "html",
        "kind": "formatting",
        "a": "<p>a</p> <p>b</p>\n",
        "b": "<p>a</p>\n<p>b</p>\n",
        "simpler": "b",
        "confidence": "medium",
        "rationale": "sibling elements on their own lines (avg_sq_line right)",
    },
    {
        "id": "avgsq-hurts-c-blank-in-body",
        "lang": "c",
        "kind": "formatting",
        "a": "void f() {}\n",
        "b": "void f()\n\n{}",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b puts a blank line inside an empty body; avg_sq_line wrongly "
        "prefers it because more lines lowers the average",
    },
    {
        "id": "avgsq-hurts-python-mid-expression",
        "lang": "python",
        "kind": "formatting",
        "a": "z = a + b\n",
        "b": "z = a +\nb\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b splits the expression after '+'; avg_sq_line wrongly prefers "
        "the split",
    },
    {
        "id": "avgsq-hurts-c-split-assignment",
        "lang": "c",
        "kind": "formatting",
        "a": "int xy = 0;\n",
        "b": "int xy =\n0;\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b breaks the assignment across two lines; avg_sq_line wrongly "
        "prefers it",
    },
    {
        "id": "avgsq-hurts-html-split-content",
        "lang": "html",
        "kind": "formatting",
        "a": "<p>hello</p>\n",
        "b": "<p>\nhello</p>",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b splits short inline content onto its own line; avg_sq_line "
        "wrongly prefers it",
    },
    # "oversplit" cases mined from the gathered corpus, where production and the
    # convex60 candidate disagree: a MEDIUM-length line (~15-25 chars) is broken
    # across lines. avg_sq_line and convex line_cost both reward the split
    # (breaking a long-ish line lowers the metric), but the one-line form is
    # clearly better. These are the residual "split quality is semantic" cases.
    {
        "id": "split-c-declaration",
        "lang": "c",
        "kind": "formatting",
        "a": "int main(void){}\n",
        "b": "int\nmain(void)\n{}\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b breaks a short declaration across three lines (between the "
        "return type and name, and the body); one line is clearly better",
    },
    {
        "id": "split-sql-dangling-paren",
        "lang": "sql",
        "kind": "formatting",
        "a": "create table logs (id);\n",
        "b": "create table logs (id\n);\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b splits off a dangling ');' onto its own line; one line is "
        "clearly better",
    },
    {
        "id": "split-html-doctype",
        "lang": "html",
        "kind": "formatting",
        "a": "<!DOCTYPE html>\n",
        "b": "<!DOCTYPE\nhtml>\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b splits the doctype across two lines; it belongs on one",
    },
    {
        "id": "split-python-expression",
        "lang": "python",
        "kind": "formatting",
        "a": "return alpha + beta\n",
        "b": "return alpha +\nbeta\n",
        "simpler": "a",
        "confidence": "high",
        "rationale": "b splits the expression after '+'; the whole expression fits "
        "comfortably on one line",
    },
]


def main() -> int:
    pairs = real_cpp_formatting_pairs() + HAND_PAIRS
    # sanity: unique ids, valid simpler field
    ids = [p["id"] for p in pairs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for p in pairs:
        assert p["simpler"] in ("a", "b"), p["id"]
    out = Path("evaluation/sortkey/ordering_pairs.json")
    out.write_text(json.dumps(pairs, indent=2) + "\n")
    print(f"wrote {len(pairs)} pairs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
