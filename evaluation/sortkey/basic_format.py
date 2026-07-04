"""A cheap, general-purpose reflow formatter (no parsing, no subprocess).

The point is CANONICALISATION for a sort key: every whitespace variant of the
same structure maps to one string, so a key on basic_format(x) is immune to
cramped-vs-readable, blank lines, and over/under-splitting -- it sidesteps the
"is this split good?" question by erasing raw whitespace entirely.

It dispatches on a cheap family heuristic (no real parsing):
  * indent  (: + block) - Python: keep logical newlines, normalise indentation to
                          a canonical depth (immune to tabs / 2-vs-4 spaces); one
                          statement per line (';' split)
  * tag     (<...>)     - HTML, XML: one tag per line, indent by nesting, leaf
                          (text-only) elements kept inline
  * brace   (default)   - C, C++, JSON, SQL, bare expressions: collapse whitespace
                          and re-layout from {} depth, ';', '#' directives; a
                          fragment with no structure flattens onto one line

String/char literals and // and /* */ comments are preserved. A small set of
unambiguous binary operators is canonicalised to " op " so operands stay on one
line (bare < > - * / are left alone: tags, templates, pointers, unary, comments).
"""

from __future__ import annotations

import re

WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
OPS = ["==", "!=", "<=", ">=", "+=", "-=", "&&", "||", "+", "="]
INDENT = "  "

# HTML void elements have no close tag; do not increase depth for them.
VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


# --- shared inline normaliser (no newlines) --------------------------------


def _inline(s: str) -> str:
    """Collapse whitespace and canonicalise operator spacing on a single line.

    Preserves string/char literals and // and /* */ comments. Never emits a
    newline (callers own line structure).
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                out.append(s[i])
                if s[i] == "\\" and i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                    continue
                if s[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            out.append(s[i:])
            break
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(s[i:j])
            i = j
            continue
        if c.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            prev = out[-1][-1] if out and out[-1] else ""
            nxt = s[j] if j < n else ""
            if prev in WORD and nxt in WORD:
                out.append(" ")
            i = j
            continue
        op = next((o for o in OPS if s.startswith(o, i)), None)
        if op is not None:
            while out and out[-1] == " ":
                out.pop()
            if out and out[-1] not in ("",):
                out.append(" ")
            out.append(op)
            out.append(" ")
            i += len(op)
            continue
        out.append(c)
        i += 1
    return "".join(out).strip()


# --- brace family ({}/;) ---------------------------------------------------


def _reflow_brace(s: str) -> str:
    out: list[str] = []
    depth = 0
    i, n = 0, len(s)

    def at_line_start() -> bool:
        k = len(out) - 1
        while k >= 0 and out[k] == " ":
            k -= 1
        return k < 0 or out[k] == "\n"

    def newline() -> None:
        while out and out[-1] == " ":
            out.pop()
        out.append("\n")
        out.append(INDENT * depth)

    while i < n:
        c = s[i]
        if c in "\"'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                out.append(s[i])
                if s[i] == "\\" and i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                    continue
                if s[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                out.append(s[i])
                i += 1
            newline()
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            while i < n and not (s[i] == "*" and i + 1 < n and s[i + 1] == "/"):
                out.append(s[i])
                i += 1
            out.append("*/")
            i += 2
            continue
        if c.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            prev = out[-1] if out else ""
            nxt = s[j] if j < n else ""
            if prev in WORD and nxt in WORD:
                out.append(" ")
            i = j
            continue
        if c == "{":
            if not at_line_start() and out and out[-1] != " ":
                out.append(" ")
            out.append("{")
            depth += 1
            i += 1
            newline()
            continue
        if c == "}":
            depth = max(0, depth - 1)
            if not at_line_start():
                newline()
            out.append("}")
            i += 1
            newline()
            continue
        if c == ";":
            out.append(";")
            i += 1
            newline()
            continue
        if c == "#":  # preprocessor directive: one per line
            if not at_line_start():
                newline()
            out.append("#")
            i += 1
            continue
        if c == ",":
            out.append(", ")
            i += 1
            continue
        op = next((o for o in OPS if s.startswith(o, i)), None)
        if op is not None:
            while out and out[-1] == " ":
                out.pop()
            if out and out[-1] != "\n":
                out.append(" ")
            out.append(op)
            out.append(" ")
            i += len(op)
            continue
        out.append(c)
        i += 1
    return _finish(out)


# --- tag family (<...>) ----------------------------------------------------

_TAG = re.compile(r"<(/?)([A-Za-z][\w:-]*)?", re.S)


def _tag_end(s: str, i: int) -> int:
    """Index just past the '>' of the tag starting at s[i] == '<'."""
    n = len(s)
    j = i + 1
    while j < n and s[j] != ">":
        if s[j] in "\"'":
            q = s[j]
            j += 1
            while j < n and s[j] != q:
                j += 1
        j += 1
    return min(j + 1, n)


def _reflow_tag(s: str) -> str:
    lines: list[str] = []
    depth = 0
    i, n = 0, len(s)

    def put(text: str) -> None:
        if text:
            lines.append(INDENT * max(0, depth) + text)

    while i < n:
        if s[i].isspace():
            i += 1
            continue
        if s[i] == "<":
            if s.startswith("!--", i + 1):
                j = s.find("-->", i)
                j = n if j < 0 else j + 3
                put(re.sub(r"\s+", " ", s[i:j]).strip())
                i = j
                continue
            if i + 1 < n and s[i + 1] in "!?":  # <!DOCTYPE ...>, <?xml ...?>
                j = s.find(">", i)
                j = n if j < 0 else j + 1
                put(re.sub(r"\s+", " ", s[i:j]).strip())
                i = j
                continue
            j = _tag_end(s, i)
            raw = s[i:j]
            tag = re.sub(r"\s+", " ", raw).strip()
            m = _TAG.match(raw)
            name = (m.group(2) or "").lower() if m else ""
            closing = bool(m and m.group(1))
            self_closing = raw.rstrip().endswith("/>") or name in VOID
            if closing:
                depth = max(0, depth - 1)
                put(tag)
                i = j
                continue
            if self_closing:
                put(tag)
                i = j
                continue
            # Open tag. If the element is a leaf (its content up to the matching
            # close tag has no child tags), keep it inline: <tag>text</tag>.
            k = s.find("<", j)
            close = f"</{name}"
            if (
                name
                and k != -1
                and s[k:].lstrip()[: len(close)].lower() == close
            ):
                ce = _tag_end(s, s.find("<", j))
                closeraw = re.sub(r"\s+", " ", s[k:ce]).strip()
                put(tag + _inline(s[j:k]) + closeraw)
                i = ce
                continue
            put(tag)
            depth += 1
            i = j
            continue
        # text node up to next '<'
        j = s.find("<", i)
        j = n if j < 0 else j
        put(_inline(s[i:j]))
        i = j
    return "\n".join(lines) + "\n" if lines else "\n"


# --- indentation family (Python) -------------------------------------------


def _logical_lines(s: str) -> list[str]:
    """Split into logical lines: newlines inside () [] {} or strings don't split."""
    lines: list[str] = []
    cur: list[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            q = c
            cur.append(c)
            i += 1
            while i < n:
                cur.append(s[i])
                if s[i] == "\\" and i + 1 < n:
                    cur.append(s[i + 1])
                    i += 2
                    continue
                if s[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if c == "\n" and depth == 0:
            lines.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        lines.append("".join(cur))
    return lines


def _split_top(s: str, seps: str) -> list[str]:
    """Split on any char in `seps` at bracket depth 0, outside string literals."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            q = c
            cur.append(c)
            i += 1
            while i < n:
                cur.append(s[i])
                if s[i] == "\\" and i + 1 < n:
                    cur.append(s[i + 1])
                    i += 2
                    continue
                if s[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if c in seps and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _reflow_indent(s: str) -> str:
    out: list[str] = []
    stack = [0]  # raw-indentation widths at each depth
    for raw in _logical_lines(s):
        if not raw.strip():
            continue
        lead = len(raw) - len(raw.lstrip(" \t"))
        width = len(raw[:lead].replace("\t", "        "))
        if width > stack[-1]:
            stack.append(width)
        else:
            while len(stack) > 1 and width < stack[-1]:
                stack.pop()
            if width > stack[-1]:  # inconsistent dedent; treat as new level
                stack.append(width)
        depth = len(stack) - 1
        # ';' separates statements in Python: one per line at the same depth.
        for stmt in _split_top(raw.strip(), ";"):
            if stmt.strip():
                out.append(INDENT * depth + _inline(stmt.strip()))
    return "\n".join(out) + "\n" if out else "\n"


# --- dispatch --------------------------------------------------------------


def _finish(out: list[str]) -> str:
    lines = [ln.rstrip() for ln in "".join(out).split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines) + "\n" if lines else "\n"


def detect_family(s: str) -> str:
    brace = "{" in s or "}" in s
    # Python: a line ending in ':' followed by an indented line (a block). Wins
    # even with braces (dict/set literals), since indentation is the structure.
    py = bool(re.search(r":[ \t]*(\n|$)", s)) and bool(re.search(r"\n[ \t]+\S", s))
    tag = bool(re.search(r"<[A-Za-z!/?][^>]*>", s))
    if py:
        return "indent"
    if tag and not brace:
        return "tag"
    # Default: brace mode. With no {}/; it simply collapses whitespace and
    # op-spaces, flattening a bare expression/statement onto one line -- which is
    # what we want for reduced fragments like `z = a +\nb`.
    return "brace"


def basic_format(s: str, indent: str = "  ") -> str:
    fam = detect_family(s)
    if fam == "tag":
        return _reflow_tag(s)
    if fam == "indent":
        return _reflow_indent(s)
    return _reflow_brace(s)


if __name__ == "__main__":
    samples = [
        "int main(void){}\n",
        "int\nmain(void)\n{}\n",
        "z = a +\nb\n",
        "def f(x):\n    y = x + 1\n    return y\n",
        "def f(x):\n\ty=x+1\n\treturn y\n",
        "<ul>\n <li>a</li>\n <li>b</li>\n</ul>\n",
        "<ul><li>a</li><li>b</li></ul>\n",
        "<!DOCTYPE html>\n",
        "<!DOCTYPE\nhtml>\n",
        "SELECT a, b FROM t WHERE x = 1;\n",
    ]
    for t in samples:
        print(f"[{detect_family(t):7}] {t!r}\n     -> {basic_format(t)!r}")
