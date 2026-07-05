"""A cheap, general-purpose reflow formatter (no parsing, no subprocess).

Used by the sort key: the point is CANONICALISATION, not pretty output. Every
whitespace variant of the same structure maps to one string, so ordering test
cases by ``basic_format(x)`` is immune to cramped-vs-readable, blank lines, and
over/under-splitting -- it sidesteps the "is this split good?" question by
erasing raw whitespace entirely. The sort key then breaks ties toward the raw
form closest to its canonical (see ``reflow_sort_key`` in ``problem.py``).

It dispatches on a cheap family heuristic (no real parsing):
  * indent  (``:`` + block) - Python: keep logical newlines, normalise
      indentation to a canonical depth (immune to tabs / 2-vs-4 spaces); one
      statement per line (``;`` split).
  * tag     (``<...>``)     - HTML, XML: one tag per line, indent by nesting,
      leaf (text-only) elements kept inline.
  * brace   (default)       - C, C++, JSON, SQL, and bare expressions: collapse
      whitespace and re-layout from ``{}`` depth, ``;`` and ``#`` directives; a
      fragment with no structure flattens onto one line.

String/char literals and ``//`` and ``/* */`` comments are preserved. A small
set of unambiguous binary operators is canonicalised to ``" op "`` so operands
stay on one line (bare ``< > - * /`` are left alone: tags, templates, pointers,
unary minus, comments).
"""

from __future__ import annotations

import re


OPS = ["==", "!=", "<=", ">=", "+=", "-=", "&&", "||", "+", "="]
INDENT = "  "


def _is_word(c: str) -> bool:
    """Whether ``c`` is an identifier character (unicode-aware).

    Used to decide when whitespace between two tokens is significant (a space
    between two identifier characters must be kept; whitespace next to
    punctuation can be dropped). Accepts unicode letters/digits so arbitrary,
    non-ASCII text is not mangled by merging adjacent words.
    """
    return c.isalnum() or c == "_"

# HTML void elements have no close tag; do not increase depth for them.
VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

_TAG = re.compile(r"<(/?)([A-Za-z][\w:-]*)?", re.S)
_TAG_ANY = re.compile(r"<[A-Za-z!/?][^>]*>")


def _has_python_block(s: str) -> bool:
    """Whether ``s`` contains a Python-style block: a ``:``-terminated line
    directly followed by a strictly more-indented line.

    Requiring the body to be *more* indented than the header (rather than just
    "a colon somewhere and an indented line somewhere") is what keeps detection
    stable under reflow: brace output has uniform indentation per depth, so a
    trailing label colon (``public:``, ``case 1:``, or a bare ``:``) is never
    followed by a deeper line and stays in the brace family. This makes
    ``basic_format`` idempotent and stops C/C++ labels reading as Python.
    """
    lines = s.split("\n")
    for k, header in enumerate(lines[:-1]):
        if not header.rstrip().endswith(":"):
            continue
        header_indent = len(header) - len(header.lstrip(" \t"))
        for body in lines[k + 1 :]:
            if body.strip() == "":
                continue
            body_indent = len(body) - len(body.lstrip(" \t"))
            if body_indent > header_indent:
                return True
            break
    return False


def _finish(out: list[str]) -> str:
    lines = [ln.rstrip() for ln in "".join(out).split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines) + "\n" if lines else "\n"


# --- shared inline normaliser (never emits a newline) ----------------------


def _inline(s: str) -> str:
    """Collapse whitespace and canonicalise operator spacing on a single line.

    Used for tag text nodes and Python statements, where '//' is not a comment,
    so only string/char literals are preserved verbatim (not comments).
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
        if c.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            prev = out[-1][-1] if out and out[-1] else ""
            nxt = s[j] if j < n else ""
            if _is_word(prev) and _is_word(nxt):
                out.append(" ")
            i = j
            continue
        op = next((o for o in OPS if s.startswith(o, i)), None)
        if op is not None:
            while out and out[-1] == " ":
                out.pop()
            if out:
                out.append(" ")
            out.append(op)
            out.append(" ")
            i += len(op)
            continue
        out.append(c)
        i += 1
    return "".join(out).strip()


# --- brace family ({} / ; / #) ---------------------------------------------


def _reflow_brace(s: str) -> str:
    out: list[str] = []
    depth = 0
    line_started = False  # has real content been emitted on the current line?
    i, n = 0, len(s)

    def emit(text: str) -> None:
        # Indentation is applied lazily, when content first arrives on a line,
        # so it reflects the depth *after* any dedent from a leading '}'.
        nonlocal line_started
        if not line_started:
            if depth:
                out.append(INDENT * depth)
            line_started = True
        out.append(text)

    def newline() -> None:
        nonlocal line_started
        while out and out[-1] == " ":
            out.pop()
        out.append("\n")
        line_started = False

    def last_char() -> str:
        return out[-1][-1] if out and out[-1] else ""

    while i < n:
        c = s[i]
        if c in "\"'":
            j = i + 1
            while j < n:
                if s[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if s[j] == c:
                    j += 1
                    break
                j += 1
            emit(s[i:j])
            i = j
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            j = i
            while j < n and s[j] != "\n":
                j += 1
            emit(s[i:j])
            newline()
            i = j
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            emit(s[i:j])
            i = j
            continue
        if c.isspace():
            j = i
            while j < n and s[j].isspace():
                j += 1
            nxt = s[j] if j < n else ""
            if line_started and _is_word(last_char()) and _is_word(nxt):
                out.append(" ")
            i = j
            continue
        if c == "{":
            # empty braces stay together on one line: "{}"
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if line_started and last_char() not in ("", " "):
                out.append(" ")
            if j < n and s[j] == "}":
                emit("{}")
                i = j + 1
            else:
                emit("{")
                depth += 1
                i += 1
            newline()
            continue
        if c == "}":
            depth = max(0, depth - 1)
            if line_started:
                newline()
            emit("}")
            newline()
            i += 1
            continue
        if c == ";":
            # keep a semicolon attached to a preceding "}" on the same line
            # (e.g. a struct/class/enum declaration: "};")
            if not line_started and len(out) >= 2 and out[-1] == "\n" and out[-2] == "}":
                out.pop()
                out.append(";")
            else:
                emit(";")
            newline()
            i += 1
            continue
        if c == "#":  # preprocessor directive: one per line
            if line_started:
                newline()
            emit("#")
            i += 1
            continue
        if c == ",":
            emit(", ")
            i += 1
            continue
        op = next((o for o in OPS if s.startswith(o, i)), None)
        if op is not None:
            while out and out[-1] == " ":
                out.pop()
            if line_started:
                out.append(" ")
            emit(op)
            out.append(" ")
            i += len(op)
            continue
        emit(c)
        i += 1
    return _finish(out)


# --- tag family (<...>) ----------------------------------------------------


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
            if name and k != -1 and s[k:].lstrip()[: len(close)].lower() == close:
                ce = _tag_end(s, k)
                closeraw = re.sub(r"\s+", " ", s[k:ce]).strip()
                put(tag + _inline(s[j:k]) + closeraw)
                i = ce
                continue
            put(tag)
            depth += 1
            i = j
            continue
        j = s.find("<", i)
        j = n if j < 0 else j
        put(_inline(s[i:j]))
        i = j
    return "\n".join(lines) + "\n" if lines else "\n"


# --- indentation family (Python) -------------------------------------------


def _scan_literal(s: str, i: int, cur: list[str]) -> int:
    q = s[i]
    cur.append(q)
    i += 1
    n = len(s)
    while i < n:
        cur.append(s[i])
        if s[i] == "\\" and i + 1 < n:
            cur.append(s[i + 1])
            i += 2
            continue
        if s[i] == q:
            return i + 1
        i += 1
    return i


def _logical_lines(s: str) -> list[str]:
    """Split into logical lines; newlines inside () [] {} or strings don't split."""
    lines: list[str] = []
    cur: list[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            i = _scan_literal(s, i, cur)
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


def _split_top(s: str, sep: str) -> list[str]:
    """Split on `sep` at bracket depth 0, outside string literals."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in "\"'":
            i = _scan_literal(s, i, cur)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        if c == sep and depth == 0:
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
            if width > stack[-1]:  # inconsistent dedent; treat as a new level
                stack.append(width)
        depth = len(stack) - 1
        for stmt in _split_top(raw.strip(), ";"):
            if stmt.strip():
                out.append(INDENT * depth + _inline(stmt.strip()))
    return "\n".join(out) + "\n" if out else "\n"


# --- dispatch --------------------------------------------------------------


def detect_family(s: str) -> str:
    # Python wins even with braces (dict/set literals): a ':'-terminated line
    # directly followed by a more-indented line is a block, and indentation is
    # the structure.
    if _has_python_block(s):
        return "indent"
    if _TAG_ANY.search(s) and "{" not in s and "}" not in s:
        return "tag"
    # Default brace mode: with no {}/; it collapses whitespace and op-spaces,
    # flattening a bare fragment (e.g. `z = a +\nb`) onto one line.
    return "brace"


def basic_format(s: str) -> str:
    fam = detect_family(s)
    if fam == "tag":
        return _reflow_tag(s)
    if fam == "indent":
        return _reflow_indent(s)
    return _reflow_brace(s)


def canonical_distance(x: str, canonical: str) -> int:
    """A cheap O(n) whitespace-aware distance from `x` to its canonical form.

    ``basic_format`` only changes whitespace, so a greedy two-pointer walk that
    charges for each whitespace insertion/deletion/substitution (and, rarely, a
    non-whitespace mismatch) approximates the edit distance well enough to order
    candidates that share a canonical form by how much reformatting they need.
    """
    i = j = d = 0
    nx, nc = len(x), len(canonical)
    while i < nx and j < nc:
        cx, cc = x[i], canonical[j]
        if cx == cc:
            i += 1
            j += 1
        elif cx.isspace() and not cc.isspace():
            i += 1
            d += 1
        elif cc.isspace() and not cx.isspace():
            j += 1
            d += 1
        else:
            i += 1
            j += 1
            d += 1
    return d + (nx - i) + (nc - j)
