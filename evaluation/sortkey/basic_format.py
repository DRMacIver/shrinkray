"""A very basic, cheap, language-agnostic reflow (no parsing, single pass).

Preserves string/char literals and // # /* */ comments verbatim. Outside those,
discards existing insignificant whitespace and re-emits a canonical layout from
brackets + ';' only:
  - indent by {} block depth
  - newline after '{' and ';'; '}' on its own dedented line
  - '()' and '[]' stay inline
  - one space after ',' and between two word characters; otherwise none

The point is CANONICALISATION for ordering: every whitespace variant of the
same {}/;/() structure maps to the same string, so a sort key on
basic_format(x) is immune to cramped-vs-readable, blank lines, and over/under-
splitting. Targets the brace family (C, C++, JSON, much SQL). Indentation
languages (Python) and tag languages (HTML/XML) would need their own cheap
rules; this prototype shows how far the brace family gets.
"""

from __future__ import annotations

WORD = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Binary operators to canonicalise as " op " (collapsing any surrounding
# whitespace, so operands stay on one line). Kept deliberately small and
# unambiguous: bare < > - * / are left alone (tags, templates, pointers, unary,
# comments), but their two-char comparison/assignment forms are safe. Longest
# match first.
OPS = ["==", "!=", "<=", ">=", "+=", "-=", "&&", "||", "+", "="]


def basic_format(s: str, indent: str = "  ") -> str:
    out: list[str] = []
    depth = 0
    i, n = 0, len(s)

    def at_line_start() -> bool:
        k = len(out) - 1
        while k >= 0 and out[k] in " ":
            k -= 1
        return k < 0 or out[k] == "\n"

    def newline() -> None:
        while out and out[-1] in " ":
            out.pop()
        out.append("\n")
        out.append(indent * depth)

    while i < n:
        c = s[i]
        if c in "\"'":  # literal
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
        if c == "/" and i + 1 < n and s[i + 1] == "/":  # line comment
            # NB: '#' is NOT treated as a comment -- in C it is a preprocessor
            # directive (#include<x> vs #include <x> must canonicalise equal).
            while i < n and s[i] != "\n":
                out.append(s[i])
                i += 1
            newline()
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":  # block comment
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
            if not at_line_start() and out and out[-1] not in " ":
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

    lines = [ln.rstrip() for ln in "".join(out).split("\n")]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    pairs = [
        ("int main(void){}\n", "int\nmain(void)\n{}\n"),
        ("void f() {}\n", "void f()\n\n{}"),
        ("create table logs (id);\n", "create table logs (id\n);\n"),
        ('{"a": 1, "b": 2}\n', '{\n  "a": 1,\n  "b": 2\n}\n'),
        ("z = a + b\n", "z = a +\nb\n"),
    ]
    for a, b in pairs:
        fa, fb = basic_format(a), basic_format(b)
        print(f"MATCH={fa == fb}")
        print("  a  ->", repr(fa))
        print("  b  ->", repr(fb))
    print("=== cramped real C++ ===")
    print(
        basic_format(
            "struct c{c b(int);};class a:c{c c();};"
            "c a::c(){auto a=[&](auto a){b(a);};a(0);}"
        )
    )
