"""Per-language parse + format registry for sort-key tuning.

Each language provides:
  * ``ext``          - file extension (no dot)
  * ``parse(data)``  - full-front-end validity check (bytes -> bool)
  * ``format(data)`` - canonical pretty-print (bytes -> bytes | None)

``parse`` is deliberately strict ("full front end"): it accepts a candidate
only if a real parser/compiler front end accepts it, so the gathered corpus
consists of genuinely valid programs. ``format`` returns ``None`` when the
formatter fails or refuses the input.

Run ``python evaluation/sortkey/langs.py`` for a self-test that every seed
parses and formats (and that formatting is idempotent on the seeds).
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# sqlglot logs a warning whenever it falls back to an opaque Command node; we
# already reject those in _sql_parse, so silence the noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


@dataclass(frozen=True)
class Language:
    name: str
    ext: str
    parse: Callable[[bytes], bool]
    format: Callable[[bytes], bytes | None]
    formatter: str  # human-readable description of the formatter used


# --- C / C++ ---------------------------------------------------------------


def _clang_syntax(lang_flag: str, std: str) -> Callable[[bytes], bool]:
    def parse(data: bytes) -> bool:
        try:
            proc = subprocess.run(
                ["clang", "-fsyntax-only", f"-std={std}", "-x", lang_flag, "-"],
                input=data,
                capture_output=True,
                timeout=20,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return proc.returncode == 0

    return parse


def _clang_format(data: bytes) -> bytes | None:
    try:
        proc = subprocess.run(
            ["clang-format"], input=data, capture_output=True, timeout=20
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


# --- Python ----------------------------------------------------------------


def _python_parse(data: bytes) -> bool:
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


def _python_format(data: bytes) -> bytes | None:
    import black

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        formatted = black.format_str(source, mode=black.Mode())
    except Exception:
        # black raises InvalidInput / assorted errors on code it can't handle.
        return None
    return formatted.encode("utf-8")


# --- JSON ------------------------------------------------------------------


def _json_parse(data: bytes) -> bool:
    try:
        json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return False
    return True


def _json_format(data: bytes) -> bytes | None:
    try:
        obj = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")


# --- SQL -------------------------------------------------------------------


# Real SQL is dialect-specific, so the "full front end" check is: does any real
# SQL dialect's parser fully accept this? We try these in order; the first that
# parses cleanly is also used to pretty-print (format).
SQL_DIALECTS = ["", "postgres", "mysql", "sqlite", "tsql", "duckdb", "snowflake"]


def _sql_dialect_for(source: str) -> str | None:
    """Return the first dialect that fully parses ``source``, else None."""
    import sqlglot
    from sqlglot import expressions as exp
    from sqlglot.errors import ErrorLevel, ParseError, TokenError

    for dialect in SQL_DIALECTS:
        try:
            statements = sqlglot.parse(
                source, read=dialect or None, error_level=ErrorLevel.RAISE
            )
        except (ParseError, TokenError, RecursionError):
            continue
        # A None entry is an empty statement (e.g. a stray ';'); a Command node
        # means sqlglot fell back to opaque text rather than truly parsing.
        if not statements or any(s is None for s in statements):
            continue
        if any(isinstance(s, exp.Command) for s in statements):
            continue
        return dialect
    return None


def _sql_parse(data: bytes) -> bool:
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return _sql_dialect_for(source) is not None


def _sql_format(data: bytes) -> bytes | None:
    import sqlglot
    from sqlglot.errors import ErrorLevel, ParseError, TokenError

    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    dialect = _sql_dialect_for(source)
    if dialect is None:
        return None
    try:
        parts = sqlglot.transpile(
            source, read=dialect or None, pretty=True, error_level=ErrorLevel.RAISE
        )
    except (ParseError, TokenError, RecursionError):
        return None
    if not parts:
        return None
    return (";\n".join(parts) + ";\n").encode("utf-8")


# --- XML -------------------------------------------------------------------


def _xml_parse(data: bytes) -> bool:
    from lxml import etree

    try:
        etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False))
    except etree.XMLSyntaxError:
        return False
    except ValueError:
        return False
    return True


def _xml_format(data: bytes) -> bytes | None:
    from lxml import etree

    try:
        root = etree.fromstring(
            data, parser=etree.XMLParser(resolve_entities=False, remove_blank_text=True)
        )
    except (etree.XMLSyntaxError, ValueError):
        return None
    return etree.tostring(root, pretty_print=True, encoding="utf-8")


# --- HTML ------------------------------------------------------------------


def _html_parse(data: bytes) -> bool:
    import html5lib

    parser = html5lib.HTMLParser(strict=True)
    try:
        parser.parse(data)
    except html5lib.html5parser.ParseError:
        return False
    except (ValueError, LookupError):
        # e.g. an unknown/empty encoding declaration.
        return False
    return True


def _html_format(data: bytes) -> bytes | None:
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(data, "html5lib")
    except Exception:
        return None
    return soup.prettify().encode("utf-8")


LANGUAGES: dict[str, Language] = {
    "c": Language(
        "c", "c", _clang_syntax("c", "c11"), _clang_format, "clang-format"
    ),
    "cpp": Language(
        "cpp", "cpp", _clang_syntax("c++", "c++17"), _clang_format, "clang-format"
    ),
    "python": Language(
        "python", "py", _python_parse, _python_format, "black"
    ),
    "sql": Language(
        "sql", "sql", _sql_parse, _sql_format, "sqlglot (pretty)"
    ),
    "json": Language(
        "json", "json", _json_parse, _json_format, "json.dumps(indent=2)"
    ),
    "xml": Language(
        "xml", "xml", _xml_parse, _xml_format, "lxml (pretty_print)"
    ),
    "html": Language(
        "html", "html", _html_parse, _html_format, "BeautifulSoup.prettify"
    ),
}


def seeds_dir() -> Path:
    return Path(__file__).parent / "seeds"


def load_seeds(lang: str) -> list[tuple[str, bytes]]:
    """Return (name, contents) for every seed file of a language."""
    d = seeds_dir() / lang
    if not d.exists():
        return []
    return [(p.name, p.read_bytes()) for p in sorted(d.glob(f"*.{LANGUAGES[lang].ext}"))]


def _self_test() -> int:
    failures = 0
    for name, lang in LANGUAGES.items():
        seeds = load_seeds(name)
        if not seeds:
            print(f"{name:8} no seeds yet")
            continue
        for seed_name, data in seeds:
            ok_parse = lang.parse(data)
            formatted = lang.format(data)
            ok_format = formatted is not None
            reparse = ok_format and lang.parse(formatted)
            status = "ok"
            if not (ok_parse and ok_format and reparse):
                status = "FAIL"
                failures += 1
            print(
                f"{name:8} {seed_name:24} parse={ok_parse!s:5} "
                f"format={ok_format!s:5} reparse={reparse!s:5} {status}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
