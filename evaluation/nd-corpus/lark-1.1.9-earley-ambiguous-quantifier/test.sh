#!/usr/bin/env bash
# Interesting iff the grammar in $1 parses the fixed input "a.?" and the
# resulting tree under Python's default hash randomisation differs from the
# tree produced with PYTHONHASHSEED=0 (lark-parser/lark#1434: Earley
# ambiguity resolution depends on set iteration order).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/bin/python"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

candidate=$(env -u PYTHONHASHSEED "$T" 30 "$PY" "$HERE/parse.py" "$1") || exit 1
reference=$(PYTHONHASHSEED=0 "$T" 30 "$PY" "$HERE/parse.py" "$1") || exit 1
[ "$candidate" != "$reference" ]
