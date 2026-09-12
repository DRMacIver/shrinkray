#!/usr/bin/env bash
# Interesting iff the grammar in $1 parses input.txt with the Earley parser
# and the tree under Python's default hash randomisation differs from the
# tree produced with PYTHONHASHSEED=0 (lark-parser/lark#595: the derivation
# chosen for an ambiguous grammar depends on set iteration order).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/bin/python"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

candidate=$(env -u PYTHONHASHSEED "$T" 60 "$PY" "$HERE/parse.py" "$1") || exit 1
reference=$(PYTHONHASHSEED=0 "$T" 60 "$PY" "$HERE/parse.py" "$1") || exit 1
[ "$candidate" != "$reference" ]
