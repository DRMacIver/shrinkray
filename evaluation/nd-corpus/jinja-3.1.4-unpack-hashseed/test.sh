#!/usr/bin/env bash
# Interesting iff the template in $1 compiles and the Python source Jinja
# generates for it under Python's default hash randomisation differs from
# the source generated with PYTHONHASHSEED=0 (pallets/jinja#2021: the
# order in which a tuple-unpacking `set` exports its names depends on set
# iteration order).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/bin/python"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

candidate=$(env -u PYTHONHASHSEED "$T" 30 "$PY" "$HERE/compile.py" "$1") || exit 1
reference=$(PYTHONHASHSEED=0 "$T" 30 "$PY" "$HERE/compile.py" "$1") || exit 1
[ "$candidate" != "$reference" ]
