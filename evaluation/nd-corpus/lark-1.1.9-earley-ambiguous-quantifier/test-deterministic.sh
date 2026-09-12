#!/usr/bin/env bash
# Deterministic variant of test.sh: the candidate run is pinned to
# PYTHONHASHSEED=1, which on the original grammar picks the other derivation
# than PYTHONHASHSEED=0 does. Interesting iff the two fixed-seed trees differ.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/bin/python"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

candidate=$(PYTHONHASHSEED=1 "$T" 30 "$PY" "$HERE/parse.py" "$1") || exit 1
reference=$(PYTHONHASHSEED=0 "$T" 30 "$PY" "$HERE/parse.py" "$1") || exit 1
[ "$candidate" != "$reference" ]
