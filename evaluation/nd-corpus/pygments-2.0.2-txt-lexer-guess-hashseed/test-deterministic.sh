#!/usr/bin/env bash
# Deterministic variant of test.sh: PYTHONHASHSEED=1, a seed on which the
# original input is misdetected as RobotFramework every time.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/python/bin/python3.5"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

# The tie-break ignores the content, so an empty or whitespace-only file
# is misdetected just the same; require some real content so the reduction
# has a non-trivial minimum to find.
grep -q '[^[:space:]]' "$1" || exit 1

name=$(PYTHONHASHSEED=1 PYTHONPATH="$HERE/.tool/pygments" "$T" 60 "$PY" "$HERE/guess.py" "$1" 2>/dev/null) || exit 1
[ -n "$name" ] && [ "$name" != "Text only" ]
