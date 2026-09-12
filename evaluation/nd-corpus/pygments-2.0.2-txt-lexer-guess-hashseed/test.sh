#!/usr/bin/env bash
# Interesting iff Pygments 2.0.2 (on Python 3.5, hash randomisation on)
# picks a lexer other than plain text for the candidate content named
# test.txt (pygments/pygments#852). The wrong answers seen are
# RobotFramework and ResourceBundle; "Text only" is the correct one.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/python/bin/python3.5"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

# The tie-break ignores the content, so an empty or whitespace-only file
# is misdetected just the same; require some real content so the reduction
# has a non-trivial minimum to find.
grep -q '[^[:space:]]' "$1" || exit 1

name=$(env -u PYTHONHASHSEED PYTHONPATH="$HERE/.tool/pygments" "$T" 60 "$PY" "$HERE/guess.py" "$1" 2>/dev/null) || exit 1
[ -n "$name" ] && [ "$name" != "Text only" ]
