#!/usr/bin/env bash
# Interesting iff reading the CSV in $1 with names=['', ''] either crashes
# the duckdb library (process killed by a signal) or returns cell values
# containing NUL bytes, i.e. uninitialised memory leaking into the result
# (duckdb/duckdb#14428). A merely wrong-but-clean answer is not counted: on
# this file that happens on every run, so it carries no nondeterminism.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.tool/bin/python"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

out=$("$T" 60 "$PY" "$HERE/query.py" "$1" 2>/dev/null)
rc=$?
if [ $rc -ge 129 ] && [ $rc -ne 124 ]; then
    exit 0  # killed by a signal (SIGSEGV = 139, SIGABRT = 134, SIGBUS = 138)
fi
[ $rc -eq 0 ] && printf '%s' "$out" | grep -q '\\x00'
