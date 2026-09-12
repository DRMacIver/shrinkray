#!/usr/bin/env bash
# Interesting iff jq 1.5 running the program in $1 on the input "ab" either
# dies with the jv_print.c assertion (SIGABRT) or exits 0 with output that
# contains the signature of uninitialised memory in a string: an escaped
# NUL (\u0000) or a replacement character (\ufffd, invalid UTF-8)
# (jqlang/jq#2192: repeated string multiplication reads freed memory).
# Neither escape can come out of a well-behaved program derived from
# `.*1024*1024` by deletion, so the oracle does not depend on what the
# candidate computes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi
INPUT='"ab"'

out=$(printf '%s\n' "$INPUT" | "$T" 60 "$HERE/.tool/jq-1.5" -f "$1" 2>/dev/null)
rc=$?
if [ $rc -eq 134 ]; then
    exit 0
fi
[ $rc -eq 0 ] && printf '%s' "$out" | grep -q '\\u0000\|\\ufffd'
