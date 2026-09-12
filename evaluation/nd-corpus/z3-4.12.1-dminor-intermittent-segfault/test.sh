#!/usr/bin/env bash
# Interesting iff z3 4.12.1 dies with SIGSEGV on the SMT-LIB file in $1
# (Z3Prover/z3#6615). Any other outcome, including unsat/sat/unknown, parse
# errors and timeouts, is not interesting.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
Z3="$HERE/.tool/bin/z3"
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

"$T" 60 "$Z3" "$1" >/dev/null 2>&1
[ $? -eq 139 ]
