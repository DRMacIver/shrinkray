#!/usr/bin/env bash
# Interesting iff compiling $1 with the parallel front-end (-Zthreads=8)
# exits 1 reporting the E0391 "cycle detected" diagnostic
# (rust-lang/rust#142063). The same file also produces two other outcomes
# depending on thread scheduling: three E0782 errors, and an internal
# compiler error (exit 101, rust-lang/rust#142064) that prints E0391 first
# and then panics; only the clean E0391 outcome counts.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export RUSTUP_HOME="$HERE/.tool/rustup"
export RUSTC_ICE=0  # do not litter the working directory with rustc-ice-*.txt dumps
if command -v timeout >/dev/null 2>&1; then T=timeout; else T=gtimeout; fi

out=$("$T" 120 rustc +nightly-2025-06-05 --edition=2024 -Zthreads=8 --crate-type=lib \
    --out-dir "$(mktemp -d)" "$1" 2>&1)
rc=$?
[ $rc -eq 1 ] && printf '%s' "$out" | grep -q 'error\[E0391\]'
