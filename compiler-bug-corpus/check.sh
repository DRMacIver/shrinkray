#!/bin/bash
# Interestingness test for a compiler-bug corpus entry.
#
# shrink-ray invokes this with the candidate file as $1. We compile it
# with the specific old compiler (running in a persistent Docker
# container for speed) and report "interesting" (exit 0) iff the
# compiler still crashes with this bug's signature.
#
# Configuration comes from the environment, set by run.py:
#   SR_CONTAINER  name of the running compiler container
#   SR_WORKDIR    host directory bind-mounted at /w inside the container
#   SR_COMPILER   compiler executable inside the container (g++ / clang++)
#   SR_STD        -std= value
#   SR_SIGNATURE  substring that identifies this specific crash
set -u

candidate="$1"

# Each parallel test call needs its own file in the shared work dir so
# concurrent compiles don't clobber one another. mktemp with trailing
# X's is portable across GNU and BSD; the compiler is told the language
# explicitly with -x so the extension-less name is fine.
tmp=$(mktemp "$SR_WORKDIR/candXXXXXXXX")
trap 'rm -f "$tmp"' EXIT
cp "$candidate" "$tmp"
base=$(basename "$tmp")

out=$(docker exec "$SR_CONTAINER" \
    "$SR_COMPILER" -std="$SR_STD" -x c++ -c "/w/$base" -o /dev/null 2>&1)

# grep -F -q: exit 0 (interesting) iff the signature is present.
printf '%s' "$out" | grep -Fq -- "$SR_SIGNATURE"
