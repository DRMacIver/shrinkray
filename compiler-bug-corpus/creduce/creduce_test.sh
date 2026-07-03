#!/bin/bash
# Interestingness test for c-reduce, run inside the creduce-host
# container. c-reduce invokes it in a temp cwd containing the current
# candidate under its original basename. We copy the candidate into the
# shared work dir (visible to the compiler container) and docker-exec
# the real old compiler, reporting interesting iff the crash signature
# is still present.
#
# Env (set by the docker run in run_creduce.sh):
#   SR_FILE       candidate basename in the cwd
#   SR_SHARED     shared work dir path (same host dir mounted at /w in
#                 the compiler container)
#   SR_CONTAINER  name of the running compiler container
#   SR_COMPILER   g++ / clang++
#   SR_STD        -std= value
#   SR_SIGNATURE  crash signature substring
set -u
export DOCKER_API_VERSION=1.44

tmp=$(mktemp "$SR_SHARED/credXXXXXXXX")
trap 'rm -f "$tmp"' EXIT
cp "$SR_FILE" "$tmp"
base=$(basename "$tmp")

out=$(docker exec "$SR_CONTAINER" \
    "$SR_COMPILER" -std="$SR_STD" -x c++ -c "/w/$base" -o /dev/null 2>&1)

printf '%s' "$out" | grep -Fq -- "$SR_SIGNATURE"
