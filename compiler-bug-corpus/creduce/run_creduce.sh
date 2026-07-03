#!/bin/bash
# Run c-reduce on one corpus entry, using the same real-compiler oracle
# (in a persistent container) that shrink ray used. Produces
# <entry>/creduce_reduced.cpp.
#
# Usage: run_creduce.sh <entry-id>
set -eu

ENTRY="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"   # compiler-bug-corpus/creduce
CORPUS="$(cd "$HERE/.." && pwd)"        # compiler-bug-corpus
META="$CORPUS/$ENTRY/meta.json"

j() { python3 -c "import json,sys;print(json.load(open('$META'))['$1'])"; }
IMAGE=$(j image); STD=$(j std); SIG=$(j signature); COMPILER_KIND=$(j compiler)
COMPILER=g++; [ "$COMPILER_KIND" = clang ] && COMPILER=clang++

CONTAINER="cred-oracle-$ENTRY"
SHARED="$HERE/shared-$ENTRY"
rm -rf "$SHARED"; mkdir -p "$SHARED"

# Persistent compiler container, shared work dir mounted at /w.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --platform linux/amd64 --name "$CONTAINER" \
    -v "$SHARED:/w" "$IMAGE" sleep infinity >/dev/null

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# creduce work dir: the input file plus the test script.
CWD="$HERE/creduce-work-$ENTRY"
rm -rf "$CWD"; mkdir -p "$CWD"
cp "$CORPUS/$ENTRY/original.cpp" "$CWD/input.cpp"
cp "$HERE/creduce_test.sh" "$CWD/test.sh"
chmod +x "$CWD/test.sh"

# Run creduce inside the creduce-host image. It needs the docker socket
# (to exec the compiler container), the shared dir (so its candidates
# reach the compiler container), and its own work dir.
docker run --rm --platform linux/amd64 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$SHARED:$SHARED" \
    -v "$CWD:/cred" \
    -e DOCKER_API_VERSION=1.44 \
    -e SR_FILE=input.cpp \
    -e SR_SHARED="$SHARED" \
    -e SR_CONTAINER="$CONTAINER" \
    -e SR_COMPILER="$COMPILER" \
    -e SR_STD="$STD" \
    -e SR_SIGNATURE="$SIG" \
    creduce-host bash -c '
        cd /cred
        # Verify the original reproduces before reducing.
        if ! ./test.sh; then echo "ORIGINAL DOES NOT REPRODUCE" >&2; exit 3; fi
        # creduce updates input.cpp in place as it finds reductions, and
        # front-loads its high-value passes; under emulation its long
        # tail is prohibitively slow, so we cap it with a time budget
        # (default 15 min) and take whatever it reached.
        timeout "${CREDUCE_BUDGET:-900}" creduce --n 1 ./test.sh input.cpp >/dev/null 2>&1
        cat input.cpp
    ' > "$CORPUS/$ENTRY/creduce_reduced.cpp"

echo "creduce done: $ENTRY -> $(wc -c < "$CORPUS/$ENTRY/creduce_reduced.cpp") bytes"
