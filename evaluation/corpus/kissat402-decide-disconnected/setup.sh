#!/bin/bash
set -eu
DEST="$TOOLS_DIR/kissat-4.0.2"
[ -x "$DEST/build/kissat" ] && exit 0
rm -rf "$DEST"
git clone --depth 1 --branch rel-4.0.2 https://github.com/arminbiere/kissat "$DEST"
cd "$DEST"
# Build with assertion checking enabled (-c) so the bug manifests as the
# specific decide.c assertion abort rather than a generic memory fault.
./configure -c
make -j
