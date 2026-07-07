#!/bin/bash
set -eu
DEST="$TOOLS_DIR/typescript-5.8.2"
[ -x "$DEST/node_modules/typescript/bin/tsc" ] && exit 0
rm -rf "$DEST"
mkdir -p "$DEST"
cd "$DEST"
# Pin exactly 5.8.2: the crash was introduced in 5.8 (PR #60052) and
# fixed in 5.9.3, so newer releases no longer reproduce it.
npm install --no-save --no-audit --no-fund typescript@5.8.2
