#!/bin/bash
set -eu
DEST="$TOOLS_DIR/terser-5.15.1"
[ -x "$DEST/node_modules/.bin/terser" ] && exit 0
rm -rf "$DEST"
mkdir -p "$DEST"
cd "$DEST"
# Pin exactly 5.15.1: the crash is fixed in later releases.
npm install --no-save --no-audit --no-fund terser@5.15.1
