#!/bin/bash
set -eu
DEST="$TOOLS_DIR/prettier-3.6.2"
[ -f "$DEST/node_modules/prettier/bin/prettier.cjs" ] && exit 0
rm -rf "$DEST"
mkdir -p "$DEST"
cd "$DEST"
# Pin exactly 3.6.2: the crash was fixed by PR #17899 in a later release.
npm install --no-save --no-audit --no-fund prettier@3.6.2
