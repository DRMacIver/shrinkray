#!/bin/bash
set -eu
DEST="$TOOLS_DIR/splr-0.17.2"
[ -x "$DEST/bin/splr" ] && exit 0
# Build the pinned splr with debug assertions turned on in the release profile,
# so the internal debug_assert_ne! in the variable eliminator (which triggers
# this bug) is compiled in and aborts deterministically.
RUSTFLAGS="-C debug-assertions=on" cargo install splr --version 0.17.2 \
  --root "$DEST" --locked
