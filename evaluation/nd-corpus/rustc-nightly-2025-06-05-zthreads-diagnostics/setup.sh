#!/usr/bin/env bash
# Install rustc nightly-2025-06-05 into a private RUSTUP_HOME at .tool/rustup
# (idempotent). Uses the rustup binary already on PATH but never touches the
# user's own ~/.rustup.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
export RUSTUP_HOME="$TOOL/rustup"
TOOLCHAIN=nightly-2025-06-05
if [ -x "$RUSTUP_HOME/toolchains/$TOOLCHAIN-$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')/bin/rustc" ]; then
    exit 0
fi
mkdir -p "$RUSTUP_HOME"
rustup toolchain install "$TOOLCHAIN" --profile minimal --no-self-update
