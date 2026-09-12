#!/usr/bin/env bash
# Download the official jq 1.5 release binary into .tool/ (idempotent).
#
# jq 1.5 was never released for macOS arm64; on Apple silicon the x86_64
# build is used and runs under Rosetta 2.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
if [ -x "$TOOL/jq-1.5" ] && [ "$("$TOOL/jq-1.5" --version 2>/dev/null)" = "jq-1.5" ]; then
    exit 0
fi
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64|Darwin-x86_64) ASSET=jq-osx-amd64 ;;
    Linux-x86_64) ASSET=jq-linux64 ;;
    *) echo "no jq 1.5 release binary for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
rm -rf "$TOOL"
mkdir -p "$TOOL"
curl -sSL -o "$TOOL/jq-1.5" "https://github.com/jqlang/jq/releases/download/jq-1.5/$ASSET"
chmod +x "$TOOL/jq-1.5"
