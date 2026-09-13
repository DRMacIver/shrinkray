#!/usr/bin/env bash
# Download the official z3 4.12.1 release binary into .tool/ (idempotent).
# Picks the asset for this platform; only macOS arm64/x86_64 and Linux x86_64
# are covered because those are the assets that release ships.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
if [ -x "$TOOL/bin/z3" ] && "$TOOL/bin/z3" --version 2>/dev/null | grep -q "4.12.1"; then
    exit 0
fi
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) ASSET=z3-4.12.1-arm64-osx-11.0 ;;
    Darwin-x86_64) ASSET=z3-4.12.1-x64-osx-10.16 ;;
    Linux-x86_64) ASSET=z3-4.12.1-x64-glibc-2.35 ;;
    *) echo "no z3 4.12.1 release asset for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
rm -rf "$TOOL"
mkdir -p "$TOOL"
curl -sSL -o "$TOOL/z3.zip" "https://github.com/Z3Prover/z3/releases/download/z3-4.12.1/$ASSET.zip"
unzip -q "$TOOL/z3.zip" -d "$TOOL"
mv "$TOOL/$ASSET/bin" "$TOOL/bin"
rm -rf "$TOOL/z3.zip" "$TOOL/$ASSET"
