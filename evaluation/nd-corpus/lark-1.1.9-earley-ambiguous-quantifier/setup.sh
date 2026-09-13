#!/usr/bin/env bash
# Install lark 1.1.9 into a local venv at .tool/ (idempotent).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
if [ -x "$TOOL/bin/python" ] && "$TOOL/bin/python" -c 'import lark, sys; sys.exit(lark.__version__ != "1.1.9")' 2>/dev/null; then
    exit 0
fi
rm -rf "$TOOL"
uv venv --quiet --python 3.12 "$TOOL"
uv pip install --quiet --python "$TOOL/bin/python" lark==1.1.9
