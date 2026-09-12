#!/usr/bin/env bash
# Install duckdb 1.1.2 into a local venv at .tool/ (idempotent).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
if [ -x "$TOOL/bin/python" ] && "$TOOL/bin/python" -c 'import duckdb, sys; sys.exit(duckdb.__version__ != "1.1.2")' 2>/dev/null; then
    exit 0
fi
rm -rf "$TOOL"
uv venv --quiet --python 3.12 "$TOOL"
uv pip install --quiet --python "$TOOL/bin/python" duckdb==1.1.2
