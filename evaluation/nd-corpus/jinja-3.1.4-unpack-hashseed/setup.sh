#!/usr/bin/env bash
# Install Jinja2 3.1.4 into a local venv at .tool/ (idempotent).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
if [ -x "$TOOL/bin/python" ] && "$TOOL/bin/python" -c 'import jinja2, sys; sys.exit(jinja2.__version__ != "3.1.4")' 2>/dev/null; then
    exit 0
fi
rm -rf "$TOOL"
uv venv --quiet --python 3.12 "$TOOL"
uv pip install --quiet --python "$TOOL/bin/python" jinja2==3.1.4 markupsafe==3.0.2
