#!/usr/bin/env bash
# Build CPython 3.5.10 from source into .tool/python and unpack the
# Pygments 2.0.2 sdist into .tool/pygments (idempotent, nothing global).
#
# Python 3.5 is required: the bug is a tie-break that follows dict iteration
# order, which stopped depending on the hash seed in CPython 3.6. Neither uv
# nor python-build-standalone ship 3.5, hence the source build (~4 minutes).
# 3.5's configure predates arm64 macOS, so its `arch` check is patched to
# accept arm64, and implicit-function-declaration errors from modern clang
# are downgraded to warnings.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOOL="$HERE/.tool"
PY="$TOOL/python/bin/python3.5"
if [ -x "$PY" ] && [ -d "$TOOL/pygments/pygments" ] && \
   PYTHONPATH="$TOOL/pygments" "$PY" -c 'import pygments, sys; sys.exit(pygments.__version__ != "2.0.2")' 2>/dev/null; then
    exit 0
fi
rm -rf "$TOOL"
mkdir -p "$TOOL/src"
cd "$TOOL/src"
curl -sSL -o Python-3.5.10.tgz https://www.python.org/ftp/python/3.5.10/Python-3.5.10.tgz
tar xzf Python-3.5.10.tgz
cd Python-3.5.10
sed -i.bak 's/^    \ti386)$/    \ti386|arm64)/' configure
CFLAGS="-Wno-error=implicit-function-declaration -Wno-implicit-function-declaration" \
    ./configure --prefix="$TOOL/python" --without-ensurepip > "$TOOL/src/configure.log" 2>&1
make -j"$(getconf _NPROCESSORS_ONLN)" > "$TOOL/src/make.log" 2>&1
make install > "$TOOL/src/install.log" 2>&1
cd "$TOOL"
curl -sSL -o pygments.tar.gz "https://files.pythonhosted.org/packages/source/P/Pygments/Pygments-2.0.2.tar.gz"
tar xzf pygments.tar.gz
mv Pygments-2.0.2 pygments
rm -rf "$TOOL/src" pygments.tar.gz
