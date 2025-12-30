#!/bin/bash

set -eux

# Check if valid python
python -c '
import ast

with open("hello.py") as f:
    ast.parse(f.read())
'

# If it crashes libcst, that's also interesting
uv run --with 'libcst==1.8.6' python -c '
import libcst as cst

with open("hello.py") as f:
    cst.parse_module(f.read())
' || exit 101

# Run it. If it fails, uninteresting.
python hello.py > hello.log

# If ranning it said hello, interesting
grep "hello" hello.log
