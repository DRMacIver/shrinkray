#!/bin/bash

set -eux

python hello.py > hello.log

uv run --with 'libcst==1.8.6' python -c '
import libcst as cst

with open("hello.py") as f:
    cst.parse_module(f.read())
' || exit 101

grep "hello" hello.log
