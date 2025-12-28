#!/bin/bash

set -eux

python hello.py > hello.log

grep "hello" hello.log
