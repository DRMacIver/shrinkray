#!/bin/bash
# Test if the program outputs "hello world"
python hello.py 2>/dev/null | grep -q "^hello world$"