#!/bin/bash
set -eu
# Pre-download the pinned toolchain into the module cache so the first
# oracle call doesn't pay the download; a no-op once cached.
GOTOOLCHAIN=go1.18.10 go version
