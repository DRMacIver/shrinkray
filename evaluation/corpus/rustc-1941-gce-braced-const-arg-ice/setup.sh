#!/bin/bash
set -eu
# The oracle invokes the pinned toolchain via `rustc +1.94.1`, which
# requires rustup. Installing an already-present toolchain is a no-op.
rustup toolchain install 1.94.1 --profile minimal
