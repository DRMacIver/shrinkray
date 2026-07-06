- Fixed quitting the TUI during a reduction occasionally reporting a
  spurious error and exiting with a nonzero status.
- When `--memory-limit` is enforced, the interestingness test is now run
  through a small `/bin/sh` wrapper that applies the limit with `ulimit`.
