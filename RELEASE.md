- Shrink Ray now requires Python 3.14 or later.
- The interactive TUI now runs in a single process, using a subinterpreter
  for the interface instead of a separate worker process. Behaviour is
  unchanged, but `shrinkray` no longer shows up as two processes and the
  `shrinkray-worker` command no longer exists.
- When `--memory-limit` is enforced, the interestingness test is now run
  through a small `/bin/sh` wrapper that applies the limit with `ulimit`.
