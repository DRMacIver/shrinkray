- Shrink Ray now handles nondeterministic interestingness tests (tests that
  only sometimes reproduce the bug). By default it replays the initial test
  case a few times at startup, the current one occasionally, and the result at
  the end; if any replay disagrees, it switches to confirming candidates by
  repeated runs before adopting them, so that reduction cannot walk the test
  case down to something that no longer reproduces the bug, and backtracks
  through the run's history to a reproducing test case when the switch comes
  late. The TUI shows the estimated reproduction rate and the calls spent on
  replays, and the final report says how often the result reproduced.
- New `--assume-deterministic` flag skips this detection and takes every run
  of the interestingness test as a verdict.
