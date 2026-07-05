- Added `--reduce-with '<command>'` (repeatable) to plug in your own external
  reducer: a program that reduces the test case by talking to Shrink Ray over a
  small JSON protocol on its stdin/stdout.
- The built-in Python reducer now runs as one of these external reducers. Use
  `--no-python-reducer` to turn it off. When history is enabled, each reducer's
  log is written under the run's `.shrinkray` directory.
- Reduction can now replace an identifier with `0` (e.g. `assert x` becomes
  `assert 0`), which drops dependencies and unblocks further reduction such as
  deleting the definition the name referred to.
