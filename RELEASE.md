- Reduction now restarts from the original input when it stops making
  progress, constrained to improve on the result found so far. This escapes
  situations where an early reduction step made a smaller final result
  unreachable, and makes results more consistent across runs.
- Fixed several bugs where reduction could stop early or get stuck without
  finding reductions it should have found:
  - Single-byte test cases are now minimized using the same ordering as the
    rest of the reduction, instead of raw byte order.
  - Reduction no longer stops before running its main passes when a single
    byte or the empty file triggers the interestingness test without being
    an improvement.
  - The empty file is now correctly preferred to whitespace-only files.
  - Format-specific passes (for example JSON) no longer crash when the test
    case stops parsing in that format mid-reduction.
- Byte lowering passes can now move bytes to preferred characters
  (whitespace, "0", "a", "z") that are larger numerically but smaller in
  the ordering used for text, so e.g. a control character can become a
  letter.
- Reduction passes that give up early when making no progress now attempt
  the same candidates at every parallelism level, making results more
  reproducible across different `--parallelism` settings.
- Reduction is significantly faster on text inputs: the internal reformatter
  used by the reduction ordering is around 10x faster on realistic files, and
  byte replacement passes apply their candidates much more cheaply.
- Interestingness test results are now cached for the whole reduction rather
  than being discarded after every successful reduction, so the restart phase
  answers replayed candidates from the cache instead of re-running the test.
- Fixed reduction hanging without making progress on OpenBSD (#56). The
  interestingness test's stdin is now the test file itself rather than data
  piped from Shrink Ray, avoiding an OpenBSD kqueue deadlock that triggered
  whenever the file was bigger than a pipe buffer and the test exited
  without reading stdin. This also stops copying the whole file to the test
  on every call, on all platforms.
- The `run.sh` reproduction script that `--history` writes now uses
  `#!/bin/sh` instead of `#!/bin/bash`, so it works on systems without
  bash (such as OpenBSD).
- Fixed two more OpenBSD problems: killing a timed-out interestingness
  test no longer fails with a permission error, and `--memory-limit` now
  caps test memory there too (OpenBSD has no address-space limit, so the
  data-segment limit is used instead).
- When a tree-sitter grammar cannot be loaded (for example because it
  has to be fetched at runtime and the download fails, or no grammar
  exists for the platform), Shrink Ray now prints a warning saying what
  went wrong and reduces without tree-sitter passes, instead of crashing
  partway through the reduction.
- Fixed a crash on macOS when shutting down an external reducer whose
  subprocess had just exited on its own (a permission error from
  signalling an already-exited process group).
