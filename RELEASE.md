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
