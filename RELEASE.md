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
- Byte lowering passes are much more thorough: they can move bytes to values
  that are larger numerically but smaller in the text ordering, lower a byte
  while raising or stripping the bytes after it (generalised carrying), and
  propose whitespace-padded layouts of the current content.
- Reduction passes that give up early when making no progress now attempt
  the same candidates at every parallelism level, and always try every
  candidate when there are few of them, making results more reproducible.
