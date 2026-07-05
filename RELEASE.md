- The final reduced file is now tidied up for readability: Shrink Ray re-indents
  and normalises whitespace for C-like, HTML/XML, and Python-style inputs, and
  prefers clearer layouts while reducing, so results are easier to read without
  being meaningfully larger.
- Added `--memory-limit` to cap the memory each interestingness-test run may use,
  so a runaway test cannot exhaust your machine's RAM. It defaults to the
  machine's physical RAM, and is not enforced on macOS (where it only warns if the
  initial test already exceeds the limit).
- If a configured formatter crashes on an input, Shrink Ray now disables it and
  keeps reducing instead of aborting.
- Fixed crashes when reducing deeply nested JSON or Python inputs.
