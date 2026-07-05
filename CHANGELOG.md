# Changelog

This is the changelog for [Shrink Ray](https://github.com/DRMacIver/shrinkray), a
fast multi-format test-case reducer. It describes changes that are visible to
people using Shrink Ray. Versions are calendar-based (`YY.M.D.N`).

Each release is assembled from the `RELEASE.md` files that ship with the changes
going into it. See the "Changelog" section of `CLAUDE.md` if you are contributing.

## 26.7.5.0 — 2026-07-05

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

## 26.7.3.1 — 2026-07-03

- C and C++ reduction no longer needs `clang_delta` or C-Reduce installed: it now
  uses built-in reduction passes, including type replacement and namespace-qualifier
  simplification.
- Shrink Ray no longer leaves temporary candidate files behind in your working
  directory.

## 26.7.3.0 — 2026-07-03

- Added support for Python 3.14.
- Invalid command-line arguments now produce a clear error message instead of a
  traceback.
- Fixed a number of crashes and incorrect reductions across the C/C++, DIMACS/CNF,
  JSON, and expression passes.
- Fixed in-place reduction sometimes leaving the original file in the wrong state.
- Fixed the interactive UI occasionally misreporting whether a test passed or failed.
- Shrink Ray now shuts down cleanly when it is terminated.

## 26.4.14.0 — 2026-04-14

- Fixed a race condition that could make "restart from a history point" behave
  incorrectly.

## 26.4.10.0 — 2026-04-10

- Fixed a rare crash during parallel reduction.

## 26.3.17.0 — 2026-03-17

- Fixed a "Directory not empty" crash caused by leftover subprocesses when a
  reduction step was cancelled.

## 26.2.20.0 — 2026-02-20

- Updated the interactive UI to Textual 8.0.

## 26.2.4.1 — 2026-02-04

- Fixed a crash on exit and made quitting with Ctrl+Q respond immediately.

## 26.2.4.0 — 2026-02-04

- Fixed display of test-case content containing characters that look like markup.

## 26.1.1.0 — 2026-01-01

- Added a history explorer: browse the reduction's progress over time and restart
  from any earlier point.
- Added `--also-interesting` to record notable test cases that are not interesting
  in their own right.
- History recording and restart now work in directory mode.
- Reduction progress is now logged under `.shrinkray/` for later inspection.
- Numerous stability fixes for the interactive UI.

## 25.12.29.0 — 2025-12-29

- Added a size-history graph to the interactive UI.
- Added a selectable, expandable panel for inspecting reduction details and test
  output.

## 25.12.28.0 — 2025-12-28

- No user-visible changes.

## 25.12.27.3 — 2025-12-27

- The interactive UI now shows live output from your interestingness test.
- Your interestingness test is validated up front, with its output shown, so
  mistakes are caught immediately.

## 25.12.27.2 — 2025-12-27

- Improved the ordering used to decide which of two test cases is simpler, giving
  more consistent and more readable minimal results.

## 25.12.27.1 — 2025-12-27

- When `--timeout` is not given, a suitable per-test timeout is now chosen
  automatically.
- Fixed the keyboard-shortcut hints shown in the UI title bar.

## 25.12.27.0 — 2025-12-27

- No user-visible changes.

## 25.12.26.2 — 2025-12-26

- Added lower version bounds to Shrink Ray's dependencies so installation resolves
  a working set even with a minimal-version resolver.

## 25.12.26.1 — 2025-12-26

- Added a pass-statistics view to the interactive UI, and controls for enabling or
  disabling individual reduction passes.

## 25.12.26 — 2025-12-26

- Initial release of Shrink Ray, a fast multi-format test-case reducer with an
  interactive terminal UI. It reduces C/C++, Python, JSON, DIMACS/CNF, and
  arbitrary text or binary inputs, with live statistics, parallel reduction, and
  automatic formatting of the final result. Requires Python 3.12 or later.
