- Fixed several ways a reduction could hang or never finish: certain malformed
  inputs could wedge the tree-sitter passes, a user-supplied external reducer
  could hang or be wrongly killed while Shrink Ray was answering it, and
  `--timeout 0` could keep retrying forever after a reduction had otherwise
  converged.
- The LLM no longer aborts a whole reduction when the model can't be downloaded
  or loaded (for example when you're offline) or when an input is too large for
  the model's context — it just carries on without the model.
- When the default memory limit blocks the initial interestingness test (as
  happens with sanitizer builds, which reserve huge amounts of address space),
  Shrink Ray now detects this, disables the limit for the run, and warns,
  instead of reporting the initial test as uninteresting. An explicitly set
  `--memory-limit` is still respected.
- Reducing a directory no longer stops early, so directory reductions now reach
  smaller results.
- Much faster C and C++ reduction on large or preprocessed inputs, which could
  previously stall for minutes per pass or exhaust memory; namespace removal
  also succeeds in more cases.
- Lower memory use when reducing large text inputs.
- Cleaner handling of `--memory-limit`: invalid values (non-finite or smaller
  than 1 MiB) are now rejected with a clear message instead of a traceback or a
  silently unusable limit, and the "cannot be enforced on this platform" warning
  on macOS only appears when you actually pass the flag.
- The final reduced size is now always shown in the interactive UI when a
  reduction finishes, and quitting while it is still starting up no longer exits
  with an error.
- If a tree-sitter grammar fails to load, Shrink Ray now falls back to reducing
  without its grammar-based passes instead of crashing.
