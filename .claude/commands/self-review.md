---
description: Review code changes for quality issues before human review
---

Perform a thorough self-review of the code changes on this branch compared to main. Act as if you were reviewing a PR from a junior developer who needs mentoring.

## Review Process

1. **Get the diff**: Run `git diff main` to see all changes

2. **Check for code smells**:
   - Brittle code that could fail in edge cases
   - Missing error handling or swallowed exceptions
   - Code duplication that should be extracted
   - Magic numbers that should be constants
   - Imports inside functions (should be at module level)
   - Mutable default arguments
   - Shortcuts or workarounds instead of proper fixes
   - Backward compatibility shims or re-exports (shrinkray is an app, not a library)
   - Dead code (unused functions, imports, variables)

3. **Evaluate test quality**:
   - Are the tests actually testing what they claim?
   - Are edge cases covered?
   - Is there code that's only tested indirectly when it should have direct tests?
   - Do tests rely on timing/delays instead of checking actual state?

4. **Check for suppressions** (these are code smells per CLAUDE.md):
   - `# type: ignore` - fix the type error properly
   - `# pragma: no cover` - write a test for this code
   - `# noqa` - fix the lint error

5. **Look for generalizable patterns**:
   - If you find a bug pattern that could recur, consider adding a lint rule to `scripts/extra_lints.py`

## Actions

For each issue found:
1. Create a todo list item describing the fix
2. Implement the fix
3. Verify with `just lint` and `just test`

## Completion

Repeat the review process until you are satisfied that a pedantic human reviewer would not find additional issues. Then commit your improvements with a clear commit message explaining what was fixed and why.
