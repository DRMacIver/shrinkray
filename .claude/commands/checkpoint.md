---
description: Run lint, tests, self-review, and commit changes
---

# Checkpoint

Perform a complete quality checkpoint before committing.

## Process

1. **Lint**: Run `just lint` and fix any issues found
2. **Test**: Run relevant tests (`just test` or specific test files if changes are focused)
3. **Self-review**: Review all changes critically:
   - Is this sloppy? Did I take shortcuts?
   - Are there type errors being suppressed instead of fixed?
   - Are there tests missing for new code paths?
   - If I were reviewing someone else's PR with this code, what would I flag?
4. **Fix issues**: Address anything found in self-review
5. **Commit**: Create a small, logically self-contained commit with a clear message

## Commit Message Format

```
<type>: <short description>

<optional body explaining why, not what>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Types: fix, feat, refactor, test, docs, chore

## When to Use

Use `/checkpoint` when you've:
- Completed a logical unit of work
- Made changes you want to verify and commit
- Hit a natural stopping point

The goal is ensuring each commit is lint-clean, test-passing, and high-quality.
