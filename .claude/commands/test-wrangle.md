---
description: Ensure test suite quality, performance, and coverage
---

# Test Wrangling

Comprehensive test suite health check and improvement.

## Goals

### Performance Targets
- **Individual fast tests**: Each non-slow test completes in < 1 second (ideally < 500ms)
- **Total fast test time**: Under 60 seconds (ideally well under)
- **Individual slow tests**: Each slow test completes in < 30 seconds (ideally < 10 seconds)

### Quality Targets
- **Coverage**: Fast tests achieve 100% code coverage
- **Structure**: All tests are module-level functions, not class-based
- **Stability**: No unstable references (line numbers, timestamps, random values without seeds)

## Process

### 1. Performance Check

Run fast tests with timing:
```bash
uv run pytest tests/ -m "not slow" --durations=50 -q 2>&1 | tail -60
```

**Check for violations:**
- Any test > 1 second: Investigate and optimize or mark as slow
- Any test 500ms-1s: Acceptable, but consider optimization if easy
- Total time > 60 seconds: Identify optimization opportunities

**Common fixes:**
- Remove unnecessary `pilot.pause()` calls in textual async tests with update loops (pause() has a 1s timeout that triggers when messages keep arriving)
- Use `autojump_clock` fixture for trio tests with sleeps (makes trio.sleep instant)
- Reduce iteration counts where fewer would suffice
- Use mocking for slow external operations

### 2. Slow Test Check

Run slow tests with timing (sort to see slowest first):
```bash
uv run pytest tests/ -m slow --durations=100 -q 2>&1 | grep -E "^[0-9]+\.[0-9]+s" | sort -rn | head -30
```

**Check for violations:**
- Any test > 30 seconds: Must be optimized, split, or removed
- Any test > 10 seconds: Consider optimization

**Common fixes:**
- Use subset of reduction passes when full set isn't needed
- Reduce hypothesis max_examples
- Split large parametrized tests
- Add timeouts to prevent runaway tests

**When to delete instead of fix:**
If a slow test provides marginal value (e.g., running the same code path on many similar inputs), deletion may be better than optimization. Ask: "What bug would this catch that faster tests wouldn't?"

### 3. Coverage Check

Run fast tests with coverage:
```bash
uv run coverage run -m pytest tests/ -m "not slow" -q
uv run coverage report --show-missing | grep -E "^(src/|TOTAL)" | tail -25
```

**Check for violations:**
- Any file < 100%: Add tests for missing lines
- TOTAL < 100%: Investigate gaps

**If coverage gaps exist:**
1. Identify uncovered lines with `uv run coverage report --show-missing`
2. Write targeted tests for those code paths
3. Consider if code is dead and should be removed

### 4. Structure Check

Find class-based tests:
```bash
grep -rn "^class Test" tests/ || echo "No class-based tests found"
```

**If class-based tests exist:**
- Convert to module-level functions
- Move class fixtures to module-level fixtures
- Use section comments (e.g., `# === Section name ===`) to group related tests

**After converting snapshot tests:** Verify snapshot files were renamed to match new test names. Old files like `TestClassName.test_name.raw` should become `test_name.raw`.

### 5. Stability Check

Find unstable references:
```bash
# Line number references in comments
grep -rn "line [0-9]" tests/ --include="*.py" | grep -v "^Binary"

# Hardcoded line numbers in docstrings/comments
grep -rn "covers line\|tests line\|at line" tests/ --include="*.py"
```

**If unstable references exist:**
- Replace "line 123" with descriptions like "the timeout handler" or "the error path"
- Use `pytest.raises(match=...)` instead of checking line numbers in tracebacks

### 6. Cleanup Check

After deleting or modifying tests, check for orphaned code:
```bash
# Find potentially unused test fixtures
grep -rn "^def [a-z_]*(" tests/conftest.py | cut -d: -f2 | while read fn; do
  name=$(echo "$fn" | sed 's/def \([a-z_]*\).*/\1/')
  count=$(grep -r "$name" tests/ --include="*.py" | wc -l)
  if [ "$count" -lt 2 ]; then echo "Possibly unused: $name"; fi
done

# Find orphaned snapshot files (class-based names after module conversion)
ls tests/__snapshots__/*/ 2>/dev/null | grep -E "^Test[A-Z]" && echo "Found class-based snapshot files"
```

### 7. Test Quality Review

First, review any tests modified during this session:
```bash
git diff --name-only HEAD~1 | grep "^tests/test_" || echo "No test files changed"
```

Then run the sampling script to select additional tests for review:
```bash
uv run python scripts/sample-tests.py --count 10 --changed
```

Review all modified tests AND the random sample for:

**Comprehensibility:**
- [ ] Test name clearly describes what's being tested
- [ ] Test has a docstring or is self-explanatory
- [ ] Assertions are clear about what's expected

**Realistic failure modes:**
- [ ] Test could actually fail if the code is broken
- [ ] Test isn't just checking implementation details
- [ ] Test exercises meaningful behavior

**Robustness:**
- [ ] No unnecessary sleeps or timing dependencies
- [ ] No flaky assertions (ordering, floating point equality)
- [ ] Mocks are minimal and focused

**Code quality:**
- [ ] No redundant if/else branches
- [ ] No unnecessary `or` conditions that can never trigger
- [ ] No commented-out code
- [ ] No TODO comments that should be addressed

**If issues found:**
- Fix the specific test
- Consider if the pattern exists elsewhere

### 8. Final Verification

Run the full fast test suite:
```bash
just test
```

Confirm:
- All tests pass
- No new warnings
- Coverage remains at 100%

## When to Use

Run `/test-wrangle` when:
- After significant test changes
- Periodically (weekly/monthly) as maintenance
- Before major releases
- When test suite feels slow or unreliable

## Commit Guidelines

If changes are made, commit in logical chunks:
- Performance fixes in one commit
- Structural changes (class→module) in another
- Cleanup (deletions, dead code removal) separately

Use this format:
```
test: <description of improvements>

- Performance: X tests optimized
- Coverage: Added tests for Y
- Structure: Converted Z classes to module-level
- Quality: Fixed issues in N tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
