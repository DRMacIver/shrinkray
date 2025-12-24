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
uv run pytest tests/ -m "not slow" --durations=0 -q
```

**Check for violations:**
- Any test > 1 second: Investigate and optimize or mark as slow
- Any test > 500ms: Consider if it can be optimized
- Total time > 60 seconds: Identify optimization opportunities

**Common fixes:**
- Remove unnecessary `pilot.pause()` calls in async tests with update loops
- Use `autojump_clock` fixture for trio tests with sleeps
- Reduce iteration counts where fewer would suffice
- Use mocking for slow external operations

### 2. Slow Test Check

Run slow tests with timing:
```bash
uv run pytest tests/ -m slow --durations=0 -q
```

**Check for violations:**
- Any test > 30 seconds: Must be split or optimized
- Any test > 10 seconds: Consider optimization

**Common fixes:**
- Use subset of reduction passes when full set isn't needed
- Reduce hypothesis max_examples
- Split large parametrized tests

### 3. Coverage Check

Run fast tests with coverage:
```bash
uv run coverage run -m pytest tests/ -m "not slow" -q
uv run coverage report --show-missing | grep -E "^(src/|TOTAL)" | tail -20
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

### 5. Stability Check

Find unstable references:
```bash
# Line number references
grep -rn "line [0-9]" tests/ --include="*.py" | grep -v "^Binary"

# Check for hardcoded line numbers in docstrings/comments
grep -rn "covers line\|tests line\|at line" tests/ --include="*.py"
```

**If unstable references exist:**
- Replace line number references with behavior descriptions
- Use `pytest.raises(match=...)` instead of checking line numbers in tracebacks

### 6. Test Quality Review

First, review any tests modified during this session. Use git to identify them:
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

### 7. Final Verification

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

If changes are made, commit with:
```
test: <description of improvements>

- Performance: X tests optimized
- Coverage: Added tests for Y
- Structure: Converted Z classes to module-level
- Quality: Fixed issues in N tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
