---
name: writing-a-changelog-entry
description: Write or update the RELEASE.md changelog entry that Shrink Ray requires for any change touching src/ or pyproject.toml. Use when opening or preparing a PR, adding a user-facing feature or fix, or when CI / `just check-release` reports a missing RELEASE.md.
---

# Writing a changelog entry

Shrink Ray assembles its user-facing `CHANGELOG.md` from per-change `RELEASE.md`
files, in the style of Hypothesis's release system. Every change that touches
`src/` or `pyproject.toml` must ship a `RELEASE.md` at the repository root.

## How the system works

1. You add a `RELEASE.md` at the repo root in your branch/PR.
2. CI's `changelog` job (and `just check-release` locally) fails if a
   source-affecting change has no `RELEASE.md`.
3. When the change lands on `main`, the auto-release job:
   - bumps the calver version,
   - prepends your `RELEASE.md` body to `CHANGELOG.md` under the new version's
     `## YY.M.D.N — DATE` heading,
   - deletes `RELEASE.md`,
   - commits, tags, and publishes to PyPI.

You never edit `CHANGELOG.md` by hand, and there is **no release-type flag** to
set — calver decides the version, so `RELEASE.md` is purely the entry body.

## What to write

`RELEASE.md` is Markdown. It is the body of one changelog entry. Write it for
**someone who uses the `shrinkray` command**, not for a developer of Shrink Ray.

Rules:

- **User-visible effects only.** New or changed CLI options, changed behaviour,
  changed output, bug fixes users could hit. Never mention Python modules,
  classes, functions, refactors, tests, type checking, or coverage.
- **Concise.** One or two sentences per change.
- **Bullet list** when there is more than one change; no long paragraphs.
- **Backticks** for CLI flags and literal values (e.g. `` `--memory-limit` ``).
- Purely internal changes still need a `RELEASE.md`. Its body is simply:
  `- No user-visible changes.`

## Good examples

```markdown
- Added `--memory-limit` to cap the memory each interestingness-test run may use,
  so a runaway test can't exhaust your machine's RAM.
- Fixed a crash when reducing deeply nested JSON inputs.
```

```markdown
- C and C++ reduction no longer needs `clang_delta` installed; it now uses
  built-in reduction passes.
```

## Bad examples (do not do this)

- `Refactored problem.py to extract reflow_sort_key into reformat.py.`
  (Internal; means nothing to a user.)
- `Bumped test coverage to 100% and fixed basedpyright errors.`
  (Internal; not a release note.)
- A three-paragraph essay explaining the implementation. (Too long; use a bullet
  and describe the effect, not the mechanism.)

## Checklist

- [ ] `RELEASE.md` exists at the repository root.
- [ ] Every bullet describes something a user would notice.
- [ ] No references to internal code, tests, or types.
- [ ] Flags/values are in backticks.
- [ ] `just check-release` passes.
