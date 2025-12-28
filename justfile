# Default recipe to list available commands
default:
    @just --list

# Install dev dependencies
install:
    uv sync --extra dev

# Run tests with coverage enforcement (skips slow tests when no args given)
test *args: install
    uv run coverage run -m pytest {{ if args == "" { "tests  -m 'not slow' --durations=10" } else { args } }}
    {{ if args == "" { "uv run coverage report" } else { "" } }}

# Run tests without coverage (faster for development)
test-quick *args: install
    uv run python -m pytest tests/ {{ args }}

# Lint and type-check
lint: install
    uv run ruff check src tests
    uv run python scripts/extra_lints.py
    uv run python scripts/check_import_cycles.py
    uv run basedpyright src tests

# Format code with ruff
format: install
    uv run ruff format src tests
    uv run ruff check --fix src tests

# Check dependencies for security issues
safety:
    uv run safety check --full-report

# Runtime type checking with typeguard
typeguard *args:
    uv run pytest --typeguard-packages=shrinkray {{ args }}

# Run doctest examples
xdoctest *args:
    uv run python -m xdoctest --modname=shrinkray --command=all {{ args }}

# Build documentation
docs-build:
    rm -rf docs/_build
    uv run sphinx-build docs docs/_build

# Serve documentation with live reloading
docs:
    rm -rf docs/_build
    uv run sphinx-autobuild --open-browser docs docs/_build

# Run the default CI checks (lint, tests)
ci: lint test

# Update version to calver (YY.M.D.N)
release-version:
    python scripts/release.py {{ if env("DRY_RUN", "") != "" { "--dry-run" } else { "" } }}

# Build package for release
release-build:
    uv build

# Publish package to PyPI
release-publish:
    uv publish {{ if env("DRY_RUN", "") != "" { "--dry-run" } else { "" } }}

# Full release: update version, build, and publish
release: release-version release-build release-publish
    @echo "Release complete!"

# Update gallery GIFs from VHS tape files (only if needed)
gallery:
    uv run python scripts/update_gallery.py

# Check if gallery GIFs need updating (returns non-zero if updates needed)
gallery-check:
    uv run python scripts/update_gallery.py --check

# Force regeneration of all gallery GIFs
gallery-force:
    uv run python scripts/update_gallery.py --force
