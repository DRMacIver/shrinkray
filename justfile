# Default recipe to list available commands
default:
    @just --list

# Install dev dependencies
install:
    uv sync --extra dev

# Run tests with coverage enforcement (skips slow tests when no args given)
test *args:
    uv run coverage run -m pytest tests/ {{ if args == "" { "-m 'not slow'" } else { args } }}
    {{ if args == "" { "uv run coverage report" } else { "" } }}

# Run tests without coverage (faster for development)
test-quick *args:
    uv run python -m pytest tests/ {{ args }}

# Lint and type-check
lint:
    uv run ruff check src tests
    uv run basedpyright src tests

# Format code with ruff
format:
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
