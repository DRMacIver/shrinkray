# Default recipe to list available commands
default:
    @just --list

# Install dev dependencies
install:
    uv sync --extra dev

# Run all tests
test *args:
    uv run python -m pytest tests/ {{ args }}

# Run tests with coverage
test-cov *args:
    uv run coverage run --parallel -m pytest tests/ {{ args }}

# Produce coverage report
coverage *args='report':
    uv run coverage combine || true
    uv run coverage {{ args }}

# Lint and type-check
lint:
    uv run flake8 src tests --ignore=C901,E203,E501,E7,W503,B007,B014,B023,B904,B950
    uv run basedpyright src tests

# Format code with shed
format:
    uv run shed --refactor src/ tests/

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
