#!/usr/bin/env python3
"""Release script for shrinkray.

Updates version to calver (YY.M.D), builds package, and publishes to PyPI.
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_calver() -> str:
    """Generate calver version string in YY.M.D format."""
    now = datetime.now()
    year = now.strftime("%y")
    month = str(now.month)  # No leading zero
    day = str(now.day)  # No leading zero
    return f"{year}.{month}.{day}"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.stdout:
            print(result.stdout)
        if check:
            sys.exit(result.returncode)

    return result


def check_git_status() -> None:
    """Ensure git working directory is clean."""
    result = run_command(["git", "status", "--porcelain"], check=False)
    if result.stdout.strip():
        print("Error: Working directory has uncommitted changes", file=sys.stderr)
        print("Please commit or stash your changes first", file=sys.stderr)
        sys.exit(1)


def update_version_in_pyproject(new_version: str) -> str:
    """Update version in pyproject.toml and return old version."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    content = pyproject_path.read_text()

    # Safety check: verify this is shrinkray's pyproject.toml
    name_pattern = r'^name = "([^"]+)"'
    name_match = re.search(name_pattern, content, flags=re.MULTILINE)
    if not name_match or name_match.group(1) != "shrinkray":
        print("Error: pyproject.toml does not belong to shrinkray", file=sys.stderr)
        print(f"Found project name: {name_match.group(1) if name_match else 'unknown'}", file=sys.stderr)
        sys.exit(1)

    # Find and replace the version line
    pattern = r'^version = "[^"]+"'
    replacement = f'version = "{new_version}"'

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        print("Error: Could not find version line in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    if count > 1:
        print("Error: Found multiple version lines in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    # Extract old version for display
    old_version_match = re.search(pattern, content, flags=re.MULTILINE)
    old_version = old_version_match.group(0).split('"')[1] if old_version_match else "unknown"

    # Write updated content
    pyproject_path.write_text(new_content)
    print(f"Updated version: {old_version} → {new_version}")
    return old_version


def create_release_commit_and_tag(version: str) -> None:
    """Commit version change and create git tag."""
    # Commit the version change
    run_command(["git", "add", "pyproject.toml"])
    run_command(["git", "commit", "-m", f"Release {version}"])
    print(f"Created commit for release {version}")

    # Create tag
    tag_name = f"v{version}"
    run_command(["git", "tag", "-a", tag_name, "-m", f"Release {version}"])
    print(f"Created tag {tag_name}")

    # Push commit and tag (set upstream if needed)
    result = run_command(["git", "push"], check=False)
    if result.returncode != 0:
        # Try setting upstream
        branch_result = run_command(["git", "branch", "--show-current"], check=True)
        branch = branch_result.stdout.strip()
        print(f"Setting upstream for branch {branch}")
        run_command(["git", "push", "--set-upstream", "origin", branch])

    run_command(["git", "push", "--tags"])
    print(f"Pushed commit and tag to GitHub")


def main() -> None:
    """Main release script."""
    # Check for --dry-run flag
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("DRY RUN MODE - no files will be modified")

    # Generate calver version
    new_version = get_calver()
    print(f"Calver version: {new_version}")

    if dry_run:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        pattern = r'^version = "[^"]+"'
        old_version_match = re.search(pattern, content, flags=re.MULTILINE)
        old_version = old_version_match.group(0).split('"')[1] if old_version_match else "unknown"
        print(f"Would update version: {old_version} → {new_version}")
        print(f"Would create commit and tag: v{new_version}")
        print("Would push to GitHub")
    else:
        # Check git status first
        check_git_status()

        # Update version in pyproject.toml
        update_version_in_pyproject(new_version)

        # Create commit and tag
        create_release_commit_and_tag(new_version)

        print("\n✓ Version updated and committed")
        print(f"✓ Tag v{new_version} created and pushed")
        print("\nNext steps:")
        print("  1. Build the package: just release-build")
        print("  2. Publish to PyPI: just release-publish")
        print("\nOr run everything at once: just release")


if __name__ == "__main__":
    main()
