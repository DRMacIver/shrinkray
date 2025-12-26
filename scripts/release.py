#!/usr/bin/env python3
"""Release script for shrinkray.

Updates version to calver (YY.M.D.N), builds package, and publishes to PyPI.
N is the release number for the day (0 for first release, 1 for second, etc.).
"""

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_current_pyproject_version() -> str | None:
    """Get the current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    content = pyproject_path.read_text()
    pattern = r'^version = "([^"]+)"'
    match = re.search(pattern, content, flags=re.MULTILINE)
    return match.group(1) if match else None


def parse_version_release_number(version: str, base_version: str) -> int | None:
    """Parse the release number from a version string.

    Returns the release number if the version matches base_version.N format,
    or 0 if it matches base_version exactly (old versioning scheme where .0 is implicit).
    Returns None if the version doesn't match the base_version date.
    """
    # Check for new format: base_version.N
    match = re.match(rf"^{re.escape(base_version)}\.(\d+)$", version)
    if match:
        return int(match.group(1))

    # Check for old format: base_version (without .N suffix)
    # PyPI treats 25.12.26 as equivalent to 25.12.26.0
    if version == base_version:
        return 0

    return None


def get_calver() -> str:
    """Generate calver version string in YY.M.D.N format.

    N is the release number for the day (0 for first, 1 for second, etc.).
    """
    now = datetime.now()
    year = now.strftime("%y")
    month = str(now.month)  # No leading zero
    day = str(now.day)  # No leading zero
    base_version = f"{year}.{month}.{day}"

    release_numbers: list[int] = []

    # Check pyproject.toml for current version (handles old versioning scheme)
    current_version = get_current_pyproject_version()
    if current_version:
        pyproject_release = parse_version_release_number(current_version, base_version)
        if pyproject_release is not None:
            release_numbers.append(pyproject_release)

    # Find existing tags for today (new format v{base}.N)
    result = run_command(["git", "tag", "-l", f"v{base_version}.*"], check=False)
    if result.returncode == 0 and result.stdout.strip():
        for tag in result.stdout.strip().split("\n"):
            # Tag format is v{year}.{month}.{day}.{release_number}
            match = re.match(rf"^v{re.escape(base_version)}\.(\d+)$", tag)
            if match:
                release_numbers.append(int(match.group(1)))

    # Also check for old format tag (v{base} without .N)
    result = run_command(["git", "tag", "-l", f"v{base_version}"], check=False)
    if result.returncode == 0 and result.stdout.strip() == f"v{base_version}":
        release_numbers.append(0)

    if not release_numbers:
        # No releases today yet, start at 0
        return f"{base_version}.0"

    # Next release number is max + 1
    next_release = max(release_numbers) + 1
    return f"{base_version}.{next_release}"


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
