#!/usr/bin/env python3
"""Update gallery GIFs from VHS tape files.

This script regenerates gallery GIFs when:
- The tape file has changed since the GIF was last generated
- Any file in src/ has changed since the GIF was last generated
- Any source file in the gallery subdirectory (excluding .gif/.png) has changed

It uses git to efficiently check for changes, falling back to file
modification times if there are uncommitted changes.

Additionally, it generates MP4 versions for GIFs that need them (for README embeds),
publishing them to the gh-pages branch for GitHub Pages hosting.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Gallery items that need MP4 versions generated for video embeds
# Maps gallery item name to base filename (without hash)
ITEMS_NEEDING_MP4 = {
    "enterprise-hello": "hello",
}

# GitHub Pages base URL for video assets
GITHUB_PAGES_BASE = "https://drmaciver.github.io/shrinkray/assets"


def get_content_hash(path: Path) -> str:
    """Get a short hash of a file's contents."""
    with open(path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:8]
    return content_hash


def get_gh_pages_checkout() -> Path:
    """Get or create a gh-pages checkout in .cache directory.

    Returns the path to the checkout directory.
    """
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    gh_pages_dir = cache_dir / "gh-pages"

    if gh_pages_dir.exists():
        # Update existing checkout
        subprocess.run(
            ["git", "fetch", "origin", "gh-pages"],
            cwd=gh_pages_dir,
            capture_output=True,
        )
        subprocess.run(
            ["git", "reset", "--hard", "origin/gh-pages"],
            cwd=gh_pages_dir,
            capture_output=True,
        )
    else:
        # Clone gh-pages branch into cache
        result = subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "gh-pages",
                "--single-branch",
                "--depth",
                "1",
                ".",
                str(gh_pages_dir),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Failed to clone gh-pages: {result.stderr.decode()}")
            raise RuntimeError("Failed to clone gh-pages branch")

    return gh_pages_dir


def publish_to_gh_pages(mp4_path: Path, target_filename: str) -> str:
    """Publish an MP4 file to the gh-pages branch.

    Args:
        mp4_path: Path to the MP4 file to publish
        target_filename: Filename to use on gh-pages (e.g., "hello-aef13.mp4")

    Returns:
        The GitHub Pages URL for the published video.
    """
    gh_pages_dir = get_gh_pages_checkout()
    assets_dir = gh_pages_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Remove old versions with same base name
    base_name = target_filename.rsplit("-", 1)[0]  # "hello-aef13" -> "hello"
    for old_file in assets_dir.glob(f"{base_name}-*.mp4"):
        old_file.unlink()

    # Copy the new file
    target_path = assets_dir / target_filename
    shutil.copy2(mp4_path, target_path)

    # Commit and push
    subprocess.run(
        ["git", "add", "-A"],
        cwd=gh_pages_dir,
        capture_output=True,
    )

    # Check if there are changes to commit
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=gh_pages_dir,
        capture_output=True,
    )
    if result.returncode != 0:
        # There are changes to commit
        subprocess.run(
            ["git", "commit", "-m", f"Update {target_filename}"],
            cwd=gh_pages_dir,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "push", "origin", "gh-pages"],
            cwd=gh_pages_dir,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Failed to push gh-pages: {result.stderr.decode()}")
            raise RuntimeError("Failed to push to gh-pages branch")
        print(f"Published {target_filename} to gh-pages")
    else:
        print(f"{target_filename} unchanged on gh-pages")

    return f"{GITHUB_PAGES_BASE}/{target_filename}"


def git_checkout(paths: list[str]) -> None:
    """Checkout files from git, restoring them to their committed state.

    Silently ignores paths that aren't tracked or don't exist in git.
    """
    if not paths:
        return

    # Filter to only paths that exist in git
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--"] + paths,
        capture_output=True,
    )
    # Even if some files aren't tracked, try to checkout the ones that are
    subprocess.run(
        ["git", "checkout", "--"] + paths,
        capture_output=True,
    )


def get_resettable_gallery_paths(tapes: list[Path], include_outputs: bool) -> list[str]:
    """Get paths to gallery files that should be reset to git version.

    Args:
        tapes: List of tape file paths
        include_outputs: If True, include generated outputs (gif, png).
                        If False, only include source files that may be modified
                        during demos (like hello.py being reduced by shrinkray).

    Returns all non-executable files except .tape files.
    Note: MP4 files are published to gh-pages, not stored locally.
    """
    paths = []
    output_extensions = {".gif", ".png"}

    for tape in tapes:
        gallery_dir = tape.parent
        for f in gallery_dir.iterdir():
            if not f.is_file():
                continue
            # Skip tape files (the source of truth for regeneration)
            if f.suffix.lower() == ".tape":
                continue
            # Skip executable files (test scripts)
            if os.access(f, os.X_OK):
                continue
            # Skip outputs if not including them
            if not include_outputs and f.suffix.lower() in output_extensions:
                continue
            paths.append(str(f))

    return paths


def get_git_commit_time(path: str) -> int | None:
    """Get the Unix timestamp of the last commit that touched a path.

    Returns None if the path has uncommitted changes or isn't tracked.
    """
    # Check if the path has uncommitted changes
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None  # Has uncommitted changes

    # Check if the path has staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", path],
        capture_output=True,
    )
    if result.returncode != 0:
        return None  # Has staged changes

    # Get the commit time
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    return int(result.stdout.strip())


def get_file_mtime(path: str) -> int:
    """Get the modification time of a file."""
    return int(os.path.getmtime(path))


def get_latest_src_time() -> tuple[int, bool]:
    """Get the latest modification time of any file in src/.

    Returns (timestamp, used_git) where used_git indicates whether
    git commit times were used (True) or file mtimes (False).
    """
    # Check if src/ has any uncommitted changes
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", "src/"],
        capture_output=True,
    )
    has_uncommitted = result.returncode != 0

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "src/"],
        capture_output=True,
    )
    has_staged = result.returncode != 0

    if has_uncommitted or has_staged:
        # Fall back to file modification times
        latest = 0
        for root, _, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    mtime = get_file_mtime(path)
                    latest = max(latest, mtime)
        return latest, False

    # Use git commit time for src/
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", "src/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        # No commits touching src/ yet, use file times
        latest = 0
        for root, _, files in os.walk("src"):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    mtime = get_file_mtime(path)
                    latest = max(latest, mtime)
        return latest, False

    return int(result.stdout.strip()), True


def get_gallery_dir_time(gallery_dir: Path, use_git: bool) -> int:
    """Get the latest modification time of source files in a gallery subdirectory.

    Source files are any files that are not .gif or .png (the generated outputs).
    """
    source_files = [
        f
        for f in gallery_dir.iterdir()
        if f.is_file() and f.suffix.lower() not in (".gif", ".png")
    ]

    if not source_files:
        return 0

    latest = 0
    for f in source_files:
        if use_git:
            t = get_git_commit_time(str(f))
            if t is None:
                t = get_file_mtime(str(f))
        else:
            t = get_file_mtime(str(f))
        latest = max(latest, t)

    return latest


def needs_update(tape_path: Path, gif_path: Path, src_time: int, use_git: bool) -> bool:
    """Check if a GIF needs to be regenerated."""
    if not gif_path.exists():
        return True

    # Get gif modification time
    if use_git:
        gif_time = get_git_commit_time(str(gif_path))
        if gif_time is None:
            # GIF has uncommitted changes or isn't tracked, use mtime
            gif_time = get_file_mtime(str(gif_path))
    else:
        gif_time = get_file_mtime(str(gif_path))

    # Get tape modification time
    if use_git:
        tape_time = get_git_commit_time(str(tape_path))
        if tape_time is None:
            tape_time = get_file_mtime(str(tape_path))
    else:
        tape_time = get_file_mtime(str(tape_path))

    # Get gallery directory source files time
    gallery_dir_time = get_gallery_dir_time(tape_path.parent, use_git)

    # Check if tape, src, or gallery source files are newer than gif
    return tape_time > gif_time or src_time > gif_time or gallery_dir_time > gif_time


def find_tapes() -> list[Path]:
    """Find all tape files in the gallery directory."""
    gallery = Path("gallery")
    if not gallery.exists():
        return []
    return sorted(gallery.rglob("*.tape"))


def run_vhs(tape_path: Path) -> bool:
    """Run VHS on a tape file. Returns True on success."""
    print(f"Generating: {tape_path}")

    # VHS generates output files relative to the tape's directory
    tape_dir = tape_path.parent

    result = subprocess.run(
        ["vhs", tape_path.name],
        cwd=tape_dir,
    )
    return result.returncode == 0


def convert_gif_to_mp4(gif_path: Path, mp4_path: Path) -> bool:
    """Convert a GIF to MP4 using ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg not found, skipping MP4 generation")
        return False

    print(f"Converting to MP4: {gif_path} -> {mp4_path}")

    # Ensure output directory exists
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            str(gif_path),
            "-movflags",
            "faststart",  # Enable streaming
            "-pix_fmt",
            "yuv420p",  # Compatible pixel format
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
            str(mp4_path),
        ],
        capture_output=True,
    )
    return result.returncode == 0


def generate_and_publish_mp4s(updated_tapes: list[Path]) -> dict[str, str]:
    """Generate MP4 versions and publish them to gh-pages.

    Returns dict mapping item name to GitHub Pages URL.
    Raises RuntimeError if any step fails.
    """
    published_urls: dict[str, str] = {}

    for tape in updated_tapes:
        item_name = tape.parent.name
        if item_name not in ITEMS_NEEDING_MP4:
            continue

        gif_path = tape.with_suffix(".gif")
        if not gif_path.exists():
            continue

        base_name = ITEMS_NEEDING_MP4[item_name]

        # Convert to MP4 in a temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mp4 = Path(temp_dir) / f"{base_name}.mp4"

            if not convert_gif_to_mp4(gif_path, temp_mp4):
                raise RuntimeError(f"Failed to convert {gif_path} to MP4")

            # Generate hashed filename based on content
            content_hash = get_content_hash(temp_mp4)
            hashed_filename = f"{base_name}-{content_hash}.mp4"

            # Publish to gh-pages
            url = publish_to_gh_pages(temp_mp4, hashed_filename)
            published_urls[item_name] = url

    return published_urls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update gallery GIFs from VHS tape files"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if updates are needed without running VHS",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all GIFs",
    )
    args = parser.parse_args()

    tapes = find_tapes()
    if not tapes:
        print("No tape files found in gallery/")
        return 0

    # Checkout gallery files from git before checking timestamps
    # This ensures we compare against committed versions, not local modifications
    all_resettable = get_resettable_gallery_paths(tapes, include_outputs=True)

    print("Restoring gallery files from git...")
    git_checkout(all_resettable)

    src_time, used_git = get_latest_src_time()
    if used_git:
        print("Using git commit times for change detection")
    else:
        print("Using file modification times (src/ has uncommitted changes)")

    updates_needed = []
    for tape in tapes:
        # Derive gif path from tape path
        gif_path = tape.with_suffix(".gif")

        if args.force or needs_update(tape, gif_path, src_time, used_git):
            updates_needed.append(tape)

    if not updates_needed:
        print("All gallery GIFs are up to date")
        return 0

    print(f"\nGIFs needing update: {len(updates_needed)}")
    for tape in updates_needed:
        print(f"  - {tape}")

    if args.check:
        return 1  # Indicate updates are needed

    # Run VHS on each tape
    print()
    failed = []
    successful = []
    for tape in updates_needed:
        if not run_vhs(tape):
            failed.append(tape)
        else:
            successful.append(tape)

    if failed:
        print(f"\nFailed to generate {len(failed)} GIF(s):")
        for tape in failed:
            print(f"  - {tape}")
        # Restore all files before returning (including outputs, since generation failed)
        print("Restoring gallery files from git...")
        git_checkout(all_resettable)
        return 1

    # Generate MP4 versions and publish to gh-pages
    try:
        published_urls = generate_and_publish_mp4s(successful)
        if published_urls:
            print("\nPublished video URLs:")
            for item_name, url in published_urls.items():
                print(f"  {item_name}: {url}")
    except RuntimeError as e:
        print(f"\nFailed to generate/publish MP4: {e}")
        # Restore all files before returning
        print("Restoring gallery files from git...")
        git_checkout(all_resettable)
        return 1

    # Restore source files that may have been modified during demos
    # (e.g., shrinkray reducing hello.py), but NOT outputs which we just generated
    source_paths = get_resettable_gallery_paths(tapes, include_outputs=False)
    print("Restoring gallery source files from git...")
    git_checkout(source_paths)

    print(f"\nSuccessfully updated {len(updates_needed)} GIF(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
