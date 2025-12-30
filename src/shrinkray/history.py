"""History management for shrink ray reduction sessions.

Records initial state and all successful reductions to a .shrinkray directory,
allowing analysis and reproduction of reduction runs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
from datetime import datetime

from attrs import define


def sanitize_for_filename(s: str) -> str:
    """Replace unsafe characters with underscores, limit length.

    Keeps alphanumeric characters, dashes, underscores, and dots.
    Collapses multiple underscores and limits to 50 characters.
    """
    # Replace unsafe characters with underscore
    safe = re.sub(r"[^\w\-.]", "_", s)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Strip leading/trailing underscores and limit length
    return safe[:50].strip("_")


@define
class HistoryManager:
    """Manages history directory structure for reduction sessions.

    Creates and maintains a directory structure like:
        .shrinkray/<run_id>/
            initial/
                <target_file>     - Original file
                <test_file>       - Copy of interestingness test (if local)
                run.sh            - Wrapper script with exact original args
            reductions/
                0001/
                    <target_file> - Reduced file
                    <target_file>.out - stdout+stderr output
                0002/
                    ...
    """

    run_id: str
    history_dir: str  # Path to .shrinkray/<run_id>
    target_basename: str
    reduction_counter: int = 0
    also_interesting_counter: int = 0
    initialized: bool = False
    record_reductions: bool = True  # If False, only record also-interesting
    is_directory: bool = False  # True if target is a directory

    @classmethod
    def create(
        cls,
        test: list[str],
        filename: str,
        *,
        record_reductions: bool = True,
        is_directory: bool = False,
    ) -> HistoryManager:
        """Create a new HistoryManager with a unique run ID.

        Args:
            test: The interestingness test command (list of strings)
            filename: Path to the target file being reduced
            record_reductions: If True, record successful reductions. If False,
                only record also-interesting cases (useful when --no-history
                but --also-interesting is explicitly passed).
            is_directory: If True, the target is a directory and test cases
                are dict[str, bytes] instead of bytes.
        """
        # Generate run ID: (test-basename)-(filename)-(datetime)-(random hex)
        test_name = sanitize_for_filename(os.path.basename(test[0]))
        file_name = sanitize_for_filename(os.path.basename(filename))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_hex = os.urandom(4).hex()
        run_id = f"{test_name}-{file_name}-{timestamp}-{random_hex}"

        history_dir = os.path.join(os.getcwd(), ".shrinkray", run_id)
        target_basename = os.path.basename(filename)

        return cls(
            run_id=run_id,
            history_dir=history_dir,
            target_basename=target_basename,
            record_reductions=record_reductions,
            is_directory=is_directory,
        )

    def initialize(
        self, initial_content: bytes, test: list[str], filename: str
    ) -> None:
        """Create initial directory structure and copy files.

        Args:
            initial_content: The original file content
            test: The interestingness test command
            filename: Path to the original target file
        """
        if self.initialized:
            return

        # Create directories
        initial_dir = os.path.join(self.history_dir, "initial")
        os.makedirs(initial_dir, exist_ok=True)
        if self.record_reductions:
            os.makedirs(os.path.join(self.history_dir, "reductions"), exist_ok=True)

        # Copy original target file
        target_path = os.path.join(initial_dir, self.target_basename)
        with open(target_path, "wb") as f:
            f.write(initial_content)

        # Copy interestingness test if it's a local file
        test_path = test[0]
        copied_test_basename: str | None = None
        if os.path.isfile(test_path):
            copied_test_basename = os.path.basename(test_path)
            shutil.copy2(test_path, os.path.join(initial_dir, copied_test_basename))

        # Create wrapper script
        self._create_wrapper_script(test, copied_test_basename, initial_dir)

        self.initialized = True

    def _create_wrapper_script(
        self,
        test: list[str],
        copied_test_basename: str | None,
        initial_dir: str,
    ) -> None:
        """Create run.sh wrapper script.

        Args:
            test: The original test command
            copied_test_basename: Basename of copied test file, or None if not copied
            initial_dir: Path to the initial directory
        """
        script_path = os.path.join(initial_dir, "run.sh")

        # Build command, referencing test file via $(dirname "$0") if it was copied
        if copied_test_basename is not None:
            test_ref = f'"$(dirname "$0")/{copied_test_basename}"'
        else:
            test_ref = shlex.quote(test[0])

        # Quote remaining arguments
        args = " ".join(shlex.quote(arg) for arg in test[1:])
        if args:
            command = f"{test_ref} {args}"
        else:
            command = test_ref

        script_content = f'''#!/bin/bash
# Shrink Ray interestingness test wrapper
# Run with: ./run.sh [target_file]
# If no target_file specified, uses the original

DIR="$(dirname "$0")"
TARGET="${{1:-"$DIR/{self.target_basename}"}}"

{command} "$TARGET"
'''
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

    def record_reduction(
        self, test_case: bytes, output: bytes | None = None
    ) -> None:
        """Record a successful reduction.

        Args:
            test_case: The reduced file content (or serialized directory content)
            output: Combined stdout/stderr from the test, or None if not captured
        """
        if not self.record_reductions:
            return
        self.reduction_counter += 1
        subdir = os.path.join(
            self.history_dir,
            "reductions",
            f"{self.reduction_counter:04d}",
        )
        os.makedirs(subdir, exist_ok=True)

        if self.is_directory:
            # Deserialize and write directory structure
            content = self._deserialize_directory(test_case)
            target_dir = os.path.join(subdir, self.target_basename)
            self._write_directory_content(target_dir, content)
        else:
            # Write reduced file
            with open(os.path.join(subdir, self.target_basename), "wb") as f:
                f.write(test_case)

        # Write output if available
        if output is not None:
            output_name = f"{self.target_basename}.out"
            with open(os.path.join(subdir, output_name), "wb") as f:
                f.write(output)

    def record_also_interesting(
        self, test_case: bytes, output: bytes | None = None
    ) -> None:
        """Record an also-interesting test case.

        These are test cases that don't satisfy the main interestingness test
        but have some other interesting property indicated by a special exit code.

        Args:
            test_case: The file content (or serialized directory content)
            output: Combined stdout/stderr from the test, or None if not captured
        """
        self.also_interesting_counter += 1
        subdir = os.path.join(
            self.history_dir,
            "also-interesting",
            f"{self.also_interesting_counter:04d}",
        )
        os.makedirs(subdir, exist_ok=True)

        if self.is_directory:
            # Deserialize and write directory structure
            content = self._deserialize_directory(test_case)
            target_dir = os.path.join(subdir, self.target_basename)
            self._write_directory_content(target_dir, content)
        else:
            # Write file
            with open(os.path.join(subdir, self.target_basename), "wb") as f:
                f.write(test_case)

        # Write output if available
        if output is not None:
            output_name = f"{self.target_basename}.out"
            with open(os.path.join(subdir, output_name), "wb") as f:
                f.write(output)

    def get_reduction_content(self, reduction_number: int) -> bytes:
        """Get the content of a specific reduction.

        Args:
            reduction_number: The reduction number (e.g., 3 for 0003)

        Returns:
            The file content as bytes. For directory mode, returns the serialized
            directory content for use with _set_initial_for_restart.

        Raises:
            FileNotFoundError: If the reduction doesn't exist
        """
        target_path = os.path.join(
            self.history_dir,
            "reductions",
            f"{reduction_number:04d}",
            self.target_basename,
        )
        if self.is_directory:
            # For directory mode, read and serialize the directory content
            content = self._read_directory_content(target_path)
            return self._serialize_directory(content)
        else:
            with open(target_path, "rb") as f:
                return f.read()

    def restart_from_reduction(
        self, reduction_number: int
    ) -> tuple[bytes, set[bytes]]:
        """Move reductions after reduction_number to also-interesting.

        This is used for the "restart from this point" feature. All reductions
        after the specified point are moved to also-interesting (so they won't
        be lost) and their contents are returned for exclusion from future
        interestingness tests.

        Args:
            reduction_number: The reduction to restart from (e.g., 3 for 0003)

        Returns:
            tuple: (content_of_reduction_N, set_of_excluded_test_cases)

        Raises:
            FileNotFoundError: If the reduction doesn't exist
        """
        excluded_test_cases: set[bytes] = set()
        reductions_dir = os.path.join(self.history_dir, "reductions")
        also_interesting_dir = os.path.join(self.history_dir, "also-interesting")
        os.makedirs(also_interesting_dir, exist_ok=True)

        # Find all reductions after the restart point
        entries_to_move: list[tuple[int, str]] = []
        for entry_name in os.listdir(reductions_dir):
            try:
                entry_num = int(entry_name)
            except ValueError:
                continue  # Skip non-numeric entries
            if entry_num > reduction_number:
                entries_to_move.append((entry_num, entry_name))

        # Sort to process in order
        entries_to_move.sort()

        for _, entry_name in entries_to_move:
            entry_path = os.path.join(reductions_dir, entry_name)
            target_path = os.path.join(entry_path, self.target_basename)

            # Read content for exclusion set
            if self.is_directory:
                # For directories, read and serialize content
                content = self._read_directory_content(target_path)
                excluded_test_cases.add(self._serialize_directory(content))
            else:
                with open(target_path, "rb") as f:
                    excluded_test_cases.add(f.read())

            # Move to also-interesting with new numbering
            self.also_interesting_counter += 1
            new_path = os.path.join(
                also_interesting_dir,
                f"{self.also_interesting_counter:04d}",
            )
            shutil.move(entry_path, new_path)

        # Update reduction counter
        self.reduction_counter = reduction_number

        # Return content to restart from
        restart_content = self.get_reduction_content(reduction_number)
        return restart_content, excluded_test_cases

    # === Directory mode methods ===

    def initialize_directory(
        self, initial_content: dict[str, bytes], test: list[str], filename: str
    ) -> None:
        """Create initial directory structure for directory test cases.

        Args:
            initial_content: The original directory content as {path: bytes}
            test: The interestingness test command
            filename: Path to the original target directory
        """
        if self.initialized:
            return

        # Create directories
        initial_dir = os.path.join(self.history_dir, "initial")
        os.makedirs(initial_dir, exist_ok=True)
        if self.record_reductions:
            os.makedirs(os.path.join(self.history_dir, "reductions"), exist_ok=True)

        # Copy original target directory structure
        target_dir = os.path.join(initial_dir, self.target_basename)
        self._write_directory_content(target_dir, initial_content)

        # Copy interestingness test if it's a local file
        test_path = test[0]
        copied_test_basename: str | None = None
        if os.path.isfile(test_path):
            copied_test_basename = os.path.basename(test_path)
            shutil.copy2(test_path, os.path.join(initial_dir, copied_test_basename))

        # Create wrapper script
        self._create_wrapper_script(test, copied_test_basename, initial_dir)

        self.initialized = True

    def _write_directory_content(
        self, target_dir: str, content: dict[str, bytes]
    ) -> None:
        """Write directory content to disk.

        Args:
            target_dir: The directory to write to
            content: The directory content as {relative_path: bytes}
        """
        os.makedirs(target_dir, exist_ok=True)
        for rel_path, file_content in content.items():
            file_path = os.path.join(target_dir, rel_path)
            # parent_dir is always non-empty since file_path includes target_dir
            parent_dir = os.path.dirname(file_path)
            os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file_content)

    def _read_directory_content(self, target_dir: str) -> dict[str, bytes]:
        """Read directory content from disk.

        Args:
            target_dir: The directory to read from

        Returns:
            The directory content as {relative_path: bytes}
        """
        content: dict[str, bytes] = {}
        for root, _, files in os.walk(target_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, target_dir)
                with open(file_path, "rb") as f:
                    content[rel_path] = f.read()
        return content

    @staticmethod
    def _deserialize_directory(data: bytes) -> dict[str, bytes]:
        """Deserialize bytes back to directory content.

        Args:
            data: JSON-encoded directory content with base64 file contents

        Returns:
            The directory content as {relative_path: bytes}
        """
        serialized = json.loads(data.decode())
        return {k: base64.b64decode(v) for k, v in serialized.items()}

    @staticmethod
    def _serialize_directory(content: dict[str, bytes]) -> bytes:
        """Serialize directory content to bytes.

        Args:
            content: The directory content as {relative_path: bytes}

        Returns:
            JSON-encoded directory content with base64 file contents
        """
        serialized = {
            k: base64.b64encode(v).decode() for k, v in sorted(content.items())
        }
        return json.dumps(serialized, sort_keys=True).encode()
