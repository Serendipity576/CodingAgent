from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from agent.change_tracker import GitStatusSnapshot


class GitStatusSnapshotTests(unittest.TestCase):
    def test_capture_records_porcelain_entries_from_injected_runner(self) -> None:
        """Use a fake runner so the test never invokes the local Git executable."""

        calls: list[tuple[object, ...]] = []

        def runner(*args: object, **kwargs: object) -> SimpleNamespace:
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout=" M existing.py\n?? notes.md\n")

        with tempfile.TemporaryDirectory() as directory:
            snapshot = GitStatusSnapshot.capture(Path(directory), runner=runner)

        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.entries, (" M existing.py", "?? notes.md"))
        self.assertEqual(calls[0][0], ["git", "status", "--porcelain"])

    def test_capture_reports_an_unavailable_baseline_on_runner_failure(self) -> None:
        def runner(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(returncode=128, stdout="")

        with tempfile.TemporaryDirectory() as directory:
            snapshot = GitStatusSnapshot.capture(Path(directory), runner=runner)

        self.assertFalse(snapshot.available)
        self.assertEqual(snapshot.entries, ())
        self.assertIsNotNone(snapshot.error)
