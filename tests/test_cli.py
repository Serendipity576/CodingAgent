from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from agent.cli import main


class CliTests(unittest.TestCase):
    def test_print_config_returns_sanitized_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--workspace",
                        directory,
                        "--model",
                        "test-model",
                        "--max-steps",
                        "7",
                        "--print-config",
                    ]
                )

        settings = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(settings["workspace"], str(Path(directory).resolve()))
        self.assertEqual(settings["model"], "test-model")
        self.assertEqual(settings["limits"]["max_steps"], 7)
        self.assertNotIn("api_key", settings)

    def test_default_command_prints_help(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("usage: coding-agent", output.getvalue())
