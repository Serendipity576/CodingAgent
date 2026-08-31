from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_print_config_returns_sanitized_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch(
                "agent.config._load_connection_file",
                return_value={
                    "CODING_AGENT_PROVIDER": "responses",
                    "CODING_AGENT_API_KEY": "test-key",
                    "CODING_AGENT_BASE_URL": "https://llm.example.test/v1",
                    "CODING_AGENT_MODEL": "test-model",
                    "CODING_AGENT_MAX_OUTPUT_TOKENS": "128",
                },
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "--workspace",
                            directory,
                            "--max-steps",
                            "7",
                            "--print-config",
                        ]
                    )

        settings = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(settings["workspace"], str(Path(directory).resolve()))
        self.assertEqual(settings["provider"], "responses")
        self.assertEqual(settings["model"], "test-model")
        self.assertEqual(settings["max_output_tokens"], 128)
        self.assertEqual(settings["max_output_tokens"], 128)
        self.assertEqual(settings["limits"]["max_steps"], 7)
        self.assertNotIn("api_key", settings)

    def test_default_command_prints_help(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("usage: coding-agent", output.getvalue())

    def test_print_config_reads_llm_connection_values_without_cli_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch(
                "agent.config._load_connection_file",
                return_value={
                    "CODING_AGENT_PROVIDER": "deepseek",
                    "CODING_AGENT_API_KEY": "test-key",
                    "CODING_AGENT_BASE_URL": "https://llm.example.test/v1",
                    "CODING_AGENT_MODEL": "compatible-model",
                },
            ):
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "--workspace",
                            directory,
                            "--print-config",
                        ]
                    )

        settings = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(settings["provider"], "deepseek")
        self.assertEqual(settings["base_url"], "https://llm.example.test/v1")
        self.assertEqual(settings["model"], "compatible-model")
        self.assertTrue(settings["api_key_configured"])
        self.assertNotIn("test-key", output.getvalue())

    def test_cli_rejects_llm_connection_options(self) -> None:
        for option in ("--model", "--base-url", "--config", "--max-output-tokens", "--provider"):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args([option, "value"])
