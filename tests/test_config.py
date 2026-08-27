from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.config import ConfigurationError, load_settings


class SettingsTests(unittest.TestCase):
    def test_explicit_values_override_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(
                workspace=directory,
                model="explicit-model",
                max_steps=8,
                environment={
                    "CODING_AGENT_WORKSPACE": "/does/not/exist",
                    "MODEL_NAME": "environment-model",
                    "CODING_AGENT_MAX_STEPS": "4",
                    "OPENAI_API_KEY": "test-key",
                },
            )

        self.assertEqual(settings.workspace, Path(directory).resolve())
        self.assertEqual(settings.model, "explicit-model")
        self.assertEqual(settings.limits.max_steps, 8)
        self.assertEqual(settings.api_key, "test-key")
        self.assertTrue(settings.public_dict()["api_key_configured"])
        self.assertNotIn("api_key", settings.public_dict())

    def test_workspace_must_be_an_existing_directory(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "workspace does not exist"):
            load_settings(workspace="/definitely/not/a/workspace", environment={})

    def test_limits_must_be_positive_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigurationError, "max_steps must be greater"):
                load_settings(
                    workspace=directory,
                    max_steps="0",
                    environment={},
                )

            with self.assertRaisesRegex(ConfigurationError, "max_output_chars must be an integer"):
                load_settings(
                    workspace=directory,
                    max_output_chars="large",
                    environment={},
                )
