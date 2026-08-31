from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.config import ConfigurationError, load_settings


class SettingsTests(unittest.TestCase):
    def test_explicit_values_override_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".env"
            config_file.write_text(
                "\n".join(
                    (
                        "CODING_AGENT_PROVIDER=responses",
                        "CODING_AGENT_API_KEY=file-key",
                        "CODING_AGENT_BASE_URL=https://file.example.test/v1",
                        "CODING_AGENT_MODEL=file-model",
                    )
                ),
                encoding="utf-8",
            )
            settings = load_settings(
                workspace=directory,
                model="explicit-model",
                base_url="https://explicit.example.test/v1",
                config_file=config_file,
                max_output_tokens=256,
                max_steps=8,
                environment={
                    "CODING_AGENT_WORKSPACE": "/does/not/exist",
                    "CODING_AGENT_MODEL": "environment-model",
                    "CODING_AGENT_MAX_STEPS": "4",
                    "CODING_AGENT_API_KEY": "test-key",
                },
            )

        self.assertEqual(settings.workspace, Path(directory).resolve())
        self.assertEqual(settings.model, "explicit-model")
        self.assertEqual(settings.base_url, "https://explicit.example.test/v1")
        self.assertEqual(settings.provider, "responses")
        self.assertEqual(settings.limits.max_steps, 8)
        self.assertEqual(settings.max_output_tokens, 256)
        self.assertEqual(settings.api_key, "file-key")
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

            with self.assertRaisesRegex(
                ConfigurationError,
                "max_output_tokens must be greater",
            ):
                load_settings(
                    workspace=directory,
                    max_output_tokens="0",
                    environment={},
                )

    def test_connection_configuration_comes_from_the_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".env"
            config_file.write_text(
                "\n".join(
                    (
                        "CODING_AGENT_PROVIDER=responses",
                        "CODING_AGENT_API_KEY=generic-key",
                        "CODING_AGENT_BASE_URL=https://llm.example.test/v1",
                        "CODING_AGENT_MODEL=compatible-model",
                    )
                ),
                encoding="utf-8",
            )
            settings = load_settings(
                workspace=directory,
                config_file=config_file,
                environment={},
            )

        self.assertEqual(settings.api_key, "generic-key")
        self.assertEqual(settings.model, "compatible-model")
        self.assertEqual(settings.base_url, "https://llm.example.test/v1")
        self.assertEqual(settings.provider, "responses")
        self.assertNotIn("generic-key", str(settings.public_dict()))

    def test_connection_fields_load_from_untracked_local_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".env"
            config_file.write_text(
                "\n".join(
                    (
                        "CODING_AGENT_API_KEY='file-key'",
                        "CODING_AGENT_BASE_URL=https://llm.example.test/v1",
                        "export CODING_AGENT_MODEL=file-model",
                        "CODING_AGENT_PROVIDER=deepseek",
                        "CODING_AGENT_MAX_OUTPUT_TOKENS=256",
                    )
                ),
                encoding="utf-8",
            )
            settings = load_settings(
                workspace=directory,
                config_file=config_file,
                environment={},
            )

        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.base_url, "https://llm.example.test/v1")
        self.assertEqual(settings.model, "file-model")
        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.max_output_tokens, 256)

    def test_explicit_library_values_override_local_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".env"
            config_file.write_text(
                "\n".join(
                    (
                        "CODING_AGENT_API_KEY=file-key",
                        "CODING_AGENT_BASE_URL=https://file.example.test/v1",
                        "CODING_AGENT_MODEL=file-model",
                        "CODING_AGENT_PROVIDER=responses",
                    )
                ),
                encoding="utf-8",
            )
            settings = load_settings(
                workspace=directory,
                config_file=config_file,
                model="explicit-model",
                base_url="https://explicit.example.test/v1",
                max_output_tokens=128,
                environment={},
            )

        self.assertEqual(settings.api_key, "file-key")
        self.assertEqual(settings.base_url, "https://explicit.example.test/v1")
        self.assertEqual(settings.model, "explicit-model")
        self.assertEqual(settings.provider, "responses")
        self.assertEqual(settings.max_output_tokens, 128)

    def test_provider_must_be_a_supported_explicit_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / ".env"
            config_file.write_text("CODING_AGENT_PROVIDER=unknown", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "provider must be one of"):
                load_settings(
                    workspace=directory,
                    config_file=config_file,
                    environment={},
                )

    def test_connection_fields_remain_unset_until_explicitly_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = load_settings(workspace=directory, environment={})

        self.assertIsNone(settings.api_key)
        self.assertIsNone(settings.base_url)
        self.assertIsNone(settings.model)
        self.assertIsNone(settings.provider)
