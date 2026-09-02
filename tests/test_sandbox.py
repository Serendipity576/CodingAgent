"""Coverage for the mandatory local command sandbox."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.config import RuntimeLimits
from agent.sandbox import BubblewrapSandbox, SandboxUnavailableError
from agent.tools.base import ToolContext
from agent.tools.shell import RunCommandTool


class BubblewrapSandboxTests(unittest.TestCase):
    """Verify that default command execution remains isolated or fails closed."""

    def test_missing_bubblewrap_never_runs_the_host_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            marker = workspace / "host-command-ran.txt"
            result = RunCommandTool(
                sandbox=BubblewrapSandbox("/missing/bwrap")
            ).execute(
                {
                    "command": [
                        "/usr/bin/python3",
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                    ]
                },
                _context(workspace),
            )
            marker_exists = marker.exists()

        self.assertFalse(result.success)
        self.assertIn("default command sandbox unavailable", result.error or "")
        self.assertFalse(marker_exists)
        self.assertFalse(result.metadata["sandbox_available"])

    def test_sandbox_hides_sensitive_paths_and_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "host-secret.txt"
            outside.write_text("host secret", encoding="utf-8")
            (workspace / ".env").write_text("TOKEN=workspace secret", encoding="utf-8")
            credentials = workspace / "credentials"
            credentials.mkdir()
            (credentials / "token.txt").write_text("credential secret", encoding="utf-8")
            try:
                BubblewrapSandbox().prepare(("/usr/bin/true",), workspace)
            except SandboxUnavailableError as error:
                self.skipTest(f"Bubblewrap sandbox unavailable on this host: {error}")

            program = "\n".join(
                (
                    "from pathlib import Path",
                    "import os",
                    "import socket",
                    "print('api=' + repr(os.getenv('CODING_AGENT_API_KEY')))",
                    "for label, path in [('env', Path('.env')), ('outside', Path("
                    + repr(str(outside))
                    + ")), ('credentials', Path('credentials/token.txt'))]:",
                    "    try:",
                    "        path.read_text()",
                    "    except OSError:",
                    "        print(label + '=blocked')",
                    "    else:",
                    "        print(label + '=visible')",
                    "try:",
                    "    socket.create_connection(('1.1.1.1', 53), timeout=0.5)",
                    "except OSError:",
                    "    print('network=blocked')",
                    "else:",
                    "    print('network=visible')",
                    "Path('sandbox-output.txt').write_text('sandbox output')",
                )
            )
            with patch.dict(os.environ, {"CODING_AGENT_API_KEY": "host-api-secret"}):
                result = RunCommandTool().execute(
                    {"command": ["/usr/bin/python3", "-c", program]}, _context(workspace)
                )
            sandbox_output = (workspace / "sandbox-output.txt").read_text(encoding="utf-8")

        self.assertTrue(result.success, result.error)
        self.assertIn("api=None", result.output)
        self.assertIn("env=blocked", result.output)
        self.assertIn("outside=blocked", result.output)
        self.assertIn("credentials=blocked", result.output)
        self.assertIn("network=blocked", result.output)
        self.assertNotIn("host-api-secret", result.output)
        self.assertNotIn("workspace secret", result.output)
        self.assertEqual(sandbox_output, "sandbox output")
        self.assertEqual(result.metadata["execution_scope"], "sandbox")
        self.assertEqual(result.metadata["network"], "disabled")


def _context(workspace: Path) -> ToolContext:
    """Create a short-lived command context for sandbox tests."""

    return ToolContext(
        workspace=workspace.resolve(),
        limits=RuntimeLimits(command_timeout_seconds=10, max_task_seconds=30),
    )
