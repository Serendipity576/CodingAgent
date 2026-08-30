"""Read-only Git baseline capture for separating existing and Agent changes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True, slots=True)
class GitStatusSnapshot:
    """The workspace's pre-task Git status, captured without changing Git state."""

    available: bool
    entries: tuple[str, ...]
    error: str | None = None

    @classmethod
    def capture(
        cls,
        workspace: Path,
        *,
        runner: Callable[..., object] = subprocess.run,
    ) -> "GitStatusSnapshot":
        """Capture ``git status --porcelain`` with a short, read-only subprocess."""

        try:
            result = runner(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return cls.unavailable(f"could not capture Git baseline: {error}")

        if getattr(result, "returncode", 1) != 0:
            return cls.unavailable("Git status command was unavailable in the workspace")

        # Store the exact porcelain lines for auditability, but do not treat them
        # as Agent changes. Agent-owned paths come from successful patch results.
        output = str(getattr(result, "stdout", "") or "")
        return cls(available=True, entries=tuple(line for line in output.splitlines() if line))

    @classmethod
    def unavailable(cls, error: str) -> "GitStatusSnapshot":
        """Represent a non-Git workspace without making change tracking fail."""

        return cls(available=False, entries=(), error=error)
