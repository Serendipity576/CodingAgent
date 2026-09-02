"""Bubblewrap-backed isolation for local command execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from agent.security.sensitive import SensitiveDataGuard


_RUNTIME_PATHS = ("/usr", "/lib", "/lib64", "/bin", "/sbin")
_SANDBOX_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp/home",
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
}
_PREFLIGHT_TIMEOUT_SECONDS = 5


class SandboxUnavailableError(RuntimeError):
    """Raised when the required command sandbox cannot be created safely."""


@dataclass(frozen=True, slots=True)
class SandboxInvocation:
    """A prepared sandbox command and its safe execution metadata."""

    command: tuple[str, ...]
    metadata: Mapping[str, object]


class CommandSandbox(Protocol):
    """Prepare one command for execution in a mandatory isolated environment."""

    def prepare(self, command: Sequence[str], workspace: Path) -> SandboxInvocation:
        """Return a sandbox-wrapped command or raise when isolation is unavailable."""


class BubblewrapSandbox:
    """Run commands in a Bubblewrap sandbox with no host network or home directory."""

    def __init__(self, executable: str | None = None) -> None:
        self._executable = executable
        self._verified_workspaces: set[Path] = set()

    def prepare(self, command: Sequence[str], workspace: Path) -> SandboxInvocation:
        """Verify Bubblewrap once per workspace, then wrap the requested command."""

        resolved_workspace = workspace.resolve()
        executable = self._resolve_executable()
        if resolved_workspace not in self._verified_workspaces:
            self._verify_workspace(executable, resolved_workspace)
            self._verified_workspaces.add(resolved_workspace)
        return self._build_invocation(executable, command, resolved_workspace)

    def _resolve_executable(self) -> str:
        """Locate an executable Bubblewrap binary without falling back to the host."""

        candidate = self._executable or shutil.which("bwrap")
        if candidate is None:
            raise SandboxUnavailableError("Bubblewrap executable 'bwrap' was not found")
        path = Path(candidate).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SandboxUnavailableError(f"Bubblewrap executable is unavailable: {path}")
        return str(path.resolve())

    def _verify_workspace(self, executable: str, workspace: Path) -> None:
        """Fail before user work runs when this host cannot create the sandbox."""

        preflight = self._build_invocation(executable, ("/usr/bin/true",), workspace)
        try:
            result = subprocess.run(
                preflight.command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                text=True,
                timeout=_PREFLIGHT_TIMEOUT_SECONDS,
            )
        except OSError as error:
            raise SandboxUnavailableError(f"could not start Bubblewrap: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise SandboxUnavailableError("Bubblewrap preflight timed out") from error
        if result.returncode != 0:
            detail = _failure_detail(result.stderr)
            raise SandboxUnavailableError(f"Bubblewrap preflight failed: {detail}")

    def _build_invocation(
        self,
        executable: str,
        command: Sequence[str],
        workspace: Path,
    ) -> SandboxInvocation:
        """Build a minimal filesystem and environment view for one command."""

        hidden_paths = _sensitive_workspace_paths(workspace)
        arguments = [
            executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--clearenv",
        ]
        for name, value in _SANDBOX_ENVIRONMENT.items():
            arguments.extend(("--setenv", name, value))
        for runtime_path in _RUNTIME_PATHS:
            if Path(runtime_path).exists():
                arguments.extend(("--ro-bind", runtime_path, runtime_path))
        # Home and temporary files are isolated before the real workspace is mounted.
        arguments.extend(("--tmpfs", "/tmp", "--tmpfs", "/home", "--dir", "/tmp/home"))
        for directory in _workspace_parent_directories(workspace):
            arguments.extend(("--dir", str(directory)))
        arguments.extend(("--proc", "/proc", "--dev", "/dev"))
        # Keep the original absolute path so project virtual-environment scripts work.
        arguments.extend(("--bind", str(workspace), str(workspace)))
        for path in hidden_paths:
            arguments.extend(_mask_path_arguments(path))
        arguments.extend(("--chdir", str(workspace), "--", *command))
        return SandboxInvocation(
            command=tuple(arguments),
            metadata={
                "execution_scope": "sandbox",
                "sandbox": "bubblewrap",
                "network": "disabled",
                "workspace_access": "read_write",
                "masked_path_count": len(hidden_paths),
            },
        )


def _workspace_parent_directories(workspace: Path) -> tuple[Path, ...]:
    """Create only destination parents required to mount the real workspace."""

    roots = {Path("/"), Path("/tmp"), Path("/home")}
    return tuple(
        directory
        for directory in reversed(workspace.parents)
        if directory not in roots
    )


def _sensitive_workspace_paths(workspace: Path) -> tuple[Path, ...]:
    """Find protected paths and stop walking once a protected directory appears."""

    guard = SensitiveDataGuard()
    hidden: list[Path] = []
    for current_text, directory_names, file_names in os.walk(
        workspace, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        for name in tuple(directory_names):
            candidate = current / name
            if guard.reason(candidate.relative_to(workspace)) is None:
                continue
            hidden.append(candidate)
            directory_names.remove(name)
        for name in file_names:
            candidate = current / name
            if guard.reason(candidate.relative_to(workspace)) is not None:
                hidden.append(candidate)
    return tuple(sorted(hidden, key=lambda path: (len(path.parts), str(path))))


def _mask_path_arguments(path: Path) -> tuple[str, ...]:
    """Hide directories with an empty mount and files with an empty read-only device."""

    if path.is_dir() and not path.is_symlink():
        return ("--tmpfs", str(path))
    return ("--ro-bind", "/dev/null", str(path))


def _failure_detail(stderr: str) -> str:
    """Keep Bubblewrap diagnostics concise enough for a model-visible tool error."""

    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown Bubblewrap initialization error"
