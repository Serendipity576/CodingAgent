"""Real-path workspace containment checks."""

from __future__ import annotations

from pathlib import Path


class PathAccessError(ValueError):
    """Raised when a requested path cannot safely remain in the workspace."""


class WorkspacePathGuard:
    """Resolve paths once and reject traversal, absolute escape, and symlink escape."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    @property
    def workspace(self) -> Path:
        """Return the normalized workspace root used for every check."""

        return self._workspace

    def resolve(self, requested_path: object) -> Path:
        """Return a real path only when it remains inside the workspace."""

        if not isinstance(requested_path, str) or not requested_path.strip():
            raise PathAccessError("path must be a non-empty string")

        candidate = Path(requested_path).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except RuntimeError as error:
            raise PathAccessError(f"could not resolve path: {error}") from error

        # ``resolve`` follows existing symlinks, so a link such as
        # ``workspace/link -> /outside`` cannot pass this containment check.
        try:
            resolved.relative_to(self._workspace)
        except ValueError as error:
            raise PathAccessError("path must remain inside the workspace") from error
        return resolved
