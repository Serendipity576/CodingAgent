"""Filename-based protection for common local credential material."""

from __future__ import annotations

from pathlib import Path


class SensitiveDataGuard:
    """Identify high-confidence credential paths before tools can access them."""

    _PRIVATE_KEY_NAMES = frozenset({"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})
    _SENSITIVE_DIRECTORY_NAMES = frozenset({".agent", "credentials", "secrets"})
    _SENSITIVE_SUFFIXES = (".pem", ".key")

    def reason(self, path: Path) -> str | None:
        """Return why a path is protected, or ``None`` when it is not matched."""

        name = path.name.casefold()
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            return "environment files may contain credentials"
        if name in self._PRIVATE_KEY_NAMES:
            return "private-key filename"
        if name.endswith(self._SENSITIVE_SUFFIXES):
            return "private-key file extension"
        if any(part.casefold() in self._SENSITIVE_DIRECTORY_NAMES for part in path.parts):
            return "protected local state or credential directory name"
        if "credential" in name or "secret" in name:
            return "credential-like filename"
        return None
