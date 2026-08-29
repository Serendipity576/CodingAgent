"""Risk levels used by deterministic tool authorization."""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Increasing impact of a requested local operation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
