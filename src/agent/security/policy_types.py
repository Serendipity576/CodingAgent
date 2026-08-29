"""Shared policy types kept separate to avoid approval-policy import cycles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.security.risk import RiskLevel


class Decision(str, Enum):
    """Authorization outcomes understood by the tool registry."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The deterministic explanation for one tool-call authorization result."""

    decision: Decision
    risk: RiskLevel
    reason: str
    policy: str
