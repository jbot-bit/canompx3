"""Hard policy gate for the passive sidecar.

The passive sidecar is blocked by default and may only be enabled after written
Topstep confirmation that passive, non-executing tooling is permitted for the
intended account stage.
"""

from __future__ import annotations

import os

_ENV_VAR = "LIVE_PASSIVE_SIDECAR_ALLOWED"
_TRUTHY = frozenset({"1", "on", "true", "yes"})


class PassiveSidecarPolicyError(RuntimeError):
    """Raised when the passive sidecar is started without explicit allowance."""


def passive_sidecar_allowed() -> bool:
    raw = os.environ.get(_ENV_VAR, "")
    return raw.strip().lower() in _TRUTHY


def policy_gate_status() -> str:
    return "allowed" if passive_sidecar_allowed() else "blocked"


def assert_passive_sidecar_allowed() -> None:
    if passive_sidecar_allowed():
        return
    raise PassiveSidecarPolicyError(
        "Passive sidecar startup blocked. Set LIVE_PASSIVE_SIDECAR_ALLOWED=true "
        "only after written Topstep confirmation that passive, non-executing "
        "tooling is permitted for the intended account stage. No API connection "
        "is allowed before that confirmation."
    )
