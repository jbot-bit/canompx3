"""Passive, non-executing TopstepX/ProjectX sidecar surfaces.

This package is intentionally read-only and policy-gated. It must never place,
cancel, or modify orders.
"""

from .data_consumer import PassiveSidecarDataConsumer, PassiveSidecarProjection
from .policy_gate import (
    PassiveSidecarPolicyError,
    assert_passive_sidecar_allowed,
    passive_sidecar_allowed,
    policy_gate_status,
)

__all__ = [
    "PassiveSidecarDataConsumer",
    "PassiveSidecarPolicyError",
    "PassiveSidecarProjection",
    "assert_passive_sidecar_allowed",
    "passive_sidecar_allowed",
    "policy_gate_status",
]
