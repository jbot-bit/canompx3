"""Filter eligibility context for operational transparency.

Produces an `EligibilityReport` for any deployed lane or validated strategy,
surfacing explicit statuses (PASS / FAIL / PENDING / DATA_MISSING /
NOT_APPLICABLE_* / RULES_NOT_LOADED / STALE_VALIDATION) instead of silent defaults.

Two consumer views:
- Trade sheet pre-session brief (deployed lanes from prop_profiles)
- Dashboard live signal strip (event-driven updates)

Design: docs/plans/2026-04-07-eligibility-context-design.md

DISCOVERY SAFETY: This module is READ-ONLY against canonical layers
(daily_features, validated_setups). It does not mutate any data or schema.
"""

from trading_app.eligibility.builder import build_eligibility_report
from trading_app.eligibility.types import (
    ConditionCategory,
    ConditionRecord,
    ConditionStatus,
    ConfidenceTier,
    EligibilityReport,
    FreshnessStatus,
    OverallStatus,
    ResolvesAt,
)

__all__ = [
    "build_eligibility_report",
    "ConditionCategory",
    "ConditionRecord",
    "ConditionStatus",
    "ConfidenceTier",
    "EligibilityReport",
    "FreshnessStatus",
    "OverallStatus",
    "ResolvesAt",
]
