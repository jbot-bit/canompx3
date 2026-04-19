"""Canonical lifecycle semantics for the deployable validated shelf.

`validated_setups` stores both deployable shelf rows and non-deployable
historical/research rows. Downstream production code should not infer shelf
semantics from `status='active'` alone forever; use the helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.db_contracts import (
    ACTIVE_VALIDATED_VIEW,
    DEPLOYABLE_VALIDATED_VIEW,
    DEPLOYMENT_SCOPE_DEPLOYABLE,
    active_validated_relation,
    deployable_validated_predicate,
    deployable_validated_relation,
    validated_setups_has_deployment_scope,
)

DEPLOYMENT_SCOPE_NON_DEPLOYABLE = "non_deployable"
NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON = "research-only / non-tradeable instrument (not in ACTIVE_ORB_INSTRUMENTS)"
__all__ = [
    "ACTIVE_VALIDATED_VIEW",
    "DEPLOYABLE_VALIDATED_VIEW",
    "DEPLOYMENT_SCOPE_DEPLOYABLE",
    "DEPLOYMENT_SCOPE_NON_DEPLOYABLE",
    "NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON",
    "ShelfLifecycle",
    "active_validated_relation",
    "deployable_validated_predicate",
    "deployable_validated_relation",
    "validated_setups_has_deployment_scope",
    "validated_shelf_lifecycle",
]


@dataclass(frozen=True)
class ShelfLifecycle:
    """Lifecycle semantics for a promoted validated row."""

    status: str
    deployment_scope: str
    retirement_reason: str | None = None


def validated_shelf_lifecycle(instrument: str) -> ShelfLifecycle:
    """Return canonical lifecycle semantics for a promoted instrument."""
    if instrument in ACTIVE_ORB_INSTRUMENTS:
        return ShelfLifecycle(
            status="active",
            deployment_scope=DEPLOYMENT_SCOPE_DEPLOYABLE,
        )
    return ShelfLifecycle(
        status="retired",
        deployment_scope=DEPLOYMENT_SCOPE_NON_DEPLOYABLE,
        retirement_reason=NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON,
    )
