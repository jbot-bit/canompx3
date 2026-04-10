"""Canonical lifecycle semantics for the deployable validated shelf.

`validated_setups` stores both deployable shelf rows and non-deployable
historical/research rows. Downstream production code should not infer shelf
semantics from `status='active'` alone forever; use the helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

DEPLOYMENT_SCOPE_DEPLOYABLE = "deployable"
DEPLOYMENT_SCOPE_NON_DEPLOYABLE = "non_deployable"
NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON = (
    "research-only / non-tradeable instrument (not in ACTIVE_ORB_INSTRUMENTS)"
)


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


def validated_setups_has_deployment_scope(con) -> bool:
    """Return True when the connected DB has the deployment_scope column."""
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'validated_setups'
          AND column_name = 'deployment_scope'
        LIMIT 1
        """
    ).fetchone()
    return row is not None


def deployable_validated_predicate(con, alias: str = "validated_setups") -> str:
    """Return SQL predicate for the deployable shelf on this connection.

    Older minimal test schemas may not have ``deployment_scope`` yet. In that
    case, fall back to the historical contract of ``status='active'``.
    """
    prefix = f"{alias}." if alias else ""
    status_predicate = f"LOWER({prefix}status) = 'active'"
    if not validated_setups_has_deployment_scope(con):
        return status_predicate
    return (
        f"{status_predicate} AND "
        f"LOWER(COALESCE({prefix}deployment_scope, '{DEPLOYMENT_SCOPE_DEPLOYABLE}')) = "
        f"'{DEPLOYMENT_SCOPE_DEPLOYABLE}'"
    )
