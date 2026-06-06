"""Published cross-layer database contracts.

These names and SQL helpers define stable read surfaces that both `pipeline/`
and `trading_app/` may depend on without breaking the one-way dependency rule.
"""

from __future__ import annotations

ACTIVE_VALIDATED_VIEW = "active_validated_setups"
DEPLOYABLE_VALIDATED_VIEW = "deployable_validated_setups"
DEPLOYMENT_SCOPE_DEPLOYABLE = "deployable"

# ── paper_trades execution-source contract ───────────────────────────────────
# `paper_trades` is a shared table discriminated by `execution_source`. Live /
# monitoring consumers must NOT see forward-monitoring `'shadow'` rows (REGIME
# would-have trades, never taken). Rather than rely on every reader REMEMBERING
# to add a predicate (a vigilance contract that leaks by construction — proven by
# the 6→7 unguarded-reader drift found in the Stage-1 adversarial review), the
# canonical read surface is the VIEW below. It is the single source of truth for
# "which execution_source values are real trades", reused by:
#   - trading_app.db_manager.init_trading_app_schema (creates the VIEW)
#   - scripts.tools.regime_shadow_runner.SHADOW_SOURCE (the excluded value)
#   - pipeline.check_drift.check_paper_trades_reads_are_shadow_safe (static guard)
# A positive allowlist (IN (...)) is fail-CLOSED: a future new source value is
# excluded from the live view until explicitly added here, never silently counted
# as a real trade.
PAPER_TRADES_SHADOW_SOURCE = "shadow"
LIVE_PAPER_TRADES_SOURCES = ("live", "backfill")
LIVE_PAPER_TRADES_VIEW = "live_paper_trades"


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


def _relation_exists(con, relation_name: str) -> bool:
    """Return True when a table/view exists in the main schema."""
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        LIMIT 1
        """,
        [relation_name],
    ).fetchone()
    return row is not None


def _alias_relation(relation_sql: str, alias: str | None) -> str:
    """Alias a relation or subquery if the caller requests one."""
    if not alias:
        return relation_sql
    if relation_sql.startswith("("):
        return f"{relation_sql} AS {alias}"
    return f"{relation_sql} {alias}"


def deployable_validated_predicate(con, alias: str = "validated_setups") -> str:
    """Return SQL predicate for the deployable validated shelf."""
    prefix = f"{alias}." if alias else ""
    status_predicate = f"LOWER({prefix}status) = 'active'"
    if not validated_setups_has_deployment_scope(con):
        return status_predicate
    return (
        f"{status_predicate} AND "
        f"LOWER(COALESCE({prefix}deployment_scope, '{DEPLOYMENT_SCOPE_DEPLOYABLE}')) = "
        f"'{DEPLOYMENT_SCOPE_DEPLOYABLE}'"
    )


def active_validated_relation(con, alias: str | None = None) -> str:
    """Return canonical SQL relation for active validated rows."""
    if _relation_exists(con, ACTIVE_VALIDATED_VIEW):
        return _alias_relation(ACTIVE_VALIDATED_VIEW, alias)
    return _alias_relation(
        "(SELECT * FROM validated_setups WHERE LOWER(status) = 'active')",
        alias,
    )


def deployable_validated_relation(con, alias: str | None = None) -> str:
    """Return canonical SQL relation for the deployable validated shelf."""
    if _relation_exists(con, DEPLOYABLE_VALIDATED_VIEW):
        return _alias_relation(DEPLOYABLE_VALIDATED_VIEW, alias)
    return _alias_relation(
        f"(SELECT * FROM validated_setups WHERE {deployable_validated_predicate(con)})",
        alias,
    )
