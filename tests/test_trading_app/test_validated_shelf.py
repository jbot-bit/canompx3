"""Tests for canonical validated shelf lifecycle helpers."""

import duckdb

from trading_app.validated_shelf import (
    DEPLOYMENT_SCOPE_DEPLOYABLE,
    DEPLOYMENT_SCOPE_NON_DEPLOYABLE,
    NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON,
    deployable_validated_predicate,
    validated_shelf_lifecycle,
)


class TestValidatedShelfLifecycle:
    def test_active_instrument_is_deployable(self):
        lifecycle = validated_shelf_lifecycle("MNQ")
        assert lifecycle.status == "active"
        assert lifecycle.deployment_scope == DEPLOYMENT_SCOPE_DEPLOYABLE
        assert lifecycle.retirement_reason is None

    def test_non_active_instrument_is_non_deployable(self):
        lifecycle = validated_shelf_lifecycle("GC")
        assert lifecycle.status == "retired"
        assert lifecycle.deployment_scope == DEPLOYMENT_SCOPE_NON_DEPLOYABLE
        assert lifecycle.retirement_reason == NON_ACTIVE_INSTRUMENT_RETIREMENT_REASON

    def test_predicate_falls_back_without_scope_column(self):
        con = duckdb.connect(":memory:")
        try:
            con.execute("CREATE TABLE validated_setups (strategy_id VARCHAR, status VARCHAR)")
            assert deployable_validated_predicate(con, "vs") == "LOWER(vs.status) = 'active'"
        finally:
            con.close()

    def test_predicate_uses_explicit_scope_when_present(self):
        con = duckdb.connect(":memory:")
        try:
            con.execute(
                """
                CREATE TABLE validated_setups (
                    strategy_id VARCHAR,
                    status VARCHAR,
                    deployment_scope VARCHAR
                )
                """
            )
            assert deployable_validated_predicate(con, "vs") == (
                "LOWER(vs.status) = 'active' AND "
                "LOWER(COALESCE(vs.deployment_scope, 'deployable')) = 'deployable'"
            )
        finally:
            con.close()
