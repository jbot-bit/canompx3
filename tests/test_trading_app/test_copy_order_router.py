"""Tests for CopyOrderRouter — fan-out order routing to N accounts."""

from unittest.mock import MagicMock

import pytest

from trading_app.live.copy_order_router import CopyOrderRouter


def _make_mock_router(account_id=1):
    """Create a mock BrokerRouter with standard interface."""
    router = MagicMock()
    router.account_id = account_id
    router.auth = MagicMock()
    return router


class TestInit:
    def test_calls_super_init(self):
        primary = _make_mock_router(account_id=100)
        shadow = _make_mock_router(account_id=200)
        copy = CopyOrderRouter(primary, [shadow])
        assert copy.account_id == 100
        assert copy.auth is primary.auth

    def test_exposes_primary_and_shadows(self):
        primary = _make_mock_router(1)
        shadows = [_make_mock_router(2), _make_mock_router(3)]
        copy = CopyOrderRouter(primary, shadows)
        assert copy.primary is primary
        assert copy.shadows is shadows
        assert copy.shadow_count == 2

    def test_all_account_ids(self):
        primary = _make_mock_router(10)
        shadows = [_make_mock_router(20), _make_mock_router(30)]
        copy = CopyOrderRouter(primary, shadows)
        assert copy.all_account_ids == [10, 20, 30]


class TestSubmitStatusCheck:
    """submit() should copy to shadows on success, skip on known failures."""

    def test_projectx_filled_copies_to_shadows(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "Filled"}
        shadow = _make_mock_router(2)
        shadow.submit.return_value = {"order_id": "def", "status": "Filled"}

        copy = CopyOrderRouter(primary, [shadow])
        result = copy.submit({"spec": True})

        assert result["status"] == "Filled"
        shadow.submit.assert_called_once_with({"spec": True})

    def test_projectx_working_copies_to_shadows(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "Working"}
        shadow = _make_mock_router(2)
        shadow.submit.return_value = {"order_id": "def", "status": "Working"}

        copy = CopyOrderRouter(primary, [shadow])
        copy.submit({"spec": True})
        shadow.submit.assert_called_once()

    def test_rithmic_submitted_copies_to_shadows(self):
        """Rithmic returns 'submitted' — must NOT be skipped."""
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "submitted"}
        shadow = _make_mock_router(2)
        shadow.submit.return_value = {"order_id": "def", "status": "submitted"}

        copy = CopyOrderRouter(primary, [shadow])
        copy.submit({"spec": True})
        shadow.submit.assert_called_once()

    def test_rejected_skips_shadows(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "rejected"}
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        copy.submit({"spec": True})
        shadow.submit.assert_not_called()

    def test_error_skips_shadows(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "error"}
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        copy.submit({"spec": True})
        shadow.submit.assert_not_called()

    def test_cancelled_skips_shadows(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "Cancelled"}
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        copy.submit({"spec": True})
        shadow.submit.assert_not_called()

    def test_shadow_failure_does_not_affect_primary(self):
        primary = _make_mock_router(1)
        primary.submit.return_value = {"order_id": "abc", "status": "Filled"}
        shadow = _make_mock_router(2)
        shadow.submit.side_effect = RuntimeError("network error")

        copy = CopyOrderRouter(primary, [shadow])
        result = copy.submit({"spec": True})
        assert result["status"] == "Filled"

    def test_primary_raise_propagates(self):
        primary = _make_mock_router(1)
        primary.submit.side_effect = RuntimeError("auth failed")
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        with pytest.raises(RuntimeError, match="auth failed"):
            copy.submit({"spec": True})
        shadow.submit.assert_not_called()


class TestCancelShadows:
    """cancel() must cancel on primary + all shadows."""

    def test_cancel_reaches_primary_and_shadows(self):
        primary = _make_mock_router(1)
        shadow1 = _make_mock_router(2)
        shadow2 = _make_mock_router(3)

        copy = CopyOrderRouter(primary, [shadow1, shadow2])
        copy.cancel(99)

        primary.cancel.assert_called_once_with(99)
        shadow1.cancel.assert_called_once_with(99)
        shadow2.cancel.assert_called_once_with(99)

    def test_primary_cancel_failure_propagates(self):
        primary = _make_mock_router(1)
        primary.cancel.side_effect = RuntimeError("cancel failed")
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        with pytest.raises(RuntimeError, match="cancel failed"):
            copy.cancel(99)
        # Shadow should NOT be attempted if primary fails (fail-closed)
        shadow.cancel.assert_not_called()

    def test_shadow_cancel_failure_does_not_propagate(self):
        primary = _make_mock_router(1)
        shadow = _make_mock_router(2)
        shadow.cancel.side_effect = RuntimeError("shadow down")

        copy = CopyOrderRouter(primary, [shadow])
        copy.cancel(99)  # Should not raise
        primary.cancel.assert_called_once_with(99)


class TestUpdateMarketPrice:
    """update_market_price() must forward to primary + all shadows."""

    def test_forwards_to_primary_and_shadows(self):
        primary = _make_mock_router(1)
        shadow1 = _make_mock_router(2)
        shadow2 = _make_mock_router(3)

        copy = CopyOrderRouter(primary, [shadow1, shadow2])
        copy.update_market_price(5200.50)

        primary.update_market_price.assert_called_once_with(5200.50)
        shadow1.update_market_price.assert_called_once_with(5200.50)
        shadow2.update_market_price.assert_called_once_with(5200.50)

    def test_shadow_failure_does_not_block(self):
        primary = _make_mock_router(1)
        shadow = _make_mock_router(2)
        shadow.update_market_price.side_effect = RuntimeError("boom")

        copy = CopyOrderRouter(primary, [shadow])
        copy.update_market_price(5200.50)  # Should not raise
        primary.update_market_price.assert_called_once()


class TestDelegation:
    """Methods that delegate to primary should pass through correctly."""

    def test_build_order_spec_delegates(self):
        primary = _make_mock_router(1)
        primary.build_order_spec.return_value = {"order_type": 2}

        copy = CopyOrderRouter(primary, [])
        result = copy.build_order_spec("long", "E1", 5200.0, "MESM6")
        assert result == {"order_type": 2}

    def test_build_exit_spec_delegates(self):
        primary = _make_mock_router(1)
        primary.build_exit_spec.return_value = {"exit": True}

        copy = CopyOrderRouter(primary, [])
        result = copy.build_exit_spec("long", "MESM6")
        assert result == {"exit": True}

    def test_supports_native_brackets_delegates(self):
        primary = _make_mock_router(1)
        primary.supports_native_brackets.return_value = True

        copy = CopyOrderRouter(primary, [])
        assert copy.supports_native_brackets() is True

    def test_query_order_status_queries_primary_only(self):
        primary = _make_mock_router(1)
        primary.query_order_status.return_value = {"status": "Filled"}
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        result = copy.query_order_status(99)
        assert result["status"] == "Filled"
        shadow.query_order_status.assert_not_called()

    def test_verify_bracket_legs_delegates_to_primary(self):
        """REGRESSION: BrokerRouter base default returns (None, None), which the
        session_orchestrator caller interprets as 'BRACKET LEGS MISSING' and fires
        a false CRITICAL alarm with empty bracket_order_ids. The wrapper MUST
        delegate so the active TopStep+CopyOrderRouter+ProjectX path gets the real
        SL/TP order IDs from the primary.
        """
        primary = _make_mock_router(1)
        primary.verify_bracket_legs.return_value = (101, 102)
        shadow = _make_mock_router(2)

        copy = CopyOrderRouter(primary, [shadow])
        sl_id, tp_id = copy.verify_bracket_legs(entry_order_id=100, contract_id="MESM6")
        assert sl_id == 101
        assert tp_id == 102
        primary.verify_bracket_legs.assert_called_once_with(100, "MESM6")
        # Shadows are best-effort — verification queries primary only.
        shadow.verify_bracket_legs.assert_not_called()

    def test_has_queryable_bracket_legs_delegates_to_primary_true(self):
        """Wrapper must inherit primary's bracket-leg queryability flag.

        When the primary is ProjectX (separately-queryable bracket legs), the
        wrapper must return True so session_orchestrator runs verify_bracket_legs
        in the active TopStep+CopyOrderRouter+ProjectX path.
        """
        primary = _make_mock_router(1)
        primary.has_queryable_bracket_legs.return_value = True

        copy = CopyOrderRouter(primary, [])
        assert copy.has_queryable_bracket_legs() is True
        primary.has_queryable_bracket_legs.assert_called_once()

    def test_has_queryable_bracket_legs_delegates_to_primary_false(self):
        """When wrapping an atomic-bracket broker (Rithmic, Tradovate), the
        wrapper must return False so session_orchestrator skips the verify call.
        """
        primary = _make_mock_router(1)
        primary.has_queryable_bracket_legs.return_value = False

        copy = CopyOrderRouter(primary, [])
        assert copy.has_queryable_bracket_legs() is False
        primary.has_queryable_bracket_legs.assert_called_once()
