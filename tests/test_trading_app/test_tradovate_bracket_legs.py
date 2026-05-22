"""Tests for TradovateOrderRouter.verify_bracket_legs and has_queryable_bracket_legs.

Covers the Stage 2 implementation that lets session_orchestrator.py dispatch the
bracket-leg verification path on Tradovate brokers (Tradeify, MFFU, direct).

Identification contract (per ``verify_bracket_legs`` docstring):

- Same ``symbol`` as the entry contract.
- ``orderType == "Stop"``  -> SL leg.
- ``orderType == "Limit"`` -> TP leg.
- ``orderId > entry_order_id`` to filter out unrelated working orders.

Contract-correctness note: the (sl_id, tp_id) return ordering is enforced by
``test_does_not_swap_stop_and_target``. Per the adversarial audit on commit
``58abc30a``: a swap does NOT produce naked-position exposure (both legs are
cancelled by iteration in ``session_orchestrator._cancel_bracket_orders``;
``record.bracket_order_ids`` is never index-accessed), but it would mis-label
SL vs TP in operator-visible logs and metadata. The test guards the
``orderType``-based dispatch contract.
"""

from unittest.mock import MagicMock

import pytest

from trading_app.live.tradovate.auth import TradovateAuth
from trading_app.live.tradovate.http import RateLimitExhausted
from trading_app.live.tradovate.order_router import TradovateOrderRouter


@pytest.fixture()
def mock_auth():
    auth = MagicMock(spec=TradovateAuth)
    auth.base_url = "https://demo.tradovateapi.com/v1"
    auth.headers.return_value = {"Authorization": "Bearer test-token"}
    return auth


@pytest.fixture()
def router(mock_auth):
    return TradovateOrderRouter(account_id=12345, auth=mock_auth, tick_size=0.25)


class TestHasQueryableBracketLegs:
    def test_returns_true(self, router):
        assert router.has_queryable_bracket_legs() is True


class TestVerifyBracketLegsHappyPath:
    def test_returns_stop_then_limit_from_open_orders(self, router):
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 1001, "symbol": "MNQM6", "orderType": "Stop"},
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id == 1002

    def test_accepts_legacy_id_and_contract_symbol_keys(self, router):
        router.query_open_orders = MagicMock(
            return_value=[
                {"id": 1001, "contractSymbol": "MNQM6", "orderType": "Stop"},
                {"id": 1002, "contractSymbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id == 1002

    def test_does_not_swap_stop_and_target(self, router):
        """Contract guard: SL must always come back in slot 0 (Stop), TP in slot 1 (Limit).

        Per the broker_base.verify_bracket_legs contract and the adversarial-audit
        finding on commit 58abc30a: a swap would mis-label SL vs TP in operator
        logs (``"Bracket SL identified: orderId=..."``) and in any future indexed
        consumer of ``record.bracket_order_ids``. Today every consumer iterates
        rather than indexes, so a swap is not currently exposure-creating — but
        the contract still requires correct ordering.
        """
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 7777, "symbol": "MGCM6", "orderType": "Limit"},
                {"orderId": 7778, "symbol": "MGCM6", "orderType": "Stop"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=7000, contract_id="MGCM6")
        assert sl_id == 7778, "Stop order must be returned as sl_id (slot 0)"
        assert tp_id == 7777, "Limit order must be returned as tp_id (slot 1)"


class TestVerifyBracketLegsPartialAndFiltering:
    def test_missing_stop_returns_none_for_sl(self, router):
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id is None
        assert tp_id == 1002

    def test_missing_limit_returns_none_for_tp(self, router):
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 1001, "symbol": "MNQM6", "orderType": "Stop"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id is None

    def test_skips_orders_below_entry_id(self, router):
        """Pre-existing working orders on the same contract must NOT be picked up."""
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 999, "symbol": "MNQM6", "orderType": "Stop"},  # pre-existing — skip
                {"orderId": 1000, "symbol": "MNQM6", "orderType": "Stop"},  # equals entry — skip
                {"orderId": 1001, "symbol": "MNQM6", "orderType": "Stop"},  # bracket SL — pick
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Limit"},  # bracket TP — pick
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id == 1002

    def test_skips_orders_on_different_contract(self, router):
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 1001, "symbol": "MESM6", "orderType": "Stop"},  # wrong contract
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Stop"},
                {"orderId": 1003, "symbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1002
        assert tp_id == 1003

    def test_picks_first_matching_leg_when_duplicates(self, router):
        """If two open Stop orders match, the first encountered wins (and we log)."""
        router.query_open_orders = MagicMock(
            return_value=[
                {"orderId": 1001, "symbol": "MNQM6", "orderType": "Stop"},
                {"orderId": 1003, "symbol": "MNQM6", "orderType": "Stop"},  # second Stop — should be ignored
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id == 1002


class TestVerifyBracketLegsFailureModes:
    def test_rate_limit_propagates(self, router):
        # RateLimitExhausted now requires error_class kwarg (2026-05-18 baseline).
        router.query_open_orders = MagicMock(
            side_effect=RateLimitExhausted("429 exhausted", error_class="E"),
        )
        with pytest.raises(RateLimitExhausted):
            router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")

    def test_generic_exception_returns_none_none(self, router):
        router.query_open_orders = MagicMock(side_effect=RuntimeError("network down"))
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id is None
        assert tp_id is None

    def test_missing_auth_raises(self, mock_auth):
        r = TradovateOrderRouter(account_id=12345, auth=mock_auth, tick_size=0.25)
        r.auth = None
        with pytest.raises(RuntimeError, match="No auth"):
            r.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")

    def test_empty_open_orders_returns_none_none(self, router):
        router.query_open_orders = MagicMock(return_value=[])
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id is None
        assert tp_id is None

    def test_skips_orders_with_missing_id_keys(self, router):
        """Audit gap (auditor 2026-05-12): an order dict with neither ``orderId``
        nor ``id`` resolves to ``oid is None`` at the loop guard. The implementation
        must continue past such orders without raising.
        """
        router.query_open_orders = MagicMock(
            return_value=[
                {"symbol": "MNQM6", "orderType": "Stop"},  # no orderId/id at all
                {"orderId": 1001, "symbol": "MNQM6", "orderType": "Stop"},
                {"orderId": 1002, "symbol": "MNQM6", "orderType": "Limit"},
            ]
        )
        sl_id, tp_id = router.verify_bracket_legs(entry_order_id=1000, contract_id="MNQM6")
        assert sl_id == 1001
        assert tp_id == 1002


class TestSupportsSequentialBracketIds:
    """Tradovate placeOSO returns API-assigned non-sequential IDs.

    The session_orchestrator emergency fallback at lines 2455-2474 reads
    ``supports_sequential_bracket_ids()`` to decide whether to apply the
    ProjectX ``entry_id+1`` / ``entry_id+2`` heuristic. Tradovate must
    inherit the base-class default (False) so we do not store guessed IDs.
    """

    def test_tradovate_returns_false(self, router):
        assert router.supports_sequential_bracket_ids() is False
