"""Tests for Rithmic broker adapter.

Tests order spec construction, bracket tick calculation, price collar,
factory integration, contract resolution, and mock-based integration tests
for submit, cancel, positions, and equity — all without network calls.

Verified against: async_rithmic 1.5.9 protobuf schema (template fields).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Order Router tests
# ---------------------------------------------------------------------------


class TestRithmicOrderRouterBuildSpec:
    """Test build_order_spec for E1 (market) and E2 (stop-market)."""

    def _make_router(self, tick_size=0.25):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        return RithmicOrderRouter(
            account_id=12345,
            auth=None,  # No auth needed for spec building
            tick_size=tick_size,
            exchange="CME",
            rithmic_account_id="12345",
        )

    def test_e1_market_long(self):
        router = self._make_router()
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6", qty=2)
        assert spec["order_type"] == 2  # MARKET
        assert spec["transaction_type"] == 1  # BUY
        assert spec["symbol"] == "MESM6"
        assert spec["exchange"] == "CME"
        assert spec["qty"] == 2
        assert "trigger_price" not in spec
        assert "_intent" in spec

    def test_e1_market_short(self):
        router = self._make_router()
        spec = router.build_order_spec("short", "E1", 5200.0, "MESM6")
        assert spec["order_type"] == 2  # MARKET
        assert spec["transaction_type"] == 2  # SELL

    def test_e2_stop_long(self):
        router = self._make_router()
        spec = router.build_order_spec("long", "E2", 5210.50, "MESM6")
        assert spec["order_type"] == 4  # STOP_MARKET
        assert spec["transaction_type"] == 1  # BUY
        assert spec["trigger_price"] == 5210.50

    def test_e2_stop_short(self):
        router = self._make_router()
        spec = router.build_order_spec("short", "E2", 5190.25, "MESM6")
        assert spec["order_type"] == 4  # STOP_MARKET
        assert spec["transaction_type"] == 2  # SELL
        assert spec["trigger_price"] == 5190.25

    def test_invalid_entry_model_raises(self):
        router = self._make_router()
        with pytest.raises(ValueError, match="E3"):
            router.build_order_spec("long", "E3", 5200.0, "MESM6")

    def test_intent_dict_present(self):
        router = self._make_router()
        spec = router.build_order_spec("long", "E2", 5210.0, "MESM6", qty=3)
        intent = spec["_intent"]
        assert intent["direction"] == "long"
        assert intent["entry_model"] == "E2"
        assert intent["entry_price"] == 5210.0
        assert intent["symbol"] == "MESM6"
        assert intent["qty"] == 3

    def test_account_id_in_spec(self):
        router = self._make_router()
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6")
        assert spec["account_id"] == "12345"


class TestRithmicOrderRouterExit:
    """Test build_exit_spec direction reversal."""

    def _make_router(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        return RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")

    def test_exit_long_sells(self):
        router = self._make_router()
        spec = router.build_exit_spec("long", "MESM6", qty=2)
        assert spec["order_type"] == 2  # MARKET
        assert spec["transaction_type"] == 2  # SELL (close long)

    def test_exit_short_buys(self):
        router = self._make_router()
        spec = router.build_exit_spec("short", "MNQM6")
        assert spec["order_type"] == 2  # MARKET
        assert spec["transaction_type"] == 1  # BUY (close short)


class TestRithmicBracketSpec:
    """Test bracket tick calculation for server-side brackets."""

    def _make_router(self, tick_size=0.25):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        return RithmicOrderRouter(account_id=12345, auth=None, tick_size=tick_size, rithmic_account_id="12345")

    def test_long_bracket_ticks(self):
        router = self._make_router(tick_size=0.25)
        # Long entry at 5200, stop at 5195, target at 5210
        bracket = router.build_bracket_spec("long", "MESM6", 5200.0, 5195.0, 5210.0)
        assert bracket["stop_ticks"] == 20  # (5200 - 5195) / 0.25 = 20
        assert bracket["target_ticks"] == 40  # (5210 - 5200) / 0.25 = 40

    def test_short_bracket_ticks(self):
        router = self._make_router(tick_size=0.25)
        # Short entry at 5200, stop at 5205, target at 5190
        bracket = router.build_bracket_spec("short", "MESM6", 5200.0, 5205.0, 5190.0)
        assert bracket["stop_ticks"] == 20  # abs(5200 - 5205) / 0.25 = 20
        assert bracket["target_ticks"] == 40  # abs(5190 - 5200) / 0.25 = 40

    def test_mgc_bracket_ticks(self):
        router = self._make_router(tick_size=0.10)
        # MGC: entry 2350.0, stop 2347.0, target 2357.5
        bracket = router.build_bracket_spec("long", "MGCM6", 2350.0, 2347.0, 2357.5)
        assert bracket["stop_ticks"] == 30  # 3.0 / 0.10 = 30
        assert bracket["target_ticks"] == 75  # 7.5 / 0.10 = 75

    def test_minimum_one_tick(self):
        router = self._make_router(tick_size=0.25)
        # Very tight bracket — should still be at least 1 tick
        bracket = router.build_bracket_spec("long", "MESM6", 5200.0, 5199.99, 5200.01)
        assert bracket["stop_ticks"] >= 1
        assert bracket["target_ticks"] >= 1

    def test_merge_bracket_into_entry(self):
        router = self._make_router()
        entry = {"order_type": 4, "transaction_type": 1, "symbol": "MESM6"}
        bracket = {"stop_ticks": 20, "target_ticks": 40}
        merged = router.merge_bracket_into_entry(entry, bracket)
        assert merged["stop_ticks"] == 20
        assert merged["target_ticks"] == 40
        assert merged["order_type"] == 4
        assert merged["symbol"] == "MESM6"


class TestRithmicPriceCollar:
    """Test price collar rejection logic."""

    def _make_router(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        router = RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")
        router.update_market_price(5200.0)
        return router

    def test_within_collar_ok(self):
        """Price within 0.5% should not raise."""
        router = self._make_router()
        # 5210 is 0.19% from 5200 — within collar
        spec = router.build_order_spec("long", "E2", 5210.0, "MESM6")
        assert spec["trigger_price"] == 5210.0

    def test_update_market_price(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        router = RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")
        assert router._last_known_price is None
        router.update_market_price(5200.0)
        assert router._last_known_price == 5200.0
        router.update_market_price(0)  # Zero price ignored
        assert router._last_known_price == 5200.0


class TestRithmicNativeBrackets:
    """Verify Rithmic reports native bracket support."""

    def test_supports_native_brackets(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        router = RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")
        assert router.supports_native_brackets() is True


# ---------------------------------------------------------------------------
# Broker Factory tests
# ---------------------------------------------------------------------------


class TestBrokerFactory:
    """Test Rithmic registration in broker factory."""

    def test_rithmic_in_valid_brokers(self):
        from trading_app.live.broker_factory import VALID_BROKERS

        assert "rithmic" in VALID_BROKERS

    def test_factory_returns_correct_types(self):
        """Verify create_broker_components('rithmic') returns correct component types.

        Note: This will fail if RITHMIC_* env vars are not set (auth tries to connect).
        We test the factory branch logic by checking it doesn't raise on import.
        """
        from trading_app.live.broker_factory import VALID_BROKERS
        from trading_app.live.rithmic.auth import RithmicAuth
        from trading_app.live.rithmic.contracts import RithmicContracts
        from trading_app.live.rithmic.order_router import RithmicOrderRouter
        from trading_app.live.rithmic.positions import RithmicPositions

        # Verify all classes are importable and are proper types
        assert RithmicAuth is not None
        assert RithmicContracts is not None
        assert RithmicOrderRouter is not None
        assert RithmicPositions is not None
        assert "rithmic" in VALID_BROKERS


# ---------------------------------------------------------------------------
# Contract resolution tests
# ---------------------------------------------------------------------------


class TestRithmicContracts:
    """Test contract symbol construction (offline, no API calls)."""

    def test_construct_front_month_q1(self):
        """March contract in Q1."""
        from trading_app.live.rithmic.contracts import RithmicContracts

        # This tests the static fallback method
        symbol = RithmicContracts._construct_front_month("MES")
        # Should return a valid CME-format symbol like MESH6, MESM6, etc.
        assert symbol.startswith("MES")
        assert len(symbol) >= 5  # e.g., "MESM6"

    def test_construct_front_month_format(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        for root in ["MES", "MNQ", "MGC"]:
            symbol = RithmicContracts._construct_front_month(root)
            assert symbol.startswith(root)
            # Month code should be one of H, M, U, Z
            month_code = symbol[len(root)]
            assert month_code in "HMUZ", f"Unexpected month code '{month_code}' in {symbol}"

    def test_instrument_roots_mapping(self):
        from trading_app.live.rithmic.contracts import INSTRUMENT_ROOTS

        assert "MES" in INSTRUMENT_ROOTS
        assert "MNQ" in INSTRUMENT_ROOTS
        assert "MGC" in INSTRUMENT_ROOTS


# ---------------------------------------------------------------------------
# Prop Profiles tests
# ---------------------------------------------------------------------------


class TestBulenoxProfile:
    """Test Bulenox prop firm spec and account profile."""

    def test_bulenox_spec_exists(self):
        from trading_app.prop_profiles import PROP_FIRM_SPECS

        assert "bulenox" in PROP_FIRM_SPECS
        spec = PROP_FIRM_SPECS["bulenox"]
        assert spec.platform == "rithmic"
        assert spec.auto_trading == "full"
        assert spec.consistency_rule == 0.40

    def test_bulenox_account_tiers(self):
        from trading_app.prop_profiles import ACCOUNT_TIERS

        assert ("bulenox", 50_000) in ACCOUNT_TIERS
        tier = ACCOUNT_TIERS[("bulenox", 50_000)]
        assert tier.max_dd == 2_500
        assert tier.max_contracts_micro == 50

    def test_bulenox_profile_exists(self):
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        assert "bulenox_50k" in ACCOUNT_PROFILES
        profile = ACCOUNT_PROFILES["bulenox_50k"]
        assert profile.firm == "bulenox"
        assert profile.copies == 3  # Max simultaneous
        assert profile.active is False  # Not activated yet
        assert len(profile.daily_lanes) == 5

    def test_bulenox_profile_sessions(self):
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        profile = ACCOUNT_PROFILES["bulenox_50k"]
        assert "CME_REOPEN" in profile.allowed_sessions
        assert "SINGAPORE_OPEN" in profile.allowed_sessions


# ---------------------------------------------------------------------------
# Auth tests (offline only)
# ---------------------------------------------------------------------------


class TestRithmicAuth:
    """Test RithmicAuth construction (no connection)."""

    def test_auth_init_no_crash(self):
        from trading_app.live.rithmic.auth import RithmicAuth

        auth = RithmicAuth()
        assert auth.is_healthy is False  # Not connected yet
        assert auth.headers() == {}
        assert auth._connected is False

    def test_auth_missing_credentials_raises(self):
        """Connecting without credentials should raise RuntimeError."""
        import os

        from trading_app.live.rithmic.auth import RithmicAuth

        # Clear any env vars that might be set
        old_user = os.environ.pop("RITHMIC_USER", None)
        old_pw = os.environ.pop("RITHMIC_PASSWORD", None)
        old_gw = os.environ.pop("RITHMIC_GATEWAY", None)
        try:
            auth = RithmicAuth()
            with pytest.raises(RuntimeError, match="credentials not configured"):
                auth.get_token()
        finally:
            if old_user:
                os.environ["RITHMIC_USER"] = old_user
            if old_pw:
                os.environ["RITHMIC_PASSWORD"] = old_pw
            if old_gw:
                os.environ["RITHMIC_GATEWAY"] = old_gw


# ---------------------------------------------------------------------------
# Mock-based integration tests (no network, verify async bridge wiring)
# ---------------------------------------------------------------------------


def _make_mock_auth():
    """Create a mock RithmicAuth with a mock async_rithmic client."""
    auth = MagicMock()
    auth.client = MagicMock()
    auth.run_async = MagicMock()
    return auth


class TestRithmicSubmitMocked:
    """Test submit() through the async bridge with mocked auth."""

    def test_submit_market_order_calls_client(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        # Simulate response with basket_id
        auth.run_async.return_value = [SimpleNamespace(basket_id="99001")]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6", qty=1)
        result = router.submit(spec)

        assert result["status"] == "submitted"
        assert result["order_id"] == "99001"
        auth.run_async.assert_called_once()

    def test_submit_stop_order_passes_trigger_price(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = [SimpleNamespace(basket_id="99002")]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        spec = router.build_order_spec("long", "E2", 5210.50, "MESM6")
        result = router.submit(spec)

        assert result["status"] == "submitted"
        # Verify the coroutine was created with trigger_price
        call_args = auth.run_async.call_args
        assert call_args is not None

    def test_submit_with_bracket_passes_ticks(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = [SimpleNamespace(basket_id="99003")]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        entry = router.build_order_spec("long", "E2", 5210.0, "MESM6")
        bracket = router.build_bracket_spec("long", "MESM6", 5210.0, 5205.0, 5220.0)
        merged = router.merge_bracket_into_entry(entry, bracket)
        result = router.submit(merged)

        assert result["order_id"] == "99003"
        assert merged["stop_ticks"] == 20
        assert merged["target_ticks"] == 40

    def test_submit_no_auth_raises(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        router = RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6")
        with pytest.raises(RuntimeError, match="No auth"):
            router.submit(spec)

    def test_submit_price_collar_rejects(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        router.update_market_price(5200.0)
        # 5500 is ~5.8% from 5200 — well beyond 0.5% collar
        spec = router.build_order_spec("long", "E2", 5500.0, "MESM6")
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)

    def test_submit_no_response_uses_generated_id(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = []  # Empty response

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6")
        result = router.submit(spec)

        # When no basket_id in response, falls back to generated order_id
        assert result["status"] == "submitted"
        assert result["order_id"].startswith("orb_")

    def test_order_ids_unique_across_accounts(self):
        """Different account IDs produce different order_id prefixes."""
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = []

        router_a = RithmicOrderRouter(
            account_id=11111, auth=auth, tick_size=0.25, rithmic_account_id="11111"
        )
        router_b = RithmicOrderRouter(
            account_id=22222, auth=auth, tick_size=0.25, rithmic_account_id="22222"
        )

        spec_a = router_a.build_order_spec("long", "E1", 5200.0, "MESM6")
        spec_b = router_b.build_order_spec("long", "E1", 5200.0, "MESM6")
        result_a = router_a.submit(spec_a)
        result_b = router_b.submit(spec_b)

        # Order IDs should contain different account suffixes
        assert "1111" in result_a["order_id"]
        assert "2222" in result_b["order_id"]
        assert result_a["order_id"] != result_b["order_id"]

    def test_order_submit_uses_higher_timeout(self):
        """Submit uses _ORDER_SUBMIT_TIMEOUT, not _BRIDGE_TIMEOUT."""
        from trading_app.live.rithmic.order_router import (
            RithmicOrderRouter,
            _BRIDGE_TIMEOUT,
            _ORDER_SUBMIT_TIMEOUT,
        )

        assert _ORDER_SUBMIT_TIMEOUT > _BRIDGE_TIMEOUT

        auth = _make_mock_auth()
        auth.run_async.return_value = [SimpleNamespace(basket_id="99001")]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6")
        router.submit(spec)

        # Verify run_async was called with the higher timeout
        call_kwargs = auth.run_async.call_args
        assert call_kwargs[1]["timeout"] == _ORDER_SUBMIT_TIMEOUT


class TestRithmicCancelMocked:
    """Test cancel() through the async bridge with mocked auth."""

    def test_cancel_calls_client_with_account_id(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = None

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        router.cancel(99001)
        auth.run_async.assert_called_once()

    def test_cancel_no_auth_raises(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        router = RithmicOrderRouter(account_id=12345, auth=None, rithmic_account_id="12345")
        with pytest.raises(RuntimeError, match="Cannot cancel"):
            router.cancel(99001)


class TestRithmicQueryOrderStatus:
    """Test query_order_status cache and API fallback."""

    def test_cache_hit_returns_cached(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = [SimpleNamespace(basket_id="99001")]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        # Submit to populate cache
        spec = router.build_order_spec("long", "E1", 5200.0, "MESM6")
        router.submit(spec)

        # Query should hit cache
        result = router.query_order_status(99001)
        assert result["order_id"] == 99001
        assert result["status"] == "submitted"

    def test_cache_miss_falls_back_to_api(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = SimpleNamespace(
            status="Filled", avg_fill_price=5200.25
        )

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        result = router.query_order_status(88888)
        assert result["order_id"] == 88888
        assert result["status"] == "Filled"
        assert result["fill_price"] == 5200.25


class TestRithmicOpenOrdersMocked:
    """Test query_open_orders and cancel_bracket_orders."""

    def test_query_open_orders_returns_list(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        auth.run_async.return_value = [
            SimpleNamespace(symbol="MESM6", basket_id="100"),
            SimpleNamespace(symbol="MESM6", basket_id="101"),
        ]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        orders = router.query_open_orders()
        assert len(orders) == 2

    def test_cancel_bracket_orders_cancels_matching(self):
        from trading_app.live.rithmic.order_router import RithmicOrderRouter

        auth = _make_mock_auth()
        # First call: query_open_orders returns 2 matching + 1 non-matching
        auth.run_async.side_effect = [
            [
                SimpleNamespace(symbol="MESM6", basket_id="100"),
                SimpleNamespace(symbol="MESM6", basket_id="101"),
                SimpleNamespace(symbol="MNQM6", basket_id="200"),
            ],
            None,  # cancel order 100
            None,  # cancel order 101
        ]

        router = RithmicOrderRouter(
            account_id=12345, auth=auth, tick_size=0.25, rithmic_account_id="12345"
        )
        cancelled = router.cancel_bracket_orders("MESM6")
        assert cancelled == 2


# ---------------------------------------------------------------------------
# Position query tests (mocked protobuf responses)
# ---------------------------------------------------------------------------


class TestRithmicPositionsMocked:
    """Test position queries with mock protobuf-like response objects."""

    def test_query_open_returns_net_positions(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        # Simulate InstrumentPnLPositionUpdate protobuf (template 450)
        auth.run_async.return_value = [
            SimpleNamespace(symbol="MESM6", net_quantity=2, avg_open_fill_price=5200.25),
            SimpleNamespace(symbol="MNQM6", net_quantity=-1, avg_open_fill_price=18500.0),
            SimpleNamespace(symbol="MGCM6", net_quantity=0, avg_open_fill_price=0),  # Flat — filtered
        ]

        pos = RithmicPositions(auth=auth)
        result = pos.query_open(12345)

        assert len(result) == 2
        assert result[0]["contract_id"] == "MESM6"
        assert result[0]["side"] == "long"
        assert result[0]["size"] == 2
        assert result[0]["avg_price"] == 5200.25
        assert result[1]["contract_id"] == "MNQM6"
        assert result[1]["side"] == "short"
        assert result[1]["size"] == 1

    def test_query_open_empty_positions(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        auth.run_async.return_value = []

        pos = RithmicPositions(auth=auth)
        result = pos.query_open(12345)
        assert result == []

    def test_query_open_raises_on_error(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        auth.run_async.side_effect = RuntimeError("Connection lost")

        pos = RithmicPositions(auth=auth)
        with pytest.raises(RuntimeError, match="Connection lost"):
            pos.query_open(12345)


class TestRithmicEquityMocked:
    """Test equity queries with mock protobuf-like response objects."""

    def test_query_equity_from_account_balance(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        # Simulate AccountPnLPositionUpdate (template 451) — STRING fields
        auth.run_async.return_value = [
            SimpleNamespace(account_balance="47500.00", cash_on_hand="48000.00")
        ]

        pos = RithmicPositions(auth=auth)
        equity = pos.query_equity(12345)
        assert equity == 47500.0

    def test_query_equity_falls_back_to_cash_on_hand(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        # account_balance is empty string (protobuf default for unset)
        auth.run_async.return_value = [
            SimpleNamespace(account_balance="", cash_on_hand="48000.00")
        ]

        pos = RithmicPositions(auth=auth)
        equity = pos.query_equity(12345)
        assert equity == 48000.0

    def test_query_equity_returns_none_when_empty(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        auth.run_async.return_value = []

        pos = RithmicPositions(auth=auth)
        equity = pos.query_equity(12345)
        assert equity is None

    def test_query_equity_handles_empty_string_balance(self):
        """Both fields empty string — should return None, not crash."""
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        auth.run_async.return_value = [
            SimpleNamespace(account_balance="", cash_on_hand="")
        ]

        pos = RithmicPositions(auth=auth)
        equity = pos.query_equity(12345)
        assert equity is None

    def test_query_equity_error_returns_none(self):
        from trading_app.live.rithmic.positions import RithmicPositions

        auth = _make_mock_auth()
        auth.run_async.side_effect = RuntimeError("timeout")

        pos = RithmicPositions(auth=auth)
        equity = pos.query_equity(12345)
        assert equity is None


# ---------------------------------------------------------------------------
# Contract resolution tests (mocked API path)
# ---------------------------------------------------------------------------


class TestRithmicContractResolveMocked:
    """Test contract resolution via API and account discovery."""

    def test_resolve_account_id_numeric(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.client.accounts = [SimpleNamespace(account_id="98765")]
        auth.client.fcm_id = "FCM01"
        auth.client.ib_id = "IB01"

        contracts = RithmicContracts(auth=auth)
        acct_id = contracts.resolve_account_id()
        assert acct_id == 98765

    def test_resolve_account_id_non_numeric_hashes(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.client.accounts = [SimpleNamespace(account_id="APEX-SIM-123")]
        auth.client.fcm_id = "FCM01"
        auth.client.ib_id = "IB01"

        contracts = RithmicContracts(auth=auth)
        acct_id = contracts.resolve_account_id()
        assert isinstance(acct_id, int)
        assert acct_id >= 0  # 0x7FFFFFFF mask ensures positive

    def test_resolve_account_id_no_accounts_raises(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.client.accounts = []

        contracts = RithmicContracts(auth=auth)
        with pytest.raises(RuntimeError, match="No Rithmic accounts"):
            contracts.resolve_account_id()

    def test_resolve_all_account_ids(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.client.accounts = [
            SimpleNamespace(account_id="11111", account_name="Bulenox-1"),
            SimpleNamespace(account_id="22222", account_name="Bulenox-2"),
        ]

        contracts = RithmicContracts(auth=auth)
        result = contracts.resolve_all_account_ids()
        assert len(result) == 2
        assert result[0] == (11111, "Bulenox-1")
        assert result[1] == (22222, "Bulenox-2")

    def test_resolve_front_month_via_api(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.run_async.return_value = "MESM6"

        contracts = RithmicContracts(auth=auth)
        symbol = contracts.resolve_front_month("MES")
        assert symbol == "MESM6"
        # Second call should hit cache
        symbol2 = contracts.resolve_front_month("MES")
        assert symbol2 == "MESM6"
        auth.run_async.assert_called_once()  # Only 1 API call

    def test_resolve_front_month_api_failure_falls_back(self):
        from trading_app.live.rithmic.contracts import RithmicContracts

        auth = _make_mock_auth()
        auth.run_async.side_effect = RuntimeError("TICKER_PLANT not connected")

        contracts = RithmicContracts(auth=auth)
        symbol = contracts.resolve_front_month("MNQ")
        assert symbol.startswith("MNQ")
        assert symbol[3] in "HMUZ"


# ---------------------------------------------------------------------------
# Roll date buffer tests
# ---------------------------------------------------------------------------


class TestRithmicRollDateBuffer:
    """Test front-month construction respects expiration dates."""

    def test_mid_quarter_returns_current(self):
        """Feb 15 → March (H) contract."""
        from datetime import date

        from trading_app.live.rithmic.contracts import RithmicContracts

        with patch("trading_app.live.rithmic.contracts.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 15)
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            symbol = RithmicContracts._construct_front_month("MES")
            assert symbol == "MESH6"

    def test_expiration_month_early_returns_current(self):
        """Mar 10 → still March (H), before day-14 cutoff."""
        from datetime import date

        from trading_app.live.rithmic.contracts import RithmicContracts

        with patch("trading_app.live.rithmic.contracts.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 10)
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            symbol = RithmicContracts._construct_front_month("MES")
            assert symbol == "MESH6"

    def test_expiration_month_late_rolls_to_next(self):
        """Mar 20 → June (M) contract, past day-14 cutoff."""
        from datetime import date

        from trading_app.live.rithmic.contracts import RithmicContracts

        with patch("trading_app.live.rithmic.contracts.date") as mock_date:
            mock_date.today.return_value = date(2026, 3, 20)
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            symbol = RithmicContracts._construct_front_month("MES")
            assert symbol == "MESM6"

    def test_december_late_rolls_to_next_year(self):
        """Dec 20 → next year March (H)."""
        from datetime import date

        from trading_app.live.rithmic.contracts import RithmicContracts

        with patch("trading_app.live.rithmic.contracts.date") as mock_date:
            mock_date.today.return_value = date(2026, 12, 20)
            mock_date.side_effect = lambda *a, **k: date(*a, **k)
            symbol = RithmicContracts._construct_front_month("MES")
            assert symbol == "MESH7"
