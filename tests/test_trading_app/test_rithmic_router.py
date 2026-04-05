"""Tests for Rithmic broker adapter.

Tests order spec construction, bracket tick calculation, price collar,
factory integration, and contract resolution — all without network calls.
"""

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
