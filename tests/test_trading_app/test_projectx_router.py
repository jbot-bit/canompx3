"""Test ProjectX order routing — mocked HTTP."""

from unittest.mock import MagicMock, patch


def test_projectx_market_buy():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "orderId": 9056,
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        from trading_app.live.projectx.order_router import ProjectXOrderRouter

        router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
        spec = router.build_order_spec(
            direction="long",
            entry_model="E1",
            entry_price=2950.0,
            symbol="CON.F.US.MGC.M26",
            qty=1,
        )
        result = router.submit(spec)
        assert result["order_id"] == 9056
        call_body = mock_post.call_args[1]["json"]
        assert call_body["accountId"] == 123
        assert call_body["type"] == 2  # Market
        assert call_body["side"] == 0  # Bid (buy)
        assert call_body["size"] == 1


def test_projectx_stop_sell():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    spec = router.build_order_spec(
        direction="short",
        entry_model="E2",
        entry_price=2950.0,
        symbol="CON.F.US.MGC.M26",
        qty=1,
    )
    assert spec["type"] == 4  # Stop
    assert spec["side"] == 1  # Ask (sell)
    assert spec["stopPrice"] == 2950.0


def test_projectx_exit_reverses_direction():
    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    # Close a long = sell
    exit_spec = router.build_exit_spec("long", "CON.F.US.MGC.M26", qty=1)
    assert exit_spec["side"] == 1  # Ask (sell)
    assert exit_spec["type"] == 2  # Market

    # Close a short = buy
    exit_spec = router.build_exit_spec("short", "CON.F.US.MGC.M26", qty=1)
    assert exit_spec["side"] == 0  # Bid (buy)


def test_projectx_supports_brackets():
    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert router.supports_native_brackets() is True


def test_projectx_has_queryable_bracket_legs():
    """ProjectX AutoBracket creates separate SL/TP child orders queryable via
    searchOpen. Flag must be True so session_orchestrator runs the
    verify_bracket_legs path to capture real order IDs for later cancellation.
    Regression guard — if this flips to False, the active TopStep production
    path loses bracket leg tracking.
    """
    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert router.has_queryable_bracket_legs() is True


def test_projectx_is_broker_router():
    mock_auth = MagicMock()
    from trading_app.live.broker_base import BrokerRouter
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert isinstance(router, BrokerRouter)


def test_projectx_e3_blocked():
    import pytest

    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    with pytest.raises(ValueError, match="not supported"):
        router.build_order_spec("long", "E3", 2950.0, "CON.F.US.MGC.M26")


# ---- fill_price parsing (OR2) ----
# Validates the is-None guard introduced in iter 21 (OR1 fix).
# Key case: fill_price=0.0 must not fall through to fallback field.


def _px_mock_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = data
    return resp


def _px_auth() -> MagicMock:
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    return auth


def test_px_submit_fill_price_primary():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99, "fillPrice": 2010.5})
        result = router.submit(spec)
    assert result["fill_price"] == 2010.5


def test_px_submit_fill_price_fallback():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99, "averagePrice": 2010.5})
        result = router.submit(spec)
    assert result["fill_price"] == 2010.5


def test_px_submit_fill_price_none():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99})
        result = router.submit(spec)
    assert result["fill_price"] is None


def test_px_submit_fill_price_zero_not_falsy():
    """fillPrice=0.0 must not fall through to averagePrice — guards the is-None fix."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp(
            {"success": True, "orderId": 99, "fillPrice": 0.0, "averagePrice": 2010.5}
        )
        result = router.submit(spec)
    assert result["fill_price"] == 0.0


def test_px_query_fill_price_primary():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "fillPrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 2011.0


def test_px_query_fill_price_fallback():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "averagePrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 2011.0


def test_px_query_fill_price_zero_not_falsy():
    """fillPrice=0.0 must not fall through to averagePrice in query path."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "fillPrice": 0.0, "averagePrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 0.0


# ---- Bracket order tests (native atomic brackets) ----


def test_px_bracket_spec_format_mgc():
    """build_bracket_spec returns stopLossBracket/takeProfitBracket with tick offsets."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.10)  # MGC
    spec = router.build_bracket_spec(
        direction="long",
        symbol="CON.F.US.MGC.M26",
        entry_price=2950.0,
        stop_price=2940.0,  # 10 pts = 100 ticks
        target_price=2970.0,  # 20 pts = 200 ticks
    )
    assert spec is not None
    # Long: SL negative (below entry), TP positive (above entry)
    assert spec["stopLossBracket"] == {"ticks": -100, "type": 4}
    assert spec["takeProfitBracket"] == {"ticks": 200, "type": 1}
    # No accountId or orders array — just bracket fields
    assert "accountId" not in spec
    assert "orders" not in spec


def test_px_bracket_spec_format_mnq():
    """Tick calculation with MNQ tick_size=0.25."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.25)  # MNQ
    spec = router.build_bracket_spec(
        direction="short",
        symbol="CON.F.US.MNQ.M26",
        entry_price=21000.0,
        stop_price=21010.0,  # 10 pts = 40 ticks
        target_price=20980.0,  # 20 pts = 80 ticks
    )
    assert spec is not None
    # Short: SL positive (above entry), TP negative (below entry)
    assert spec["stopLossBracket"] == {"ticks": 40, "type": 4}
    assert spec["takeProfitBracket"] == {"ticks": -80, "type": 1}


def test_px_bracket_tick_min_clamp():
    """Bracket ticks never go below 1 (minimum 1 tick)."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.10)
    spec = router.build_bracket_spec(
        direction="long",
        symbol="TEST",
        entry_price=100.0,
        stop_price=99.999,  # < 1 tick → clamp to 1
        target_price=100.001,  # < 1 tick → clamp to 1
    )
    # Long: SL negative, TP positive, both at least 1 tick magnitude
    assert abs(spec["stopLossBracket"]["ticks"]) >= 1
    assert abs(spec["takeProfitBracket"]["ticks"]) >= 1
    assert spec["stopLossBracket"]["ticks"] < 0  # Long SL is negative
    assert spec["takeProfitBracket"]["ticks"] > 0  # Long TP is positive


def test_px_bracket_tick_rounding():
    """Tick calculation rounds to nearest integer."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.25)
    spec = router.build_bracket_spec(
        direction="long",
        symbol="TEST",
        entry_price=100.0,
        stop_price=98.60,  # 1.40 / 0.25 = 5.6 → rounds to 6
        target_price=102.90,  # 2.90 / 0.25 = 11.6 → rounds to 12
    )
    # Long: SL negative, TP positive
    assert spec["stopLossBracket"]["ticks"] == -6
    assert spec["takeProfitBracket"]["ticks"] == 12


def test_px_merge_bracket_into_entry():
    """merge_bracket_into_entry produces combined payload for atomic submission."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=_px_auth(), tick_size=0.10)
    entry_spec = {
        "accountId": 123,
        "contractId": "CON.F.US.MGC.M26",
        "type": 4,
        "side": 0,
        "size": 1,
        "stopPrice": 2950.0,
    }
    bracket_spec = {
        "stopLossBracket": {"ticks": -100, "type": 4},
        "takeProfitBracket": {"ticks": 200, "type": 1},
    }
    combined = router.merge_bracket_into_entry(entry_spec, bracket_spec)
    # All entry fields preserved
    assert combined["accountId"] == 123
    assert combined["type"] == 4
    assert combined["stopPrice"] == 2950.0
    # Bracket fields attached (signed ticks)
    assert combined["stopLossBracket"] == {"ticks": -100, "type": 4}
    assert combined["takeProfitBracket"] == {"ticks": 200, "type": 1}
    # Original entry_spec not mutated
    assert "stopLossBracket" not in entry_spec


def test_px_submit_combined_bracket_entry():
    """Submit with bracket fields sends correct combined payload to API."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=_px_auth(), tick_size=0.10)
    entry_spec = router.build_order_spec("long", "E2", 2950.0, "CON.F.US.MGC.M26", qty=1)
    bracket = router.build_bracket_spec("long", "CON.F.US.MGC.M26", 2950.0, 2940.0, 2970.0)
    combined = router.merge_bracket_into_entry(entry_spec, bracket)

    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 5001, "fillPrice": 2950.1})
        result = router.submit(combined)

    assert result["order_id"] == 5001
    assert result["fill_price"] == 2950.1
    # Verify the payload sent to the API
    call_body = mock_req.post.call_args[1]["json"]
    assert call_body["type"] == 4  # Stop entry
    assert call_body["stopPrice"] == 2950.0
    # Long: SL negative, TP positive
    assert call_body["stopLossBracket"] == {"ticks": -100, "type": 4}
    assert call_body["takeProfitBracket"] == {"ticks": 200, "type": 1}


def test_px_bracket_short_direction():
    """Bracket tick calculation is symmetric for short direction."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.10)
    # Short: entry=2950, stop=2960 (above), target=2930 (below)
    spec = router.build_bracket_spec(
        direction="short",
        symbol="CON.F.US.MGC.M26",
        entry_price=2950.0,
        stop_price=2960.0,
        target_price=2930.0,
    )
    # Short: SL positive (above entry), TP negative (below entry)
    assert spec["stopLossBracket"]["ticks"] == 100  # abs(2950 - 2960) / 0.10, positive for short SL
    assert spec["takeProfitBracket"]["ticks"] == -200  # abs(2930 - 2950) / 0.10, negative for short TP


def test_px_tick_size_validation():
    """Router rejects non-positive tick_size."""
    import pytest

    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    with pytest.raises(ValueError, match="tick_size must be positive"):
        ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=0.0)
    with pytest.raises(ValueError, match="tick_size must be positive"):
        ProjectXOrderRouter(account_id=1, auth=_px_auth(), tick_size=-0.25)


def test_px_base_merge_is_noop():
    """BrokerRouter base class merge_bracket_into_entry is a no-op."""
    from trading_app.live.broker_base import BrokerRouter

    class StubRouter(BrokerRouter):
        def build_order_spec(self, *a, **kw):
            return {}

        def submit(self, spec):
            return {}

        def build_exit_spec(self, *a, **kw):
            return {}

        def cancel(self, oid):
            pass

        def supports_native_brackets(self):
            return False

    router = StubRouter(account_id=1, auth=None)
    entry = {"type": 2, "side": 0}
    bracket = {"stopLossBracket": {"ticks": 10, "type": 4}}
    result = router.merge_bracket_into_entry(entry, bracket)
    # Base class returns entry unchanged — no bracket merged
    assert result == {"type": 2, "side": 0}
    assert "stopLossBracket" not in result
