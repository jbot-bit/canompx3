"""Tests for price collar on order routers."""

from unittest.mock import MagicMock, patch

import pytest

from trading_app.live.tradovate.order_router import OrderSpec, TradovateOrderRouter


class TestTradovatePriceCollar:
    def _make_router(self, collar_pct=0.005):
        return TradovateOrderRouter(
            account_id=12345,
            auth=MagicMock(),
            demo=True,
            price_collar_pct=collar_pct,
        )

    def test_collar_rejects_distant_stop(self):
        """Stop price 2% from market should be rejected (default collar 0.5%)."""
        router = self._make_router()
        router.update_market_price(20000.0)
        spec = OrderSpec(
            action="Buy",
            order_type="Stop",
            symbol="MNQM6",
            qty=1,
            account_id=12345,
            stop_price=20500.0,  # 2.5% away
        )
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)

    def test_collar_allows_close_stop(self):
        """Stop price 0.1% from market should pass collar."""
        router = self._make_router()
        router.update_market_price(20000.0)
        spec = OrderSpec(
            action="Buy",
            order_type="Stop",
            symbol="MNQM6",
            qty=1,
            account_id=12345,
            stop_price=20020.0,  # 0.1%
        )
        # Will fail on HTTP (no real broker) but should NOT fail on collar
        with pytest.raises(Exception) as exc_info:
            router.submit(spec)
        assert "PRICE_COLLAR" not in str(exc_info.value)

    def test_collar_skips_market_orders(self):
        """Exit (market) orders should never be collared."""
        router = self._make_router()
        router.update_market_price(20000.0)
        spec = OrderSpec(
            action="Sell",
            order_type="Market",
            symbol="MNQM6",
            qty=1,
            account_id=12345,
            stop_price=None,
        )
        # Will fail on HTTP but should NOT fail on collar
        with pytest.raises(Exception) as exc_info:
            router.submit(spec)
        assert "PRICE_COLLAR" not in str(exc_info.value)

    def test_collar_skips_when_no_market_price(self):
        """If no market price known yet, collar cannot check — allow through."""
        router = self._make_router()
        # Do NOT call update_market_price
        spec = OrderSpec(
            action="Buy",
            order_type="Stop",
            symbol="MNQM6",
            qty=1,
            account_id=12345,
            stop_price=20500.0,
        )
        # Will fail on HTTP but NOT on collar
        with pytest.raises(Exception) as exc_info:
            router.submit(spec)
        assert "PRICE_COLLAR" not in str(exc_info.value)

    def test_update_market_price(self):
        router = self._make_router()
        assert router._last_known_price is None
        router.update_market_price(20000.0)
        assert router._last_known_price == 20000.0
        router.update_market_price(0.0)  # zero ignored
        assert router._last_known_price == 20000.0

    def test_custom_collar_pct(self):
        """Custom collar of 1% should allow 0.8% deviation."""
        router = self._make_router(collar_pct=0.01)
        router.update_market_price(20000.0)
        spec = OrderSpec(
            action="Buy",
            order_type="Stop",
            symbol="MNQM6",
            qty=1,
            account_id=12345,
            stop_price=20160.0,  # 0.8%
        )
        # Should pass collar (0.8% < 1.0%)
        with pytest.raises(Exception) as exc_info:
            router.submit(spec)
        assert "PRICE_COLLAR" not in str(exc_info.value)


class TestProjectXPriceCollar:
    def test_collar_rejects_distant_stop(self):
        from trading_app.live.projectx.order_router import ProjectXOrderRouter

        router = ProjectXOrderRouter(account_id=12345, auth=MagicMock(), tick_size=0.25)
        router.update_market_price(20000.0)
        spec = {
            "accountId": 12345,
            "contractId": "MNQM6",
            "type": 4,
            "side": 0,
            "size": 1,
            "stopPrice": 20500.0,  # 2.5%
        }
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)

    def test_collar_skips_market_exit(self):
        from trading_app.live.projectx.order_router import ProjectXOrderRouter

        router = ProjectXOrderRouter(account_id=12345, auth=MagicMock(), tick_size=0.25)
        router.update_market_price(20000.0)
        spec = {
            "accountId": 12345,
            "contractId": "MNQM6",
            "type": 2,  # Market
            "side": 1,
            "size": 1,
        }
        # No stopPrice -> no collar check -> fails on HTTP, not collar
        with pytest.raises(Exception) as exc_info:
            router.submit(spec)
        assert "PRICE_COLLAR" not in str(exc_info.value)
