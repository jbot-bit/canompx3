"""ProjectX real-time market data via SignalR.

Connects to the Market Hub, subscribes to contract quotes,
aggregates into 1-minute OHLCV bars via BarAggregator.

Market Hub URL: https://rtc.thefuturesdesk.projectx.com/hubs/market
Subscribe: invoke SubscribeContractQuotes with contract ID
Events: GatewayQuote (lastPrice, bestBid, bestAsk, volume),
        GatewayTrade (price, volume, type)
Auth: JWT passed as access_token_factory to pysignalr.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..bar_aggregator import Bar, BarAggregator
from ..broker_base import BrokerAuth, BrokerFeed

log = logging.getLogger(__name__)

MARKET_HUB = "https://rtc.thefuturesdesk.projectx.com/hubs/market"

# Stop-file for graceful Windows shutdown (same as Tradovate)
_STOP_FILE = Path(__file__).parent.parent.parent.parent / "live_session.stop"

# Reconnect settings
_MAX_RECONNECTS = 20
_BACKOFF_INITIAL = 5.0  # seconds before first retry
_BACKOFF_MAX = 60.0  # cap at 60s


class ProjectXDataFeed(BrokerFeed):
    """Stream ProjectX quotes -> 1-minute bars -> on_bar async callback.

    Uses pysignalr to connect to the ProjectX Market Hub SignalR endpoint.
    Falls back to signalrcore if pysignalr is unavailable.

    Usage:
        feed = ProjectXDataFeed(auth, on_bar=my_async_callback)
        await feed.run("12345")  # contract ID
    """

    def __init__(
        self,
        auth: BrokerAuth,
        on_bar,
        **kwargs,
    ):
        super().__init__(auth, on_bar, **kwargs)
        self._agg = BarAggregator()
        self._symbol: str = ""
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def parse_quote(self, quote: dict) -> tuple[float, int]:
        """Extract price and volume from a GatewayQuote message.

        Falls back to bestBid/bestAsk if lastPrice is absent.
        Raises ValueError when no price field is found.
        """
        price = quote.get("lastPrice")
        if price is None:
            price = quote.get("bestBid") or quote.get("bestAsk")
        if price is None:
            raise ValueError(f"No price in quote: {quote}")
        vol = quote.get("volume", 1)
        return float(price), int(vol) if vol else 1

    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end."""
        bar = self._agg.flush()
        if bar is not None:
            bar.symbol = symbol or self._symbol
        return bar

    # ------------------------------------------------------------------
    # run() entry point — tries pysignalr, falls back to signalrcore
    # ------------------------------------------------------------------

    async def run(self, symbol: str) -> None:
        """Connect to ProjectX Market Hub and stream bars.

        Args:
            symbol: ProjectX contract ID (numeric string or int).
        """
        self._symbol = symbol
        self._stop_requested = False

        try:
            await self._run_pysignalr(symbol)
        except ImportError:
            log.info("pysignalr not available, trying signalrcore")
            await self._run_signalrcore(symbol)

    # ------------------------------------------------------------------
    # pysignalr backend
    # ------------------------------------------------------------------

    async def _run_pysignalr(self, symbol: str) -> None:
        from pysignalr.client import SignalRClient

        backoff = _BACKOFF_INITIAL

        for attempt in range(_MAX_RECONNECTS + 1):
            if self._stop_requested:
                return

            try:
                log.info(
                    "Connecting to ProjectX Market Hub (pysignalr, attempt %d)",
                    attempt + 1,
                )

                client = SignalRClient(
                    MARKET_HUB,
                    access_token_factory=lambda: self.auth.get_token(),
                    headers={"Accept": "text/plain"},
                )

                client.on("GatewayQuote", self._on_quote)
                client.on("GatewayTrade", self._on_trade)
                client.on_open(lambda _c=client, _s=symbol: self._on_connected(_c, _s))

                # Run feed with stop-file watcher
                stop_task = asyncio.create_task(self._stop_file_watcher())
                try:
                    await client.run()
                finally:
                    stop_task.cancel()

                log.info("Feed closed cleanly for %s", symbol)
                return

            except ImportError:
                raise  # let caller fall back to signalrcore
            except Exception as e:
                log.warning("ProjectX feed error (pysignalr): %s", e)
                if attempt < _MAX_RECONNECTS:
                    log.info("Reconnecting in %.0fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX)
                else:
                    log.error(
                        "Max reconnects (%d) exhausted for %s",
                        _MAX_RECONNECTS,
                        symbol,
                    )

    # ------------------------------------------------------------------
    # signalrcore fallback backend
    # ------------------------------------------------------------------

    async def _run_signalrcore(self, symbol: str) -> None:
        from signalrcore.hub_connection_builder import HubConnectionBuilder

        backoff = _BACKOFF_INITIAL

        for attempt in range(_MAX_RECONNECTS + 1):
            if self._stop_requested:
                return

            try:
                log.info(
                    "Connecting to ProjectX Market Hub (signalrcore, attempt %d)",
                    attempt + 1,
                )

                token = self.auth.get_token()
                hub = (
                    HubConnectionBuilder()
                    .with_url(f"{MARKET_HUB}?access_token={token}")
                    .with_automatic_reconnect({"type": "interval", "intervals": [5, 10, 30, 60]})
                    .build()
                )

                hub.on("GatewayQuote", lambda args: self._on_quote_sync(args))
                hub.on("GatewayTrade", lambda args: self._on_trade_sync(args))
                hub.on_open(lambda: log.info("Connected to ProjectX Market Hub"))

                hub.start()
                hub.send("SubscribeContractQuotes", [symbol])
                log.info("Subscribed to quotes: %s", symbol)

                # Block until stop file
                while not _STOP_FILE.exists():
                    await asyncio.sleep(2.5)
                    self.auth.refresh_if_needed()

                log.info("Stop file detected — shutting down")
                _STOP_FILE.unlink(missing_ok=True)
                hub.stop()
                return

            except ImportError:
                raise
            except Exception as e:
                log.warning("ProjectX feed error (signalrcore): %s", e)
                if attempt < _MAX_RECONNECTS:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX)
                else:
                    log.error(
                        "Max reconnects (%d) exhausted for %s",
                        _MAX_RECONNECTS,
                        symbol,
                    )

    # ------------------------------------------------------------------
    # SignalR event handlers — pysignalr (async context)
    # ------------------------------------------------------------------

    def _on_connected(self, client, symbol: str) -> None:
        """Called when pysignalr connects — subscribe to contract quotes."""
        log.info("Connected to ProjectX Market Hub")
        asyncio.create_task(client.send("SubscribeContractQuotes", [symbol]))
        log.info("Subscribed to quotes: %s", symbol)

    def _on_quote(self, args: list[Any]) -> None:
        """Handle GatewayQuote event (pysignalr — called in async context)."""
        for quote in args if isinstance(args, list) else [args]:
            if not isinstance(quote, dict):
                continue
            try:
                price, vol = self.parse_quote(quote)
                bar = self._agg.on_tick(price, vol, datetime.now(UTC))
                if bar is not None:
                    bar.symbol = self._symbol
                    asyncio.create_task(self.on_bar(bar))
            except (ValueError, KeyError) as e:
                log.debug("Skipping quote: %s", e)

    def _on_trade(self, args: list[Any]) -> None:
        """Handle GatewayTrade event (pysignalr)."""
        for trade in args if isinstance(args, list) else [args]:
            if not isinstance(trade, dict):
                continue
            price = trade.get("price")
            vol = trade.get("volume", 1)
            if price is not None:
                bar = self._agg.on_tick(float(price), int(vol) if vol else 1, datetime.now(UTC))
                if bar is not None:
                    bar.symbol = self._symbol
                    asyncio.create_task(self.on_bar(bar))

    # ------------------------------------------------------------------
    # SignalR event handlers — signalrcore (synchronous callbacks)
    # ------------------------------------------------------------------

    def _on_quote_sync(self, args: Any) -> None:
        """Handle GatewayQuote event (signalrcore — synchronous callback)."""
        quotes = args if isinstance(args, list) else [args]
        for quote in quotes:
            if not isinstance(quote, dict):
                continue
            try:
                price, vol = self.parse_quote(quote)
                bar = self._agg.on_tick(price, vol, datetime.now(UTC))
                if bar is not None:
                    bar.symbol = self._symbol
                    # signalrcore is sync — can't await. Log instead.
                    log.info("BAR: %s", bar)
            except (ValueError, KeyError) as e:
                log.debug("Skipping quote: %s", e)

    def _on_trade_sync(self, args: Any) -> None:
        """Handle GatewayTrade (signalrcore — sync)."""
        trades = args if isinstance(args, list) else [args]
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            price = trade.get("price")
            vol = trade.get("volume", 1)
            if price is not None:
                bar = self._agg.on_tick(float(price), int(vol) if vol else 1, datetime.now(UTC))
                if bar is not None:
                    bar.symbol = self._symbol
                    log.info("BAR: %s", bar)

    # ------------------------------------------------------------------
    # Stop-file watcher
    # ------------------------------------------------------------------

    async def _stop_file_watcher(self) -> None:
        """Watch for stop file — used with pysignalr backend."""
        while True:
            await asyncio.sleep(2.5)
            if _STOP_FILE.exists():
                log.info("Stop file detected — requesting graceful shutdown")
                _STOP_FILE.unlink(missing_ok=True)
                self._stop_requested = True
                return
