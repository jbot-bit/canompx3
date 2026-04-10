"""ProjectX real-time market data via SignalR.

Connects to the Market Hub, subscribes to contract quotes,
aggregates into 1-minute OHLCV bars via BarAggregator.

Market Hub URL: configurable via PROJECTX_BASE_URL env var
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
from .auth import MARKET_HUB_URL

log = logging.getLogger(__name__)

# Stop-file for graceful Windows shutdown (same as Tradovate)
_STOP_FILE = Path(__file__).parent.parent.parent.parent / "live_session.stop"

# Reconnect settings
_MAX_RECONNECTS = 20
_BACKOFF_INITIAL = 5.0  # seconds before first retry
_BACKOFF_MAX = 60.0  # cap at 60s

# Liveness monitoring — detect "connected but silent" state
_STALE_TIMEOUT = 90.0  # seconds with no data before first warning
_MAX_STALE_BEFORE_RECONNECT = 2  # consecutive stale checks before forcing reconnect


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
        self._force_reconnect = False
        self._bar_queue: asyncio.Queue = asyncio.Queue()
        # Liveness tracking
        self._last_data_at: datetime | None = None
        self._stale_count: int = 0
        self._quote_count: int = 0

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
        error_attempts = 0  # only ERROR reconnects count toward budget (not stale)

        while error_attempts <= _MAX_RECONNECTS:
            if self._stop_requested:
                return

            try:
                log.info(
                    "Connecting to ProjectX Market Hub (pysignalr, errors=%d/%d)",
                    error_attempts,
                    _MAX_RECONNECTS,
                )

                token = self.auth.get_token()
                url = f"{MARKET_HUB_URL}?access_token={token}"
                client = SignalRClient(
                    url,
                    access_token_factory=lambda: self.auth.get_token(),
                    headers={"Accept": "text/plain"},
                )

                client.on("GatewayQuote", self._on_quote)
                client.on("GatewayTrade", self._on_trade)
                client.on_open(lambda _c=client, _s=symbol: self._on_connected_async(_c, _s))

                # Race client.run() against stop-file watcher
                run_task = asyncio.create_task(client.run())
                stop_task = asyncio.create_task(self._stop_file_watcher())
                try:
                    await asyncio.wait(
                        [run_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for t in (run_task, stop_task):
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass

                if self._stop_requested:
                    log.info("Feed stopped cleanly via stop flag for %s", symbol)
                    return
                if self._force_reconnect:
                    log.warning("Feed stale — forcing reconnect for %s", symbol)
                    self._force_reconnect = False
                    self._stale_count = 0
                    self._last_data_at = None  # prevent immediate re-trigger
                    backoff = _BACKOFF_INITIAL
                    # stale reconnect does NOT increment error_attempts
                    continue
                log.info("Feed closed cleanly for %s", symbol)
                return

            except ImportError:
                raise  # let caller fall back to signalrcore
            except Exception as e:
                error_attempts += 1
                log.warning("ProjectX feed error (pysignalr, %d/%d): %s", error_attempts, _MAX_RECONNECTS, e)
                if error_attempts <= _MAX_RECONNECTS:
                    log.info("Reconnecting in %.0fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX)

        # Exhausted error budget
        log.critical("FEED DEAD: max error reconnects (%d) exhausted for %s", _MAX_RECONNECTS, symbol)
        if self.on_stale is not None:
            try:
                self.on_stale(0.0, -1)
            except Exception:
                log.exception("FEED DEAD notification failed")

    # ------------------------------------------------------------------
    # signalrcore fallback backend
    # ------------------------------------------------------------------

    async def _run_signalrcore(self, symbol: str) -> None:
        from signalrcore.hub_connection_builder import HubConnectionBuilder

        backoff = _BACKOFF_INITIAL
        error_attempts = 0  # only ERROR reconnects count toward budget (not stale)

        # R2-C2: capture event loop for thread-safe callback bridging
        self._loop = asyncio.get_running_loop()

        while error_attempts <= _MAX_RECONNECTS:
            if self._stop_requested:
                return

            try:
                log.info(
                    "Connecting to ProjectX Market Hub (signalrcore, errors=%d/%d)",
                    error_attempts,
                    _MAX_RECONNECTS,
                )

                token = self.auth.get_token()
                hub = (
                    HubConnectionBuilder()
                    .with_url(f"{MARKET_HUB_URL}?access_token={token}")
                    .with_automatic_reconnect({"type": "interval", "intervals": [5, 10, 30, 60]})
                    .build()
                )

                hub.on("GatewayQuote", lambda args: self._on_quote_sync(args))
                hub.on("GatewayTrade", lambda args: self._on_trade_sync(args))
                # Check 10 fix: re-subscribe after automatic reconnect.
                # The spec requires re-invoking Subscribe* methods after reconnect.
                # Without this, signalrcore's automatic reconnect restores the WebSocket
                # but the quote subscription is lost → silent data loss.
                hub.on_open(
                    lambda _hub=hub: (
                        log.info("Connected to ProjectX Market Hub — subscribing to %s", symbol),
                        _hub.send("SubscribeContractQuotes", [symbol]),
                    )
                )

                hub.start()
                hub.send("SubscribeContractQuotes", [symbol])
                log.info("Subscribed to quotes: %s", symbol)

                # Start async consumer for bars from sync callbacks
                drain_task = asyncio.create_task(self._drain_bar_queue())
                _stale_break = False
                try:
                    while not _STOP_FILE.exists():
                        await asyncio.sleep(2.5)
                        await asyncio.get_running_loop().run_in_executor(
                            None, self.auth.refresh_if_needed
                        )
                        # Liveness check (same logic as pysignalr watcher)
                        if self._last_data_at is not None:
                            gap = (datetime.now(UTC) - self._last_data_at).total_seconds()
                            if gap > _STALE_TIMEOUT:
                                self._stale_count += 1
                                log.warning(
                                    "LIVENESS: %.0fs since last data (%d quotes, stale %d/%d)",
                                    gap,
                                    self._quote_count,
                                    self._stale_count,
                                    _MAX_STALE_BEFORE_RECONNECT,
                                )
                                if self.on_stale is not None:
                                    try:
                                        self.on_stale(gap, self._stale_count)
                                    except Exception:
                                        log.exception("on_stale callback error")
                                if self._stale_count >= _MAX_STALE_BEFORE_RECONNECT:
                                    log.critical("FEED STALE: forcing reconnect (signalrcore)")
                                    hub.stop()
                                    self._stale_count = 0
                                    self._last_data_at = None
                                    _stale_break = True
                                    break  # break inner loop → reconnect via outer loop
                            else:
                                self._stale_count = 0
                    else:
                        # while condition was False — stop file detected
                        log.info("Stop file detected — shutting down")
                        hub.stop()
                        return
                finally:
                    drain_task.cancel()

                if _stale_break:
                    # Stale reconnect does NOT increment error_attempts
                    backoff = _BACKOFF_INITIAL
                    continue
                log.info("Feed closed cleanly for %s", symbol)
                return

            except ImportError:
                raise
            except Exception as e:
                error_attempts += 1
                log.warning("ProjectX feed error (signalrcore, %d/%d): %s", error_attempts, _MAX_RECONNECTS, e)
                if error_attempts <= _MAX_RECONNECTS:
                    log.info("Reconnecting in %.0fs...", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX)

        # Exhausted error budget
        log.critical("FEED DEAD: max error reconnects (%d) exhausted for %s", _MAX_RECONNECTS, symbol)
        if self.on_stale is not None:
            try:
                self.on_stale(0.0, -1)
            except Exception:
                log.exception("FEED DEAD notification failed")

    # ------------------------------------------------------------------
    # Sync → async queue bridge
    # ------------------------------------------------------------------

    async def _drain_bar_queue(self) -> None:
        """Consume bars from the sync→async queue bridge."""
        while True:
            bar = await self._bar_queue.get()
            try:
                await self.on_bar(bar)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("on_bar error in drain queue — bar dropped, drain continues")

    # ------------------------------------------------------------------
    # SignalR event handlers — pysignalr (async context)
    # ------------------------------------------------------------------

    async def _on_connected_async(self, client, symbol: str) -> None:
        """Called when pysignalr connects — subscribe to contract quotes."""
        log.info("Connected to ProjectX Market Hub")
        await client.send("SubscribeContractQuotes", [symbol])
        log.info("Subscribed to quotes: %s", symbol)

    async def _on_quote(self, args: list[Any]) -> None:
        """Handle GatewayQuote event (pysignalr — async callback)."""
        for quote in args if isinstance(args, list) else [args]:
            if not isinstance(quote, dict):
                continue
            try:
                price, vol = self.parse_quote(quote)
                now = datetime.now(UTC)
                self._last_data_at = now
                self._quote_count += 1
                bar = self._agg.on_tick(price, vol, now)
                if bar is not None:
                    bar.symbol = self._symbol
                    await self.on_bar(bar)
            except (ValueError, KeyError) as e:
                log.debug("Skipping quote: %s", e)

    async def _on_trade(self, args: list[Any]) -> None:
        """Handle GatewayTrade event (pysignalr — async callback)."""
        for trade in args if isinstance(args, list) else [args]:
            if not isinstance(trade, dict):
                continue
            price = trade.get("price")
            vol = trade.get("volume", 1)
            if price is not None:
                now = datetime.now(UTC)
                self._last_data_at = now
                self._quote_count += 1
                bar = self._agg.on_tick(float(price), int(vol) if vol else 1, now)
                if bar is not None:
                    bar.symbol = self._symbol
                    await self.on_bar(bar)

    # ------------------------------------------------------------------
    # SignalR event handlers — signalrcore (synchronous callbacks)
    # ------------------------------------------------------------------

    def _on_quote_sync(self, args: Any) -> None:
        """Handle GatewayQuote event (signalrcore — synchronous callback on foreign thread).

        R2-C1/C2: signalrcore fires this on its own thread. All state mutations
        and queue operations are routed through the asyncio event loop via
        call_soon_threadsafe to prevent cross-thread data corruption.
        BarAggregator.on_tick() is additionally protected by its own threading.Lock.
        """
        quotes = args if isinstance(args, list) else [args]
        for quote in quotes:
            if not isinstance(quote, dict):
                continue
            try:
                price, vol = self.parse_quote(quote)
                now = datetime.now(UTC)
                # on_tick is thread-safe (has internal lock)
                bar = self._agg.on_tick(price, vol, now)
                # Route state updates and queue puts through event loop thread
                loop = getattr(self, "_loop", None)
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._apply_tick_state, now, bar)
                else:
                    # Fallback: direct update (acceptable if loop is gone)
                    self._apply_tick_state(now, bar)
            except (ValueError, KeyError) as e:
                log.debug("Skipping quote: %s", e)

    def _on_trade_sync(self, args: Any) -> None:
        """Handle GatewayTrade (signalrcore — sync on foreign thread).

        R2-C1/C2: same thread-safety pattern as _on_quote_sync.
        """
        trades = args if isinstance(args, list) else [args]
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            price = trade.get("price")
            vol = trade.get("volume", 1)
            if price is not None:
                now = datetime.now(UTC)
                bar = self._agg.on_tick(float(price), int(vol) if vol else 1, now)
                loop = getattr(self, "_loop", None)
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._apply_tick_state, now, bar)
                else:
                    self._apply_tick_state(now, bar)

    def _apply_tick_state(self, now: datetime, bar: "Bar | None") -> None:
        """Apply tick state updates on the event loop thread (thread-safe target).

        Called via call_soon_threadsafe from signalrcore callbacks.
        Updates liveness tracking and enqueues completed bars.
        """
        self._last_data_at = now
        self._quote_count += 1
        if bar is not None:
            bar.symbol = self._symbol
            log.info("BAR: %s", bar)
            try:
                self._bar_queue.put_nowait(bar)
            except Exception:
                log.error("Bar queue full — dropping bar %s", bar.ts_utc)

    # ------------------------------------------------------------------
    # Stop-file watcher
    # ------------------------------------------------------------------

    async def _stop_file_watcher(self) -> None:
        """Watch for stop file + feed liveness — used with pysignalr backend."""
        while True:
            await asyncio.sleep(2.5)
            # Stop-file check (priority: stop overrides reconnect)
            if _STOP_FILE.exists():
                log.info("Stop file detected — requesting graceful shutdown")
                self._stop_requested = True
                return
            # Liveness check: detect "connected but silent" state
            if self._last_data_at is not None:
                gap = (datetime.now(UTC) - self._last_data_at).total_seconds()
                if gap > _STALE_TIMEOUT:
                    self._stale_count += 1
                    log.warning(
                        "LIVENESS: %.0fs since last data (%d quotes total, stale check %d/%d)",
                        gap,
                        self._quote_count,
                        self._stale_count,
                        _MAX_STALE_BEFORE_RECONNECT,
                    )
                    if self.on_stale is not None:
                        try:
                            self.on_stale(gap, self._stale_count)
                        except Exception:
                            log.exception("on_stale callback error — continuing liveness monitoring")
                    if self._stale_count >= _MAX_STALE_BEFORE_RECONNECT:
                        log.critical(
                            "FEED STALE: %d consecutive stale checks — forcing reconnect",
                            self._stale_count,
                        )
                        self._force_reconnect = True
                        return  # break out of watcher → triggers reconnect in _run_pysignalr
                else:
                    self._stale_count = 0  # reset on any data
