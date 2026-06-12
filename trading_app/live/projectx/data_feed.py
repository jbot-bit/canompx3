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
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..bar_aggregator import Bar, BarAggregator
from ..broker_base import BrokerAuth, BrokerFeed
from ..spread_accumulator import SpreadAccumulator
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

# Spread capture (Defect B) — off unless this env flag is truthy.
_SPREAD_CAPTURE_ENV = "CANOMPX_CAPTURE_SPREAD"


def _spread_capture_enabled() -> bool:
    """True iff CANOMPX_CAPTURE_SPREAD is set to a truthy value (default OFF)."""
    return os.environ.get(_SPREAD_CAPTURE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


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
        on_quote_minute=None,
        **kwargs,
    ):
        super().__init__(auth, on_bar, **kwargs)
        self._agg = BarAggregator()
        self._symbol: str = ""
        # Spread capture (Defect B) — OFF by default. When CANOMPX_CAPTURE_SPREAD
        # is set, every quote's (bestBid, bestAsk) is accumulated into per-minute
        # spread summaries and emitted via on_quote_minute. Runs in its own
        # try/except branch after the bar/order path and CANNOT perturb trading.
        # When off, neither object exists and no spread branch is taken — the
        # capital path is byte-identical to a feed without this feature.
        self._capture_spread_enabled: bool = _spread_capture_enabled()
        self._spread_acc: SpreadAccumulator | None = SpreadAccumulator() if self._capture_spread_enabled else None
        # Plain synchronous callback (QuotePersister.append is lock-guarded), so
        # it is safe to invoke from both the async and foreign-thread quote paths.
        self._on_quote_minute = on_quote_minute
        self._stop_requested = False
        self._force_reconnect = False
        self._bar_queue: asyncio.Queue = asyncio.Queue()
        # Liveness tracking
        self._last_data_at: datetime | None = None
        self._stale_count: int = 0
        self._quote_count: int = 0
        # Cumulative-volume → per-tick delta conversion (Defect A fix, 2026-06-10).
        # GatewayQuote.volume is the contract's running session-cumulative total, NOT a
        # per-quote delta. BarAggregator's contract is "volume arg = per-tick delta", so
        # we diff consecutive cumulative readings here at the ProjectX boundary before
        # feeding on_tick. The lock guards _last_cum_volume against the two-thread reality
        # of this feed (pysignalr fires _on_quote on the loop thread; signalrcore fires
        # _on_quote_sync on a foreign thread). See
        # docs/audit/2026-06-10-data-pipeline-gap-report.md Defect A.
        self._last_cum_volume: int | None = None
        self._cum_volume_lock = threading.Lock()
        # DIAGNOSTIC (2026-06-13, temporary): the live MNQ feed oscillates the
        # cumulative-volume field between the true session total (~1.28M) and a
        # `1` sentinel, ping-ponging _cum_to_delta's reset branch against the
        # aggregator's per-tick cap. To learn WHAT the `1` is (lastSize vs a real
        # volume field), _cum_to_delta sets _last_reset_fired when it re-baselines
        # on a negative delta; the caller (which has the raw quote dict) logs the
        # full quote, rate-limited via _last_reset_log_ts. Remove once the feed
        # schema is confirmed and the delta fix lands. Guarded by _cum_volume_lock.
        self._last_reset_fired: bool = False
        self._last_reset_log_ts: float = 0.0

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
        # volume MUST distinguish explicit 0 from a missing field. Under cumulative
        # semantics (see _cum_to_delta), an explicit 0 is a real zero-cumulative reading
        # (pre-open / no trades yet) and must pass through as 0 so the delta baseline is
        # correct — coercing 0→1 silently corrupts the first real delta by one contract.
        # A MISSING volume field (price-only bestBid/bestAsk quote) defaults to 1.
        vol = quote.get("volume")
        return float(price), int(vol) if vol is not None else 1

    @staticmethod
    def parse_bid_ask(quote: dict) -> tuple[float | None, float | None]:
        """Extract (bestBid, bestAsk) from a GatewayQuote — pure, no fallback.

        Returns (None, None)-tolerant tuple: a missing or non-numeric side
        becomes None so the accumulator's crossed/one-sided guard drops it.
        Spread capture only (Defect B); never feeds the price/bar path.
        """

        def _num(v: Any) -> float | None:
            if v is None or isinstance(v, bool):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return _num(quote.get("bestBid")), _num(quote.get("bestAsk"))

    def _capture_spread(self, quote: dict, now: datetime) -> None:
        """Feed one quote's bid/ask to the spread accumulator (Defect B).

        Shared by BOTH quote paths (_on_quote async + _on_quote_sync foreign
        thread). Self-contained try/except: a failure here must NEVER propagate
        into the bar/order path. No-op when spread capture is disabled.

        The accumulator is thread-safe (its own lock) and QuotePersister.append
        is lock-guarded, so this is safe to call from either thread.
        """
        acc = self._spread_acc
        if acc is None:
            return
        try:
            bid, ask = self.parse_bid_ask(quote)
            qm = acc.add(bid, ask, now)
            if qm is not None and self._on_quote_minute is not None:
                qm.symbol = self._symbol
                self._on_quote_minute(qm)
        except Exception:
            log.warning("Spread capture failed (trading unaffected)", exc_info=True)

    def _cum_to_delta(self, cum_volume: int) -> int:
        """Convert a cumulative session-volume reading into a per-tick delta.

        GatewayQuote.volume is the contract's running cumulative total. BarAggregator
        expects per-tick deltas (it sums them). This diffs consecutive readings so the
        aggregator's delta contract holds. Thread-safe via _cum_volume_lock.

        Edge cases (institutional-rigor §6 — no silent reset):
        - First reading (no baseline): set baseline, return 0. We cannot know the
          in-minute increment for the very first quote, and 0 is honest, not a guess.
        - delta < 0 (session reset or contract rollover re-bases the cumulative
          counter): re-baseline to the new reading, return 0, log at WARNING. Never
          emit negative volume.
        """
        with self._cum_volume_lock:
            if self._last_cum_volume is None:
                self._last_cum_volume = cum_volume
                return 0
            delta = cum_volume - self._last_cum_volume
            if delta < 0:
                log.warning(
                    "Cumulative volume reset detected (%d -> %d) — re-baselining, emitting 0",
                    self._last_cum_volume,
                    cum_volume,
                )
                self._last_cum_volume = cum_volume
                # DIAGNOSTIC (2026-06-13): mark that a reset just fired so the
                # caller logs the raw quote (it has the dict; we only see the int).
                self._last_reset_fired = True
                return 0
            self._last_cum_volume = cum_volume
            return delta

    def _quote_volume_delta(self, quote: dict, cum_vol: int) -> int:
        """Return the per-tick volume delta for a GatewayQuote.

        ProjectX sometimes emits price-only quote updates with no ``volume``
        field. Those are valid price ticks, but they are NOT cumulative-volume
        observations and must not reset the cumulative baseline. Treat them as
        zero-volume ticks so price OHLC stays live while volume stays honest.
        """
        if quote.get("volume") is None:
            return 0
        return self._cum_to_delta(cum_vol)

    def _maybe_log_raw_quote_on_reset(self, quote: dict) -> None:
        """DIAGNOSTIC (2026-06-13, temporary): log the full raw quote dict when a
        cumulative-volume reset just fired, rate-limited to ~1/10s.

        Called from BOTH quote paths (_on_quote async + _on_quote_sync foreign
        thread) so the capture has no parity gap. Reads/clears _last_reset_fired
        and reads/updates _last_reset_log_ts under _cum_volume_lock (same lock
        _cum_to_delta uses to set the flag). Logged at INFO so it surfaces under
        the live bot's INFO level (run_live_session.py). No-op when no reset fired
        or when within the rate-limit window. Remove with the delta fix.
        """
        with self._cum_volume_lock:
            if not self._last_reset_fired:
                return
            self._last_reset_fired = False
            now = time.monotonic()
            if now - self._last_reset_log_ts < 10.0:
                return
            self._last_reset_log_ts = now
        # Log outside the lock — formatting a dict shouldn't hold the feed lock.
        log.info("RAW QUOTE on cumulative-volume reset (diagnostic): %r", quote)

    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end.

        Also force-closes the spread accumulator's in-progress minute (Defect B)
        so the final partial minute of the session is not lost — mirrors the bar
        flush. The QuoteMinute is emitted via on_quote_minute; failures here
        never affect the returned bar (own try/except).
        """
        bar = self._agg.flush()
        if bar is not None:
            bar.symbol = symbol or self._symbol
        if self._spread_acc is not None:
            try:
                qm = self._spread_acc.flush(symbol or self._symbol)
                if qm is not None and self._on_quote_minute is not None:
                    self._on_quote_minute(qm)
            except Exception:
                log.warning("Spread accumulator flush failed (trading unaffected)", exc_info=True)
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
                        await asyncio.get_running_loop().run_in_executor(None, self.auth.refresh_if_needed)
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
                price, cum_vol = self.parse_quote(quote)
                vol = self._quote_volume_delta(quote, cum_vol)
                self._maybe_log_raw_quote_on_reset(quote)
                now = datetime.now(UTC)
                self._last_data_at = now
                self._quote_count += 1
                bar = self._agg.on_tick(price, vol, now)
                if bar is not None:
                    bar.symbol = self._symbol
                    await self.on_bar(bar)
            except (ValueError, KeyError) as e:
                log.debug("Skipping quote: %s", e)
            # Spread capture runs AFTER the bar/order path, in its own branch
            # inside _capture_spread (Defect B). Outside the except above so a
            # price-parse failure does not skip a capturable bid/ask quote; its
            # own try/except means it can never raise into this handler.
            if self._spread_acc is not None:
                self._capture_spread(quote, datetime.now(UTC))

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
                price, cum_vol = self.parse_quote(quote)
                # _quote_volume_delta is thread-safe through _cum_to_delta's internal
                # lock when a real cumulative field is present. Missing-volume quote
                # updates are price-only and must not perturb that baseline.
                vol = self._quote_volume_delta(quote, cum_vol)
                # _maybe_log_raw_quote_on_reset is thread-safe (same lock).
                self._maybe_log_raw_quote_on_reset(quote)
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
            # Spread capture (Defect B) — runs on this foreign thread directly:
            # SpreadAccumulator.add and QuotePersister.append are both
            # lock-guarded, so no event-loop hop is needed (unlike the bar
            # path, which must route the queue put through the loop). Own
            # try/except inside _capture_spread — cannot raise into this handler.
            if self._spread_acc is not None:
                self._capture_spread(quote, datetime.now(UTC))

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
