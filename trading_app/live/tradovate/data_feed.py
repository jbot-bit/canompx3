"""
Tradovate market data WebSocket feed.

Connects, subscribes to real-time quotes, aggregates into 1-minute OHLCV bars
via BarAggregator, then calls on_bar(bar) callback for each completed bar.

Heartbeat every 2.5s required per Tradovate docs.
Reconnects automatically on disconnect with exponential backoff (5s -> 60s max).

Stop-file: create live_session.stop in project root for graceful shutdown.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import websockets

from ..bar_aggregator import Bar, BarAggregator
from ..broker_base import BrokerAuth, BrokerFeed

log = logging.getLogger(__name__)

MD_WS_LIVE = "wss://md.tradovateapi.com/v1/websocket"
MD_WS_DEMO = "wss://md.tradovateapi.com/v1/websocket"

# Reconnect settings
_BACKOFF_INITIAL = 5.0  # seconds before first retry
_BACKOFF_MAX = 60.0  # cap at 60s
_MAX_RECONNECTS = 20  # give up after this many consecutive failures

# Stop-file for graceful Windows shutdown (proc.terminate() is a hard kill on Windows)
_STOP_FILE = Path(__file__).parent.parent.parent.parent / "live_session.stop"


class TradovateDataFeed(BrokerFeed):
    """
    Streams Tradovate quotes -> 1-minute bars -> on_bar async callback.

    on_bar must be an async coroutine: ``async def on_bar(bar: Bar) -> None``.
    This allows the event loop to remain responsive (heartbeat, stop-file check)
    even when on_bar awaits blocking I/O (e.g. order submission via run_in_executor).

    Reconnects automatically on WebSocket disconnect.

    Usage:
        feed = TradovateDataFeed(auth, on_bar=my_async_callback, demo=True)
        await feed.run("MGCM6")
    """

    def __init__(
        self,
        auth: BrokerAuth,
        on_bar: Callable[[Bar], Coroutine[Any, Any, None]],
        demo: bool = True,
    ):
        super().__init__(auth, on_bar)
        self.demo = demo
        self._agg = BarAggregator()
        self._heartbeat_task: asyncio.Task | None = None
        self._stop_requested = False

    async def run(self, symbol: str) -> None:
        """Connect and stream bars. Reconnects on disconnect up to _MAX_RECONNECTS times."""
        self._stop_requested = False
        url = MD_WS_DEMO if self.demo else MD_WS_LIVE
        backoff = _BACKOFF_INITIAL

        for attempt in range(_MAX_RECONNECTS + 1):
            try:
                log.info("Connecting to %s for %s (attempt %d)", url, symbol, attempt + 1)
                async with websockets.connect(url, ping_interval=None) as ws:
                    backoff = _BACKOFF_INITIAL  # reset on successful connect
                    await self._session(ws, symbol)
                    log.info("Feed closed cleanly for %s", symbol)
                    return

            except websockets.exceptions.ConnectionClosedError as e:
                log.warning("WebSocket connection dropped: %s", e)
            except websockets.exceptions.ConnectionClosedOK:
                log.info("WebSocket closed normally")
                return
            except OSError as e:
                log.error("Network error: %s", e)
            except Exception as e:
                log.error("Unexpected feed error: %s", e, exc_info=True)

            if attempt < _MAX_RECONNECTS:
                log.info("Reconnecting in %.0fs...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)
            else:
                log.error("Max reconnects (%d) exhausted for %s — giving up", _MAX_RECONNECTS, symbol)

    async def _session(self, ws, symbol: str) -> None:
        """Run a single authenticated WebSocket session."""
        # Auth handshake
        await ws.send(
            json.dumps(
                {
                    "url": "auth/accesstokenrequest",
                    "body": {"token": self.auth.get_token()},
                }
            )
        )
        resp_raw = await ws.recv()
        try:
            resp = json.loads(resp_raw) if resp_raw and resp_raw != "[]" else {}
        except json.JSONDecodeError:
            resp = {}
        # Tradovate returns either a dict or an array; unwrap arrays
        if isinstance(resp, list) and resp:
            resp = resp[0]
        status = resp.get("s") if isinstance(resp, dict) else None
        if status is None:
            raise RuntimeError(f"Tradovate auth returned no status field: {resp_raw[:200]}")
        elif status not in (200,):
            raise RuntimeError(f"Tradovate auth failed: {resp}")
        log.info("Authenticated to Tradovate MD feed")

        # Subscribe to real-time quotes
        await ws.send(
            json.dumps(
                {
                    "url": "md/subscribeQuote",
                    "body": {"symbol": symbol},
                }
            )
        )
        log.info("Subscribed to quotes: %s", symbol)

        # Stop event — set by heartbeat when stop-file is detected
        stop_event = asyncio.Event()

        # Heartbeat task (required every 2.5s per Tradovate docs)
        async def heartbeat():
            while not stop_event.is_set():
                await asyncio.sleep(2.5)
                if _STOP_FILE.exists():
                    log.info("Stop file detected — requesting graceful shutdown")
                    _STOP_FILE.unlink(missing_ok=True)
                    self._stop_requested = True
                    stop_event.set()
                    return
                try:
                    await ws.send("[]")
                except Exception as exc:
                    log.warning("Heartbeat send failed: %s", exc)
                    break

        self._heartbeat_task = asyncio.create_task(heartbeat())
        try:
            async for message in ws:
                if stop_event.is_set():
                    break
                if not message or message == "[]":
                    continue
                try:
                    frames = json.loads(message)
                except json.JSONDecodeError:
                    continue
                for frame in frames if isinstance(frames, list) else [frames]:
                    await self._handle_frame(frame, symbol)
        finally:
            self._heartbeat_task.cancel()

    async def _handle_frame(self, frame: dict, symbol: str) -> None:
        if not isinstance(frame, dict):
            return
        for q in frame.get("d", {}).get("quotes", []):
            price = q.get("bidPrice")
            if price is None:
                price = q.get("price")
            if price is None:
                continue
            bar = self._agg.on_tick(float(price), 1, datetime.now(UTC))
            if bar is not None:
                bar.symbol = symbol
                await self.on_bar(bar)

    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end."""
        bar = self._agg.flush()
        if bar is not None:
            bar.symbol = symbol
        return bar
