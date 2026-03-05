"""
Tradovate market data WebSocket feed.

Connects, subscribes to real-time quotes, aggregates into 1-minute OHLCV bars
via BarAggregator, then calls on_bar(bar) callback for each completed bar.

Heartbeat every 2.5s required per Tradovate docs.
Reconnects automatically on disconnect with exponential backoff (5s → 60s max).
"""
import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone

import websockets

from .bar_aggregator import Bar, BarAggregator
from .tradovate_auth import TradovateAuth

log = logging.getLogger(__name__)

MD_WS_LIVE = "wss://md.tradovate.com/v1/websocket"
MD_WS_DEMO = "wss://md-demo.tradovate.com/v1/websocket"

# Reconnect settings
_BACKOFF_INITIAL = 5.0   # seconds before first retry
_BACKOFF_MAX = 60.0      # cap at 60s
_MAX_RECONNECTS = 20     # give up after this many consecutive failures


class DataFeed:
    """
    Streams Tradovate quotes → 1-minute bars → on_bar callback.

    Reconnects automatically on WebSocket disconnect.

    Usage:
        feed = DataFeed(auth, on_bar=my_callback, demo=True)
        await feed.run("MGCM6")
    """

    def __init__(self, auth: TradovateAuth, on_bar: Callable[[Bar], None], demo: bool = True):
        self.auth = auth
        self.on_bar = on_bar
        self.demo = demo
        self._agg = BarAggregator()
        self._heartbeat_task: asyncio.Task | None = None

    async def run(self, symbol: str) -> None:
        """Connect and stream bars. Reconnects on disconnect up to _MAX_RECONNECTS times."""
        url = MD_WS_DEMO if self.demo else MD_WS_LIVE
        backoff = _BACKOFF_INITIAL

        for attempt in range(_MAX_RECONNECTS + 1):
            try:
                log.info("Connecting to %s for %s (attempt %d)", url, symbol, attempt + 1)
                async with websockets.connect(url, ping_interval=None) as ws:
                    backoff = _BACKOFF_INITIAL  # reset on successful connect
                    await self._session(ws, symbol)
                    # _session returns normally only if the feed is cleanly closed
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
        await ws.send(json.dumps({
            "url": "auth/accesstokenrequest",
            "body": {"token": self.auth.get_token()},
        }))
        resp_raw = await ws.recv()
        resp = json.loads(resp_raw) if resp_raw and resp_raw != "[]" else {}
        if isinstance(resp, dict) and resp.get("s") not in (200, None):
            raise RuntimeError(f"Tradovate auth failed: {resp}")
        log.info("Authenticated to Tradovate MD feed")

        # Subscribe to real-time quotes
        await ws.send(json.dumps({
            "url": "md/subscribeQuote",
            "body": {"symbol": symbol},
        }))
        log.info("Subscribed to quotes: %s", symbol)

        # Heartbeat task (required every 2.5s per Tradovate docs)
        async def heartbeat():
            while True:
                await asyncio.sleep(2.5)
                try:
                    await ws.send("[]")
                except Exception:
                    break

        self._heartbeat_task = asyncio.create_task(heartbeat())
        try:
            async for message in ws:
                if not message or message == "[]":
                    continue
                try:
                    frames = json.loads(message)
                except json.JSONDecodeError:
                    continue
                for frame in (frames if isinstance(frames, list) else [frames]):
                    self._handle_frame(frame, symbol)
        finally:
            self._heartbeat_task.cancel()

    def _handle_frame(self, frame: dict, symbol: str) -> None:
        if not isinstance(frame, dict):
            return
        for q in frame.get("d", {}).get("quotes", []):
            price = q.get("bidPrice") or q.get("price")
            if price is None:
                continue
            bar = self._agg.on_tick(float(price), 1, datetime.now(timezone.utc))
            if bar is not None:
                bar.symbol = symbol
                self.on_bar(bar)

    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end."""
        bar = self._agg.flush()
        if bar is not None:
            bar.symbol = symbol
        return bar
