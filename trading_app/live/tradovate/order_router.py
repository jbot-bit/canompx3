"""
Maps ExecutionEngine TradeEvent -> Tradovate REST orders.

CRITICAL API facts (verified against execution_engine.py):
- TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
- TradeEvent.price = entry fill on ENTRY events, exit fill on EXIT events
- There is NO entry_model, pnl_r, or expectancy_r on TradeEvent
- entry_model must be looked up from Portfolio.strategies by strategy_id

Entry model -> order type mapping:
  E1 -> Market order (fill immediately at current price)
  E2 -> Stop-market order (fill when price reaches stop_price)
  E3 -> BLOCKED (no timeout mechanism in live session; use E1 or E2 only)
"""

import logging
import random
import time
from dataclasses import dataclass

import requests

from ..broker_base import BrokerAuth, BrokerRouter
from .auth import DEMO_BASE, LIVE_BASE

log = logging.getLogger(__name__)

# Rate-limit retry configuration.
# REQUIRES_SIM_RETEST after any change to these values.
_429_MAX_RETRIES = 3
_429_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4
_429_BACKOFF_MAX = 30.0
_429_JITTER_FACTOR = 0.2  # ±20%


def _backoff_wait(attempt: int) -> float:
    """Exponential backoff with jitter for 429 retries."""
    base_wait = min(_429_BACKOFF_BASE * (2**attempt), _429_BACKOFF_MAX)
    jitter = base_wait * _429_JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.1, base_wait + jitter)


def _submit_with_429_retry(
    method: str,
    url: str,
    auth_headers: dict,
    timeout: float = 5,
    json_body: dict | None = None,
    params: dict | None = None,
) -> requests.Response:
    """HTTP request with 429 rate-limit detection and exponential backoff.

    On non-429 errors, raises immediately (no retry).
    On 429 after max retries, raises with clear error message.

    Note: auth_headers is a snapshot from caller. If token expires during
    retry backoff (~7s worst case), subsequent retries will get 401 (not 429),
    which raises immediately. TradovateAuth refreshes 60s before expiry so
    this is only an issue if the first request was made <7s before expiry.
    Accepted risk: failure mode is correct (raises -> FILL_UNKNOWN).
    """
    last_resp = None
    for attempt in range(_429_MAX_RETRIES + 1):
        if method == "POST":
            resp = requests.post(url, json=json_body, headers=auth_headers, timeout=timeout)
        else:
            resp = requests.get(url, params=params, headers=auth_headers, timeout=timeout)

        if resp.status_code != 429:
            return resp

        last_resp = resp
        if attempt < _429_MAX_RETRIES:
            wait = _backoff_wait(attempt)
            log.warning(
                "HTTP 429 rate-limited on %s (attempt %d/%d) — retrying in %.1fs",
                url.split("/")[-1],
                attempt + 1,
                _429_MAX_RETRIES + 1,
                wait,
            )
            time.sleep(wait)

    # Exhausted retries — raise with context
    log.error(
        "HTTP 429 rate limit EXHAUSTED after %d attempts on %s",
        _429_MAX_RETRIES + 1,
        url.split("/")[-1],
    )
    assert last_resp is not None
    last_resp.raise_for_status()
    return last_resp  # unreachable, but satisfies type checker


@dataclass
class OrderSpec:
    action: str  # "Buy" | "Sell"
    order_type: str  # "Market" | "Stop"
    symbol: str
    qty: int
    account_id: int
    stop_price: float | None = None


@dataclass
class OrderResult:
    order_id: int
    status: str
    fill_price: float | None = None


# Default price collar: reject entry orders >0.5% from last known price.
# Configurable per-instrument via price_collar_pct kwarg.
_DEFAULT_PRICE_COLLAR_PCT = 0.005


class TradovateOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, demo: bool = True, **kwargs):
        super().__init__(account_id, auth, **kwargs)
        self.base = DEMO_BASE if demo else LIVE_BASE
        self._price_collar_pct: float = kwargs.get("price_collar_pct", _DEFAULT_PRICE_COLLAR_PCT)
        self._last_known_price: float | None = None

    def build_order_spec(
        self,
        direction: str,  # "long" | "short"
        entry_model: str,  # "E1" | "E2" only -- E3 blocked live
        entry_price: float,  # TradeEvent.price on ENTRY events
        symbol: str,
        qty: int = 1,
    ) -> OrderSpec:
        """
        Build an order spec from a TradeEvent.

        direction: event.direction ("long" or "short")
        entry_model: look up from PortfolioStrategy.entry_model (NOT TradeEvent)
        entry_price: event.price on ENTRY events (fill price)
        """
        action = "Buy" if direction == "long" else "Sell"

        if entry_model == "E1":
            return OrderSpec(
                action=action,
                order_type="Market",
                symbol=symbol,
                qty=qty,
                account_id=self.account_id,
            )
        elif entry_model == "E2":
            return OrderSpec(
                action=action,
                order_type="Stop",
                symbol=symbol,
                qty=qty,
                account_id=self.account_id,
                stop_price=entry_price,
            )
        else:
            raise ValueError(
                f"Entry model '{entry_model}' not supported for live trading. "
                f"E3 has no timeout mechanism -- use E1 or E2 only."
            )

    def update_market_price(self, price: float) -> None:
        """Update last known market price for price collar validation."""
        if price > 0:
            self._last_known_price = price

    def submit(self, spec: OrderSpec) -> OrderResult:
        """Submit an order to Tradovate. Requires auth.

        Synchronous HTTP call — the orchestrator wraps this in run_in_executor
        so the async event loop is never blocked.

        Handles HTTP 429 (rate limit) with exponential backoff + jitter.
        After exhausting retries, raises — caller must handle as FILL_UNKNOWN.

        Price collar: entry orders (Stop type) are rejected if stop_price deviates
        more than collar_pct from last known market price. Exit orders (Market type)
        are not collared — we must always be able to close.
        """
        if self.auth is None:
            raise RuntimeError("No auth -- cannot submit live orders without TradovateAuth")

        # Price collar — entry orders only (Stop type)
        if (
            spec.stop_price is not None
            and self._last_known_price is not None
            and self._last_known_price > 0
        ):
            deviation = abs(spec.stop_price - self._last_known_price) / self._last_known_price
            if deviation > self._price_collar_pct:
                msg = (
                    f"PRICE_COLLAR_REJECTED: {spec.action} {spec.symbol} stop={spec.stop_price:.2f} "
                    f"deviates {deviation:.2%} from market {self._last_known_price:.2f} "
                    f"(collar={self._price_collar_pct:.2%})"
                )
                log.critical(msg)
                raise ValueError(msg)

        body = {
            "accountId": spec.account_id,
            "action": spec.action,
            "symbol": spec.symbol,
            "orderQty": spec.qty,
            "orderType": spec.order_type,
            "isAutomated": True,
        }
        if spec.stop_price is not None:
            body["stopPrice"] = spec.stop_price

        t0 = time.monotonic()
        resp = _submit_with_429_retry(
            "POST",
            f"{self.base}/order/placeOrder",
            self.auth.headers(),
            timeout=5,
            json_body=body,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        order_id = data.get("orderId")
        if order_id is None or order_id <= 0:
            log.error("Order placement returned no valid orderId: %s", data)
            raise RuntimeError(f"Order placement returned no valid orderId: {data}")
        log.info(
            "Order placed: %s %s qty=%d -> orderId=%d (%.0fms)",
            spec.action,
            spec.symbol,
            spec.qty,
            order_id,
            elapsed_ms,
        )
        if elapsed_ms > 1000:
            log.warning("Order HTTP round-trip took %.0fms", elapsed_ms)
        fill_price = data.get("avgPx")
        if fill_price is None:
            fill_price = data.get("fillPrice")
        return OrderResult(
            order_id=order_id,
            status="submitted",
            fill_price=float(fill_price) if fill_price is not None else None,
        )

    def build_exit_spec(
        self,
        direction: str,  # original trade direction ("long" or "short")
        symbol: str,
        qty: int = 1,
    ) -> OrderSpec:
        """
        Build a market order to close a position.

        Exits are ALWAYS market orders regardless of entry model.
        Direction is the ORIGINAL trade direction -- we reverse it for the close.
        """
        # Close a long by selling, close a short by buying
        action = "Sell" if direction == "long" else "Buy"
        return OrderSpec(
            action=action,
            order_type="Market",
            symbol=symbol,
            qty=qty,
            account_id=self.account_id,
        )

    def cancel(self, order_id: int) -> None:
        """Cancel an open order by ID. Handles HTTP 429 with backoff."""
        if self.auth is None:
            raise RuntimeError(f"Cannot cancel order {order_id} -- no auth configured")
        resp = _submit_with_429_retry(
            "POST",
            f"{self.base}/order/cancelOrder",
            self.auth.headers(),
            timeout=5,
            json_body={"orderId": order_id},
        )
        resp.raise_for_status()
        log.info("Order cancelled: orderId=%d", order_id)

    def supports_native_brackets(self) -> bool:
        """Tradovate does not support native bracket orders via this router."""
        return False

    def query_order_status(self, order_id: int) -> dict:
        """Query order status from Tradovate REST API. Handles HTTP 429 with backoff."""
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")
        resp = _submit_with_429_retry(
            "GET",
            f"{self.base}/order/item",
            self.auth.headers(),
            timeout=5,
            params={"id": order_id},
        )
        resp.raise_for_status()
        data = resp.json()
        # Map Tradovate ordStatus to standard format
        status_map = {
            "Filled": "Filled",
            "Working": "Working",
            "Cancelled": "Cancelled",
            "Rejected": "Rejected",
        }
        raw_status = data.get("ordStatus", "Unknown")
        fill_price = data.get("avgPx")
        if fill_price is None:
            fill_price = data.get("fillPrice")
        return {
            "order_id": order_id,
            "status": status_map.get(raw_status, raw_status),
            "fill_price": float(fill_price) if fill_price is not None else None,
        }
