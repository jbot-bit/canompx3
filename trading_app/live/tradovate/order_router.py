"""Tradovate order routing via REST API.

Order types: "Market", "Limit", "Stop", "StopLimit", "TrailingStop"
Actions: "Buy", "Sell"
Brackets: placeOSO with bracket1 (target) and bracket2 (stop) on entry order.
isAutomated: MUST be true for all bot-placed orders (Tradovate requirement).
"""

import logging
import random
import time

import requests

from ..broker_base import BrokerAuth, BrokerRouter

log = logging.getLogger(__name__)

_DEFAULT_PRICE_COLLAR_PCT = 0.005
_429_MAX_RETRIES = 3
_429_BACKOFF_BASE = 1.0
_429_BACKOFF_MAX = 30.0
_429_JITTER_FACTOR = 0.2


class RateLimitExhausted(Exception):
    """Raised when 429 retries are exhausted."""

    pass


def _backoff_wait(attempt: int) -> float:
    base_wait = min(_429_BACKOFF_BASE * (2**attempt), _429_BACKOFF_MAX)
    jitter = base_wait * _429_JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.1, base_wait + jitter)


def _request_with_retry(
    method: str,
    url: str,
    headers: dict,
    json_body: dict | None = None,
    timeout: float = 5,
) -> requests.Response:
    """HTTP request with 429 rate-limit retry and exponential backoff."""
    func = requests.post if method == "POST" else requests.get
    kwargs: dict = {"headers": headers, "timeout": timeout}
    if json_body is not None:
        kwargs["json"] = json_body
    for attempt in range(_429_MAX_RETRIES + 1):
        resp = func(url, **kwargs)
        if resp.status_code != 429:
            return resp
        if attempt < _429_MAX_RETRIES:
            wait = _backoff_wait(attempt)
            log.warning("HTTP 429 on %s (attempt %d) — retrying in %.1fs", url.split("/")[-1], attempt + 1, wait)
            time.sleep(wait)
    raise RateLimitExhausted(f"429 exhausted after {_429_MAX_RETRIES + 1} attempts on {url.split('/')[-1]}")


class TradovateOrderRouter(BrokerRouter):
    """Route orders to Tradovate REST API.

    Compatible with Tradeify, MFFU, and direct Tradovate accounts.
    Same BrokerRouter interface as ProjectXOrderRouter.
    """

    def __init__(
        self,
        account_id: int,
        auth: BrokerAuth | None,
        account_spec: str = "",
        tick_size: float = 0.25,
        **kwargs,
    ):
        super().__init__(account_id, auth, **kwargs)
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size
        self.account_spec = account_spec
        self._price_collar_pct = kwargs.get("price_collar_pct", _DEFAULT_PRICE_COLLAR_PCT)
        self._last_known_price: float | None = None

    def _url(self, path: str) -> str:
        """Build full API URL from auth base."""
        if self.auth is None:
            raise RuntimeError("No auth configured")
        base = getattr(self.auth, "base_url", "https://live.tradovateapi.com/v1")
        return f"{base}/{path.lstrip('/')}"

    def update_market_price(self, price: float) -> None:
        if price > 0:
            self._last_known_price = price

    def build_order_spec(
        self,
        direction: str,
        entry_model: str,
        entry_price: float,
        symbol: str,
        qty: int = 1,
    ) -> dict:
        action = "Buy" if direction == "long" else "Sell"

        base = {
            "accountId": self.account_id,
            "accountSpec": self.account_spec,
            "action": action,
            "symbol": symbol,
            "orderQty": qty,
            "isAutomated": True,
        }

        if entry_model == "E1":
            base["orderType"] = "Market"
        elif entry_model == "E2":
            base["orderType"] = "Stop"
            base["stopPrice"] = entry_price
        else:
            raise ValueError(f"Entry model '{entry_model}' not supported live. Use E1 or E2.")

        return base

    def submit(self, spec: dict) -> dict:
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit orders")

        # Price collar for stop orders
        stop_price = spec.get("stopPrice")
        if stop_price is not None and self._last_known_price is not None and self._last_known_price > 0:
            deviation = abs(stop_price - self._last_known_price) / self._last_known_price
            if deviation > self._price_collar_pct:
                msg = (
                    f"PRICE_COLLAR_REJECTED: symbol={spec.get('symbol')} "
                    f"stop={stop_price:.2f} deviates {deviation:.2%} from market "
                    f"{self._last_known_price:.2f} (collar={self._price_collar_pct:.2%})"
                )
                log.critical(msg)
                raise ValueError(msg)

        import json as _json

        log.info("TRADOVATE ORDER SUBMIT: %s", _json.dumps(spec, default=str))

        # Use placeOSO if bracket fields present, otherwise placeorder
        has_bracket = "bracket1" in spec or "bracket2" in spec
        endpoint = "order/placeOSO" if has_bracket else "order/placeorder"

        t0 = time.monotonic()
        resp = _request_with_retry("POST", self._url(endpoint), self.auth.headers(), json_body=spec)
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        log.info("TRADOVATE ORDER RESPONSE: %s", _json.dumps(data, default=str))

        order_id = data.get("orderId")
        if order_id is None:
            raise RuntimeError(f"Tradovate order returned no orderId: {data}")

        fill_price = data.get("fillPrice") or data.get("avgFillPrice")

        log.info(
            "Tradovate order placed: %s %s qty=%d bracket=%s -> orderId=%d (%.0fms)",
            spec.get("action", "?"),
            spec.get("orderType", "?"),
            spec.get("orderQty", 0),
            has_bracket,
            order_id,
            elapsed_ms,
        )
        if elapsed_ms > 1000:
            log.warning("Order submission took %.0fms", elapsed_ms)

        return {
            "order_id": order_id,
            "status": "submitted",
            "fill_price": float(fill_price) if fill_price is not None else None,
        }

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        action = "Sell" if direction == "long" else "Buy"
        return {
            "accountId": self.account_id,
            "accountSpec": self.account_spec,
            "action": action,
            "symbol": symbol,
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
        }

    def cancel(self, order_id: int) -> None:
        if self.auth is None:
            raise RuntimeError(f"Cannot cancel order {order_id} — no auth")
        resp = _request_with_retry(
            "POST",
            self._url("order/cancelorder"),
            self.auth.headers(),
            json_body={"orderId": order_id},
        )
        resp.raise_for_status()
        log.info("Tradovate order cancelled: orderId=%d", order_id)

    def supports_native_brackets(self) -> bool:
        return True

    def build_bracket_spec(
        self,
        direction: str,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        qty: int = 1,
    ) -> dict | None:
        """Build Tradovate OSO bracket fields.

        Tradovate uses bracket1/bracket2 on the entry order for placeOSO.
        bracket1 = profit target, bracket2 = stop loss.
        """
        if direction == "long":
            target_spec = {
                "action": "Sell",
                "orderType": "Limit",
                "price": target_price,
            }
            stop_spec = {
                "action": "Sell",
                "orderType": "Stop",
                "stopPrice": stop_price,
            }
        else:
            target_spec = {
                "action": "Buy",
                "orderType": "Limit",
                "price": target_price,
            }
            stop_spec = {
                "action": "Buy",
                "orderType": "Stop",
                "stopPrice": stop_price,
            }

        return {
            "bracket1": target_spec,
            "bracket2": stop_spec,
        }

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        return {**entry_spec, **bracket_spec}

    def query_order_status(self, order_id: int) -> dict:
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")
        resp = _request_with_retry("GET", self._url(f"order/item?id={order_id}"), self.auth.headers())
        resp.raise_for_status()
        data = resp.json()
        status = data.get("ordStatus", "Unknown")
        fill_price = data.get("avgPx") or data.get("fillPrice")
        return {
            "order_id": order_id,
            "status": status,
            "fill_price": float(fill_price) if fill_price is not None else None,
        }

    def query_open_orders(self) -> list[dict]:
        if self.auth is None:
            return []
        resp = _request_with_retry(
            "GET",
            self._url(f"order/ldeps?masterid={self.account_id}"),
            self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() if isinstance(resp.json(), list) else []
