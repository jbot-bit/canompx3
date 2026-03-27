"""ProjectX order routing via REST API.

Order types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
Sides: 0=Bid (buy), 1=Ask (sell)
Bracket: stopLossBracket/takeProfitBracket fields on entry order (ticks from fill).
"""

import logging
import random
import time

import requests

from ..broker_base import BrokerAuth, BrokerRouter
from .auth import BASE_URL

log = logging.getLogger(__name__)


_DEFAULT_PRICE_COLLAR_PCT = 0.005

# Rate-limit retry configuration (matches Tradovate order_router).
_429_MAX_RETRIES = 3
_429_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4
_429_BACKOFF_MAX = 30.0
_429_JITTER_FACTOR = 0.2  # ±20%


class RateLimitExhausted(Exception):
    """Raised when 429 retries are exhausted on any ProjectX API call."""

    pass


def _backoff_wait(attempt: int) -> float:
    """Exponential backoff with jitter for 429 retries."""
    base_wait = min(_429_BACKOFF_BASE * (2**attempt), _429_BACKOFF_MAX)
    jitter = base_wait * _429_JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.1, base_wait + jitter)


def _request_with_429_retry(
    method: str,
    url: str,
    auth_headers: dict,
    json_body: dict | None = None,
    timeout: float = 5,
) -> requests.Response:
    """HTTP request with 429 rate-limit detection and exponential backoff.

    On non-429 errors, raises immediately (no retry).
    On 429 after max retries, raises RateLimitExhausted.
    """
    func = requests.post if method == "POST" else requests.get
    kwargs: dict = {"headers": auth_headers, "timeout": timeout}
    if json_body is not None:
        kwargs["json"] = json_body
    for attempt in range(_429_MAX_RETRIES + 1):
        resp = func(url, **kwargs)
        if resp.status_code != 429:
            return resp
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
    log.error(
        "HTTP 429 rate limit EXHAUSTED after %d attempts on %s",
        _429_MAX_RETRIES + 1,
        url.split("/")[-1],
    )
    raise RateLimitExhausted(f"429 rate limit exhausted after {_429_MAX_RETRIES + 1} attempts on {url.split('/')[-1]}")


def _submit_with_429_retry(
    url: str,
    auth_headers: dict,
    json_body: dict,
    timeout: float = 5,
) -> requests.Response:
    """HTTP POST with 429 retry. Legacy wrapper."""
    return _request_with_429_retry("POST", url, auth_headers, json_body, timeout)


class ProjectXOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, tick_size: float = 0.10, **kwargs):
        super().__init__(account_id, auth, **kwargs)
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size
        self._price_collar_pct: float = kwargs.get("price_collar_pct", _DEFAULT_PRICE_COLLAR_PCT)
        self._last_known_price: float | None = None

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
        side = 0 if direction == "long" else 1  # 0=Bid(buy), 1=Ask(sell)

        if entry_model == "E1":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 2,  # Market
                "side": side,
                "size": qty,
            }
        elif entry_model == "E2":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 4,  # Stop
                "side": side,
                "size": qty,
                "stopPrice": entry_price,
            }
        else:
            raise ValueError(f"Entry model '{entry_model}' not supported live. Use E1 or E2.")

    def submit(self, spec: dict) -> dict:
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit orders without ProjectXAuth")

        # Price collar — entry orders only (Stop type=4)
        stop_price = spec.get("stopPrice")
        if stop_price is not None and self._last_known_price is not None and self._last_known_price > 0:
            deviation = abs(stop_price - self._last_known_price) / self._last_known_price
            if deviation > self._price_collar_pct:
                msg = (
                    f"PRICE_COLLAR_REJECTED: contractId={spec.get('contractId')} "
                    f"stop={stop_price:.2f} deviates {deviation:.2%} from market "
                    f"{self._last_known_price:.2f} (collar={self._price_collar_pct:.2%})"
                )
                log.critical(msg)
                raise ValueError(msg)

        # Full payload audit trail — logged BEFORE submission
        import json as _json

        log.info("ORDER SUBMIT PAYLOAD: %s", _json.dumps(spec, default=str))

        t0 = time.monotonic()
        resp = _submit_with_429_retry(
            f"{BASE_URL}/api/Order/place",
            {**self.auth.headers(), "Content-Type": "application/json"},
            json_body=spec,
            timeout=5,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        # Full response audit trail
        log.info("ORDER RESPONSE: %s", _json.dumps(data, default=str))

        if not data.get("success"):
            raise RuntimeError(f"ProjectX order failed: {data.get('errorMessage', data)}")

        order_id = data.get("orderId")
        if order_id is None or order_id <= 0:
            log.error("ProjectX order returned no valid orderId: %s", data)
            raise RuntimeError(f"ProjectX order returned no valid orderId: {data}")
        # Spec says place response only has orderId/success/errorCode/errorMessage.
        # filledPrice may appear on immediate fills — try spec field name first.
        fill_price = data.get("filledPrice")
        if fill_price is None:
            fill_price = data.get("fillPrice")  # fallback for non-spec field name
        if fill_price is None:
            fill_price = data.get("averagePrice")

        has_bracket = "stopLossBracket" in spec or "takeProfitBracket" in spec
        log.info(
            "ProjectX order placed: side=%d type=%d qty=%d bracket=%s -> orderId=%d (%.0fms)",
            spec.get("side", -1),
            spec.get("type", -1),
            spec.get("size", 0),
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
        side = 1 if direction == "long" else 0  # Reverse: close long=sell, close short=buy
        return {
            "accountId": self.account_id,
            "contractId": symbol,
            "type": 2,  # Market
            "side": side,
            "size": qty,
        }

    def cancel(self, order_id: int) -> None:
        if self.auth is None:
            raise RuntimeError(f"Cannot cancel order {order_id} — no auth configured")
        resp = _request_with_429_retry(
            "POST",
            f"{BASE_URL}/api/Order/cancel",
            self.auth.headers(),
            json_body={"accountId": self.account_id, "orderId": order_id},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(f"ProjectX cancel failed for orderId={order_id}: {data.get('errorMessage', data)}")
        log.info("ProjectX order cancelled: orderId=%d (account=%d)", order_id, self.account_id)

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
        """Build bracket fields for ProjectX native bracket order.

        Returns stopLossBracket/takeProfitBracket dicts with tick offsets.
        These get merged into the entry order spec via merge_bracket_into_entry().
        """
        # ProjectX requires SIGNED ticks: negative = below entry, positive = above.
        # Long: SL below entry (negative), TP above (positive).
        # Short: SL above entry (positive), TP below (negative).
        stop_ticks_abs = max(1, round(abs(entry_price - stop_price) / self.tick_size))
        target_ticks_abs = max(1, round(abs(target_price - entry_price) / self.tick_size))

        if direction == "long":
            stop_ticks = -stop_ticks_abs  # SL below entry
            target_ticks = target_ticks_abs  # TP above entry
        else:
            stop_ticks = stop_ticks_abs  # SL above entry
            target_ticks = -target_ticks_abs  # TP below entry

        return {
            "stopLossBracket": {"ticks": stop_ticks, "type": 4},
            "takeProfitBracket": {"ticks": target_ticks, "type": 1},
        }

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        """Attach bracket fields to entry order for atomic submission."""
        return {**entry_spec, **bracket_spec}

    # ProjectX OrderStatus enum (per official API spec)
    _STATUS_INT_MAP = {
        0: "None",
        1: "Working",  # Open/working
        2: "Filled",
        3: "Cancelled",
        4: "Expired",
        5: "Rejected",
        6: "Pending",
    }

    def query_order_status(self, order_id: int) -> dict:
        """Query order status from ProjectX REST API.

        The API returns status as an INTEGER (OrderStatus enum), not a string.
        See docs/reference/PROJECTX_API_REFERENCE.md for canonical spec.
        """
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")
        resp = _request_with_429_retry(
            "GET",
            f"{BASE_URL}/api/Order/{order_id}",
            self.auth.headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        # Map integer status to string (spec returns int, callers expect string)
        raw_status = data.get("status", 0)
        if isinstance(raw_status, int):
            mapped = self._STATUS_INT_MAP.get(raw_status, f"Unknown({raw_status})")
        else:
            # Defensive: handle string if API ever changes
            mapped = str(raw_status)
        # Spec field is "filledPrice", not "fillPrice"
        fill_price = data.get("filledPrice")
        if fill_price is None:
            fill_price = data.get("fillPrice")  # fallback for non-spec endpoints
        if fill_price is None:
            fill_price = data.get("averagePrice")
        return {
            "order_id": order_id,
            "status": mapped,
            "fill_price": float(fill_price) if fill_price is not None else None,
        }

    def query_open_orders(self) -> list[dict]:
        """Query all open/working orders for this account.

        Returns list of order dicts with id, type, side, stopPrice, limitPrice, customTag.
        Used for bracket verification and orphan detection.
        """
        if self.auth is None:
            return []
        resp = _request_with_429_retry(
            "POST",
            f"{BASE_URL}/api/Order/searchOpen",
            self.auth.headers(),
            json_body={"accountId": self.account_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Validate application-level success — an empty list from a failed query
        # looks identical to "no open orders" and will fool bracket verification.
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX searchOpen failed: {data.get('errorMessage', data)}")
        orders = data.get("orders", []) if isinstance(data, dict) else data
        return orders

    def verify_bracket_legs(self, entry_order_id: int, contract_id: str) -> tuple[int | None, int | None]:
        """Verify bracket legs exist after entry fill.

        Returns (sl_order_id, tp_order_id). Either can be None if not found.

        Per API spec, place response returns ONE orderId — bracket legs appear
        as separate orders in searchOpen AFTER fill. We identify them by:
        1. Primary: sequential IDs (entry_id+1 for SL, entry_id+2 for TP)
        2. Fallback: type matching on same contract (type=4 for Stop/SL, type=1 for Limit/TP)
           with IDs greater than the entry order (created after it).

        See docs/reference/PROJECTX_API_REFERENCE.md — searchOpen does NOT
        return customTag, so tag-based matching is impossible.
        """
        try:
            orders = self.query_open_orders()
        except RateLimitExhausted:
            raise  # 429 exhaustion must propagate — silent (None, None) hides rate-limit failure
        except Exception as e:
            log.error("Bracket verification failed (cannot query open orders): %s", e)
            return None, None

        sl_id = None
        tp_id = None
        expected_sl = entry_order_id + 1
        expected_tp = entry_order_id + 2

        # Fallback candidates: orders on same contract with ID > entry_id
        sl_fallback = None
        tp_fallback = None

        for o in orders:
            oid = o.get("id", o.get("orderId"))
            o_contract = o.get("contractId", "")

            if o_contract != contract_id or oid is None:
                continue

            # Primary: sequential ID match
            if oid == expected_sl:
                sl_id = oid
            elif oid == expected_tp:
                tp_id = oid
            # Fallback: type-based match for orders created after entry
            elif oid > entry_order_id:
                o_type = o.get("type")
                if o_type == 4 and sl_fallback is None:  # Stop = SL
                    sl_fallback = oid
                elif o_type == 1 and tp_fallback is None:  # Limit = TP
                    tp_fallback = oid

        # Use fallback if primary didn't find both legs
        if sl_id is None and sl_fallback is not None:
            sl_id = sl_fallback
            log.info("Bracket SL found via type-match fallback: orderId=%d", sl_id)
        if tp_id is None and tp_fallback is not None:
            tp_id = tp_fallback
            log.info("Bracket TP found via type-match fallback: orderId=%d", tp_id)

        return sl_id, tp_id

    def cancel_bracket_orders(self, contract_id: str) -> int:
        """Cancel orphaned bracket orders for a contract. Returns count cancelled.

        Per API spec, searchOpen does NOT return customTag. Bracket legs are
        identified by type: Stop (type=4) for SL, Limit (type=1) for TP.
        This is used at startup for orphan cleanup — during normal exit flow,
        bracket IDs are tracked per-strategy and cancelled by specific ID.
        """
        try:
            orders = self.query_open_orders()
        except RateLimitExhausted:
            raise  # 429 exhaustion must propagate — silent 0 hides rate-limit failure
        except Exception as e:
            log.error("Cannot query open orders for bracket cleanup: %s", e)
            raise  # fail-closed: caller must know orphan cleanup was not performed

        cancelled = 0
        for o in orders:
            o_contract = o.get("contractId", "")
            oid = o.get("id", o.get("orderId"))
            o_type = o.get("type")

            # Bracket legs are Stop (type=4) or Limit (type=1) orders
            if o_contract == contract_id and o_type in (1, 4) and oid:
                try:
                    self.cancel(oid)
                    cancelled += 1
                except Exception as e:
                    log.warning("Failed to cancel bracket order %s: %s", oid, e)
        return cancelled
