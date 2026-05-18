"""ProjectX order routing via REST API.

Order types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
Sides: 0=Bid (buy), 1=Ask (sell)
Bracket: stopLossBracket/takeProfitBracket fields on entry order (ticks from fill).

Resilience (2026-05-18 baseline):
- All HTTP flows through trading_app.live.http_client.BrokerHTTPClient.
- Order-mutating calls carry a client_order_id (UUID4) sent as ProjectX
  customTag — the broker's per-account uniqueness constraint is the
  primary idempotency gate (resources/projectx_api_spec_2026_05_16.md
  line 14: "customTag must be unique per account across all orders, ever").
- On retry of a place call where the prior attempt may have succeeded,
  _reconcile_post_place inspects /Order/searchOpen for a fingerprint match
  (contract+size+side+type, created within reconcile_window_s). searchOpen
  does NOT return customTag (API spec line 22), so the match is by
  fingerprint, not by tag.
"""

import json as _json
import logging
import time
import uuid

from ..broker_base import BrokerAuth, BrokerRouter
from ..http_client import (
    ORDER_POLICY,
    READ_POLICY,
    BrokerHTTPClient,
    BrokerHTTPError,
    BrokerProtocolError,
    BrokerRateLimitExhausted,
)
from .auth import BASE_URL

log = logging.getLogger(__name__)


_DEFAULT_PRICE_COLLAR_PCT = 0.005
_RECONCILE_WINDOW_S = 5.0  # search-window for fingerprint-based reconcile


# Backwards-compat re-export so callers `from .order_router import RateLimitExhausted`
# still resolve to the unified exception type.
RateLimitExhausted = BrokerRateLimitExhausted


def generate_client_order_id() -> str:
    """UUID4 hex — used as ProjectX customTag (idempotency key).

    Per ProjectX API spec, customTag must be unique per-account across ALL
    orders, ever. UUID4 collision probability is negligible.
    """
    return uuid.uuid4().hex


class ProjectXOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, tick_size: float = 0.10, **kwargs):
        super().__init__(account_id, auth, **kwargs)
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size
        self._price_collar_pct: float = kwargs.get("price_collar_pct", _DEFAULT_PRICE_COLLAR_PCT)
        self._last_known_price: float | None = None
        # ProjectX auth may be None in test harness scenarios; only construct
        # the http client when we have a real auth (for refresh_token hook).
        # Stage 4: failure_hook is the orchestrator's CircuitBreaker, set on
        # auth before this component is built. None outside an orchestrator
        # (tests, ad-hoc scripts) — the HTTP client falls back to _NoopFailureHook.
        self._http: BrokerHTTPClient | None = None
        if auth is not None:
            failure_hook = getattr(auth, "failure_hook", None)
            kwargs_for_client: dict = {
                "base_url": BASE_URL,
                "refresh_token": auth.refresh_if_needed,
                "name": "projectx-orders",
            }
            if failure_hook is not None:
                kwargs_for_client["failure_hook"] = failure_hook
            self._http = BrokerHTTPClient(**kwargs_for_client)

    def _client(self) -> BrokerHTTPClient:
        if self._http is None:
            raise RuntimeError("No auth — order router HTTP client not initialized")
        return self._http

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

        _intent = {
            "direction": direction,
            "entry_model": entry_model,
            "entry_price": entry_price,
            "symbol": symbol,
            "qty": qty,
        }

        if entry_model == "E1":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 2,  # Market
                "side": side,
                "size": qty,
                "_intent": _intent,
            }
        elif entry_model == "E2":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 4,  # Stop
                "side": side,
                "size": qty,
                "stopPrice": entry_price,
                "_intent": _intent,
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

        # Idempotency key — UUID4, unique-per-account-forever per ProjectX spec.
        client_order_id = spec.get("customTag") or generate_client_order_id()

        # Strip internal routing fields before sending to ProjectX API
        wire_spec = {k: v for k, v in spec.items() if not k.startswith("_")}
        wire_spec["customTag"] = client_order_id

        # Full payload audit trail — logged BEFORE submission. Includes the
        # client_order_id so a journal/log reconciliation can match a duplicate
        # broker-side reject back to its origin attempt.
        log.info("ORDER SUBMIT PAYLOAD (cid=%s): %s", client_order_id, _json.dumps(wire_spec, default=str))

        contract_id = spec.get("contractId")
        side = spec.get("side")
        size = spec.get("size")
        order_type = spec.get("type")
        place_started_at = time.monotonic()

        t0 = time.monotonic()
        try:
            data = self._client().post_json(
                "/api/Order/place",
                headers={**self.auth.headers(), "Content-Type": "application/json"},
                body=wire_spec,
                policy=ORDER_POLICY,
                timeout=5,
            )
        except BrokerHTTPError as exc:
            # Transient/auth/protocol failure. The broker MAY have accepted the
            # order on a prior attempt before the connection dropped. Try to
            # reconcile via fingerprint match before bubbling the failure.
            reconciled = self._reconcile_post_place(
                contract_id=contract_id,
                side=side,
                size=size,
                order_type=order_type,
                window_started_at=place_started_at,
            )
            if reconciled is not None:
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.warning(
                    "ProjectX order RECONCILED after transient error (%s): orderId=%d cid=%s (%.0fms)",
                    exc.error_class, reconciled, client_order_id, elapsed_ms,
                )
                return {
                    "order_id": reconciled,
                    "status": "submitted_reconciled",
                    "fill_price": None,
                    "client_order_id": client_order_id,
                }
            raise

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Full response audit trail
        log.info("ORDER RESPONSE (cid=%s): %s", client_order_id, _json.dumps(data, default=str))

        order_id = data.get("orderId")
        if order_id is None or order_id <= 0:
            log.error("ProjectX order returned no valid orderId: %s", data)
            raise BrokerProtocolError(
                f"ProjectX order returned no valid orderId: {data}",
                error_class="G",
            )
        # Spec says place response only has orderId/success/errorCode/errorMessage.
        # filledPrice may appear on immediate fills — try spec field name first.
        fill_price = data.get("filledPrice")
        if fill_price is None:
            fill_price = data.get("fillPrice")  # fallback for non-spec field name
        if fill_price is None:
            fill_price = data.get("averagePrice")

        has_bracket = "stopLossBracket" in spec or "takeProfitBracket" in spec
        log.info(
            "ProjectX order placed: side=%d type=%d qty=%d bracket=%s -> orderId=%d cid=%s (%.0fms)",
            spec.get("side", -1),
            spec.get("type", -1),
            spec.get("size", 0),
            has_bracket,
            order_id,
            client_order_id,
            elapsed_ms,
        )
        if elapsed_ms > 1000:
            log.warning("Order submission took %.0fms", elapsed_ms)
        return {
            "order_id": order_id,
            "status": "submitted",
            "fill_price": float(fill_price) if fill_price is not None else None,
            "client_order_id": client_order_id,
        }

    def _reconcile_post_place(
        self,
        *,
        contract_id: str | None,
        side: int | None,
        size: int | None,
        order_type: int | None,
        window_started_at: float,
    ) -> int | None:
        """Fingerprint-based reconcile after a transient place failure.

        ProjectX searchOpen does NOT return customTag (API spec line 22), so
        post-fact tag-based matching is impossible. Instead we match by
        (contract, side, size, type) and prefer the most recently-created
        order. Window: time since the place attempt started, bounded by
        _RECONCILE_WINDOW_S.

        Returns the broker order_id if a unique fingerprint match exists,
        else None (caller will surface the original failure).
        """
        elapsed = time.monotonic() - window_started_at
        if elapsed > _RECONCILE_WINDOW_S:
            log.warning("Reconcile skipped: window expired (%.1fs > %.1fs)", elapsed, _RECONCILE_WINDOW_S)
            return None
        if contract_id is None or side is None or size is None or order_type is None:
            return None
        try:
            orders = self.query_open_orders()
        except Exception as exc:
            log.warning("Reconcile failed (cannot query open orders): %s", exc)
            return None
        candidates = [
            o for o in orders
            if o.get("contractId") == contract_id
            and o.get("side") == side
            and o.get("size") == size
            and o.get("type") == order_type
        ]
        if len(candidates) == 1:
            oid = candidates[0].get("id") or candidates[0].get("orderId")
            return int(oid) if oid is not None else None
        if len(candidates) > 1:
            log.warning(
                "Reconcile ambiguous: %d open orders match fingerprint contract=%s side=%s size=%s type=%s — refusing to adopt",
                len(candidates), contract_id, side, size, order_type,
            )
        return None

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
        data = self._client().post_json(
            "/api/Order/cancel",
            headers=self.auth.headers(),
            body={"accountId": self.account_id, "orderId": order_id},
            policy=ORDER_POLICY,
            timeout=10,
        )
        # post_json already raised on success=false / non-JSON.
        log.info("ProjectX order cancelled: orderId=%d (account=%d) resp=%s", order_id, self.account_id, data)

    def supports_native_brackets(self) -> bool:
        return True

    def has_queryable_bracket_legs(self) -> bool:
        """ProjectX AutoBracket creates separate child orders for SL/TP.

        The bracket legs are queryable via searchOpen as individual orders with
        IDs entry_id+1 (stop) and entry_id+2 (take profit), tagged with
        'AutoBracket'. verify_bracket_legs() returns the real order IDs.
        """
        return True

    def supports_sequential_bracket_ids(self) -> bool:
        """ProjectX AutoBracket guarantees ``entry_id+1`` = SL, ``entry_id+2`` = TP.

        See ``has_queryable_bracket_legs`` and ``verify_bracket_legs`` — the
        primary verification path queries searchOpen, but the orchestrator
        keeps this as an emergency fallback when the query call raises.
        """
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
        # Need raw response (status_code) for non-JSON debugging; use request().
        resp = self._client().request(
            "GET",
            f"/api/Order/{order_id}",
            headers=self.auth.headers(),
            policy=READ_POLICY,
            timeout=10,
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
        data = self._client().post_json(
            "/api/Order/searchOpen",
            headers=self.auth.headers(),
            body={"accountId": self.account_id},
            policy=READ_POLICY,
            timeout=10,
        )
        # Response might be a dict with "orders" key or a bare list.
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
