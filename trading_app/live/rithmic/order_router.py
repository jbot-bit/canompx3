"""Rithmic order routing via Protocol Buffer over WebSocket.

Order types: MARKET=2, STOP_MARKET=4 (from async_rithmic.OrderType)
Sides: TransactionType.BUY=1, TransactionType.SELL=2
Brackets: stop_ticks + target_ticks on submit_order() → atomic server-side bracket

Server-side brackets: stops/targets live on Rithmic servers, survive client crash.
This is strictly better than ProjectX/Tradovate client-managed brackets.

Verified against: async_rithmic 1.5.9 source (plants/order.py lines 209-271, enums.py)
"""

import json as _json
import logging
import random
import time
from datetime import UTC, datetime

from ..broker_base import BrokerAuth, BrokerRouter

log = logging.getLogger(__name__)

_DEFAULT_PRICE_COLLAR_PCT = 0.005
_BRIDGE_TIMEOUT = 10.0  # seconds for async→sync bridge (queries, cancels)
_ORDER_SUBMIT_TIMEOUT = 20.0  # seconds for order submission — must exceed library's
                               # internal 30s retry timeout (but library uses retries=1
                               # for orders, so effective wait = retry_settings.timeout)


class RithmicOrderRouter(BrokerRouter):
    """Route orders to Rithmic via async_rithmic.

    Compatible with Bulenox, Elite Trader Funding, and other Rithmic-based firms.
    Same BrokerRouter interface as ProjectXOrderRouter and TradovateOrderRouter.
    """

    def __init__(
        self,
        account_id: int,
        auth: BrokerAuth | None,
        tick_size: float = 0.25,
        exchange: str = "CME",
        **kwargs,
    ):
        super().__init__(account_id, auth, **kwargs)
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size
        self.exchange = exchange
        self._price_collar_pct: float = kwargs.get("price_collar_pct", _DEFAULT_PRICE_COLLAR_PCT)
        self._last_known_price: float | None = None

        # Rithmic account IDs are strings. Store the original for API calls.
        self._rithmic_account_id: str = kwargs.get("rithmic_account_id", str(account_id))

        # Order state cache: user_order_id → {basket_id, status, fill_price}
        # Updated by exchange notification callback (if registered)
        self._order_cache: dict[str, dict] = {}

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
        """Build a Rithmic-compatible order spec.

        Returns a dict consumed by submit(). Contains both Rithmic-specific
        fields and a broker-agnostic _intent dict for BrokerDispatcher routing.
        """
        # async_rithmic TransactionType: BUY=1, SELL=2
        transaction_type = 1 if direction == "long" else 2

        _intent = {
            "direction": direction,
            "entry_model": entry_model,
            "entry_price": entry_price,
            "symbol": symbol,
            "qty": qty,
        }

        # async_rithmic OrderType: MARKET=2, STOP_MARKET=4
        if entry_model == "E1":
            return {
                "order_type": 2,  # MARKET
                "transaction_type": transaction_type,
                "symbol": symbol,
                "exchange": self.exchange,
                "qty": qty,
                "account_id": self._rithmic_account_id,
                "_intent": _intent,
            }
        elif entry_model == "E2":
            return {
                "order_type": 4,  # STOP_MARKET
                "transaction_type": transaction_type,
                "trigger_price": entry_price,
                "symbol": symbol,
                "exchange": self.exchange,
                "qty": qty,
                "account_id": self._rithmic_account_id,
                "_intent": _intent,
            }
        else:
            raise ValueError(f"Entry model '{entry_model}' not supported live. Use E1 or E2.")

    def submit(self, spec: dict) -> dict:
        """Submit order to Rithmic via async bridge.

        Calls async_rithmic submit_order() through auth.run_async().
        Bracket fields (stop_ticks, target_ticks) are passed as kwargs
        to submit_order() — the library auto-switches to template 330
        for atomic server-side bracket creation.
        """
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit orders without RithmicAuth")

        # Price collar — entry stop orders only (STOP_MARKET type=4)
        trigger_price = spec.get("trigger_price")
        if trigger_price is not None and self._last_known_price is not None and self._last_known_price > 0:
            deviation = abs(trigger_price - self._last_known_price) / self._last_known_price
            if deviation > self._price_collar_pct:
                msg = (
                    f"PRICE_COLLAR_REJECTED: symbol={spec.get('symbol')} "
                    f"trigger={trigger_price:.2f} deviates {deviation:.2%} from market "
                    f"{self._last_known_price:.2f} (collar={self._price_collar_pct:.2%})"
                )
                log.critical(msg)
                raise ValueError(msg)

        # Generate unique order ID (Rithmic requires user-assigned string ID).
        # Account suffix prevents collisions when copy trading sends multiple
        # orders in the same second (3 copies = 3 orders/second).
        # 6-digit random: P(collision) < 0.0002% per second with 3 copies.
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        acct_suffix = self._rithmic_account_id[-4:] if len(self._rithmic_account_id) >= 4 else self._rithmic_account_id
        order_id = f"orb_{ts}_{acct_suffix}_{random.randint(100000, 999999)}"

        # Build kwargs for async_rithmic submit_order()
        # Verified signature: submit_order(order_id, symbol, exchange, qty, transaction_type, order_type, **kwargs)
        submit_kwargs = {}
        if spec.get("trigger_price") is not None:
            submit_kwargs["trigger_price"] = spec["trigger_price"]
        if spec.get("account_id"):
            submit_kwargs["account_id"] = spec["account_id"]

        # Bracket fields — passed as kwargs, library auto-uses template 330
        if spec.get("stop_ticks") is not None:
            submit_kwargs["stop_ticks"] = spec["stop_ticks"]
        if spec.get("target_ticks") is not None:
            submit_kwargs["target_ticks"] = spec["target_ticks"]

        # Strip internal routing fields for logging
        wire_spec = {k: v for k, v in spec.items() if not k.startswith("_")}
        log.info("RITHMIC ORDER SUBMIT: order_id=%s payload=%s", order_id, _json.dumps(wire_spec, default=str))

        # async_rithmic accepts raw int values for protobuf enums
        # OrderType: MARKET=2, STOP_MARKET=4; TransactionType: BUY=1, SELL=2
        order_type_int = spec["order_type"]
        txn_type_int = spec["transaction_type"]

        t0 = time.monotonic()
        try:
            responses = self.auth.run_async(
                self.auth.client.submit_order(
                    order_id=order_id,
                    symbol=spec["symbol"],
                    exchange=spec["exchange"],
                    qty=spec["qty"],
                    transaction_type=txn_type_int,
                    order_type=order_type_int,
                    **submit_kwargs,
                ),
                timeout=_ORDER_SUBMIT_TIMEOUT,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.error("RITHMIC ORDER FAILED: order_id=%s error=%s (%.0fms)", order_id, e, elapsed_ms)
            raise

        log.info("RITHMIC ORDER RESPONSE: order_id=%s response=%s (%.0fms)", order_id, responses, elapsed_ms)
        if elapsed_ms > 1000:
            log.warning("Order submission took %.0fms", elapsed_ms)

        # Extract basket_id and rp_code from response.
        # ResponseNewOrder (313) / ResponseBracketOrder (331) fields:
        #   basket_id (str), rp_code (str: "0"=success, else rejection), user_tag (str)
        basket_id = None
        rp_code = None
        if responses and hasattr(responses, "__iter__"):
            for r in responses if isinstance(responses, list) else [responses]:
                if hasattr(r, "basket_id"):
                    raw_bid = r.basket_id
                    if raw_bid and raw_bid != "":  # protobuf default is ""
                        basket_id = raw_bid
                if hasattr(r, "rp_code"):
                    rp_code = r.rp_code
                if basket_id is not None:
                    break

        # Detect rejection via rp_code
        status = "submitted"
        if rp_code is not None and rp_code != "" and rp_code != "0":
            status = "rejected"
            log.error(
                "RITHMIC ORDER REJECTED: order_id=%s rp_code=%s basket_id=%s",
                order_id, rp_code, basket_id,
            )

        # Cache order state — keyed by generated order_id for reliable lookup.
        # basket_id may be None for rejected orders.
        self._order_cache[order_id] = {
            "basket_id": basket_id,
            "status": status,
            "fill_price": None,
        }

        return {
            "order_id": basket_id or order_id,
            "status": status,
            "fill_price": None,
        }

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        """Build a market close order. Reverses the direction."""
        # Close long = SELL, close short = BUY
        transaction_type = 2 if direction == "long" else 1
        return {
            "order_type": 2,  # MARKET
            "transaction_type": transaction_type,
            "symbol": symbol,
            "exchange": self.exchange,
            "qty": qty,
            "account_id": self._rithmic_account_id,
        }

    def cancel(self, order_id: int) -> None:
        """Cancel an order by basket_id (Rithmic-assigned exchange ID).

        Passes account_id to avoid the library doing a full get_order() scan
        across all accounts (extra network round-trip).
        """
        if self.auth is None:
            raise RuntimeError(f"Cannot cancel order {order_id} — no auth configured")

        try:
            self.auth.run_async(
                self.auth.client.cancel_order(
                    basket_id=str(order_id),
                    account_id=self._rithmic_account_id,
                ),
                timeout=_BRIDGE_TIMEOUT,
            )
            log.info("Rithmic order cancelled: basket_id=%s", order_id)
        except Exception as e:
            log.error("Rithmic cancel failed for basket_id=%s: %s", order_id, e)
            raise

    def supports_native_brackets(self) -> bool:
        """Rithmic has SERVER-SIDE brackets — stops/targets survive client crash."""
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
        """Build bracket fields for Rithmic atomic bracket order.

        Returns stop_ticks and target_ticks (always positive integers).
        These get merged into the entry order spec via merge_bracket_into_entry()
        and passed as kwargs to submit_order(), which auto-uses template 330
        for server-side bracket creation.
        """
        stop_ticks = max(1, round(abs(entry_price - stop_price) / self.tick_size))
        target_ticks = max(1, round(abs(target_price - entry_price) / self.tick_size))

        return {
            "stop_ticks": stop_ticks,
            "target_ticks": target_ticks,
        }

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        """Merge bracket fields into entry order spec for atomic submission.

        async_rithmic submit_order() accepts stop_ticks and target_ticks as kwargs.
        When present, it auto-switches to template 330 (bracket order).
        """
        return {**entry_spec, **bracket_spec}

    def query_order_status(self, order_id: int) -> dict:
        """Query order status from Rithmic.

        First checks local cache (populated by exchange notification callbacks).
        Falls back to async get_order() if not in cache.
        """
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")

        # Check cache first — match by basket_id OR by generated order_id.
        # submit() returns basket_id when available, generated order_id otherwise.
        # The orchestrator stores whichever was returned as entry_order_id.
        order_id_str = str(order_id)
        for uid, cached in self._order_cache.items():
            if cached.get("basket_id") == order_id_str or uid == order_id_str:
                return {
                    "order_id": order_id,
                    "status": cached.get("status", "Unknown"),
                    "fill_price": cached.get("fill_price"),
                }

        # Fallback: query via API
        try:
            order = self.auth.run_async(
                self.auth.client.get_order(basket_id=order_id_str),
                timeout=_BRIDGE_TIMEOUT,
            )
            if order:
                # status is STRING (protobuf type=9), default "".
                # avg_fill_price is DOUBLE (type=1), default 0.0.
                raw_status = getattr(order, "status", "")
                status = str(raw_status) if raw_status else "Unknown"
                raw_fill = getattr(order, "avg_fill_price", 0.0)
                fill_price = float(raw_fill) if raw_fill != 0.0 else None
                return {
                    "order_id": order_id,
                    "status": status,
                    "fill_price": fill_price,
                }
        except Exception as e:
            log.error("Rithmic order query FAILED for basket_id=%s: %s", order_id, e)

        return {"order_id": order_id, "status": "Unknown", "fill_price": None}

    def query_open_orders(self) -> list[dict]:
        """Query all open/working orders for this account."""
        if self.auth is None:
            log.warning("query_open_orders called with no auth — returning empty")
            return []
        try:
            orders = self.auth.run_async(
                self.auth.client.list_orders(account_id=self._rithmic_account_id),
                timeout=_BRIDGE_TIMEOUT,
            )
            return orders if isinstance(orders, list) else []
        except Exception as e:
            log.error("Rithmic open orders query failed: %s", e)
            raise

    def cancel_bracket_orders(self, contract_id: str) -> int:
        """Cancel orphaned bracket orders for a contract.

        Rithmic brackets are server-side, so this queries open orders and cancels
        any stop/limit legs that match the contract.
        """
        try:
            orders = self.query_open_orders()
        except Exception as e:
            log.error("Cannot query open orders for bracket cleanup: %s", e)
            raise

        cancelled = 0
        for o in orders:
            o_symbol = getattr(o, "symbol", "")
            basket_id = getattr(o, "basket_id", None)

            if o_symbol == contract_id and basket_id:
                try:
                    self.auth.run_async(
                        self.auth.client.cancel_order(
                            basket_id=basket_id,
                            account_id=self._rithmic_account_id,
                        ),
                        timeout=_BRIDGE_TIMEOUT,
                    )
                    cancelled += 1
                except Exception as e:
                    log.warning("Failed to cancel bracket order %s: %s", basket_id, e)
        return cancelled
