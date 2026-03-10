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
import time
from dataclasses import dataclass

import requests

from ..broker_base import BrokerAuth, BrokerRouter

log = logging.getLogger(__name__)

LIVE_BASE = "https://live.tradovateapi.com/v1"
DEMO_BASE = "https://demo.tradovateapi.com/v1"


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


class TradovateOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, demo: bool = True):
        super().__init__(account_id, auth)
        self.base = DEMO_BASE if demo else LIVE_BASE

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

    def submit(self, spec: OrderSpec) -> OrderResult:
        """Submit an order to Tradovate. Requires auth.

        Synchronous HTTP call — the orchestrator wraps this in run_in_executor
        so the async event loop is never blocked.
        """
        if self.auth is None:
            raise RuntimeError("No auth -- cannot submit live orders without TradovateAuth")

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
        resp = requests.post(
            f"{self.base}/order/placeOrder",
            json=body,
            headers=self.auth.headers(),
            timeout=5,
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
        """Cancel an open order by ID."""
        if self.auth is None:
            raise RuntimeError(f"Cannot cancel order {order_id} -- no auth configured")
        resp = requests.post(
            f"{self.base}/order/cancelOrder",
            json={"orderId": order_id},
            headers=self.auth.headers(),
            timeout=5,
        )
        resp.raise_for_status()
        log.info("Order cancelled: orderId=%d", order_id)

    def supports_native_brackets(self) -> bool:
        """Tradovate does not support native bracket orders via this router."""
        return False

    def query_order_status(self, order_id: int) -> dict:
        """Query order status from Tradovate REST API."""
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")
        resp = requests.get(
            f"{self.base}/order/item",
            params={"id": order_id},
            headers=self.auth.headers(),
            timeout=5,
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
