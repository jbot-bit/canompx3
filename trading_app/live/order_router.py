"""
Maps ExecutionEngine TradeEvent → Tradovate REST orders.

CRITICAL API facts (verified against execution_engine.py):
- TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
- TradeEvent.price = entry fill on ENTRY events, exit fill on EXIT events
- There is NO entry_model, pnl_r, or expectancy_r on TradeEvent
- entry_model must be looked up from Portfolio.strategies by strategy_id

Entry model → order type mapping:
  E1 → Market order (fill immediately at current price)
  E2 → Stop-market order (fill when price reaches stop_price)
  E3 → BLOCKED (no timeout mechanism in live session; use E1 or E2 only)
"""
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .tradovate_auth import TradovateAuth

log = logging.getLogger(__name__)

LIVE_BASE = "https://live.tradovate.com/v1"
DEMO_BASE = "https://demo.tradovate.com/v1"


@dataclass
class OrderSpec:
    action: str              # "Buy" | "Sell"
    order_type: str          # "Market" | "Stop"
    symbol: str
    qty: int
    account_id: int
    stop_price: Optional[float] = None


@dataclass
class OrderResult:
    order_id: int
    status: str


class OrderRouter:
    def __init__(self, account_id: int, auth: Optional[TradovateAuth], demo: bool = True):
        self.account_id = account_id
        self.auth = auth
        self.base = DEMO_BASE if demo else LIVE_BASE

    def build_order_spec(
        self,
        direction: str,       # "long" | "short"
        entry_model: str,     # "E1" | "E2" only — E3 blocked live
        entry_price: float,   # TradeEvent.price on ENTRY events
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
                action=action, order_type="Market",
                symbol=symbol, qty=qty, account_id=self.account_id,
            )
        elif entry_model == "E2":
            return OrderSpec(
                action=action, order_type="Stop",
                symbol=symbol, qty=qty, account_id=self.account_id,
                stop_price=entry_price,
            )
        else:
            raise ValueError(
                f"Entry model '{entry_model}' not supported for live trading. "
                f"E3 has no timeout mechanism — use E1 or E2 only."
            )

    def submit(self, spec: OrderSpec) -> OrderResult:
        """Submit an order to Tradovate. Requires auth."""
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit live orders without TradovateAuth")

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

        resp = requests.post(
            f"{self.base}/order/placeOrder",
            json=body,
            headers=self.auth.headers(),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        order_id = data.get("orderId", -1)
        log.info(
            "Order placed: %s %s qty=%d → orderId=%d",
            spec.action, spec.symbol, spec.qty, order_id,
        )
        return OrderResult(order_id=order_id, status="submitted")

    def cancel(self, order_id: int) -> None:
        """Cancel an open order by ID."""
        if self.auth is None:
            return
        requests.post(
            f"{self.base}/order/cancelOrder",
            json={"orderId": order_id},
            headers=self.auth.headers(),
            timeout=5,
        )
