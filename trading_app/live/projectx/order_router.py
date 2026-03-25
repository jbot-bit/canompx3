"""ProjectX order routing via REST API.

Order types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
Sides: 0=Bid (buy), 1=Ask (sell)
Bracket: stopLossBracket/takeProfitBracket fields on entry order (ticks from fill).
"""

import logging
import time

import requests

from ..broker_base import BrokerAuth, BrokerRouter
from .auth import BASE_URL

log = logging.getLogger(__name__)


class ProjectXOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, tick_size: float = 0.10, **kwargs):
        super().__init__(account_id, auth, **kwargs)
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size

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

        t0 = time.monotonic()
        resp = requests.post(
            f"{BASE_URL}/api/Order/place",
            json=spec,
            headers={**self.auth.headers(), "Content-Type": "application/json"},
            timeout=5,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            raise RuntimeError(f"ProjectX order failed: {data.get('errorMessage', data)}")

        order_id = data.get("orderId")
        if order_id is None or order_id <= 0:
            log.error("ProjectX order returned no valid orderId: %s", data)
            raise RuntimeError(f"ProjectX order returned no valid orderId: {data}")
        fill_price = data.get("fillPrice")
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
        resp = requests.post(
            f"{BASE_URL}/api/Order/cancel",
            json={"accountId": self.account_id, "orderId": order_id},
            headers=self.auth.headers(),
            timeout=5,
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
            stop_ticks = -stop_ticks_abs   # SL below entry
            target_ticks = target_ticks_abs  # TP above entry
        else:
            stop_ticks = stop_ticks_abs     # SL above entry
            target_ticks = -target_ticks_abs  # TP below entry

        return {
            "stopLossBracket": {"ticks": stop_ticks, "type": 4},
            "takeProfitBracket": {"ticks": target_ticks, "type": 1},
        }

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        """Attach bracket fields to entry order for atomic submission."""
        return {**entry_spec, **bracket_spec}

    def query_order_status(self, order_id: int) -> dict:
        """Query order status from ProjectX REST API."""
        if self.auth is None:
            raise RuntimeError("No auth — cannot query order status")
        resp = requests.get(
            f"{BASE_URL}/api/Order/{order_id}",
            headers=self.auth.headers(),
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        # Map ProjectX status to standard format
        status_map = {
            "Filled": "Filled",
            "Working": "Working",
            "Cancelled": "Cancelled",
            "Rejected": "Rejected",
        }
        raw_status = data.get("status", "Unknown")
        fill_price = data.get("fillPrice")
        if fill_price is None:
            fill_price = data.get("averagePrice")
        return {
            "order_id": order_id,
            "status": status_map.get(raw_status, raw_status),
            "fill_price": float(fill_price) if fill_price is not None else None,
        }
