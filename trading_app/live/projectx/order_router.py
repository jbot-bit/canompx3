"""ProjectX order routing via REST API.

Order types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
Sides: 0=Bid (buy), 1=Ask (sell)
"""

import logging
import time

import requests

from ..broker_base import BrokerAuth, BrokerRouter
from .auth import BASE_URL

log = logging.getLogger(__name__)


class ProjectXOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, **kwargs):
        super().__init__(account_id, auth, **kwargs)

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
        log.info(
            "ProjectX order placed: side=%d type=%d qty=%d -> orderId=%d (%.0fms)",
            spec.get("side", -1),
            spec.get("type", -1),
            spec.get("size", 0),
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
            json={"orderId": order_id},
            headers=self.auth.headers(),
            timeout=5,
        )
        resp.raise_for_status()
        log.info("ProjectX order cancelled: orderId=%d", order_id)

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
        """Build OCO bracket (stop + limit target) for ProjectX."""
        # Stop side: reverse direction to close position
        stop_side = 1 if direction == "long" else 0
        target_side = stop_side  # same side for close

        return {
            "accountId": self.account_id,
            "contractId": symbol,
            "orders": [
                {
                    "type": 4,  # Stop
                    "side": stop_side,
                    "size": qty,
                    "stopPrice": stop_price,
                },
                {
                    "type": 1,  # Limit
                    "side": target_side,
                    "size": qty,
                    "price": target_price,
                },
            ],
        }

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
