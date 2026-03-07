"""Tradovate position queries for crash recovery and EOD reconciliation."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)

LIVE_BASE = "https://live.tradovateapi.com/v1"
DEMO_BASE = "https://demo.tradovateapi.com/v1"


class TradovatePositions(BrokerPositions):
    def __init__(self, auth: BrokerAuth, demo: bool = True):
        super().__init__(auth)
        self.demo = demo
        self.base = DEMO_BASE if demo else LIVE_BASE

    def query_open(self, account_id: int) -> list[dict]:
        """Query open positions from Tradovate.

        GET /position/list -> filter netPos != 0 -> normalize to standard format.
        """
        resp = requests.get(
            f"{self.base}/position/list",
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        positions = resp.json()
        result = [
            {
                "contract_id": p.get("contractId"),
                "side": "BUY" if p.get("netPos", 0) > 0 else "SELL",
                "size": abs(p.get("netPos", 0)),
                "avg_price": p.get("netPrice", 0.0),
            }
            for p in positions
            if p.get("netPos", 0) != 0 and p.get("accountId") == account_id
        ]
        if result:
            log.warning("Found %d open positions on Tradovate", len(result))
        return result
