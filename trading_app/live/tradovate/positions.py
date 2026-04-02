"""Tradovate position queries for reconciliation and crash recovery.

GET /position/list → [{id, accountId, contractId, netPos, netPrice}]
GET /cashBalance/getCashBalanceSnapshot?accountId=N → {totalCashValue}
"""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)


class TradovatePositions(BrokerPositions):
    """Query Tradovate for open positions and account equity."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._base = getattr(auth, "base_url", "https://live.tradovateapi.com/v1")

    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions: [{contract_id, side, size, avg_price}]."""
        resp = requests.get(
            f"{self._base}/position/list",
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        positions = resp.json()

        result = []
        for p in positions:
            if p.get("accountId") != account_id:
                continue
            net_pos = p.get("netPos", 0)
            if net_pos == 0:
                continue
            result.append(
                {
                    "contract_id": p.get("contractId"),
                    "side": "long" if net_pos > 0 else "short",
                    "size": abs(net_pos),
                    "avg_price": p.get("netPrice", 0),
                }
            )
        return result

    def query_equity(self, account_id: int) -> float | None:
        """Return current account equity (totalCashValue)."""
        try:
            resp = requests.get(
                f"{self._base}/cashBalance/getCashBalanceSnapshot?accountId={account_id}",
                headers=self.auth.headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("totalCashValue")
        except Exception as e:
            log.warning("Tradovate equity query failed for account %d: %s", account_id, e)
            return None
