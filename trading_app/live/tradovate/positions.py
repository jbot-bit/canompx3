"""Tradovate position queries for crash recovery and EOD reconciliation."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerPositions
from .auth import DEMO_BASE, LIVE_BASE

log = logging.getLogger(__name__)


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
                "side": "long" if p.get("netPos", 0) > 0 else "short",
                "size": abs(p.get("netPos", 0)),
                "avg_price": p.get("netPrice", 0.0),
            }
            for p in positions
            if p.get("netPos", 0) != 0 and p.get("accountId") == account_id
        ]
        if result:
            log.warning("Found %d open positions on Tradovate", len(result))
        return result

    def query_equity(self, account_id: int) -> float | None:
        """Query current account equity from Tradovate.

        Returns cashBalance (realized equity). Does NOT include unrealized PnL
        from open positions. This means:
        - When flat (no positions): accurate account equity
        - When positions open: understates drawdown by the unrealized loss amount

        KNOWN LIMITATION: Intraday with open positions, the HWM tracker may not
        see the full drawdown. Prop firm EOD DD is computed when flat, so EOD
        readings are correct. For intraday protection, rely on the R-based
        circuit breaker in RiskManager (which tracks trade-level risk).

        A more accurate approach would use /cashBalance/getCashBalanceSnapshot
        or sum cashBalance + open position P&L, but those endpoints are not
        confirmed available on all Tradovate account types.
        """
        try:
            resp = requests.get(
                f"{self.base}/account/item",
                params={"id": account_id},
                headers=self.auth.headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            cash = data.get("cashBalance")
            if cash is not None:
                return float(cash)
            log.warning("Tradovate account response missing cashBalance: %s", data)
            return None
        except Exception as e:
            log.warning("Failed to query Tradovate equity: %s", e)
            return None
