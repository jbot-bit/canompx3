"""Tradovate position queries for reconciliation and crash recovery.

GET /position/list → [{id, accountId, contractId, netPos, netPrice}]
GET /cashBalance/getCashBalanceSnapshot?accountId=N → {totalCashValue}
"""

import logging
import time

import requests

from ..broker_base import BrokerAuth, BrokerPositions
from ..http_client import BrokerHTTPError, EquityReading
from .http import request_with_retry

log = logging.getLogger(__name__)


class TradovatePositions(BrokerPositions):
    """Query Tradovate for open positions and account equity."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        if not hasattr(auth, "base_url"):
            raise RuntimeError("Auth object missing base_url — cannot determine API endpoint")
        self._base: str = auth.base_url  # type: ignore[attr-defined]
        self._last_good_equity: dict[int, tuple[float, float]] = {}

    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions: [{contract_id, side, size, avg_price}]."""
        resp = request_with_retry(
            "GET",
            f"{self._base}/position/list",
            self.auth.headers(),
            failure_hook=getattr(self.auth, "failure_hook", None),
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
        reading = self.query_equity_with_age(account_id)
        if reading.source == "live":
            return reading.value
        return None

    def query_equity_with_age(self, account_id: int) -> EquityReading:
        """Return Tradovate account equity with age-since-last-successful-fetch."""
        try:
            resp = request_with_retry(
                "GET",
                f"{self._base}/cashBalance/getCashBalanceSnapshot?accountId={account_id}",
                self.auth.headers(),
                failure_hook=getattr(self.auth, "failure_hook", None),
            )
            resp.raise_for_status()
            data = resp.json()
        except (BrokerHTTPError, requests.RequestException) as e:
            cached = self._last_good_equity.get(account_id)
            if cached is not None:
                value, ts = cached
                age = time.monotonic() - ts
                log.warning(
                    "Tradovate equity query failed for account %d (%s) — serving cache age=%.1fs",
                    account_id,
                    e,
                    age,
                )
                return EquityReading(value=value, age_s=age, source="cache")
            log.warning("Tradovate equity query failed for account %d: %s", account_id, e)
            return EquityReading(value=None, age_s=0.0, source="missing")

        value = data.get("totalCashValue")
        if value is None:
            log.warning("Tradovate equity response missing totalCashValue for account %d", account_id)
            return EquityReading(value=None, age_s=0.0, source="missing")
        equity = float(value)
        self._last_good_equity[account_id] = (equity, time.monotonic())
        return EquityReading(value=equity, age_s=0.0, source="live")
