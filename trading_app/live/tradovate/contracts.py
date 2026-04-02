"""Tradovate contract resolution and account discovery.

GET /contract/find?name=MNQM6 → {id, name, contractMaturityId}
GET /account/list → [{id, name, active}]
"""

import logging

from ..broker_base import BrokerAuth, BrokerContracts
from .http import request_with_retry

log = logging.getLogger(__name__)


class TradovateContracts(BrokerContracts):
    """Resolve Tradovate contract symbols and account IDs."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._base = getattr(auth, "base_url", "https://live.tradovateapi.com/v1")

    def resolve_account_id(self) -> int:
        """Return the first active trading account ID."""
        accounts = self.resolve_all_account_ids()
        if not accounts:
            raise RuntimeError("No active Tradovate accounts found")
        return accounts[0][0]

    def resolve_all_account_ids(self) -> list[tuple[int, str]]:
        """Return ALL active account IDs and names. For copy trading."""
        resp = request_with_retry(
            "GET",
            f"{self._base}/account/list",
            self.auth.headers(),
        )
        resp.raise_for_status()
        accounts = resp.json()
        active = [(a["id"], a.get("name", f"account_{a['id']}")) for a in accounts if a.get("active", True)]
        log.info("Tradovate: found %d active accounts", len(active))
        return active

    def resolve_front_month(self, instrument: str) -> str:
        """Return current front-month contract symbol.

        Tradovate uses the format: MNQM6 (instrument + month code + year digit).
        We query the API to find the current front month.
        """
        # Try to find by instrument name — Tradovate contract/find accepts name prefix
        resp = request_with_retry(
            "GET",
            f"{self._base}/contract/suggest?t={instrument}&l=5",
            self.auth.headers(),
        )
        resp.raise_for_status()
        contracts = resp.json()

        if not contracts:
            raise RuntimeError(f"No contracts found for {instrument} on Tradovate")

        # Return the first (most relevant) contract symbol
        best = contracts[0]
        symbol = best.get("name", best.get("contractSymbol", ""))
        log.info("Tradovate front month for %s: %s", instrument, symbol)
        return symbol
