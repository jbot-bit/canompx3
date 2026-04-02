"""Tradovate contract resolution and account discovery.

GET /contract/find?name=MNQM6 → {id, name, contractMaturityId}
GET /account/list → [{id, name, active}]
"""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerContracts

log = logging.getLogger(__name__)

# Front-month symbol format: instrument + month_code + last_digit_of_year
# E.g., MNQM6 = MNQ June 2026
_MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


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
        resp = requests.get(
            f"{self._base}/account/list",
            headers=self.auth.headers(),
            timeout=10,
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
        resp = requests.get(
            f"{self._base}/contract/suggest?t={instrument}&l=5",
            headers=self.auth.headers(),
            timeout=10,
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
