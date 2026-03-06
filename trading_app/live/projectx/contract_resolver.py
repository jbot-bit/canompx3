"""ProjectX contract resolution and account discovery."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerContracts
from .auth import BASE_URL

log = logging.getLogger(__name__)

# Search terms for each instrument. Dynamic discovery via /api/Contract/available.
INSTRUMENT_SEARCH_TERMS: dict[str, list[str]] = {
    "MGC": ["MGC", "Micro Gold"],
    "MNQ": ["MNQ", "Micro Nasdaq", "Micro E-mini Nasdaq"],
    "MES": ["MES", "Micro E-mini S&P", "Micro S&P"],
    "M2K": ["M2K", "Micro E-mini Russell", "Micro Russell"],
}


class ProjectXContracts(BrokerContracts):
    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._contract_cache: dict[str, str] = {}

    def resolve_account_id(self) -> int:
        resp = requests.post(
            f"{BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": True},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response might be a list directly or have an "accounts" key
        accounts = data if isinstance(data, list) else data.get("accounts", [])
        if not accounts:
            raise RuntimeError("No active ProjectX accounts found")
        acct = accounts[0]
        acct_id = acct.get("id") or acct.get("accountId")
        log.info("ProjectX account: %s (id=%s)", acct.get("name", "unknown"), acct_id)
        return int(acct_id)

    def resolve_front_month(self, instrument: str) -> str:
        if instrument in self._contract_cache:
            return self._contract_cache[instrument]

        resp = requests.post(
            f"{BASE_URL}/api/Contract/available",
            json={"live": False},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Response might be a list directly or have a "contracts" key
        contracts = data if isinstance(data, list) else data.get("contracts", [])

        search_terms = INSTRUMENT_SEARCH_TERMS.get(instrument, [instrument])
        for contract in contracts:
            name = contract.get("name", "")
            desc = contract.get("description", "")
            cid = str(contract.get("id", ""))
            if any(
                term.upper() in name.upper() or term.upper() in desc.upper() or term.upper() in cid.upper()
                for term in search_terms
            ):
                self._contract_cache[instrument] = cid
                log.info("ProjectX contract: %s -> %s (%s)", instrument, cid, name)
                return cid

        # Log all available for debugging
        log.error("Could not find contract for %s. Available contracts:", instrument)
        for c in contracts[:30]:
            log.error("  %s: %s (%s)", c.get("id"), c.get("name"), c.get("description"))
        raise RuntimeError(f"No active ProjectX contract found for {instrument}")
