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
    "NQ": ["NQ", "E-mini Nasdaq", "E-mini NASDAQ 100"],
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
        # Validate application-level success (HTTP 200 is not enough)
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX account search failed: {data.get('errorMessage', data)}")
        # Response might be a list directly or have an "accounts" key
        accounts = data if isinstance(data, list) else data.get("accounts", [])
        if not accounts:
            raise RuntimeError("No active ProjectX accounts found")
        acct = accounts[0]
        acct_id = acct.get("id")
        if acct_id is None:
            acct_id = acct.get("accountId")
        if acct_id is None:
            raise RuntimeError(f"ProjectX account has no id field: {acct}")
        log.info("ProjectX account: %s (id=%s)", acct.get("name", "unknown"), acct_id)
        return int(acct_id)

    def resolve_all_account_ids(self) -> list[tuple[int, str]]:
        """Return ALL active account IDs and names.

        Returns list of (account_id, account_name) tuples, sorted by id.
        Uses same /api/Account/search endpoint as resolve_account_id().
        One API key covers all linked accounts per ProjectX docs.
        """
        resp = requests.post(
            f"{BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": True},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX account search failed: {data.get('errorMessage', data)}")
        accounts = data if isinstance(data, list) else data.get("accounts", [])
        if not accounts:
            raise RuntimeError("No active ProjectX accounts found")

        result = []
        for acct in accounts:
            acct_id = acct.get("id") or acct.get("accountId")
            if acct_id is not None:
                name = acct.get("name", f"account_{acct_id}")
                result.append((int(acct_id), name))
                log.info("ProjectX account discovered: %s (id=%s)", name, acct_id)

        result.sort(key=lambda x: x[0])
        log.info("ProjectX: %d active accounts discovered", len(result))
        return result

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
        # Validate application-level success (HTTP 200 is not enough)
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX contract query failed: {data.get('errorMessage', data)}")
        # Response might be a list directly or have a "contracts" key
        contracts = data if isinstance(data, list) else data.get("contracts", [])

        search_terms = INSTRUMENT_SEARCH_TERMS.get(instrument, [instrument])
        for contract in contracts:
            name = contract.get("name") or ""
            desc = contract.get("description") or ""
            cid = str(contract.get("id") or "")
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
