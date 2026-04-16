"""ProjectX position queries for crash recovery."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerPositions
from .auth import BASE_URL

log = logging.getLogger(__name__)


class ProjectXPositions(BrokerPositions):
    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)

    def query_open(self, account_id: int) -> list[dict]:
        resp = requests.post(
            f"{BASE_URL}/api/Position/searchOpen",
            json={"accountId": account_id},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Validate application-level success — orphan detection that silently
        # returns nothing is worse than no orphan detection at all.
        if isinstance(data, dict) and data.get("success") is False:
            raise RuntimeError(f"ProjectX position query failed: {data.get('errorMessage', data)}")
        # Response might be a list directly or have a "positions" key
        positions = data if isinstance(data, list) else data.get("positions", [])
        result = []
        for p in positions:
            result.append(
                {
                    "contract_id": p.get("contractId"),
                    "side": "long" if p.get("type") == 1 else "short",
                    "size": p.get("size", 0),
                    "avg_price": p.get("averagePrice", 0),
                }
            )
        if result:
            log.warning("Found %d open positions on ProjectX", len(result))
        return result

    def query_equity(self, account_id: int) -> float | None:
        """Query current account equity from ProjectX.

        Uses POST /api/Account/search to find account by id, returns balance.
        The /api/Account/{id} GET endpoint returns 404 on TopStepX — use search instead.

        KNOWN LIMITATION: Returns realized balance. May not include unrealized PnL
        from open positions. EOD readings (when flat) are accurate for prop firm DD.
        """
        try:
            resp = requests.post(
                f"{BASE_URL}/api/Account/search",
                json={"onlyActiveAccounts": True},
                headers=self.auth.headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            accounts = data if isinstance(data, list) else data.get("accounts", [])
            for acct in accounts:
                acct_id = acct.get("id") or acct.get("accountId")
                if acct_id is not None and int(acct_id) == account_id:
                    balance = acct.get("balance") or acct.get("cashBalance")
                    if balance is not None:
                        return float(balance)
            log.warning("ProjectX account %d not found in search results", account_id)
            return None
        except Exception as e:
            log.warning("Failed to query ProjectX equity: %s", e)
            return None

    def query_account_metadata(self, account_id: int) -> dict | None:
        """Return the full account metadata dict for account_id, or None on miss.

        Used for broker-reality checks (e.g. Trading Combine vs Express Funded).
        Observed TopStep fields: id, name, balance, canTrade, isVisible, simulated.
        TC accounts have 'TC' in the name (e.g. '50KTC-V2-451890-20372221').
        """
        try:
            resp = requests.post(
                f"{BASE_URL}/api/Account/search",
                json={"onlyActiveAccounts": True},
                headers=self.auth.headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            accounts = data if isinstance(data, list) else data.get("accounts", [])
            for acct in accounts:
                acct_id = acct.get("id") or acct.get("accountId")
                if acct_id is not None and int(acct_id) == account_id:
                    return dict(acct)
            log.warning("ProjectX account %d metadata not found", account_id)
            return None
        except Exception as e:
            log.warning("Failed to query ProjectX account metadata: %s", e)
            return None
