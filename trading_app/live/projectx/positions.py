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
