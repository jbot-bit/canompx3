"""ProjectX position queries for crash recovery.

All HTTP traffic flows through trading_app.live.http_client.BrokerHTTPClient.

query_equity preserves its float | None contract for back-compat with existing
HWM/dashboard callers, BUT no longer swallows transient errors silently — it
relies on BrokerHTTPClient's classified retry. A sibling query_equity_with_age()
returns an EquityReading consumed by Stage 3's broker-health-tick to enforce
the kill-switch SLA.
"""

import logging
import time

from ..broker_base import BrokerAuth, BrokerPositions
from ..http_client import (
    READ_POLICY,
    BrokerHTTPClient,
    BrokerHTTPError,
    EquityReading,
)
from .auth import BASE_URL

log = logging.getLogger(__name__)


class ProjectXPositions(BrokerPositions):
    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._http = BrokerHTTPClient(
            base_url=BASE_URL,
            refresh_token=auth.refresh_if_needed,
            name="projectx-positions",
        )
        # Stage 3 wires from here: last_good_equity per account_id, with timestamp.
        self._last_good_equity: dict[int, tuple[float, float]] = {}  # account_id -> (value, monotonic_ts)

    def query_open(self, account_id: int) -> list[dict]:
        data = self._http.post_json(
            "/api/Position/searchOpen",
            headers=self.auth.headers(),
            body={"accountId": account_id},
            policy=READ_POLICY,
            timeout=10,
        )
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

        Behavior change (2026-05-18 resilience baseline):
        - Transient errors (A/B/C/D/E) raise BrokerHTTPError — no silent None.
        - Permanent miss (account not found / no balance field) returns None.
        - Callers that need age-aware reads should use query_equity_with_age().

        KNOWN LIMITATION: Returns realized balance. May not include unrealized PnL
        from open positions. EOD readings (when flat) are accurate for prop firm DD.
        """
        reading = self.query_equity_with_age(account_id)
        if reading.source == "live":
            return reading.value
        if reading.source == "missing":
            return None
        # Source is "cache" — caller asked for current equity; surface staleness as None.
        # Callers needing the staleness data should call query_equity_with_age directly.
        return None

    def query_equity_with_age(self, account_id: int) -> EquityReading:
        """Equity reading with age-since-last-successful-fetch.

        Stage 3 broker-health-tick uses this to drive the kill-switch SLA.

        Return semantics:
          - source="live", age_s=0.0 — fresh successful read.
          - source="cache", age_s>0 — transient error; returning last-good for the UI.
          - source="missing" — account not found / no balance field (permanent miss).
        Raises BrokerHTTPError on classified retry exhaustion ONLY when no
        last-good cache exists; otherwise the cache is returned with age_s set.
        """
        try:
            data = self._http.post_json(
                "/api/Account/search",
                headers=self.auth.headers(),
                body={"onlyActiveAccounts": True},
                policy=READ_POLICY,
                timeout=10,
            )
        except BrokerHTTPError as exc:
            cached = self._last_good_equity.get(account_id)
            if cached is not None:
                value, ts = cached
                age = time.monotonic() - ts
                log.warning(
                    "ProjectX equity transient error (%s) — serving cache age=%.1fs",
                    exc.error_class, age,
                )
                return EquityReading(value=value, age_s=age, source="cache")
            raise

        accounts = data if isinstance(data, list) else data.get("accounts", [])
        for acct in accounts:
            acct_id = acct.get("id") or acct.get("accountId")
            if acct_id is not None and int(acct_id) == account_id:
                balance = acct.get("balance")
                if balance is None:
                    balance = acct.get("cashBalance")
                if balance is not None:
                    value = float(balance)
                    self._last_good_equity[account_id] = (value, time.monotonic())
                    return EquityReading(value=value, age_s=0.0, source="live")
                log.warning(
                    "ProjectX account %d found but no balance/cashBalance field",
                    account_id,
                )
                return EquityReading(value=None, age_s=0.0, source="missing")
        log.warning("ProjectX account %d not found in search results", account_id)
        return EquityReading(value=None, age_s=0.0, source="missing")

    def query_account_metadata(self, account_id: int) -> dict | None:
        """Return the full account metadata dict for account_id, or None on miss.

        Used for broker-reality checks (e.g. Trading Combine vs Express Funded).
        Observed TopStep fields: id, name, balance, canTrade, isVisible, simulated.
        TC accounts have 'TC' in the name (e.g. '50KTC-V2-451890-20372221').
        """
        try:
            data = self._http.post_json(
                "/api/Account/search",
                headers=self.auth.headers(),
                body={"onlyActiveAccounts": True},
                policy=READ_POLICY,
                timeout=10,
            )
        except BrokerHTTPError as exc:
            log.warning("Failed to query ProjectX account metadata: %s", exc)
            return None
        accounts = data if isinstance(data, list) else data.get("accounts", [])
        for acct in accounts:
            acct_id = acct.get("id") or acct.get("accountId")
            if acct_id is not None and int(acct_id) == account_id:
                return dict(acct)
        log.warning("ProjectX account %d metadata not found", account_id)
        return None
