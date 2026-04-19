"""Rithmic contract resolution and account discovery.

Uses async_rithmic TickerPlant.get_front_month_contract() for contract lookup.
Account list from OrderPlant.accounts (populated on login).

Verified against: async_rithmic 1.5.9 source (plants/ticker.py, plants/order.py)
"""

import logging
from datetime import date
from typing import TYPE_CHECKING

from ..broker_base import BrokerAuth, BrokerContracts

if TYPE_CHECKING:
    from .auth import RithmicAuth

log = logging.getLogger(__name__)

# Rithmic uses CME exchange code for all CME Group products (MES, MNQ, MGC).
# MGC trades on COMEX division but uses "CME" in the Rithmic API.
EXCHANGE_CODE = "CME"

# Root symbols match between our instrument names and Rithmic.
# No translation needed — "MES" → "MES", "MNQ" → "MNQ", "MGC" → "MGC".
INSTRUMENT_ROOTS: dict[str, str] = {
    "MES": "MES",
    "MNQ": "MNQ",
    "NQ": "NQ",
    "MGC": "MGC",
}


class RithmicContracts(BrokerContracts):
    """Resolve accounts and front-month contracts via Rithmic API."""

    # Narrow the inherited BrokerAuth type — at runtime this is always
    # RithmicAuth, which exposes .client (RithmicClient) and .run_async.
    auth: "RithmicAuth"

    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._contract_cache: dict[str, str] = {}

    def resolve_account_id(self) -> int:
        """Return the first active account ID.

        Rithmic account IDs are strings (e.g. "12345678").
        Converted to int for BrokerContracts ABC compatibility.
        """
        accounts = self.auth.client.accounts
        if not accounts:
            raise RuntimeError("No Rithmic accounts found after login")

        acct = accounts[0]
        acct_id = acct.account_id
        log.info("Rithmic account: %s (fcm=%s, ib=%s)", acct_id, self.auth.client.fcm_id, self.auth.client.ib_id)
        try:
            return int(acct_id)
        except (ValueError, TypeError):
            # Non-numeric account ID — use hash for ABC compliance
            hashed = hash(acct_id) & 0x7FFFFFFF
            log.warning("Rithmic account ID '%s' is non-numeric — using hash %d", acct_id, hashed)
            return hashed

    def resolve_all_account_ids(self) -> list[tuple[int, str]]:
        """Return ALL active account IDs and names for copy trading.

        Bulenox allows up to 3 simultaneous Master accounts.
        Each account appears in client.accounts after ORDER_PLANT login.
        """
        accounts = self.auth.client.accounts
        if not accounts:
            raise RuntimeError("No Rithmic accounts found after login")

        result = []
        for acct in accounts:
            acct_id_str = acct.account_id
            try:
                acct_id_int = int(acct_id_str)
            except (ValueError, TypeError):
                acct_id_int = hash(acct_id_str) & 0x7FFFFFFF

            name = getattr(acct, "account_name", acct_id_str)
            result.append((acct_id_int, str(name)))
            log.info("Rithmic account discovered: %s (id=%s)", name, acct_id_str)

        result.sort(key=lambda x: x[0])
        log.info("Rithmic: %d active accounts discovered", len(result))
        return result

    def resolve_front_month(self, instrument: str) -> str:
        """Return current front-month contract symbol.

        Uses async_rithmic get_front_month_contract(root, exchange).
        Example: "MES" → "MESM6" (June 2026 Micro E-mini S&P).

        Result is cached for the session (contracts don't roll mid-session).
        """
        if instrument in self._contract_cache:
            return self._contract_cache[instrument]

        root = INSTRUMENT_ROOTS.get(instrument, instrument)

        # get_front_month_contract is on TickerPlant — requires TICKER_PLANT connection.
        # For order-only mode (no TICKER_PLANT), we construct the symbol manually.
        # Try the API first, fall back to manual construction.
        try:
            symbol = self.auth.run_async(self.auth.client.get_front_month_contract(root, EXCHANGE_CODE))
            if symbol:
                self._contract_cache[instrument] = symbol
                log.info("Rithmic contract: %s → %s", instrument, symbol)
                return symbol
        except Exception as e:
            log.warning("Rithmic front-month lookup failed for %s: %s — using manual construction", instrument, e)

        # Manual fallback: construct from root + current CME month code
        symbol = self._construct_front_month(root)
        self._contract_cache[instrument] = symbol
        log.info("Rithmic contract (manual): %s → %s", instrument, symbol)
        return symbol

    @staticmethod
    def _construct_front_month(root: str) -> str:
        """Construct front-month symbol from root + CME calendar.

        CME micro futures use quarterly cycle: H (Mar), M (Jun), U (Sep), Z (Dec).
        Roll typically happens ~1 week before expiration on 3rd Friday.
        We use a 2-week buffer before the expiration month to avoid trading
        an expired contract (e.g., if today is Mar 20 and MESH6 expired Mar 15).
        """
        today = date.today()
        month = today.month
        day = today.day
        year = today.year % 10  # Single digit year

        # Quarterly months and their codes
        quarters = [(3, "H"), (6, "M"), (9, "U"), (12, "Z")]

        for q_month, code in quarters:
            # In the expiration month, roll to next contract after day 14
            # (3rd Friday is always between 15th-21st, roll before it)
            if month < q_month or (month == q_month and day <= 14):
                return f"{root}{code}{year}"

        # Past December (or past Dec 14) → next year's March
        return f"{root}H{(year + 1) % 10}"
