"""Rithmic position queries for crash recovery and equity tracking.

Uses async_rithmic PnlPlant for position snapshots and account summaries.

Verified against: async_rithmic 1.5.9 source (plants/pnl.py lines 44-62)
"""

import logging

from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)

_BRIDGE_TIMEOUT = 10.0


class RithmicPositions(BrokerPositions):
    """Query Rithmic for open positions and account equity."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)

    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions for crash recovery and orphan detection.

        Uses PnlPlant.list_positions() which returns instrument-level PnL snapshots.
        Each snapshot includes symbol, quantity (net position), and average price.
        """
        acct_id_str = str(account_id)

        try:
            positions = self.auth.run_async(
                self.auth.client.list_positions(account_id=acct_id_str),
                timeout=_BRIDGE_TIMEOUT,
            )
        except Exception as e:
            log.error("Rithmic position query failed for account %s: %s", acct_id_str, e)
            raise

        if not positions:
            return []

        result = []
        for p in positions if isinstance(positions, list) else [positions]:
            # Extract fields from protobuf response object
            symbol = getattr(p, "symbol", "")
            quantity = getattr(p, "open_long_quantity", 0) - getattr(p, "open_short_quantity", 0)
            avg_price = getattr(p, "avg_open_fill_price", 0)

            if quantity == 0:
                continue

            result.append({
                "contract_id": symbol,
                "side": "long" if quantity > 0 else "short",
                "size": abs(quantity),
                "avg_price": float(avg_price) if avg_price else 0,
            })

        if result:
            log.warning("Found %d open positions on Rithmic (account %s)", len(result), acct_id_str)
        return result

    def query_equity(self, account_id: int) -> float | None:
        """Return current account equity for DD tracking.

        Uses PnlPlant.list_account_summary() which returns account-level PnL.
        Returns net equity (realized + unrealized) for prop firm DD monitoring.
        """
        acct_id_str = str(account_id)

        try:
            summaries = self.auth.run_async(
                self.auth.client.list_account_summary(account_id=acct_id_str),
                timeout=_BRIDGE_TIMEOUT,
            )
        except Exception as e:
            log.warning("Rithmic equity query failed for account %s: %s", acct_id_str, e)
            return None

        if not summaries:
            return None

        # Extract equity from first summary
        summary = summaries[0] if isinstance(summaries, list) else summaries
        equity = getattr(summary, "account_balance", None)
        if equity is None:
            equity = getattr(summary, "cash_on_hand", None)
        if equity is not None:
            return float(equity)

        log.warning("Rithmic account summary has no balance field for account %s", acct_id_str)
        return None
