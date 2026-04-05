"""Rithmic position queries for crash recovery and equity tracking.

Uses async_rithmic PnlPlant for position snapshots and account summaries.

Response objects are protobuf messages — financial fields (account_balance,
cash_on_hand, open_position_pnl) are STRING type, not numeric.

Verified against: async_rithmic 1.5.9 protobuf schema:
  InstrumentPnLPositionUpdate (template 450): buy_qty, sell_qty, net_quantity,
      avg_open_fill_price, symbol, exchange, open_position_pnl
  AccountPnLPositionUpdate (template 451): account_balance, cash_on_hand,
      margin_balance, open_position_pnl (all STRING type=9)
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
            # Protobuf fields from InstrumentPnLPositionUpdate (template 450):
            #   net_quantity (int), buy_qty (int), sell_qty (int),
            #   avg_open_fill_price (float), symbol (str)
            symbol = getattr(p, "symbol", "")
            quantity = getattr(p, "net_quantity", 0)
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

        # Extract equity from first summary.
        # Protobuf fields from AccountPnLPositionUpdate (template 451):
        #   account_balance, cash_on_hand are STRING type (protobuf type=9).
        #   Empty string "" is the default for unset string fields.
        summary = summaries[0] if isinstance(summaries, list) else summaries
        for field in ("account_balance", "cash_on_hand"):
            raw = getattr(summary, field, None)
            if raw is not None and raw != "":
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    log.warning("Rithmic %s not numeric: '%s' for account %s", field, raw, acct_id_str)

        log.warning("Rithmic account summary has no balance field for account %s", acct_id_str)
        return None
