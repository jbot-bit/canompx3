"""CopyOrderRouter — fan out orders to N accounts.

Single signal -> replicate orders across N accounts (one primary + N-1 shadows).
Primary account: full position tracking, fill polling, DD monitoring.
Shadow accounts: same orders, best-effort. Broker-side DD enforcement.

Architecture:
    ONE auth token -> ONE DataFeed -> ONE SessionOrchestrator -> CopyOrderRouter
    CopyOrderRouter wraps N ProjectXOrderRouters (one per account_id).
    submit() -> primary first, then shadows. Returns primary result.

Rate budget: 200 req/60s (ProjectX). 5 accounts x 1 order each = 5 requests.
"""

import logging

from .broker_base import BrokerRouter

log = logging.getLogger(__name__)


class CopyOrderRouter(BrokerRouter):
    """Wraps a primary + N shadow OrderRouters. Same interface as BrokerRouter."""

    def __init__(self, primary: BrokerRouter, shadows: list[BrokerRouter]):
        super().__init__(account_id=primary.account_id, auth=primary.auth)
        self.primary = primary
        self.shadows = shadows

    def build_order_spec(
        self,
        direction: str,
        entry_model: str,
        entry_price: float,
        symbol: str,
        qty: int = 1,
    ) -> dict:
        """Delegate to primary (all routers use same spec format)."""
        return self.primary.build_order_spec(direction, entry_model, entry_price, symbol, qty)

    def submit(self, spec: dict) -> dict:
        """Submit to primary, then copy to all shadows. Return primary result.

        Shadow failures are logged but don't affect the primary trade.
        Synchronous (requests-based) to match ProjectXOrderRouter.submit() interface.
        """
        result = self.primary.submit(spec)
        primary_status = result.get("status", "unknown")

        # Inverted check: skip shadows only on known failures. This ensures
        # compatibility across brokers (ProjectX: "Filled"/"Working",
        # Rithmic: "submitted") without maintaining a whitelist.
        _SKIP_STATUSES = ("rejected", "error", "cancelled")
        if primary_status.lower() in _SKIP_STATUSES:
            log.info("Primary status=%s — skipping shadow copies", primary_status)
        else:
            for shadow in self.shadows:
                try:
                    shadow_result = shadow.submit(spec)
                    log.info(
                        "Shadow copy account %s: %s (order_id=%s)",
                        shadow.account_id,
                        shadow_result.get("status", "unknown"),
                        shadow_result.get("order_id"),
                    )
                except Exception:
                    log.warning(
                        "Shadow copy FAILED account %s — primary unaffected",
                        shadow.account_id,
                        exc_info=True,
                    )

        return result

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        """Delegate to primary."""
        return self.primary.build_exit_spec(direction, symbol, qty)

    def cancel(self, order_id: int) -> None:
        """Cancel on primary (fail-closed), then best-effort cancel shadows."""
        self.primary.cancel(order_id)
        for shadow in self.shadows:
            try:
                shadow.cancel(order_id)
                log.info("Shadow account %s: order %s cancelled", shadow.account_id, order_id)
            except Exception:
                log.warning(
                    "Shadow account %s: cancel order %s failed — verify manually",
                    shadow.account_id,
                    order_id,
                    exc_info=True,
                )

    def cancel_bracket_orders(self, contract_id: str) -> int:
        """Cancel orphaned bracket orders on primary + all shadows.

        Primary cleanup fails-closed (raises on failure — same as single-account path).
        Shadow cleanup is best-effort (logs warning on failure, does not block startup).
        Returns total count cancelled across all accounts.
        """
        cancelled = self.primary.cancel_bracket_orders(contract_id)
        for shadow in self.shadows:
            try:
                n = shadow.cancel_bracket_orders(contract_id)
                cancelled += n
                if n > 0:
                    log.info("Shadow account %s: cancelled %d orphaned bracket orders", shadow.account_id, n)
            except Exception:
                log.warning(
                    "Shadow account %s: bracket orphan cleanup failed — verify no open bracket orders",
                    shadow.account_id,
                    exc_info=True,
                )
        return cancelled

    def update_market_price(self, price: float) -> None:
        """Forward market price to primary + all shadows for price collar checks."""
        self.primary.update_market_price(price)
        for shadow in self.shadows:
            try:
                shadow.update_market_price(price)
            except Exception:
                log.warning(
                    "Shadow account %s: update_market_price failed",
                    shadow.account_id,
                    exc_info=True,
                )

    def supports_native_brackets(self) -> bool:
        """Delegate to primary."""
        return self.primary.supports_native_brackets()

    def build_bracket_spec(
        self,
        direction: str,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        qty: int = 1,
    ) -> dict | None:
        """Delegate to primary."""
        return self.primary.build_bracket_spec(direction, symbol, entry_price, stop_price, target_price, qty)

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        """Delegate to primary."""
        return self.primary.merge_bracket_into_entry(entry_spec, bracket_spec)

    def verify_bracket_legs(
        self, entry_order_id: int, contract_id: str
    ) -> tuple[int | None, int | None]:
        """Delegate to primary.

        Without this delegate, calls fall through to BrokerRouter's default
        (None, None) which the session_orchestrator caller interprets as
        "BRACKET LEGS MISSING" — false critical alarm + empty bracket_order_ids
        in the active TopStep+CopyOrderRouter+ProjectX path.
        """
        return self.primary.verify_bracket_legs(entry_order_id, contract_id)

    def query_order_status(self, order_id: int) -> dict:
        """Query primary only (shadows are best-effort, not polled)."""
        return self.primary.query_order_status(order_id)

    @property
    def all_account_ids(self) -> list[int]:
        """All account IDs (primary + shadows)."""
        return [self.primary.account_id] + [s.account_id for s in self.shadows]

    @property
    def shadow_count(self) -> int:
        return len(self.shadows)
