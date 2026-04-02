"""BrokerDispatcher — route trade signals to multiple brokers simultaneously.

Architecture:
    SessionOrchestrator → BrokerDispatcher → [CopyOrderRouter(ProjectX), CopyOrderRouter(Tradovate), ...]

    One master signal fires. Dispatcher fans out to ALL active broker routers.
    Primary broker: full result tracking (fills, brackets, errors).
    Secondary brokers: best-effort fire-and-forget with error logging.

    This enables: 1 bot → TopStep x5 + Tradeify x5 + MFFU x5 + Bulenox x3 simultaneously.
"""

import logging

from .broker_base import BrokerRouter

log = logging.getLogger(__name__)


class BrokerDispatcher(BrokerRouter):
    """Wraps multiple BrokerRouters (one per firm). Same interface as BrokerRouter.

    Primary router returns the result. All others fire-and-forget.
    Implements BrokerRouter ABC so SessionOrchestrator treats it as a single router.
    """

    def __init__(self, primary: BrokerRouter, secondaries: list[BrokerRouter] | None = None):
        self.primary = primary
        self.secondaries = secondaries or []
        # Expose primary's fields for compatibility
        self.account_id = primary.account_id
        self.auth = primary.auth

    def build_order_spec(self, direction: str, entry_model: str, entry_price: float, symbol: str, qty: int = 1) -> dict:
        return self.primary.build_order_spec(direction, entry_model, entry_price, symbol, qty)

    def submit(self, spec: dict) -> dict:
        # Primary: full result
        result = self.primary.submit(spec)

        # Secondaries: best-effort, don't block on failures
        for router in self.secondaries:
            try:
                # Check if this is an exit order
                exit_intent = spec.get("_exit_intent")
                if exit_intent is not None:
                    sec_spec = router.build_exit_spec(**exit_intent)
                else:
                    sec_spec = self._adapt_spec(spec, router)
                sec_result = router.submit(sec_spec)
                log.info(
                    "BrokerDispatcher: secondary %s submitted orderId=%s",
                    type(router).__name__,
                    sec_result.get("order_id"),
                )
            except Exception as e:
                log.error(
                    "BrokerDispatcher: secondary %s FAILED: %s (primary succeeded, continuing)",
                    type(router).__name__,
                    e,
                )

        return result

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        spec = self.primary.build_exit_spec(direction, symbol, qty)
        # Attach intent for cross-broker exit routing
        spec["_exit_intent"] = {"direction": direction, "symbol": symbol, "qty": qty}
        return spec

    def cancel(self, order_id: int) -> None:
        self.primary.cancel(order_id)
        # Secondary cancellation handled via exit fan-out, not by order_id

    def supports_native_brackets(self) -> bool:
        return self.primary.supports_native_brackets()

    def build_bracket_spec(
        self, direction: str, symbol: str, entry_price: float, stop_price: float, target_price: float, qty: int = 1
    ) -> dict | None:
        return self.primary.build_bracket_spec(direction, symbol, entry_price, stop_price, target_price, qty)

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        return self.primary.merge_bracket_into_entry(entry_spec, bracket_spec)

    def update_market_price(self, price: float) -> None:
        """Forward market price to all routers for price collar checks."""
        if hasattr(self.primary, "update_market_price"):
            self.primary.update_market_price(price)
        for router in self.secondaries:
            if hasattr(router, "update_market_price"):
                router.update_market_price(price)

    def _adapt_spec(self, primary_spec: dict, router: BrokerRouter) -> dict:
        """Adapt a primary broker's order spec for a secondary broker.

        Uses the _intent dict attached by build_order_spec() to rebuild the spec
        in the secondary broker's format. This handles cross-broker routing
        (ProjectX fields → Tradovate fields) correctly.

        If no _intent is present (same-broker fan-out), falls back to accountId swap.
        """
        intent = primary_spec.get("_intent")
        if intent is not None:
            return router.build_order_spec(**intent)
        # Same-broker fan-out (e.g., CopyOrderRouter): just swap accountId
        return {**primary_spec, "accountId": router.account_id}

    @property
    def all_routers(self) -> list[BrokerRouter]:
        """All routers (primary + secondaries) for status/health checks."""
        return [self.primary, *self.secondaries]
