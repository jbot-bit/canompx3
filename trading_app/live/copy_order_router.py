"""CopyOrderRouter — fan out orders to N accounts.

Single signal -> replicate orders across N accounts (one primary + N-1 shadows).
Primary account: full position tracking, fill polling, DD monitoring.
Shadow accounts: same orders, best-effort. Broker-side DD enforcement.

Architecture:
    ONE auth token -> ONE DataFeed -> ONE SessionOrchestrator -> CopyOrderRouter
    CopyOrderRouter wraps N ProjectXOrderRouters (one per account_id).
    submit() -> primary first, then shadows. Returns primary result.

Rate budget: 200 req/60s (ProjectX). 5 accounts x 1 order each = 5 requests.

@audit-finding F-2b Cross-account divergence detection:
@canonical-source docs/research-input/topstep/topstep_cross_account_hedging.md
@verbatim "You remain fully responsible for all activity across your accounts,
           including positions created through... Automated trading systems.
           Any third-party tools."

Shadow submission failures used to be silently logged-and-continued, which
allowed primary and shadows to drift out of sync. A subsequent opposing-
direction entry could create cross-account hedging (prohibited under
TopStep's 3-strike progressive enforcement, ending in PERMANENT account
closure). Stage 5 fix: track divergence state, raise ShadowDivergenceError
on the NEXT submit after a shadow failure, and expose is_degraded() so the
orchestrator can halt proactively.
"""

import logging

from .broker_base import BrokerRouter

log = logging.getLogger(__name__)


class ShadowDivergenceError(RuntimeError):
    """Raised when CopyOrderRouter detects shadow account divergence.

    Indicates that one or more shadow accounts have failed an operation
    (submit, cancel, or bracket) such that their position state may differ
    from the primary. The orchestrator should halt the session and require
    manual reconciliation before resuming.
    """


class CopyOrderRouter(BrokerRouter):
    """Wraps a primary + N shadow OrderRouters. Same interface as BrokerRouter."""

    def __init__(self, primary: BrokerRouter, shadows: list[BrokerRouter]):
        super().__init__(account_id=primary.account_id, auth=primary.auth)
        self.primary = primary
        self.shadows = shadows
        # F-2b divergence tracking. Maps shadow account_id → last failure reason string.
        # An empty dict means no detected divergence. A non-empty dict means at least
        # one shadow failed an operation and the router is in DEGRADED state.
        self._shadow_failures: dict[int, str] = {}

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

        F-2b: Refuses to submit if the router is already in degraded state from
        a previous shadow failure (raises ShadowDivergenceError). On a fresh
        shadow failure, marks the router degraded so the NEXT submit will raise.

        Synchronous (requests-based) to match ProjectXOrderRouter.submit() interface.
        """
        # F-2b: Refuse new entries if any prior shadow operation diverged.
        # Continuing would risk creating asymmetric / hedged positions across copies.
        if self._shadow_failures:
            raise ShadowDivergenceError(
                f"CopyOrderRouter is DEGRADED — refusing submit. "
                f"Diverged shadow accounts: {dict(self._shadow_failures)}. "
                f"Manual reconciliation required before resuming."
            )

        result = self.primary.submit(spec)
        primary_status = result.get("status", "unknown")

        # Inverted check: skip shadows only on known failures. This ensures
        # compatibility across brokers (ProjectX: "Filled"/"Working",
        # Rithmic: "submitted") without maintaining a whitelist.
        _SKIP_STATUSES = ("rejected", "error", "cancelled")
        if primary_status.lower() in _SKIP_STATUSES:
            log.info("Primary status=%s — skipping shadow copies", primary_status)
            return result

        for shadow in self.shadows:
            try:
                shadow_result = shadow.submit(spec)
                log.info(
                    "Shadow copy account %s: %s (order_id=%s)",
                    shadow.account_id,
                    shadow_result.get("status", "unknown"),
                    shadow_result.get("order_id"),
                )
            except Exception as exc:
                # F-2b: mark router degraded so the next submit raises.
                # Do NOT raise here — primary already filled, we need to return
                # its result so the orchestrator can manage the position.
                self._shadow_failures[shadow.account_id] = f"submit: {type(exc).__name__}: {exc}"
                log.critical(
                    "F-2b SHADOW DIVERGENCE: account %s submit failed — "
                    "primary already filled (order_id=%s). Router DEGRADED. "
                    "Next submit will raise ShadowDivergenceError.",
                    shadow.account_id,
                    result.get("order_id"),
                    exc_info=True,
                )

        return result

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        """Delegate to primary."""
        return self.primary.build_exit_spec(direction, symbol, qty)

    def cancel(self, order_id: int) -> None:
        """Cancel on primary (fail-closed), then best-effort cancel shadows.

        F-2b: shadow cancel failures mark the router degraded. The next submit
        will raise. Cancel itself does not raise on shadow failure (idempotent
        cleanup is fine to retry from the orchestrator path).
        """
        self.primary.cancel(order_id)
        for shadow in self.shadows:
            try:
                shadow.cancel(order_id)
                log.info("Shadow account %s: order %s cancelled", shadow.account_id, order_id)
            except Exception as exc:
                self._shadow_failures[shadow.account_id] = (
                    f"cancel(order_id={order_id}): {type(exc).__name__}: {exc}"
                )
                log.critical(
                    "F-2b SHADOW DIVERGENCE: account %s cancel order %s failed — "
                    "router DEGRADED. Next submit will raise ShadowDivergenceError.",
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

    def has_queryable_bracket_legs(self) -> bool:
        """Delegate to primary — wrapper inherits the primary's bracket model."""
        return self.primary.has_queryable_bracket_legs()

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

    # ─── F-2b divergence detection API ───────────────────────────────
    # @canonical-source docs/research-input/topstep/topstep_cross_account_hedging.md
    # @audit-finding F-2b — orchestrator integration point for proactive halt-on-divergence

    def is_degraded(self) -> bool:
        """True if any shadow operation has failed since the last reset.

        The orchestrator should call this after every poll cycle (e.g. in the
        bar heartbeat loop) and halt the session if it returns True.
        """
        return bool(self._shadow_failures)

    def degraded_accounts(self) -> dict[int, str]:
        """Return a copy of {account_id: failure_reason} for diagnostics.

        Empty dict means no divergence detected.
        """
        return dict(self._shadow_failures)

    def clear_degraded(self) -> None:
        """Clear divergence state after MANUAL reconciliation.

        This should ONLY be called after an operator has verified that all
        shadow accounts are back in sync with the primary (e.g. all flat,
        or all holding the same position). Calling this with active divergent
        positions defeats the F-2b safety guard.
        """
        if self._shadow_failures:
            log.warning(
                "F-2b: clearing degraded state. Operator must have manually "
                "reconciled accounts: %s",
                dict(self._shadow_failures),
            )
        self._shadow_failures.clear()
