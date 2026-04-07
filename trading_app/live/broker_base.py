"""Abstract base classes for multi-broker support.

All broker implementations (Tradovate, ProjectX) must implement these ABCs.
SessionOrchestrator depends ONLY on these interfaces, never on concrete classes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from .bar_aggregator import Bar


class BrokerAuth(ABC):
    """Authenticate with a broker and manage session tokens."""

    @abstractmethod
    def get_token(self) -> str:
        """Return a valid session/access token, refreshing if needed."""
        ...

    @abstractmethod
    def headers(self) -> dict:
        """Return Authorization header dict for REST calls."""
        ...

    @abstractmethod
    def refresh_if_needed(self) -> None:
        """Proactively refresh token if near expiry."""
        ...


class BrokerFeed(ABC):
    """Stream real-time market data and produce 1-minute bars."""

    def __init__(
        self,
        auth: BrokerAuth,
        on_bar: Callable[[Bar], Coroutine[Any, Any, None]],
        on_stale: Callable[[float, int], None] | None = None,
        **kwargs,
    ):
        self.auth = auth
        self.on_bar = on_bar
        self.on_stale = on_stale  # callback(seconds_since_last_data, stale_count)

    @abstractmethod
    async def run(self, symbol: str) -> None:
        """Connect and stream bars. Reconnects on disconnect."""
        ...

    @property
    def was_stopped(self) -> bool:
        """Whether feed was stopped by user request (stop-file), not by exhausting reconnects."""
        return getattr(self, "_stop_requested", False)

    @abstractmethod
    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end."""
        ...


class BrokerRouter(ABC):
    """Route orders to the broker."""

    def __init__(self, account_id: int, auth: BrokerAuth | None, **kwargs):
        self.account_id = account_id
        self.auth = auth

    @abstractmethod
    def build_order_spec(
        self,
        direction: str,
        entry_model: str,
        entry_price: float,
        symbol: str,
        qty: int = 1,
    ) -> dict:
        """Build a broker-specific order payload."""
        ...

    @abstractmethod
    def submit(self, spec: dict) -> dict:
        """Submit order, return {order_id, status}."""
        ...

    @abstractmethod
    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        """Build a market close order."""
        ...

    @abstractmethod
    def cancel(self, order_id: int) -> None:
        """Cancel an open order."""
        ...

    @abstractmethod
    def supports_native_brackets(self) -> bool:
        """Whether broker supports stop/target brackets on entry order."""
        ...

    def update_market_price(self, price: float) -> None:  # noqa: B027
        """Update last known market price for price collar checks. Default: no-op."""

    def build_bracket_spec(
        self,
        direction: str,
        symbol: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        qty: int = 1,
    ) -> dict | None:
        """Build OCO bracket (stop + target). Returns None if not supported."""
        return None

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        """Merge bracket fields into entry spec for atomic submission. Default: no-op."""
        return entry_spec

    def query_order_status(self, order_id: int) -> dict:
        """Query order status. Returns {order_id, status, fill_price}.

        Status values: 'Filled', 'Working', 'Cancelled', 'Rejected'.
        Raises NotImplementedError if broker doesn't support it.
        """
        raise NotImplementedError

    def cancel_bracket_orders(self, contract_id: str) -> int:
        """Cancel orphaned bracket orders for a contract.

        Returns the number of orders cancelled. Used on startup to clean up
        brackets left over from a prior crashed session.

        Default implementation returns 0 (no-op). Concrete routers override
        to query and cancel broker-side bracket legs.
        """
        return 0

    def verify_bracket_legs(
        self, entry_order_id: int, contract_id: str
    ) -> tuple[int | None, int | None]:
        """Verify that bracket legs (SL + TP) were actually created at the broker.

        Returns (stop_loss_order_id, take_profit_order_id). Both None if the
        broker does not support bracket verification (e.g. Rithmic with native
        server-side brackets does not need verification).

        Only meaningful for brokers that merge brackets into the entry order
        and create separate child orders (ProjectX AutoBracket). For
        server-side bracket brokers, legs are atomic with entry — no verify
        needed.

        IMPORTANT: Callers MUST check `has_queryable_bracket_legs()` before
        calling this method. The default (None, None) return is NOT a signal
        of failure — it is the correct answer for brokers without separately-
        queryable legs. The flag check is the only way to distinguish.
        """
        return (None, None)

    def has_queryable_bracket_legs(self) -> bool:
        """Whether this broker's bracket legs are separately-queryable orders.

        True  — bracket SL/TP legs are distinct child orders with their own
                order IDs that can be fetched via `verify_bracket_legs()`.
                Example: ProjectX AutoBracket creates entry_id+1 for SL and
                entry_id+2 for TP, both returned by searchOpen.

        False — bracket legs are atomic with the entry submission and managed
                server-side. No separate SL/TP order IDs exist to query.
                Example: Rithmic native brackets. The broker manages leg
                cancellation on exit-order submission, so the bot does not
                need to track leg IDs.

        session_orchestrator checks this flag before calling verify_bracket_legs.
        When False, the verification path is skipped entirely — no alarm, no
        fallback, no tracked leg IDs. When True, verify_bracket_legs must
        return real order IDs.

        Default: True (conservative — assume broker exposes legs and failing
        verification means a real problem).
        """
        return True


class BrokerContracts(ABC):
    """Resolve accounts and contract symbols."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        self.auth = auth

    @abstractmethod
    def resolve_account_id(self) -> int:
        """Return the numeric account ID for trading."""
        ...

    @abstractmethod
    def resolve_front_month(self, instrument: str) -> str:
        """Return current front-month contract symbol for an instrument."""
        ...

    def resolve_all_account_ids(self) -> list[tuple[int, str]]:
        """Return ALL active account IDs and names for copy trading.

        Returns list of (account_id, account_name) tuples.
        Default: raises NotImplementedError. Override in broker-specific subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-account discovery. "
            "Copy trading requires a broker that implements resolve_all_account_ids()."
        )


class BrokerPositions(ABC):
    """Query broker for open positions (crash recovery + EOD reconciliation)."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        self.auth = auth

    @abstractmethod
    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions: [{contract_id, side, size, avg_price}]."""
        ...

    def query_equity(self, account_id: int) -> float | None:
        """Return current account equity in dollars. None if unavailable.

        Default: not implemented (returns None). Override in broker-specific classes.
        Used by AccountHWMTracker for cross-session DD tracking.
        """
        return None
