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
        **kwargs,
    ):
        self.auth = auth
        self.on_bar = on_bar

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


class BrokerPositions(ABC):
    """Query broker for open positions (crash recovery + EOD reconciliation)."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        self.auth = auth

    @abstractmethod
    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions: [{contract_id, side, size, avg_price}]."""
        ...
