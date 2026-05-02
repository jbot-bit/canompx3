"""ProjectX user-hub consumer for the passive sidecar."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

from trading_app.live.projectx.auth import ProjectXAuth, USER_HUB_URL

from .policy_gate import assert_passive_sidecar_allowed

log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _event_dicts(args: Any) -> list[dict[str, Any]]:
    if isinstance(args, dict):
        return [args]
    if isinstance(args, list):
        return [item for item in args if isinstance(item, dict)]
    return []


@dataclass
class PassiveSidecarProjection:
    connection_status: str = "idle"
    last_error: str | None = None
    last_event_utc: str | None = None
    accounts_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    orders_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    positions_by_contract: dict[str, dict[str, Any]] = field(default_factory=dict)
    trades_by_id: dict[int, dict[str, Any]] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_event_utc = _utc_now_iso()

    def mark_connected(self) -> None:
        self.connection_status = "connected"
        self.last_error = None

    def mark_connecting(self) -> None:
        self.connection_status = "connecting"
        self.last_error = None

    def mark_subscribed(self) -> None:
        self.connection_status = "subscribed"
        self.last_error = None

    def mark_error(self, exc: Exception) -> None:
        self.connection_status = "error"
        self.last_error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "connection_status": self.connection_status,
            "last_error": self.last_error,
            "last_event_utc": self.last_event_utc,
            "accounts_by_id": dict(self.accounts_by_id),
            "orders_by_id": dict(self.orders_by_id),
            "positions_by_contract": dict(self.positions_by_contract),
            "trades_by_id": dict(self.trades_by_id),
        }


class PassiveSidecarDataConsumer:
    """Read-only ProjectX user-hub consumer.

    This class is intentionally limited to user-hub state consumption. It has
    no order-routing behavior and no execution affordances.
    """

    def __init__(
        self,
        *,
        auth_factory: Callable[[], ProjectXAuth] | None = None,
        user_hub_url: str = USER_HUB_URL,
        projection: PassiveSidecarProjection | None = None,
    ) -> None:
        self._auth_factory = auth_factory or ProjectXAuth
        self._user_hub_url = user_hub_url.rstrip("/")
        self.projection = projection or PassiveSidecarProjection()
        self._account_ids: tuple[int, ...] = ()
        self._auth: ProjectXAuth | None = None
        self._client: Any = None

    async def start(self, account_ids: Iterable[int]) -> None:
        assert_passive_sidecar_allowed()
        ids = tuple(dict.fromkeys(int(account_id) for account_id in account_ids))
        if not ids:
            raise ValueError("Passive sidecar requires at least one account ID")

        self._account_ids = ids
        self.projection.mark_connecting()
        self._auth = self._auth_factory()
        try:
            from pysignalr.client import SignalRClient

            token = self._auth.get_token()
            url = f"{self._user_hub_url}?access_token={token}"
            client = SignalRClient(
                url=url,
                access_token_factory=lambda: self._auth.get_token(),
                headers={"Accept": "text/plain"},
                skip_negotiation=True,
            )

            client.on("GatewayUserAccount", self._on_accounts)
            client.on("GatewayUserOrder", self._on_orders)
            client.on("GatewayUserPosition", self._on_positions)
            client.on("GatewayUserTrade", self._on_trades)
            client.on_open(lambda _client=client: self._on_connected_async(_client))

            self._client = client
            await client.run()
        except Exception as exc:
            self.projection.mark_error(exc)
            raise

    async def _on_connected_async(self, client: Any) -> None:
        self.projection.mark_connected()
        log.info("Connected to ProjectX User Hub")
        await client.send("SubscribeAccounts", [])
        for account_id in self._account_ids:
            await client.send("SubscribeOrders", [account_id])
            await client.send("SubscribePositions", [account_id])
            await client.send("SubscribeTrades", [account_id])
        self.projection.mark_subscribed()
        log.info("Subscribed to user-hub streams for %d account(s)", len(self._account_ids))

    async def _on_accounts(self, args: Any) -> None:
        for payload in _event_dicts(args):
            account_id = payload.get("id") or payload.get("accountId")
            if account_id is None:
                continue
            self.projection.accounts_by_id[int(account_id)] = dict(payload)
            self.projection.touch()

    async def _on_orders(self, args: Any) -> None:
        for payload in _event_dicts(args):
            order_id = payload.get("id") or payload.get("orderId")
            if order_id is None:
                continue
            self.projection.orders_by_id[int(order_id)] = dict(payload)
            self.projection.touch()

    async def _on_positions(self, args: Any) -> None:
        for payload in _event_dicts(args):
            account_id = payload.get("accountId")
            contract_id = payload.get("contractId")
            if account_id is None or contract_id is None:
                continue
            key = f"{int(account_id)}:{contract_id}"
            self.projection.positions_by_contract[key] = dict(payload)
            self.projection.touch()

    async def _on_trades(self, args: Any) -> None:
        for payload in _event_dicts(args):
            trade_id = payload.get("id") or payload.get("tradeId")
            if trade_id is None:
                continue
            self.projection.trades_by_id[int(trade_id)] = dict(payload)
            self.projection.touch()
