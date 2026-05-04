"""Read-only rule evaluation for passive sidecar projections."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from trading_app.live.alert_engine import record_operator_alert

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, message: str) -> bool: ...


@dataclass(frozen=True)
class PassiveSidecarAlertConfig:
    stale_after_seconds: float = 120.0
    max_position_size: int | None = None
    max_loss_abs: float | None = None
    profile: str | None = None
    mode: str = "PASSIVE"


def _parse_iso_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _emit_alert(
    *,
    message: str,
    config: PassiveSidecarAlertConfig,
    notifier: Notifier | None,
) -> dict[str, object] | None:
    log.warning(message)
    alert = record_operator_alert(
        message=message,
        profile=config.profile,
        mode=config.mode,
        source="passive_sidecar",
    )
    if notifier is not None:
        try:
            notifier.send(message)
        except Exception:
            log.warning("passive sidecar notifier failed", exc_info=True)
    return alert


def evaluate_projection(
    projection: Mapping[str, Any],
    *,
    config: PassiveSidecarAlertConfig | None = None,
    notifier: Notifier | None = None,
) -> list[dict[str, object]]:
    cfg = config or PassiveSidecarAlertConfig()
    alerts: list[dict[str, object]] = []

    connection_status = str(projection.get("connection_status") or "idle").lower()
    last_error = projection.get("last_error")
    if connection_status in {"dead", "disconnected", "error"}:
        alert = _emit_alert(
            message=f"PASSIVE SIDECAR CONNECTION DOWN: status={connection_status} error={last_error or 'n/a'}",
            config=cfg,
            notifier=notifier,
        )
        if alert is not None:
            alerts.append(alert)

    last_event = _parse_iso_utc(projection.get("last_event_utc"))
    if last_event is not None and connection_status in {"connected", "healthy", "subscribed"}:
        age_seconds = (datetime.now(UTC) - last_event).total_seconds()
        if age_seconds > cfg.stale_after_seconds:
            alert = _emit_alert(
                message=(
                    f"PASSIVE SIDECAR STALE: {age_seconds:.0f}s since last user-hub event "
                    f"(threshold={cfg.stale_after_seconds:.0f}s)"
                ),
                config=cfg,
                notifier=notifier,
            )
            if alert is not None:
                alerts.append(alert)

    accounts = projection.get("accounts_by_id") or {}
    if isinstance(accounts, Mapping):
        for account_id, account in accounts.items():
            if not isinstance(account, Mapping):
                continue
            can_trade = account.get("canTrade")
            if can_trade is False:
                alert = _emit_alert(
                    message=f"PASSIVE SIDECAR ACCOUNT BLOCKED: account={account_id} canTrade=false",
                    config=cfg,
                    notifier=notifier,
                )
                if alert is not None:
                    alerts.append(alert)
                positions = projection.get("positions_by_contract") or {}
                if isinstance(positions, Mapping):
                    account_positions = [
                        position
                        for position in positions.values()
                        if isinstance(position, Mapping) and position.get("accountId") == account.get("id", account_id)
                    ]
                    if account_positions:
                        alert = _emit_alert(
                            message=(
                                "PASSIVE SIDECAR ACCOUNT/POSITION MISMATCH: "
                                f"account={account_id} canTrade=false with {len(account_positions)} open position(s)"
                            ),
                            config=cfg,
                            notifier=notifier,
                        )
                        if alert is not None:
                            alerts.append(alert)

    positions = projection.get("positions_by_contract") or {}
    if isinstance(positions, Mapping) and cfg.max_position_size is not None:
        for key, position in positions.items():
            if not isinstance(position, Mapping):
                continue
            try:
                size = abs(int(position.get("size", 0)))
            except (TypeError, ValueError):
                continue
            if size > cfg.max_position_size:
                alert = _emit_alert(
                    message=(
                        f"PASSIVE SIDECAR POSITION THRESHOLD: key={key} size={size} limit={cfg.max_position_size}"
                    ),
                    config=cfg,
                    notifier=notifier,
                )
                if alert is not None:
                    alerts.append(alert)

    accounts_pnl = projection.get("accounts_by_id") or {}
    if isinstance(accounts_pnl, Mapping) and cfg.max_loss_abs is not None:
        for account_id, account in accounts_pnl.items():
            if not isinstance(account, Mapping):
                continue
            pnl = account.get("dailyPnL")
            if pnl is None:
                pnl = account.get("profitAndLoss")
            try:
                pnl_value = float(pnl)
            except (TypeError, ValueError):
                continue
            if pnl_value < -abs(cfg.max_loss_abs):
                alert = _emit_alert(
                    message=(
                        f"PASSIVE SIDECAR LOSS THRESHOLD: account={account_id} pnl={pnl_value:.2f} "
                        f"limit={-abs(cfg.max_loss_abs):.2f}"
                    ),
                    config=cfg,
                    notifier=notifier,
                )
                if alert is not None:
                    alerts.append(alert)

    return alerts
