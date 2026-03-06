"""
Per-strategy live P&L tracking with CUSUM drift detection.

Takes list[PortfolioStrategy] from Portfolio.strategies — NOT LiveStrategySpec.
PortfolioStrategy has .strategy_id and .expectancy_r. LiveStrategySpec does not.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from trading_app.portfolio import PortfolioStrategy

from .cusum_monitor import CUSUMMonitor

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    strategy_id: str
    trading_day: date
    direction: str
    entry_price: float
    exit_price: float
    actual_r: float
    expected_r: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class PerformanceMonitor:
    """
    Tracks live P&L per strategy and triggers CUSUM alarms on drift.

    Usage:
        portfolio, notes = build_live_portfolio(...)
        monitor = PerformanceMonitor(portfolio.strategies)
        ...
        alert = monitor.record_trade(record)
        if alert:
            send_notification(alert)
    """

    def __init__(self, strategies: list[PortfolioStrategy]):
        self._monitors: dict[str, CUSUMMonitor] = {
            s.strategy_id: CUSUMMonitor(
                expected_r=s.expectancy_r,
                std_r=1.0,
                threshold=4.0,
            )
            for s in strategies
        }
        self._strategy_map: dict[str, PortfolioStrategy] = {s.strategy_id: s for s in strategies}
        self._trades: list[TradeRecord] = []
        self._daily_r: dict[str, float] = {}

    def record_trade(self, record: TradeRecord) -> str | None:
        """
        Record a completed trade. Returns alert string if CUSUM alarm triggered, else None.
        """
        self._trades.append(record)
        self._daily_r[record.strategy_id] = self._daily_r.get(record.strategy_id, 0.0) + record.actual_r
        monitor = self._monitors.get(record.strategy_id)
        if monitor and monitor.update(record.actual_r):
            msg = (
                f"⚠ CUSUM ALARM: {record.strategy_id} "
                f"drift={monitor.drift_severity:.2f}σ after {monitor.n_trades} trades"
            )
            log.warning(msg)
            return msg
        return None

    def daily_summary(self) -> dict:
        """Return EOD summary dict for logging."""
        return {
            "date": date.today().isoformat(),
            "total_r": round(sum(self._daily_r.values()), 4),
            "by_strategy": dict(self._daily_r),
            "n_trades": len(self._trades),
            "alarms": [sid for sid, m in self._monitors.items() if m.alarm_triggered],
        }

    def reset_daily(self) -> None:
        """Clear daily R accumulator (call at EOD after logging summary)."""
        self._daily_r.clear()

    def get_cusum(self, strategy_id: str) -> CUSUMMonitor | None:
        """Return the CUSUM monitor for a specific strategy (for inspection/testing)."""
        return self._monitors.get(strategy_id)
