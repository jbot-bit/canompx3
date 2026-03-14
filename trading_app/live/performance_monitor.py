"""
Per-strategy live P&L tracking with CUSUM drift detection.

Takes list[PortfolioStrategy] from Portfolio.strategies — NOT LiveStrategySpec.
PortfolioStrategy has .strategy_id and .expectancy_r. LiveStrategySpec does not.

Trade persistence is handled by TradeJournal (live_journal.db), NOT this module.
This module is in-memory only — it tracks daily P&L and fires CUSUM alarms.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from trading_app.portfolio import PortfolioStrategy

from .cusum_monitor import CUSUMMonitor

log = logging.getLogger(__name__)


def _compute_std_r(win_rate: float, rr_target: float, expectancy_r: float) -> float:
    """Theoretical std of R outcomes for a fixed-RR strategy.

    Wins pay +RR, losses cost -1.0. Formula is binary outcome variance.
    @research-source: binary outcome variance for fixed risk-reward
    """
    return math.sqrt(win_rate * (rr_target - expectancy_r) ** 2 + (1 - win_rate) * (-1.0 - expectancy_r) ** 2)


@dataclass
class TradeRecord:
    strategy_id: str
    trading_day: date
    direction: str
    entry_price: float
    exit_price: float
    actual_r: float
    expected_r: float
    slippage_pts: float = 0.0
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

    # CUSUM alarm threshold in standard deviation units.
    # 4.0σ ≈ conservative: avoids false positives over ~100-trade windows
    # while catching genuine regime change.
    # @research-source arXiv:1509.01570 (Real-time financial surveillance via CUSUM)
    # @revalidated-for E1/E2 event-based sessions (Mar 2026)
    CUSUM_THRESHOLD: float = 4.0

    def __init__(self, strategies: list[PortfolioStrategy]):
        self._monitors: dict[str, CUSUMMonitor] = {
            s.strategy_id: CUSUMMonitor(
                expected_r=s.expectancy_r,
                std_r=_compute_std_r(s.win_rate, s.rr_target, s.expectancy_r),
                threshold=self.CUSUM_THRESHOLD,
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
        total_slippage = sum(t.slippage_pts for t in self._trades)
        return {
            "date": date.today().isoformat(),
            "total_r": round(sum(self._daily_r.values()), 4),
            "by_strategy": dict(self._daily_r),
            "n_trades": len(self._trades),
            "total_slippage_pts": round(total_slippage, 4),
            "alarms": [sid for sid, m in self._monitors.items() if m.alarm_triggered],
        }

    def reset_daily(self) -> None:
        """Clear daily accumulators and CUSUM monitors (call at EOD after logging summary).

        Design note: CUSUM monitors are intentionally reset at the daily boundary.
        Trade-off: genuine multi-day drift resets at EOD and must re-accumulate.
        Alternative (no reset) was worse — a single bad day permanently kills drift
        detection for the rest of the session. Operators see the alarm in daily_summary()
        and can investigate before the next trading day begins.
        """
        self._daily_r.clear()
        self._trades.clear()
        for monitor in self._monitors.values():
            monitor.clear()

    @property
    def trade_count(self) -> int:
        """Number of trades recorded today (resets on reset_daily)."""
        return len(self._trades)

    def get_cusum(self, strategy_id: str) -> CUSUMMonitor | None:
        """Return the CUSUM monitor for a specific strategy (for inspection/testing)."""
        return self._monitors.get(strategy_id)
