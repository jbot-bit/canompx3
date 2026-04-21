"""Alert 3 -- Win-rate Drift detector.

Fires when a strategy's rolling win rate has dropped by at least
thresholds.wr_delta_pp percentage points below its backtest baseline,
AND the rolling window has accumulated at least thresholds.wr_window_trades
trades (under-window returns [] -- UNVERIFIED, not DEAD, per
feedback_oos_power_floor.md).

Pure function. Caller (monitor_runner, sub-step 2.i) is responsible for
computing rolling_wr from paper_trades (or a future PerformanceMonitor
rolling accessor) and supplying each strategy's baseline_wr from the
PortfolioStrategy record.

Canonical classifier contract: messages carry the "WR DRIFT" marker,
which alert_engine._ALERT_RULES maps to (warning, wr_drift).

Units:
  - rolling_wr, baseline_wr: fractions in [0, 1]
  - wr_delta_pp: percentage points (10 means 10pp, i.e. 0.10 fraction drop)

@revalidated-for: Phase 6e initial build (2026-04-21)
@research-source: docs/plans/2026-04-21-phase-6e-monitoring-design.md section 4
"""

from trading_app.live.monitor_thresholds import MonitorThresholds

# Float-precision tolerance for boundary compare. Without this, a literal
# 10pp drop computed as (0.60 - 0.50) * 100 = 9.999... underflows and fails
# the >= gate. 1e-9 pp is vastly below any operator-relevant precision.
_WR_DROP_TOLERANCE_PP = 1e-9


def check_wr_drift(
    *,
    strategy_id: str,
    rolling_wr: float,
    baseline_wr: float,
    n_trades: int,
    thresholds: MonitorThresholds,
) -> list[str]:
    if n_trades < thresholds.wr_window_trades:
        return []
    drop_pp = (baseline_wr - rolling_wr) * 100.0
    if drop_pp + _WR_DROP_TOLERANCE_PP < thresholds.wr_delta_pp:
        return []
    return [
        f"WR DRIFT: {strategy_id} rolling_wr={rolling_wr * 100:.1f}% "
        f"baseline={baseline_wr * 100:.1f}% drop={drop_pp:.1f}pp after {n_trades} trades"
    ]
