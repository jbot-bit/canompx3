"""Alert 1 -- Drawdown detector.

Fires when cumulative daily R multiple falls strictly below
MonitorThresholds.daily_pnl_warn_r. Stateless / pure; caller (monitor_runner)
supplies the daily_r scalar and threshold instance, and is responsible for
per-cadence deduplication.

Canonical classifier contract: messages carry the "DRAWDOWN WARN" marker,
which alert_engine._ALERT_RULES maps to (warning, drawdown_warn).

Parallel concept (different role, do not confuse):
  trading_app.risk_manager.RiskLimits.drawdown_warning_r is a per-account
  CONFIGURABLE default used for ENFORCEMENT (trade-blocking) in the
  engine-risk path. This detector is part of the MONITORING layer and
  emits an OPERATOR ALERT via alert_engine only. Both happen to default
  to -3.0 because both derive from the 2026-02-08 Phase 6 design. The
  6e pre-reg (section 4) locks this value for the monitoring layer; risk
  manager retains per-account parametric control. If a future decision
  unifies the two, it must update docs/plans/2026-04-21-phase-6e-
  monitoring-design.md section 4 in a pre-reg-style amendment.

@revalidated-for: Phase 6e initial build (2026-04-21)
@research-source: docs/plans/2026-04-21-phase-6e-monitoring-design.md section 4
"""

from trading_app.live.monitor_thresholds import MonitorThresholds


def check_drawdown(*, daily_r: float, thresholds: MonitorThresholds) -> list[str]:
    if daily_r < thresholds.daily_pnl_warn_r:
        return [f"DRAWDOWN WARN: daily_r={daily_r:.2f} below threshold={thresholds.daily_pnl_warn_r:.1f}"]
    return []
