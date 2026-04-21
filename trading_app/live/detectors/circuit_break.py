"""Alert 2 -- Daily Circuit Break detector.

Fires when cumulative daily R multiple reaches or falls below
MonitorThresholds.daily_pnl_halt_r. At-or-below (<=) semantics --
contrast with Alert 1 Drawdown which uses strict less-than. Operator
sees both fire when the loss crosses the halt floor.

Canonical classifier contract: messages carry the "DAILY CIRCUIT BREAK"
marker, which alert_engine._ALERT_RULES maps to (critical, daily_circuit_break).

Parallel concept (different role, do not confuse):
  trading_app.risk_manager.RiskLimits.max_daily_loss_r is a per-account
  CONFIGURABLE default (-5.0) used for ENFORCEMENT (trade-blocking) in
  the engine-risk path. This detector is part of the MONITORING layer
  and emits a CRITICAL operator alert via alert_engine only. Both default
  to -5.0 because both derive from the 2026-02-08 Phase 6 design. The 6e
  pre-reg (section 4) locks this value for the monitoring layer; risk
  manager retains parametric control. Unification requires a pre-reg-
  style amendment to docs/plans/2026-04-21-phase-6e-monitoring-design.md
  section 4.

@revalidated-for: Phase 6e initial build (2026-04-21)
@research-source: docs/plans/2026-04-21-phase-6e-monitoring-design.md section 4
"""

from trading_app.live.monitor_thresholds import MonitorThresholds


def check_circuit_break(*, daily_r: float, thresholds: MonitorThresholds) -> list[str]:
    if daily_r <= thresholds.daily_pnl_halt_r:
        return [f"DAILY CIRCUIT BREAK: daily_r={daily_r:.2f} at/below threshold={thresholds.daily_pnl_halt_r:.1f}"]
    return []
