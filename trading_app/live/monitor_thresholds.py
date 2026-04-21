"""Phase 6e monitor threshold contract -- locked numeric values for operator alerts.

Source authority:
  - docs/plans/2026-04-21-phase-6e-monitoring-design.md § 4 (Pre-reg-style
    numeric contracts). That § 4 in turn cites
    docs/plans/2026-02-08-phase6-live-trading-design.md § 6e for 10 of 11 values
    and docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml
    G9 kill_criteria.shiryaev_roberts_alarm + Pepelyshev-Polunchenko 2015 Eq. 11
    for sr_alarm_arl0.

Any change to a field below requires a new pre-reg-style amendment block in the
design doc, cited by date, per docs/institutional/pre_registered_criteria.md
no-post-hoc-relaxation rule.

@revalidated-for: Phase 6e initial build (2026-04-21)
@research-source: docs/plans/2026-04-21-phase-6e-monitoring-design.md § 4
@entry-models: N/A (monitoring layer -- not strategy-level config)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorThresholds:
    # Alert 1 -- Drawdown (warn)
    daily_pnl_warn_r: float = -3.0

    # Alert 2 -- Daily Circuit Break (halt)
    daily_pnl_halt_r: float = -5.0

    # Alert 3 -- Win-rate Drift
    wr_window_trades: int = 50
    wr_delta_pp: float = 10.0

    # Alert 4 -- ExpR Drift (includes Shiryaev-Roberts alarm ARL_0 target)
    expr_window_trades: int = 50
    expr_ratio_threshold: float = 0.50
    sr_alarm_arl0: int = 1000

    # Alert 5 -- ORB Size Regime (rolling window + median ratio)
    orb_size_rolling_days: int = 30
    orb_size_median_ratio: float = 2.0

    # Alert 6 -- Missing Data (ratio of expected per-session bar count)
    missing_data_ratio: float = 0.80

    # Alert 7 -- Strategy Stale (inactivity)
    stale_days: int = 30
