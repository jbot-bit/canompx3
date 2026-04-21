"""Alert 4a -- ExpR Ratio Drift detector (CRITICAL).

Fires when a strategy's rolling mean expectancy (R per trade) has fallen
STRICTLY below thresholds.expr_ratio_threshold * baseline_expr, AND the
rolling window has accumulated at least thresholds.expr_window_trades
trades, AND the baseline is strictly positive.

Per 2026-02-08 Phase 6 spec line 425:
    "Rolling 50-trade ExpR < 50% of backtest | CRITICAL"

Pure function. Caller (monitor_runner, sub-step 2.i) is responsible for
computing rolling_expr from paper_trades or PerformanceMonitor state and
supplying each strategy's baseline expectancy from its PortfolioStrategy
record.

Alert 4 composite: this detector (4a) handles the ratio gate. The
Shiryaev-Roberts statistical alarm (4b) lives in sr_alarm.py per
Pepelyshev-Polunchenko 2015 Eq. 11 / 17-18 with ARL_0 = 1000.

Canonical classifier contract: messages carry the "EXPR DRIFT" marker,
which alert_engine._ALERT_RULES maps to (critical, expr_drift).

Edge cases (pre-registered in
docs/runtime/stages/phase-6e-detector-expr-ratio.md):
  - baseline_expr <= 0 -> returns []. Non-positive baseline makes the
    ratio gate semantically undefined. Upstream validation (portfolio
    construction / strategy discovery) is expected to reject non-
    positive-expectancy strategies before they reach monitoring. If one
    leaks through, this detector stays silent rather than emit spurious
    signals; the config gap is the upstream's alert, not Alert 4a's.
  - rolling_expr == expr_ratio_threshold * baseline_expr exactly -> no
    fire (strict `<` per spec).
  - No float-tolerance added: the RHS is a single multiplication (no
    subtraction of near-equal floats), so IEEE-754 boundary noise is
    not operator-relevant at this scale.

Units:
  - rolling_expr, baseline_expr: R multiples per trade (any real number;
    baseline typically 0.1-0.5 for viable strategies).
  - expr_ratio_threshold: dimensionless fraction in [0, 1].

@revalidated-for: Phase 6e initial build (2026-04-21)
@research-source: docs/plans/2026-04-21-phase-6e-monitoring-design.md section 4
"""

import math

from trading_app.live.monitor_thresholds import MonitorThresholds


def check_expr_ratio(
    *,
    strategy_id: str,
    rolling_expr: float,
    baseline_expr: float,
    n_trades: int,
    thresholds: MonitorThresholds,
) -> list[str]:
    # NaN input => upstream data corruption (distinct alert class); stay silent here.
    if math.isnan(rolling_expr) or math.isnan(baseline_expr):
        return []
    if n_trades < thresholds.expr_window_trades:
        return []
    if baseline_expr <= 0:
        return []
    fire_line = thresholds.expr_ratio_threshold * baseline_expr
    if rolling_expr >= fire_line:
        return []
    ratio = rolling_expr / baseline_expr
    return [
        f"EXPR DRIFT: {strategy_id} rolling_expr={rolling_expr:.3f}R "
        f"baseline={baseline_expr:.3f}R ratio={ratio:.2f} after {n_trades} trades"
    ]
