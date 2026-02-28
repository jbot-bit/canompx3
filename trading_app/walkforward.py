"""Walk-forward validation for ORB breakout strategies (Phase 4b).

Anchored expanding walk-forward with non-overlapping test windows.
Runs between Phase 4 (stress test) and promotion to validated_setups.

No new DB tables. Results written to JSONL file (append-only).
"""

import json
import logging
import calendar
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path

from trading_app.strategy_fitness import _load_strategy_outcomes
from trading_app.strategy_discovery import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation for a single strategy."""

    strategy_id: str
    instrument: str
    n_total_windows: int
    n_valid_windows: int
    n_positive_windows: int
    pct_positive: float
    agg_oos_exp_r: float
    total_oos_trades: int
    passed: bool
    rejection_reason: str | None
    windows: list[dict]
    params: dict
    window_imbalance_ratio: float | None = None
    window_imbalanced: bool = False


def _add_months(d: date, months: int) -> date:
    """Add calendar months to a date, clamping day to month end."""
    total_months = d.year * 12 + (d.month - 1) + months
    year = total_months // 12
    month = total_months % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def run_walkforward(
    con,
    strategy_id: str,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    orb_minutes: int,
    test_window_months: int = 6,
    min_train_months: int = 12,
    min_trades_per_window: int = 15,
    min_valid_windows: int = 3,
    min_pct_positive: float = 0.60,
    dst_regime: str | None = None,
    wf_start_date: date | None = None,
) -> WalkForwardResult:
    """
    Anchored walk-forward validation on existing orb_outcomes.

    Loads filtered outcomes, splits into non-overlapping test windows,
    and evaluates OOS performance consistency.

    Args:
        dst_regime: If 'winter' or 'summer', restrict DST-affected sessions to
            that regime only (matches the _W/_S strategy_id suffix from discovery).

    Pass rule (ALL 4 required, fail-closed):
      1. n_valid >= min_valid_windows
      2. pct_positive >= min_pct_positive
      3. agg_oos_exp_r > 0 (trade-weighted)
      4. total_oos_trades >= min_trades_per_window * min_valid_windows
    """
    params = {
        "test_window_months": test_window_months,
        "min_train_months": min_train_months,
        "min_trades_per_window": min_trades_per_window,
        "min_valid_windows": min_valid_windows,
        "min_pct_positive": min_pct_positive,
    }

    # Load all filtered outcomes (no date restriction)
    outcomes = _load_strategy_outcomes(
        con,
        instrument=instrument,
        orb_label=orb_label,
        orb_minutes=orb_minutes,
        entry_model=entry_model,
        rr_target=rr_target,
        confirm_bars=confirm_bars,
        filter_type=filter_type,
        dst_regime=dst_regime,
    )

    if not outcomes:
        return WalkForwardResult(
            strategy_id=strategy_id, instrument=instrument,
            n_total_windows=0, n_valid_windows=0, n_positive_windows=0,
            pct_positive=0.0, agg_oos_exp_r=0.0, total_oos_trades=0,
            passed=False, rejection_reason="No outcomes found",
            windows=[], params=params,
        )

    trading_days = [o["trading_day"] for o in outcomes]
    earliest = min(trading_days)
    latest = max(trading_days)

    # Generate non-overlapping test windows
    # Apply per-instrument WF start override (skip regime-shifted early data)
    anchor = max(earliest, wf_start_date) if wf_start_date else earliest
    windows = []
    window_start = _add_months(anchor, min_train_months)

    while window_start <= latest:
        window_end = _add_months(window_start, test_window_months)

        test_outcomes = [
            o for o in outcomes
            if window_start <= o["trading_day"] < window_end
        ]

        metrics = compute_metrics(test_outcomes)
        test_n = metrics["sample_size"]
        test_exp_r = metrics["expectancy_r"]

        windows.append({
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "test_n": test_n,
            "test_exp_r": test_exp_r,
            "test_wr": metrics["win_rate"],
            "test_sharpe": metrics["sharpe_ratio"],
            "test_pass": (
                test_n >= min_trades_per_window
                and test_exp_r is not None
                and test_exp_r > 0
            ),
        })

        logger.info(
            "WF %s window %s..%s: N=%d ExpR=%s",
            strategy_id, window_start, window_end, test_n, test_exp_r,
        )

        window_start = window_end

    if not windows:
        return WalkForwardResult(
            strategy_id=strategy_id, instrument=instrument,
            n_total_windows=0, n_valid_windows=0, n_positive_windows=0,
            pct_positive=0.0, agg_oos_exp_r=0.0, total_oos_trades=0,
            passed=False,
            rejection_reason=(
                f"All {len(outcomes)} outcomes in training period "
                f"({min_train_months}mo). No test windows."
            ),
            windows=[], params=params,
        )

    # Aggregate valid windows (test_n >= threshold)
    valid_windows = [w for w in windows if w["test_n"] >= min_trades_per_window]
    n_valid = len(valid_windows)
    n_positive = sum(
        1 for w in valid_windows
        if w["test_exp_r"] is not None and w["test_exp_r"] > 0
    )
    pct_positive = n_positive / n_valid if n_valid > 0 else 0.0
    total_oos_trades = sum(w["test_n"] for w in valid_windows)

    # Trade-weighted aggregate OOS ExpR
    if total_oos_trades > 0:
        agg_oos_exp_r = sum(
            w["test_exp_r"] * w["test_n"]
            for w in valid_windows
            if w["test_exp_r"] is not None
        ) / total_oos_trades
    else:
        agg_oos_exp_r = 0.0

    # Window imbalance detection
    window_counts = [w["test_n"] for w in valid_windows if w["test_n"] > 0]
    window_imbalance_ratio = None
    window_imbalanced = False
    if len(window_counts) >= 2:
        window_imbalance_ratio = round(max(window_counts) / max(min(window_counts), 1), 1)
        window_imbalanced = window_imbalance_ratio > 5.0

    # Pass rule (ALL 4 required, fail-closed)
    oos_trade_floor = min_trades_per_window * min_valid_windows
    rejection_reason = None

    if n_valid < min_valid_windows:
        rejection_reason = (
            f"Insufficient valid windows: {n_valid} < {min_valid_windows}"
        )
    elif pct_positive < min_pct_positive:
        rejection_reason = (
            f"Too few positive windows: {pct_positive:.0%} < {min_pct_positive:.0%}"
        )
    elif agg_oos_exp_r <= 0:
        rejection_reason = (
            f"Negative aggregate OOS ExpR: {agg_oos_exp_r:.4f}"
        )
    elif total_oos_trades < oos_trade_floor:
        rejection_reason = (
            f"Total OOS trades {total_oos_trades} < floor {oos_trade_floor}"
        )

    passed = rejection_reason is None

    if passed:
        logger.warning(
            "WF PASS %s: %d/%d windows positive (%.0f%%), "
            "agg_ExpR=%.4f, N=%d",
            strategy_id, n_positive, n_valid, pct_positive * 100,
            agg_oos_exp_r, total_oos_trades,
        )
    else:
        logger.warning("WF FAIL %s: %s", strategy_id, rejection_reason)

    return WalkForwardResult(
        strategy_id=strategy_id, instrument=instrument,
        n_total_windows=len(windows), n_valid_windows=n_valid,
        n_positive_windows=n_positive,
        pct_positive=round(pct_positive, 4),
        agg_oos_exp_r=round(agg_oos_exp_r, 4),
        total_oos_trades=total_oos_trades,
        passed=passed, rejection_reason=rejection_reason,
        windows=windows, params=params,
        window_imbalance_ratio=window_imbalance_ratio,
        window_imbalanced=window_imbalanced,
    )


def append_walkforward_result(result: WalkForwardResult, output_path: str | Path):
    """Append a WalkForwardResult as one JSON line to the output file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = asdict(result)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
