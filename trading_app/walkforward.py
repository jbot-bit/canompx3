"""Walk-forward validation for ORB breakout strategies (Phase 4b).

Anchored expanding walk-forward with non-overlapping test windows.
Runs between Phase 4 (stress test) and promotion to validated_setups.

No new DB tables. Results written to JSONL file (append-only).
"""

import calendar
import json
import logging
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from trading_app.config import apply_tight_stop
from trading_app.strategy_discovery import compute_metrics
from trading_app.strategy_fitness import _load_strategy_outcomes

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
    wfe: float | None = None  # Walk-Forward Efficiency = OOS/IS ExpR (Pardo)


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
    stop_multiplier: float = 1.0,
    cost_spec=None,
    test_window_trades: int | None = None,
    min_train_trades: int | None = None,
) -> WalkForwardResult:
    """
    Anchored walk-forward validation on existing orb_outcomes.

    Loads filtered outcomes, splits into non-overlapping test windows,
    and evaluates OOS performance consistency.

    Args:
        dst_regime: If 'winter' or 'summer', restrict DST-affected sessions to
            that regime only (matches the _W/_S strategy_id suffix from discovery).
        wf_start_date: Optional per-instrument anchor override. When set,
            window generation starts from max(earliest_outcome, wf_start_date)
            instead of earliest_outcome. Full-sample outcomes still load from
            earliest date — only the WF test window anchor shifts.

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
            strategy_id=strategy_id,
            instrument=instrument,
            n_total_windows=0,
            n_valid_windows=0,
            n_positive_windows=0,
            pct_positive=0.0,
            agg_oos_exp_r=0.0,
            total_oos_trades=0,
            passed=False,
            rejection_reason="No outcomes found",
            windows=[],
            params=params,
        )

    # Apply tight stop simulation (no-op for 1.0x)
    if stop_multiplier != 1.0 and cost_spec is not None:
        outcomes = apply_tight_stop(outcomes, stop_multiplier, cost_spec)

    # Pre-sort outcomes by trading_day (should already be sorted from DB, make explicit)
    outcomes.sort(key=lambda o: o["trading_day"])
    all_trading_days = [o["trading_day"] for o in outcomes]
    earliest = all_trading_days[0]
    latest = all_trading_days[-1]

    # Generate non-overlapping test windows
    # Apply per-instrument WF start override (skip regime-shifted early data)
    anchor = max(earliest, wf_start_date) if wf_start_date else earliest
    windows = []

    if test_window_trades is not None:
        # ── Trade-count mode (AFML Ch.2 information-driven sampling) ──────
        # @research-source Lopez de Prado AFML Ch.2 — information-driven bars;
        #   window by trade count for regime-spanning OOS validation
        # @entry-models E1/E2
        # @revalidated-for E1/E2 event-based sessions (2026-03-17)
        min_is = min_train_trades if min_train_trades is not None else test_window_trades
        # Apply anchor: skip outcomes before wf_start_date
        anchor_idx = bisect_left(all_trading_days, anchor)
        usable = outcomes[anchor_idx:]
        usable_days = all_trading_days[anchor_idx:]

        idx = min_is
        while idx + test_window_trades <= len(usable):
            is_outcomes = usable[:idx]
            oos_outcomes = usable[idx : idx + test_window_trades]

            metrics = compute_metrics(oos_outcomes)
            is_metrics = compute_metrics(is_outcomes) if len(is_outcomes) >= 15 else None
            is_exp_r = is_metrics["expectancy_r"] if is_metrics else None

            windows.append(
                {
                    "window_start": usable_days[idx].isoformat(),
                    "window_end": usable_days[idx + test_window_trades - 1].isoformat(),
                    "test_n": metrics["sample_size"],
                    "test_exp_r": metrics["expectancy_r"],
                    "test_wr": metrics["win_rate"],
                    "test_sharpe": metrics["sharpe_ratio"],
                    "test_pass": metrics["expectancy_r"] is not None and metrics["expectancy_r"] > 0,
                    "is_exp_r": is_exp_r,
                }
            )

            logger.info(
                "WF %s trade-count window [%d:%d] %s..%s: N=%d ExpR=%s",
                strategy_id,
                idx,
                idx + test_window_trades,
                usable_days[idx],
                usable_days[idx + test_window_trades - 1],
                metrics["sample_size"],
                metrics["expectancy_r"],
            )

            idx += test_window_trades
        params["mode"] = "trade_count"
        params["test_window_trades"] = test_window_trades
        params["min_train_trades"] = min_is
    else:
        # ── Calendar mode (existing, unchanged) ──────────────────────────
        window_start = _add_months(anchor, min_train_months)

        while window_start <= latest:
            window_end = _add_months(window_start, test_window_months)

            # O(log N) window slicing via bisect
            lo = bisect_left(all_trading_days, window_start)
            hi = bisect_left(all_trading_days, window_end)
            test_outcomes = outcomes[lo:hi]

            metrics = compute_metrics(test_outcomes)
            test_n = metrics["sample_size"]
            test_exp_r = metrics["expectancy_r"]

            # IS metrics: all outcomes before this window (anchored expanding)
            # @research-source Lopez de Prado AFML Ch.11 — minimum IS observations for stable estimate;
            #   consistent with wf_min_trades=15 in strategy_validator.py
            # @revalidated-for E1/E2 event-based sessions (2026-03-10)
            is_outcomes = outcomes[:lo]
            is_metrics = compute_metrics(is_outcomes) if len(is_outcomes) >= 15 else None
            is_exp_r = is_metrics["expectancy_r"] if is_metrics else None

            windows.append(
                {
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "test_n": test_n,
                    "test_exp_r": test_exp_r,
                    "test_wr": metrics["win_rate"],
                    "test_sharpe": metrics["sharpe_ratio"],
                    "test_pass": (test_n >= min_trades_per_window and test_exp_r is not None and test_exp_r > 0),
                    "is_exp_r": is_exp_r,
                }
            )

            logger.info(
                "WF %s window %s..%s: N=%d ExpR=%s",
                strategy_id,
                window_start,
                window_end,
                test_n,
                test_exp_r,
            )

            window_start = window_end

    if not windows:
        return WalkForwardResult(
            strategy_id=strategy_id,
            instrument=instrument,
            n_total_windows=0,
            n_valid_windows=0,
            n_positive_windows=0,
            pct_positive=0.0,
            agg_oos_exp_r=0.0,
            total_oos_trades=0,
            passed=False,
            rejection_reason=(
                f"All {len(outcomes)} outcomes in training period ({min_train_months}mo). No test windows."
            ),
            windows=[],
            params=params,
        )

    # Aggregate valid windows (test_n >= threshold)
    valid_windows = [w for w in windows if w["test_n"] >= min_trades_per_window]
    n_valid = len(valid_windows)
    n_positive = sum(1 for w in valid_windows if w["test_exp_r"] is not None and w["test_exp_r"] > 0)
    pct_positive = n_positive / n_valid if n_valid > 0 else 0.0
    total_oos_trades = sum(w["test_n"] for w in valid_windows)

    # Trade-weighted aggregate OOS ExpR
    if total_oos_trades > 0:
        agg_oos_exp_r = (
            sum(w["test_exp_r"] * w["test_n"] for w in valid_windows if w["test_exp_r"] is not None) / total_oos_trades
        )
    else:
        agg_oos_exp_r = 0.0

    # Walk-Forward Efficiency (Pardo): WFE = mean(OOS ExpR) / mean(IS ExpR)
    # WFE > 0.50 = healthy strategy (OOS retains half of IS performance)
    wfe = None
    wfe_windows = [
        w
        for w in valid_windows
        if w.get("is_exp_r") is not None and w["is_exp_r"] > 0 and w.get("test_exp_r") is not None
    ]
    if wfe_windows:
        mean_oos = sum(w["test_exp_r"] for w in wfe_windows) / len(wfe_windows)
        mean_is = sum(w["is_exp_r"] for w in wfe_windows) / len(wfe_windows)
        if mean_is > 0:
            wfe = round(mean_oos / mean_is, 4)

    # Window imbalance detection
    # @research-source Pardo "The Evaluation and Optimization of Trading Strategies" Ch.7 —
    #   imbalanced OOS windows inflate aggregate stats (one large window dominates);
    #   5.0x threshold flags severe regime concentration for operator review
    # @revalidated-for E1/E2 event-based sessions (2026-03-10)
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
        rejection_reason = f"Insufficient valid windows: {n_valid} < {min_valid_windows}"
    elif pct_positive < min_pct_positive:
        rejection_reason = f"Too few positive windows: {pct_positive:.0%} < {min_pct_positive:.0%}"
    elif agg_oos_exp_r <= 0:
        rejection_reason = f"Negative aggregate OOS ExpR: {agg_oos_exp_r:.4f}"
    elif total_oos_trades < oos_trade_floor:
        rejection_reason = f"Total OOS trades {total_oos_trades} < floor {oos_trade_floor}"

    passed = rejection_reason is None

    if passed:
        logger.warning(
            "WF PASS %s: %d/%d windows positive (%.0f%%), agg_ExpR=%.4f, N=%d",
            strategy_id,
            n_positive,
            n_valid,
            pct_positive * 100,
            agg_oos_exp_r,
            total_oos_trades,
        )
    else:
        logger.warning("WF FAIL %s: %s", strategy_id, rejection_reason)

    return WalkForwardResult(
        strategy_id=strategy_id,
        instrument=instrument,
        n_total_windows=len(windows),
        n_valid_windows=n_valid,
        n_positive_windows=n_positive,
        pct_positive=round(pct_positive, 4),
        agg_oos_exp_r=round(agg_oos_exp_r, 4),
        total_oos_trades=total_oos_trades,
        passed=passed,
        rejection_reason=rejection_reason,
        windows=windows,
        params=params,
        window_imbalance_ratio=window_imbalance_ratio,
        window_imbalanced=window_imbalanced,
        wfe=wfe,
    )


def append_walkforward_result(result: WalkForwardResult, output_path: str | Path):
    """Append a WalkForwardResult as one JSON line to the output file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = asdict(result)
    record["timestamp"] = datetime.now(UTC).isoformat()

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
