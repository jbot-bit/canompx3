"""Shiryaev-Roberts live drift monitor for deployed lanes.

This is the Criterion 12 monitor:
- live/paper trade stream preferred
- canonical forward outcomes allowed only as an explicitly labeled fallback
- threshold calibrated to approximately 60 trading days ARL
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.live.performance_monitor import _compute_std_r
from trading_app.live.sr_monitor import ShiryaevRobertsMonitor, calibrate_sr_threshold
from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id
from trading_app.strategy_fitness import _load_strategy_outcomes

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ARL_DAYS = 60
DEFAULT_DELTA = -1.0
DEFAULT_VARIANCE_RATIO = 1.0
BASELINE_WINDOW = 50


def _load_reference_stats() -> dict[str, dict]:
    """Load validated-setups backtest reference stats for deployed lanes."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)
    try:
        rows = con.execute(
            """
            SELECT strategy_id, win_rate, rr_target, expectancy_r
            FROM validated_setups
            WHERE LOWER(status) = 'active'
            """
        ).fetchall()
    finally:
        con.close()

    stats: dict[str, dict] = {}
    for strategy_id, win_rate, rr_target, expectancy_r in rows:
        if win_rate is None or rr_target is None or expectancy_r is None:
            continue
        mu0 = float(expectancy_r)
        stats[strategy_id] = {
            "mu0": mu0,
            "sigma": _compute_std_r(float(win_rate), float(rr_target), mu0),
        }
    return stats


def _build_lanes() -> dict[str, dict]:
    lanes = {}
    profile_id = resolve_profile_id()
    reference_stats = _load_reference_stats()

    for i, lane in enumerate(get_profile_lane_definitions(profile_id), 1):
        strategy_id = lane["strategy_id"]
        bt = reference_stats.get(strategy_id, {"mu0": 0.10, "sigma": 1.0})
        suffix = " (shadow)" if lane.get("shadow_only") else ""
        lanes[strategy_id] = {
            "mu0": bt["mu0"],
            "sigma": bt["sigma"],
            "instrument": lane["instrument"],
            "orb_label": lane["orb_label"],
            "orb_minutes": lane["orb_minutes"],
            "entry_model": lane["entry_model"],
            "rr_target": lane["rr_target"],
            "confirm_bars": lane["confirm_bars"],
            "filter_type": lane["filter_type"],
            "label": f"L{i} {lane['orb_label']} {lane['filter_type']}{suffix}",
        }

    return lanes


def _empirical_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def _load_canonical_forward_trades(
    con: duckdb.DuckDBPyConnection,
    params: dict,
) -> list[float]:
    outcomes = _load_strategy_outcomes(
        con,
        instrument=params["instrument"],
        orb_label=params["orb_label"],
        orb_minutes=params["orb_minutes"],
        entry_model=params["entry_model"],
        rr_target=params["rr_target"],
        confirm_bars=params["confirm_bars"],
        filter_type=params["filter_type"],
        start_date=date(2026, 1, 1),
    )
    return [
        o["pnl_r"]
        for o in outcomes
        if o.get("pnl_r") is not None and o.get("outcome") not in (None, "scratch")
    ]


def prepare_monitor_inputs(
    con: duckdb.DuckDBPyConnection,
    strategy_id: str,
    params: dict,
    *,
    baseline_window: int = BASELINE_WINDOW,
) -> tuple[ShiryaevRobertsMonitor, list[float], str, str]:
    """Prepare baseline + monitored stream for one lane.

    Preference order:
    1. If >= baseline_window paper trades exist, estimate pre-change mean/std
       from the first `baseline_window` paper trades and monitor the remainder.
    2. Otherwise, use validated-setups backtest stats as the baseline and
       monitor all available paper trades.
    3. If no paper trades exist, fall back to canonical forward outcomes and
       still label that source explicitly.
    """
    paper_rows = con.execute(
        """
        SELECT pnl_r
        FROM paper_trades
        WHERE strategy_id = ? AND pnl_r IS NOT NULL
        ORDER BY trading_day
        """,
        [strategy_id],
    ).fetchall()
    paper_trades = [r[0] for r in paper_rows]

    baseline_source = "validated_backtest"
    stream_source = "none"
    expected_r = float(params["mu0"])
    std_r = float(params["sigma"])
    stream: list[float] = []

    if len(paper_trades) >= baseline_window:
        baseline = paper_trades[:baseline_window]
        stream = paper_trades[baseline_window:]
        expected_r = sum(baseline) / len(baseline)
        empirical_std = _empirical_std(baseline)
        if empirical_std > 0:
            std_r = empirical_std
            baseline_source = f"paper_trades_first_{baseline_window}"
        else:
            baseline_source = f"paper_trades_first_{baseline_window}_sigma_fallback"
        stream_source = "paper_trades"
    elif paper_trades:
        stream = paper_trades
        stream_source = "paper_trades"
    else:
        stream = _load_canonical_forward_trades(con, params)
        if stream:
            stream_source = "canonical_forward"

    threshold = calibrate_sr_threshold(
        TARGET_ARL_DAYS,
        delta=DEFAULT_DELTA,
        variance_ratio=DEFAULT_VARIANCE_RATIO,
    )
    monitor = ShiryaevRobertsMonitor(
        expected_r=expected_r,
        std_r=std_r,
        threshold=threshold,
        delta=DEFAULT_DELTA,
        variance_ratio=DEFAULT_VARIANCE_RATIO,
    )
    return monitor, stream, baseline_source, stream_source


def run_monitor() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)

    threshold = calibrate_sr_threshold(
        TARGET_ARL_DAYS,
        delta=DEFAULT_DELTA,
        variance_ratio=DEFAULT_VARIANCE_RATIO,
    )

    print("=" * 120)
    print(f"SHIRYAEV-ROBERTS MONITOR | {date.today()}")
    print(
        f"Target ARL≈{TARGET_ARL_DAYS} trading days | "
        f"delta={DEFAULT_DELTA:+.1f}σ | q={DEFAULT_VARIANCE_RATIO:.1f} | "
        f"threshold={threshold:.2f}"
    )
    print("=" * 120)
    print(
        f"\n{'Lane':<40} {'N':>4} {'SR':>10} {'Thr':>8} {'Status':<10} "
        f"{'Baseline':<28} {'Stream':<18}"
    )
    print("-" * 120)

    results = []
    for strategy_id, params in _build_lanes().items():
        monitor, trades, baseline_source, stream_source = prepare_monitor_inputs(con, strategy_id, params)

        status = "NO_DATA"
        alarm_trade = None
        for i, trade_r in enumerate(trades, 1):
            if monitor.update(trade_r):
                status = "ALARM"
                alarm_trade = i
                break
        else:
            if trades:
                status = "CONTINUE"

        print(
            f"{params['label']:<40} {len(trades):>4} {monitor.sr_stat:>10.2f} "
            f"{monitor.threshold:>8.2f} {status:<10} "
            f"{baseline_source:<28} {stream_source:<18}"
        )

        results.append(
            {
                "strategy_id": strategy_id,
                "orb_label": params["orb_label"],
                "n_monitored": len(trades),
                "sr_stat": round(monitor.sr_stat, 4),
                "threshold": round(monitor.threshold, 4),
                "status": status,
                "alarm_trade": alarm_trade,
                "baseline_source": baseline_source,
                "stream_source": stream_source,
                "expected_r": round(monitor.expected_r, 4),
                "std_r": round(monitor.std_r, 4),
            }
        )

    print(f"\n{'=' * 120}")
    print("Status key:")
    print("  CONTINUE  = no SR alarm yet on the monitored stream")
    print("  ALARM     = SR threshold crossed; strategy should move to suspended/manual review")
    print("  NO_DATA   = no trades available for monitoring")
    print("Baseline key:")
    print("  paper_trades_first_50            = first 50 paper trades define the pre-change baseline")
    print("  paper_trades_first_50_sigma_fallback = live mean used, sigma fell back to validated baseline")
    print("  validated_backtest              = validated_setups backtest baseline used")
    print("Stream key:")
    print("  paper_trades      = logged live/shadow execution stream")
    print("  canonical_forward = orb_outcomes shadow fallback only")
    print("=" * 120)

    state_file = STATE_DIR / "sr_state.json"
    state_file.write_text(json.dumps({"date": str(date.today()), "results": results}, indent=2))

    con.close()


if __name__ == "__main__":
    run_monitor()
