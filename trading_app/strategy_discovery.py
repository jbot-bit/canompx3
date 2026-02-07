"""
Grid search over strategy variants and save results to experimental_strategies.

For each combination of (orb_label, rr_target, confirm_bars, filter),
queries pre-computed orb_outcomes, computes performance metrics, and
writes results to experimental_strategies.

Usage:
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31
    python trading_app/strategy_discovery.py --instrument MGC --start 2021-01-01 --end 2025-12-31 --dry-run
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from pipeline.init_db import ORB_LABELS
from trading_app.config import ALL_FILTERS
from trading_app.setup_detector import detect_setups
from trading_app.db_manager import init_trading_app_schema
from trading_app.outcome_builder import RR_TARGETS, CONFIRM_BARS_OPTIONS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


def compute_metrics(outcomes: list[dict], cost_spec) -> dict:
    """
    Compute performance metrics from a list of outcome rows.

    Returns dict with: sample_size, win_rate, avg_win_r, avg_loss_r,
    expectancy_r, sharpe_ratio, max_drawdown_r, yearly_results.
    """
    if not outcomes:
        return {
            "sample_size": 0,
            "win_rate": None,
            "avg_win_r": None,
            "avg_loss_r": None,
            "expectancy_r": None,
            "sharpe_ratio": None,
            "max_drawdown_r": None,
            "yearly_results": "{}",
        }

    # Split wins/losses (scratches excluded from W/L stats)
    wins = [o for o in outcomes if o["outcome"] == "win"]
    losses = [o for o in outcomes if o["outcome"] == "loss"]
    traded = [o for o in outcomes if o["outcome"] in ("win", "loss")]

    n_traded = len(traded)
    if n_traded == 0:
        return {
            "sample_size": len(outcomes),
            "win_rate": None,
            "avg_win_r": None,
            "avg_loss_r": None,
            "expectancy_r": None,
            "sharpe_ratio": None,
            "max_drawdown_r": None,
            "yearly_results": "{}",
        }

    win_rate = len(wins) / n_traded
    loss_rate = 1.0 - win_rate

    avg_win_r = (
        sum(o["pnl_r"] for o in wins) / len(wins) if wins else 0.0
    )
    avg_loss_r = (
        abs(sum(o["pnl_r"] for o in losses) / len(losses)) if losses else 0.0
    )

    # E = (WR * AvgWin_R) - (LR * AvgLoss_R)  [CANONICAL_LOGIC.txt section 4]
    expectancy_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    # Sharpe ratio: mean(R) / std(R)
    r_values = [o["pnl_r"] for o in traded]
    mean_r = sum(r_values) / len(r_values)
    if len(r_values) > 1:
        variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
        std_r = variance ** 0.5
        sharpe_ratio = mean_r / std_r if std_r > 0 else None
    else:
        sharpe_ratio = None

    # Max drawdown in R (cumulative R-equity curve)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for o in traded:
        cumulative += o["pnl_r"]
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Yearly breakdown
    yearly = {}
    for o in traded:
        year = str(o["trading_day"].year) if hasattr(o["trading_day"], "year") else str(o["trading_day"])[:4]
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "total_r": 0.0}
        yearly[year]["trades"] += 1
        if o["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += o["pnl_r"]

    # Compute per-year metrics
    for year_data in yearly.values():
        year_data["win_rate"] = round(
            year_data["wins"] / year_data["trades"], 4
        ) if year_data["trades"] > 0 else 0.0
        year_data["avg_r"] = round(
            year_data["total_r"] / year_data["trades"], 4
        ) if year_data["trades"] > 0 else 0.0
        year_data["total_r"] = round(year_data["total_r"], 4)

    return {
        "sample_size": len(outcomes),
        "win_rate": round(win_rate, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "expectancy_r": round(expectancy_r, 4),
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
        "max_drawdown_r": round(max_dd, 4),
        "yearly_results": json.dumps(yearly),
    }


def make_strategy_id(
    instrument: str,
    orb_label: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
) -> str:
    """Generate deterministic strategy ID."""
    return f"{instrument}_{orb_label}_RR{rr_target}_CB{confirm_bars}_{filter_type}"


def run_discovery(
    db_path: Path | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    orb_minutes: int = 5,
    dry_run: bool = False,
) -> int:
    """
    Grid search over all strategy variants.

    Returns count of strategies written.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path))
    try:
        if not dry_run:
            init_trading_app_schema(db_path=db_path)

        total_strategies = 0
        total_combos = len(ORB_LABELS) * len(RR_TARGETS) * len(CONFIRM_BARS_OPTIONS) * len(ALL_FILTERS)
        combo_idx = 0

        for filter_key, strategy_filter in ALL_FILTERS.items():
            for orb_label in ORB_LABELS:
                # Get matching days for this filter + ORB
                matching_days = detect_setups(
                    con=con,
                    strategy_filter=strategy_filter,
                    orb_label=orb_label,
                    instrument=instrument,
                    orb_minutes=orb_minutes,
                    start_date=start_date,
                    end_date=end_date,
                )

                matching_day_set = {d[0] for d in matching_days}

                for rr_target in RR_TARGETS:
                    for cb in CONFIRM_BARS_OPTIONS:
                        combo_idx += 1

                        # Query orb_outcomes for this combo, filtered to matching days
                        if not matching_day_set:
                            continue

                        # Build date list for IN clause
                        params = [instrument, orb_minutes, orb_label, rr_target, cb]
                        date_placeholders = ", ".join(["?"] * len(matching_day_set))
                        params.extend(sorted(matching_day_set))

                        rows = con.execute(
                            f"""SELECT trading_day, outcome, pnl_r, mae_r, mfe_r
                                FROM orb_outcomes
                                WHERE symbol = ? AND orb_minutes = ?
                                AND orb_label = ? AND rr_target = ? AND confirm_bars = ?
                                AND outcome IS NOT NULL
                                AND trading_day IN ({date_placeholders})
                                ORDER BY trading_day""",
                            params,
                        ).fetchall()

                        if not rows:
                            continue

                        outcomes = [
                            {
                                "trading_day": r[0],
                                "outcome": r[1],
                                "pnl_r": r[2],
                                "mae_r": r[3],
                                "mfe_r": r[4],
                            }
                            for r in rows
                        ]

                        metrics = compute_metrics(outcomes, cost_spec)
                        strategy_id = make_strategy_id(
                            instrument, orb_label, rr_target, cb, filter_key,
                        )

                        if not dry_run:
                            con.execute(
                                """INSERT OR REPLACE INTO experimental_strategies
                                   (strategy_id, instrument, orb_label, orb_minutes,
                                    rr_target, confirm_bars, filter_type, filter_params,
                                    sample_size, win_rate, avg_win_r, avg_loss_r,
                                    expectancy_r, sharpe_ratio, max_drawdown_r,
                                    yearly_results)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                [
                                    strategy_id, instrument, orb_label, orb_minutes,
                                    rr_target, cb, filter_key, strategy_filter.to_json(),
                                    metrics["sample_size"], metrics["win_rate"],
                                    metrics["avg_win_r"], metrics["avg_loss_r"],
                                    metrics["expectancy_r"], metrics["sharpe_ratio"],
                                    metrics["max_drawdown_r"], metrics["yearly_results"],
                                ],
                            )

                        total_strategies += 1

                if combo_idx % 50 == 0:
                    print(f"  Progress: {combo_idx}/{total_combos} combos, {total_strategies} strategies")

        if not dry_run:
            con.commit()

        print(f"Done: {total_strategies} strategies from {total_combos} combos")
        if dry_run:
            print("  (DRY RUN — no data written)")

        return total_strategies

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over strategy variants"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB duration")
    parser.add_argument("--dry-run", action="store_true", help="No DB writes")
    args = parser.parse_args()

    run_discovery(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        orb_minutes=args.orb_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
