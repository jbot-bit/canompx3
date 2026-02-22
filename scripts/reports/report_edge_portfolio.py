"""
Edge family portfolio report with honest aggregate stats.

Shows two stat layers:
1. Per-trade stats (from compute_metrics): total trades, win%, ExpR, avg_win/loss
2. Portfolio stats (daily ledger): daily Sharpe, annualized Sharpe, max DD
   Groups trades by calendar day to avoid inflating frequency from
   multiple sessions trading the same day.

Usage:
    python scripts/reports/report_edge_portfolio.py --instrument MGC
    python scripts/reports/report_edge_portfolio.py --all
    python scripts/reports/report_edge_portfolio.py --all --db-path C:/db/gold.db
    python scripts/reports/report_edge_portfolio.py --all --include-purged
    python scripts/reports/report_edge_portfolio.py --all --slots
"""

import sys
from math import sqrt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.strategy_discovery import compute_metrics
from trading_app.db_manager import has_edge_families

# ~252 trading days/year (standard futures approximation)
TRADING_DAYS_PER_YEAR = 252

def _load_head_trades(con, instrument, include_purged=False):
    """Load all trades for family heads, joined through edge_families.

    Returns list of dicts with strategy params + trade data.
    """
    purge_filter = "" if include_purged else "AND ef.robustness_status != 'PURGED'"

    rows = con.execute(f"""
        SELECT ef.family_hash, ef.head_strategy_id, ef.robustness_status,
               vs.orb_label, vs.entry_model, vs.rr_target, vs.confirm_bars,
               vs.orb_minutes, vs.filter_type,
               oo.trading_day, oo.outcome, oo.pnl_r,
               oo.entry_price, oo.stop_price
        FROM edge_families ef
        JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
        JOIN strategy_trade_days std ON vs.strategy_id = std.strategy_id
        JOIN orb_outcomes oo
          ON oo.symbol = vs.instrument
          AND oo.orb_label = vs.orb_label
          AND oo.orb_minutes = vs.orb_minutes
          AND oo.entry_model = vs.entry_model
          AND oo.rr_target = vs.rr_target
          AND oo.confirm_bars = vs.confirm_bars
          AND oo.trading_day = std.trading_day
        WHERE ef.instrument = ?
          {purge_filter}
          AND oo.outcome IS NOT NULL
        ORDER BY oo.trading_day, ef.head_strategy_id
    """, [instrument]).fetchall()

    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, r)) for r in rows]

def _compute_daily_ledger(trades):
    """Build daily R-return series from flat trade list.

    Groups trades by trading_day, sums pnl_r per day.
    Returns (sorted list of (day, daily_r), overlap_count).
    overlap_count = days with >1 trade (multi-session exposure).
    """
    daily_r = defaultdict(float)
    daily_trade_count = defaultdict(int)

    for t in trades:
        day = t["trading_day"]
        daily_r[day] += t["pnl_r"]
        daily_trade_count[day] += 1

    overlap_count = sum(1 for c in daily_trade_count.values() if c > 1)
    daily_returns = sorted(daily_r.items())
    return daily_returns, overlap_count

def _compute_portfolio_stats(daily_returns):
    """Compute portfolio-level stats from daily R-returns.

    Returns dict with: trading_days, sharpe_daily, sharpe_ann, max_dd_r, total_r.
    """
    if not daily_returns:
        return {
            "trading_days": 0,
            "sharpe_daily": None,
            "sharpe_ann": None,
            "max_dd_r": 0.0,
            "total_r": 0.0,
        }

    values = [r for _, r in daily_returns]
    n = len(values)
    total_r = sum(values)

    mean_daily = total_r / n
    if n > 1:
        variance = sum((v - mean_daily) ** 2 for v in values) / (n - 1)
        std_daily = variance ** 0.5
    else:
        std_daily = 0.0

    sharpe_daily = mean_daily / std_daily if std_daily > 0 else None
    sharpe_ann = (
        sharpe_daily * sqrt(TRADING_DAYS_PER_YEAR)
        if sharpe_daily is not None
        else None
    )

    # Max drawdown on cumulative daily equity
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in values:
        cumulative += v
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return {
        "trading_days": n,
        "sharpe_daily": round(sharpe_daily, 4) if sharpe_daily is not None else None,
        "sharpe_ann": round(sharpe_ann, 4) if sharpe_ann is not None else None,
        "max_dd_r": round(max_dd, 4),
        "total_r": round(total_r, 4),
    }

def _yearly_breakdown(trades):
    """Group trades by year, compute per-year summary."""
    yearly = defaultdict(lambda: {"trades": 0, "wins": 0, "total_r": 0.0})
    for t in trades:
        td = t["trading_day"]
        year = td.year if hasattr(td, "year") else int(str(td)[:4])
        yearly[year]["trades"] += 1
        if t["outcome"] == "win":
            yearly[year]["wins"] += 1
        yearly[year]["total_r"] += t["pnl_r"]
    return dict(sorted(yearly.items()))

def report_instrument(db_path, instrument, include_purged=False):
    """Generate portfolio report for one instrument.

    Returns dict with all computed stats, or None if no families.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not has_edge_families(con):
            return None

        # Count families
        purge_filter = "" if include_purged else "AND robustness_status != 'PURGED'"
        family_count = con.execute(f"""
            SELECT COUNT(*) FROM edge_families
            WHERE instrument = ? {purge_filter}
        """, [instrument]).fetchone()[0]

        if family_count == 0:
            return None

        # Load trades
        trades = _load_head_trades(con, instrument, include_purged)
        if not trades:
            return {
                "instrument": instrument,
                "family_count": family_count,
                "total_trades": 0,
            }

        # Filter to win/loss only (scratches/early_exit excluded from stats)
        traded = [t for t in trades if t["outcome"] in ("win", "loss")]

        # Per-trade stats via compute_metrics
        per_trade = compute_metrics(traded)

        # Daily ledger
        daily_returns, overlap_count = _compute_daily_ledger(traded)
        portfolio = _compute_portfolio_stats(daily_returns)

        # Unique days
        unique_days = len(daily_returns)

        # Per-ORB breakdown
        orb_groups = defaultdict(list)
        for t in traded:
            orb_groups[t["orb_label"]].append(t)

        per_orb = {}
        for orb_label in sorted(orb_groups.keys()):
            orb_metrics = compute_metrics(orb_groups[orb_label])
            per_orb[orb_label] = {
                "trades": orb_metrics["sample_size"],
                "expectancy_r": orb_metrics["expectancy_r"],
                "win_rate": orb_metrics["win_rate"],
            }

        # Yearly breakdown
        yearly = _yearly_breakdown(traded)

        return {
            "instrument": instrument,
            "family_count": family_count,
            "total_trades": per_trade["sample_size"],
            "unique_days": unique_days,
            "multi_session_days": overlap_count,
            "per_trade": per_trade,
            "portfolio": portfolio,
            "per_orb": per_orb,
            "yearly": yearly,
        }
    finally:
        con.close()

def print_report(result):
    """Print formatted report to stdout."""
    if result is None:
        return

    inst = result["instrument"]
    print(f"=== {inst} Edge Family Portfolio ===")
    print(f"Families:       {result['family_count']}")

    if result["total_trades"] == 0:
        print("No trades found.")
        print()
        return

    print(f"Total trades:   {result['total_trades']:,}")
    print(f"Unique days:    {result['unique_days']:,}  "
          f"({result['multi_session_days']:,} multi-session days)")

    pt = result["per_trade"]
    print()
    print("--- Per-Trade Stats ---")
    print(f"Win rate:        {pt['win_rate']:.1%}" if pt["win_rate"] else "Win rate:        N/A")
    print(f"Expectancy (R):  {pt['expectancy_r']:+.4f}" if pt["expectancy_r"] else "Expectancy (R):  N/A")
    print(f"Avg win:         {pt['avg_win_r']:+.4f}R | "
          f"Avg loss: -{pt['avg_loss_r']:.4f}R"
          if pt["avg_win_r"] is not None else "Avg win/loss:    N/A")
    print(f"Per-trade Sharpe: {pt['sharpe_ratio']:.4f}" if pt["sharpe_ratio"] else "Per-trade Sharpe: N/A")
    print(f"Max drawdown:    {pt['max_drawdown_r']:.2f}R" if pt["max_drawdown_r"] else "Max drawdown:    N/A")

    pf = result["portfolio"]
    print()
    print("--- Portfolio Stats (daily ledger) ---")
    print(f"Trading days:    {pf['trading_days']:,}")
    print(f"Sharpe (daily):  {pf['sharpe_daily']:.4f}" if pf["sharpe_daily"] is not None else "Sharpe (daily):  N/A")
    print(f"Sharpe (ann):    {pf['sharpe_ann']:.4f}" if pf["sharpe_ann"] is not None else "Sharpe (ann):    N/A")
    print(f"Max drawdown:    {pf['max_dd_r']:.2f}R")
    print(f"Total R:         {pf['total_r']:+.1f}")

    print()
    print("--- By ORB Session ---")
    for orb, stats in result["per_orb"].items():
        wr = f"{stats['win_rate']:.1%}" if stats["win_rate"] else "N/A"
        expr = f"{stats['expectancy_r']:+.4f}" if stats["expectancy_r"] else "N/A"
        print(f"  {orb}: {stats['trades']} trades, ExpR {expr}, WR {wr}")

    print()
    print("--- Yearly ---")
    for year, stats in result["yearly"].items():
        wr = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
        print(f"  {year}: {stats['trades']} trades, WR {wr:.1%}, {stats['total_r']:+.1f}R")
    print()

def session_slots(db_path, instrument=None):
    """Collapse edge families to one representative per instrument+session.

    Elects best family per (instrument, orb_label) by Sharpe (breaks ties
    with trade count). No new table — query-only view over edge_families.

    Returns list of dicts sorted by instrument then ExpR descending.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not has_edge_families(con):
            return []

        inst_filter = "AND ef.instrument = ?" if instrument else ""
        params = [instrument] if instrument else []

        rows = con.execute(f"""
            WITH slot_ranked AS (
                SELECT ef.instrument,
                       vs.orb_label AS session,
                       ef.family_hash,
                       ef.head_strategy_id,
                       ef.head_expectancy_r,
                       ef.head_sharpe_ann,
                       ef.trade_day_count,
                       ef.member_count,
                       ef.robustness_status,
                       ef.trade_tier,
                       ROW_NUMBER() OVER (
                           PARTITION BY ef.instrument, vs.orb_label
                           ORDER BY ef.head_sharpe_ann DESC,
                                    ef.trade_day_count DESC
                       ) AS rn
                FROM edge_families ef
                JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
                WHERE ef.robustness_status IN ('ROBUST', 'WHITELISTED', 'SINGLETON')
                  {inst_filter}
            ),
            slot_counts AS (
                SELECT ef.instrument,
                       vs.orb_label AS session,
                       COUNT(*) AS competing_families
                FROM edge_families ef
                JOIN validated_setups vs ON ef.head_strategy_id = vs.strategy_id
                WHERE ef.robustness_status IN ('ROBUST', 'WHITELISTED', 'SINGLETON')
                  {inst_filter}
                GROUP BY 1, 2
            )
            SELECT sr.instrument, sr.session, sr.head_strategy_id,
                   sr.head_expectancy_r, sr.head_sharpe_ann,
                   sr.trade_day_count, sr.trade_tier,
                   sr.robustness_status, sr.member_count,
                   sc.competing_families
            FROM slot_ranked sr
            JOIN slot_counts sc
              ON sr.instrument = sc.instrument AND sr.session = sc.session
            WHERE sr.rn = 1
            ORDER BY sr.instrument, sr.head_expectancy_r DESC
        """, params * 2).fetchall()  # params used twice (slot_ranked + slot_counts)

        cols = [
            "instrument", "session", "head_strategy_id",
            "head_expectancy_r", "head_sharpe_ann",
            "trade_day_count", "trade_tier",
            "robustness_status", "member_count",
            "competing_families",
        ]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def print_session_slots(slots):
    """Print collapsed session slot view."""
    if not slots:
        print("No session slots found (no non-PURGED edge families).")
        return

    # Group by instrument
    by_inst = defaultdict(list)
    for s in slots:
        by_inst[s["instrument"]].append(s)

    total_slots = 0
    total_families = 0
    total_trades = 0

    for inst in sorted(by_inst.keys()):
        inst_slots = by_inst[inst]
        inst_families = sum(s["competing_families"] for s in inst_slots)
        print(f"=== {inst} Session Slots "
              f"({len(inst_slots)} sessions from {inst_families} families) ===")

        for s in inst_slots:
            tier_tag = f"[{s['trade_tier']}]" if s["trade_tier"] != "CORE" else ""
            print(
                f"  {s['session']:>18}: {s['head_strategy_id']:<52s}"
                f"  ExpR={s['head_expectancy_r']:+.3f}"
                f"  ShANN={s['head_sharpe_ann']:.2f}"
                f"  N={s['trade_day_count']:>4}"
                f"  ({s['competing_families']} fam)"
                f"  {tier_tag}"
            )

        total_slots += len(inst_slots)
        total_families += inst_families
        total_trades += sum(s["trade_day_count"] for s in inst_slots)
        print()

    print(f"--- Summary: {total_slots} slots from {total_families} families, "
          f"~{total_trades:,} total trades ---")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge family portfolio report with honest aggregate stats"
    )
    parser.add_argument("--instrument", help="Instrument symbol")
    parser.add_argument("--all", action="store_true", help="Report all instruments")
    parser.add_argument(
        "--db-path", default=None, help="Database path (default: project gold.db)"
    )
    parser.add_argument(
        "--include-purged", action="store_true",
        help="Include PURGED families (default: ROBUST + WHITELISTED only)",
    )
    parser.add_argument(
        "--slots", action="store_true",
        help="Show session slots view (one representative per instrument+session)",
    )
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Either --instrument or --all is required")

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    # Session slots mode
    if args.slots:
        if args.all:
            slots = session_slots(db_path)
        else:
            slots = session_slots(db_path, args.instrument)
        print_session_slots(slots)
        return

    if args.all:
        for inst in ["MGC", "MNQ", "MES"]:
            result = report_instrument(db_path, inst, args.include_purged)
            if result is not None:
                print_report(result)
            else:
                print(f"=== {inst}: no edge families ===")
                print()
    else:
        result = report_instrument(db_path, args.instrument, args.include_purged)
        if result is not None:
            print_report(result)
        else:
            print(f"No edge families for {args.instrument}")

if __name__ == "__main__":
    main()
