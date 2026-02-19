#!/usr/bin/env python3
"""
DST Winter/Summer Split for Strategy Re-Validation.

Fixed sessions (0900, 1800, 0030, 2300) blend two DST regimes.
This script splits every validated + positive-avgR experimental strategy
by winter/summer and checks if the edge survives in both regimes.

Read-only: no writes to gold.db.

Usage:
    python research/research_dst_strategy_revalidation.py
    python research/research_dst_strategy_revalidation.py --db-path C:/db/gold.db
"""

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading_app.config import ALL_FILTERS, DirectionFilter, DIR_LONG, _MES_1000_BAND_FILTERS

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ── DST regime detection ──────────────────────────────────────────────

TZ_US_EAST = ZoneInfo("America/New_York")
TZ_UK = ZoneInfo("Europe/London")

# Which timezone governs each affected session
SESSION_DST_TZ = {
    "0900": TZ_US_EAST,
    "0030": TZ_US_EAST,
    "2300": TZ_US_EAST,
    "1800": TZ_UK,
}

AFFECTED_SESSIONS = set(SESSION_DST_TZ.keys())


def is_winter(trading_day: date, session: str) -> bool:
    """True if trading_day falls in winter (standard time) for this session's reference TZ."""
    tz = SESSION_DST_TZ[session]
    from datetime import datetime, time
    dt = datetime.combine(trading_day, time(12, 0), tzinfo=tz)
    offset_hours = dt.utcoffset().total_seconds() / 3600
    if tz == TZ_US_EAST:
        return offset_hours == -5  # EST
    else:  # UK
        return offset_hours == 0   # GMT


# ── Filter reconstruction ─────────────────────────────────────────────

# Build lookup of all known filters (base + extras)
FILTER_LOOKUP = dict(ALL_FILTERS)
FILTER_LOOKUP["DIR_LONG"] = DIR_LONG
FILTER_LOOKUP.update(_MES_1000_BAND_FILTERS)


def get_eligible_days(features: list[dict], orb_label: str, filter_type: str) -> set:
    """Return set of trading_days eligible under given filter + orb_label."""
    filt = FILTER_LOOKUP.get(filter_type)
    if filt is None:
        return set()
    days = set()
    for row in features:
        if row.get(f"orb_{orb_label}_break_dir") is None:
            continue
        if row.get(f"orb_{orb_label}_double_break"):
            continue
        if filt.matches_row(row, orb_label):
            days.add(row["trading_day"])
    return days


# ── Metrics computation ───────────────────────────────────────────────

def compute_split_metrics(outcomes: list[dict]) -> dict:
    """Compute N, avgR, totalR, WR, Sharpe, MaxDD from outcome list."""
    if not outcomes:
        return {"n": 0, "avg_r": None, "total_r": 0.0, "wr": None, "sharpe": None, "max_dd": 0.0}

    traded = [o for o in outcomes if o["outcome"] in ("win", "loss")]
    n = len(traded)
    if n == 0:
        return {"n": 0, "avg_r": None, "total_r": 0.0, "wr": None, "sharpe": None, "max_dd": 0.0}

    rs = [o["pnl_r"] for o in traded]
    wins = sum(1 for o in traded if o["outcome"] == "win")
    total_r = sum(rs)
    avg_r = total_r / n
    wr = wins / n

    sharpe = None
    if n >= 10:
        mean_r = avg_r
        if n > 1:
            var = sum((r - mean_r) ** 2 for r in rs) / (n - 1)
            std = var ** 0.5
            if std > 0:
                sharpe = mean_r / std

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {"n": n, "avg_r": round(avg_r, 4), "total_r": round(total_r, 2),
            "wr": round(wr, 4), "sharpe": round(sharpe, 4) if sharpe else None,
            "max_dd": round(max_dd, 2)}


def classify_dst_stability(winter: dict, summer: dict) -> str:
    """Assign DST stability verdict."""
    wn, sn = winter["n"], summer["n"]
    wa, sa = winter["avg_r"], summer["avg_r"]

    if wn < 10 or sn < 10:
        return "LOW-N"
    if wa is None or sa is None:
        return "LOW-N"

    diff = abs(wa - sa)
    if diff <= 0.10:
        return "STABLE"
    if wa > 0 and sa <= 0:
        return "WINTER-ONLY"
    if sa > 0 and wa <= 0:
        return "SUMMER-ONLY"
    if wa > sa + 0.10 and wn >= 15:
        return "WINTER-DOMINANT"
    if sa > wa + 0.10 and sn >= 15:
        return "SUMMER-DOMINANT"
    return "UNSTABLE"


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DST strategy re-validation")
    parser.add_argument("--db-path", default=str(PROJECT_ROOT / "gold.db"))
    args = parser.parse_args()

    con = duckdb.connect(args.db_path, read_only=True)

    # Step 1: Collect strategies to test
    print("Step 1: Loading strategies at affected sessions...")

    validated = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type,
               sample_size, expectancy_r, sharpe_ratio, 'validated' as source
        FROM validated_setups
        WHERE orb_label IN ('0900', '1800', '0030', '2300')
    """).fetchall()
    v_cols = [d[0] for d in con.description]

    experimental = con.execute("""
        SELECT strategy_id, instrument, orb_label, orb_minutes,
               entry_model, rr_target, confirm_bars, filter_type,
               sample_size, expectancy_r, sharpe_ratio, 'experimental' as source
        FROM experimental_strategies
        WHERE orb_label IN ('0900', '1800', '0030', '2300')
          AND expectancy_r > 0
          AND is_canonical = true
          AND strategy_id NOT IN (SELECT strategy_id FROM validated_setups)
    """).fetchall()

    strategies = [dict(zip(v_cols, r)) for r in validated + experimental]
    n_val = len(validated)
    n_exp = len(experimental)
    print(f"  {n_val} validated + {n_exp} positive experimental = {len(strategies)} total")

    if not strategies:
        print("No strategies to test.")
        con.close()
        return

    # Step 2: Load daily_features and outcomes
    print("Step 2: Loading daily_features and outcomes...")

    instruments = sorted(set(s["instrument"] for s in strategies))
    orb_minutes_set = sorted(set(s["orb_minutes"] for s in strategies))

    features_by_inst = {}
    for inst in instruments:
        for om in orb_minutes_set:
            rows = con.execute(
                "SELECT * FROM daily_features WHERE symbol = ? AND orb_minutes = ? ORDER BY trading_day",
                [inst, om],
            ).fetchall()
            cols = [d[0] for d in con.description]
            features_by_inst[(inst, om)] = [dict(zip(cols, r)) for r in rows]
            print(f"  {inst} orb_minutes={om}: {len(rows)} feature rows")

    # Step 3: Process each strategy
    print(f"\nStep 3: Processing {len(strategies)} strategies...")
    results = []

    for i, strat in enumerate(strategies):
        sid = strat["strategy_id"]
        inst = strat["instrument"]
        session = strat["orb_label"]
        om = strat["orb_minutes"]
        em = strat["entry_model"]
        rr = strat["rr_target"]
        cb = strat["confirm_bars"]
        ft = strat["filter_type"]

        # Get eligible days for this filter
        features = features_by_inst.get((inst, om), [])
        eligible = get_eligible_days(features, session, ft)

        # Load outcomes
        outcomes = con.execute("""
            SELECT trading_day, outcome, pnl_r
            FROM orb_outcomes
            WHERE symbol = ? AND orb_minutes = ? AND orb_label = ?
              AND entry_model = ? AND rr_target = ? AND confirm_bars = ?
              AND outcome IS NOT NULL
        """, [inst, om, session, em, rr, cb]).fetchall()
        o_cols = [d[0] for d in con.description]
        outcomes = [dict(zip(o_cols, r)) for r in outcomes]

        # Intersect with eligible filter days
        outcomes = [o for o in outcomes if o["trading_day"] in eligible]

        # Split by DST regime
        winter_outcomes = [o for o in outcomes if is_winter(o["trading_day"], session)]
        summer_outcomes = [o for o in outcomes if not is_winter(o["trading_day"], session)]

        combined = compute_split_metrics(outcomes)
        winter = compute_split_metrics(winter_outcomes)
        summer = compute_split_metrics(summer_outcomes)
        verdict = classify_dst_stability(winter, summer)

        results.append({
            "strategy_id": sid, "source": strat["source"],
            "instrument": inst, "session": session, "filter": ft,
            "combined_n": combined["n"], "combined_avgR": combined["avg_r"],
            "combined_sharpe": combined["sharpe"],
            "winter_n": winter["n"], "winter_avgR": winter["avg_r"],
            "winter_wr": winter["wr"], "winter_sharpe": winter["sharpe"],
            "winter_maxdd": winter["max_dd"],
            "summer_n": summer["n"], "summer_avgR": summer["avg_r"],
            "summer_wr": summer["wr"], "summer_sharpe": summer["sharpe"],
            "summer_maxdd": summer["max_dd"],
            "verdict": verdict,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(strategies)} processed")

    con.close()
    print(f"  {len(results)} strategies processed")

    # Step 4: Output
    print("\n" + "=" * 80)
    print("RESULTS: DST Winter/Summer Strategy Re-Validation")
    print("=" * 80)

    # Summary
    from collections import Counter
    verdict_counts = Counter(r["verdict"] for r in results)
    print("\n--- Verdict Summary ---")
    for v in ["STABLE", "WINTER-DOMINANT", "SUMMER-DOMINANT", "WINTER-ONLY", "SUMMER-ONLY", "LOW-N", "UNSTABLE"]:
        if verdict_counts.get(v, 0) > 0:
            print(f"  {v:20s}: {verdict_counts[v]}")
    print(f"  {'TOTAL':20s}: {len(results)}")

    # Detail table — sorted by instability
    def instability(r):
        wa = r["winter_avgR"] or 0
        sa = r["summer_avgR"] or 0
        return -abs(wa - sa)

    results.sort(key=instability)

    print(f"\n--- Per-Strategy Detail (sorted by |winter - summer|) ---")
    print(f"{'Strategy ID':<45s} {'Sess':>4s} {'Src':>5s} | {'Comb avgR':>9s} {'W avgR(N)':>12s} {'S avgR(N)':>12s} | Verdict")
    print("-" * 110)
    for r in results:
        wa_str = f"{r['winter_avgR']:+.3f}({r['winter_n']})" if r['winter_avgR'] is not None else f"  N/A({r['winter_n']})"
        sa_str = f"{r['summer_avgR']:+.3f}({r['summer_n']})" if r['summer_avgR'] is not None else f"  N/A({r['summer_n']})"
        ca_str = f"{r['combined_avgR']:+.3f}" if r['combined_avgR'] is not None else "  N/A"
        src = "VAL" if r["source"] == "validated" else "EXP"
        print(f"{r['strategy_id']:<45s} {r['session']:>4s} {src:>5s} | {ca_str:>9s} {wa_str:>12s} {sa_str:>12s} | {r['verdict']}")

    # Red flags
    red_flags = [r for r in results if r["verdict"] in ("WINTER-ONLY", "SUMMER-ONLY")]
    if red_flags:
        print(f"\n--- RED FLAGS: Edge dies in one regime ({len(red_flags)} strategies) ---")
        for r in red_flags:
            wa = r["winter_avgR"] or 0
            sa = r["summer_avgR"] or 0
            dead = "summer" if r["verdict"] == "WINTER-ONLY" else "winter"
            live = "winter" if dead == "summer" else "summer"
            live_avg = wa if live == "winter" else sa
            print(f"  {r['strategy_id']}")
            print(f"    Edge DIES in {dead} (avgR <= 0). Lives in {live} (avgR={live_avg:+.3f}).")
            if r["session"] == "0900":
                print(f"    -> Consider: switch to CME_OPEN or US_DATA_OPEN dynamic session")
            elif r["session"] == "1800":
                print(f"    -> Consider: switch to LONDON_OPEN dynamic session")
            elif r["session"] == "0030":
                print(f"    -> Consider: switch to US_EQUITY_OPEN dynamic session")
            elif r["session"] == "2300":
                print(f"    -> Consider: investigate if 2300 aligns with any dynamic session")
    else:
        print("\n--- No RED FLAGS (no edges that fully die in one regime) ---")

    # Hidden edges (experimental that would validate in one regime)
    hidden = [r for r in results if r["source"] == "experimental"
              and r["verdict"] in ("WINTER-ONLY", "SUMMER-ONLY", "WINTER-DOMINANT", "SUMMER-DOMINANT")
              and max(r["winter_n"] or 0, r["summer_n"] or 0) >= 30]
    if hidden:
        print(f"\n--- HIDDEN EDGES: Experimental strategies revealed by DST split ({len(hidden)}) ---")
        for r in hidden:
            better = "winter" if (r["winter_avgR"] or 0) > (r["summer_avgR"] or 0) else "summer"
            bn = r["winter_n"] if better == "winter" else r["summer_n"]
            ba = r["winter_avgR"] if better == "winter" else r["summer_avgR"]
            bs = r["winter_sharpe"] if better == "winter" else r["summer_sharpe"]
            print(f"  {r['strategy_id']}  {better} N={bn} avgR={ba:+.3f} sharpe={bs}")

    # Save CSV
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "dst_strategy_revalidation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
