#!/usr/bin/env python3
"""
E3 Fill Timing Audit -- Quantify stale retrace fills.

E3 (limit-at-ORB retrace entry) has NO time limit on when the retrace
can happen. _resolve_e3() scans ALL bars from confirm to trading day end
(up to 24 hours). A retrace at 3pm to a level that broke at 10am counts
as a valid fill. This audit quantifies how many fills are "stale" and
whether they inflate or dilute performance.

Read-only: no writes to gold.db.

Output:
  research/output/e3_fill_timing.csv         -- per-fill detail
  research/output/e3_fill_timing_summary.csv -- per-session by time bucket
  Console summary

Usage:
    python research/research_e3_fill_timing.py
    python research/research_e3_fill_timing.py --db-path C:/db/gold.db
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


# -- Time bucket definitions ------------------------------------------

TIME_BUCKETS = [
    ("0-5min",    0,    5),
    ("5-15min",   5,   15),
    ("15-30min",  15,  30),
    ("30-60min",  30,  60),
    ("60-120min", 60,  120),
    ("120-240min", 120, 240),
    ("240+min",   240, 99999),
]

DEFAULT_THRESHOLD_MIN = 60  # fills within 60 min are "fresh"


def classify_bucket(delay_min: float) -> str:
    """Classify fill delay into a time bucket label."""
    for label, lo, hi in TIME_BUCKETS:
        if lo <= delay_min < hi:
            return label
    return "240+min"


# -- Metrics computation ----------------------------------------------

def compute_metrics(pnl_rs: list[float]) -> dict:
    """Compute summary metrics from a list of pnl_r values."""
    n = len(pnl_rs)
    if n == 0:
        return {"n": 0, "avg_r": None, "total_r": 0.0, "wr": None, "sharpe": None}

    avg_r = sum(pnl_rs) / n
    total_r = sum(pnl_rs)
    wins = sum(1 for r in pnl_rs if r > 0)
    wr = wins / n

    if n >= 2:
        import statistics
        std = statistics.stdev(pnl_rs)
        sharpe = avg_r / std if std > 0 else None
    else:
        sharpe = None

    return {"n": n, "avg_r": round(avg_r, 4), "total_r": round(total_r, 2),
            "wr": round(wr, 4), "sharpe": round(sharpe, 4) if sharpe is not None else None}


# -- Main audit -------------------------------------------------------

def run_audit(db_path: Path, threshold_min: int = DEFAULT_THRESHOLD_MIN) -> None:
    output_dir = PROJECT_ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path), read_only=True) as con:
        # Get all E3 outcomes with entry_ts (filled trades)
        # Join to daily_features to get break_ts for fill delay computation
        print("Loading E3 outcomes with break timestamps...")

        # Get all orb_labels that have E3 outcomes
        labels = con.execute("""
            SELECT DISTINCT orb_label FROM orb_outcomes
            WHERE entry_model = 'E3' AND entry_ts IS NOT NULL
        """).fetchall()
        orb_labels = [r[0] for r in labels]
        print(f"  E3 sessions: {orb_labels}")

        all_fills = []

        for orb_label in orb_labels:
            break_ts_col = f"orb_{orb_label}_break_ts"

            rows = con.execute(f"""
                SELECT
                    o.trading_day,
                    o.symbol,
                    o.orb_label,
                    o.orb_minutes,
                    o.rr_target,
                    o.confirm_bars,
                    o.entry_ts,
                    o.outcome,
                    o.pnl_r,
                    df.{break_ts_col} AS break_ts
                FROM orb_outcomes o
                JOIN daily_features df
                  ON o.trading_day = df.trading_day
                  AND o.symbol = df.symbol
                  AND o.orb_minutes = df.orb_minutes
                WHERE o.entry_model = 'E3'
                  AND o.entry_ts IS NOT NULL
                  AND o.outcome IN ('win', 'loss')
                  AND o.orb_label = ?
                  AND df.{break_ts_col} IS NOT NULL
                ORDER BY o.trading_day
            """, [orb_label]).fetchall()

            col_names = ["trading_day", "symbol", "orb_label", "orb_minutes",
                         "rr_target", "confirm_bars", "entry_ts", "outcome",
                         "pnl_r", "break_ts"]

            for row in rows:
                d = dict(zip(col_names, row))

                # Compute fill delay in minutes
                entry_ts = d["entry_ts"]
                break_ts = d["break_ts"]

                if hasattr(entry_ts, 'timestamp') and hasattr(break_ts, 'timestamp'):
                    delay_sec = entry_ts.timestamp() - break_ts.timestamp()
                else:
                    # Fallback: try converting to datetime
                    from datetime import datetime
                    if isinstance(entry_ts, str):
                        entry_ts = datetime.fromisoformat(entry_ts)
                    if isinstance(break_ts, str):
                        break_ts = datetime.fromisoformat(break_ts)
                    delay_sec = (entry_ts - break_ts).total_seconds()

                delay_min = delay_sec / 60.0
                d["fill_delay_min"] = round(delay_min, 1)
                d["time_bucket"] = classify_bucket(delay_min)
                d["is_fresh"] = delay_min < threshold_min

                all_fills.append(d)

            print(f"  {orb_label}: {len(rows)} filled E3 outcomes")

        if not all_fills:
            print("No E3 fills found. Exiting.")
            return

        print(f"\nTotal E3 fills: {len(all_fills)}")

        # -- Per-fill detail CSV --------------------------------------

        detail_path = output_dir / "e3_fill_timing.csv"
        detail_fields = ["trading_day", "symbol", "orb_label", "orb_minutes",
                         "rr_target", "confirm_bars", "outcome", "pnl_r",
                         "fill_delay_min", "time_bucket", "is_fresh"]

        with open(detail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            for fill in sorted(all_fills, key=lambda x: (x["orb_label"], x["trading_day"])):
                writer.writerow(fill)

        print(f"\nDetail CSV: {detail_path}")

        # -- Deduplicate to unique fills (one per trading_day/orb_label) --
        # Multiple RR targets share the same fill event. For fill-count
        # analysis, deduplicate to unique (trading_day, symbol, orb_label, orb_minutes).
        seen_fills = set()
        unique_fills = []
        for fill in all_fills:
            key = (fill["trading_day"], fill["symbol"], fill["orb_label"], fill["orb_minutes"])
            if key not in seen_fills:
                seen_fills.add(key)
                unique_fills.append(fill)

        print(f"Unique fill events (deduped across RR targets): {len(unique_fills)}")

        # -- Summary by session x time bucket -------------------------

        print("\n" + "=" * 80)
        print("E3 FILL TIMING AUDIT")
        print("=" * 80)

        summary_rows = []

        # Group by orb_label
        sessions = sorted(set(f["orb_label"] for f in all_fills))

        for session in sessions:
            session_fills = [f for f in all_fills if f["orb_label"] == session]
            session_unique = [f for f in unique_fills if f["orb_label"] == session]

            print(f"\n{'-' * 60}")
            print(f"SESSION: {session}  (N_fills={len(session_unique)} unique, "
                  f"N_outcomes={len(session_fills)} with all RR targets)")
            print(f"{'-' * 60}")

            # Time bucket breakdown (using all outcomes for metrics, unique for fill counts)
            print(f"\n  {'Bucket':<14} {'Fills':>6} {'%Fills':>7} {'N_out':>6} {'avgR':>8} {'WR':>6} {'totR':>8} {'Sharpe':>8}")
            print(f"  {'-' * 74}")

            total_unique = len(session_unique)

            for bucket_label, _, _ in TIME_BUCKETS:
                bucket_unique = [f for f in session_unique if f["time_bucket"] == bucket_label]
                bucket_outcomes = [f for f in session_fills if f["time_bucket"] == bucket_label]

                fill_count = len(bucket_unique)
                fill_pct = fill_count / total_unique * 100 if total_unique > 0 else 0

                pnl_rs = [f["pnl_r"] for f in bucket_outcomes if f["pnl_r"] is not None]
                m = compute_metrics(pnl_rs)

                wr_str = f"{m['wr']:.1%}" if m['wr'] is not None else "N/A"
                avg_str = f"{m['avg_r']}" if m['avg_r'] is not None else "N/A"
                sh_str = f"{m['sharpe']}" if m['sharpe'] is not None else "N/A"
                print(f"  {bucket_label:<14} {fill_count:>6} {fill_pct:>6.1f}% {m['n']:>6} "
                      f"{avg_str:>8} {wr_str:>6} {m['total_r']:>8.1f} {sh_str:>8}")

                summary_rows.append({
                    "session": session,
                    "time_bucket": bucket_label,
                    "unique_fills": fill_count,
                    "fill_pct": round(fill_pct, 1),
                    "n_outcomes": m["n"],
                    "avg_r": m["avg_r"],
                    "total_r": m["total_r"],
                    "wr": m["wr"],
                    "sharpe": m["sharpe"],
                })

            # Fresh vs stale comparison
            fresh = [f for f in session_fills if f["is_fresh"]]
            stale = [f for f in session_fills if not f["is_fresh"]]

            fresh_unique = [f for f in session_unique if f["is_fresh"]]
            stale_unique = [f for f in session_unique if not f["is_fresh"]]

            fresh_m = compute_metrics([f["pnl_r"] for f in fresh if f["pnl_r"] is not None])
            stale_m = compute_metrics([f["pnl_r"] for f in stale if f["pnl_r"] is not None])

            print(f"\n  FRESH (<{threshold_min}min): {len(fresh_unique)} fills ({len(fresh_unique)/total_unique*100:.1f}%), "
                  f"avgR={fresh_m['avg_r']}, WR={fresh_m['wr']}, totR={fresh_m['total_r']}")
            print(f"  STALE (>={threshold_min}min): {len(stale_unique)} fills ({len(stale_unique)/total_unique*100:.1f}%), "
                  f"avgR={stale_m['avg_r']}, WR={stale_m['wr']}, totR={stale_m['total_r']}")

            stale_pct = len(stale_unique) / total_unique * 100 if total_unique > 0 else 0

            # RR-target breakdown for fresh vs stale
            rr_targets = sorted(set(f["rr_target"] for f in session_fills))
            if len(rr_targets) > 1:
                print(f"\n  By RR target (fresh vs stale):")
                print(f"  {'RR':>5} {'Fresh_N':>8} {'Fresh_avgR':>10} {'Stale_N':>8} {'Stale_avgR':>10} {'Delta':>8}")
                for rr in rr_targets:
                    fr = [f["pnl_r"] for f in fresh if f["rr_target"] == rr and f["pnl_r"] is not None]
                    st = [f["pnl_r"] for f in stale if f["rr_target"] == rr and f["pnl_r"] is not None]
                    fr_m = compute_metrics(fr)
                    st_m = compute_metrics(st)
                    delta = ""
                    if fr_m["avg_r"] is not None and st_m["avg_r"] is not None:
                        delta = f"{st_m['avg_r'] - fr_m['avg_r']:+.4f}"
                    print(f"  {rr:>5.1f} {fr_m['n']:>8} {fr_m['avg_r'] if fr_m['avg_r'] is not None else 'N/A':>10} "
                          f"{st_m['n']:>8} {st_m['avg_r'] if st_m['avg_r'] is not None else 'N/A':>10} {delta:>8}")

            # Instrument breakdown
            instruments = sorted(set(f["symbol"] for f in session_fills))
            if len(instruments) > 1:
                print(f"\n  By instrument (fresh vs stale):")
                print(f"  {'Inst':>5} {'Fresh_N':>8} {'Fresh_avgR':>10} {'Stale_N':>8} {'Stale_avgR':>10} {'Delta':>8}")
                for inst in instruments:
                    fr = [f["pnl_r"] for f in fresh if f["symbol"] == inst and f["pnl_r"] is not None]
                    st = [f["pnl_r"] for f in stale if f["symbol"] == inst and f["pnl_r"] is not None]
                    fr_m = compute_metrics(fr)
                    st_m = compute_metrics(st)
                    delta = ""
                    if fr_m["avg_r"] is not None and st_m["avg_r"] is not None:
                        delta = f"{st_m['avg_r'] - fr_m['avg_r']:+.4f}"
                    print(f"  {inst:>5} {fr_m['n']:>8} {fr_m['avg_r'] if fr_m['avg_r'] is not None else 'N/A':>10} "
                          f"{st_m['n']:>8} {st_m['avg_r'] if st_m['avg_r'] is not None else 'N/A':>10} {delta:>8}")

            # Decision criteria
            print(f"\n  DECISION CRITERIA:")
            if stale_pct < 5:
                print(f"  -> Stale <5% ({stale_pct:.1f}%). Document only.")
            elif stale_pct <= 20:
                if stale_m["avg_r"] is not None and fresh_m["avg_r"] is not None:
                    delta = stale_m["avg_r"] - fresh_m["avg_r"]
                    if delta > 0.1:
                        print(f"  -> Stale 5-20% ({stale_pct:.1f}%), stale avgR BETTER by {delta:+.4f}R. FIX NEEDED (artificial inflation).")
                    else:
                        print(f"  -> Stale 5-20% ({stale_pct:.1f}%), stale avgR delta={delta:+.4f}R. Document only (dilution, not inflation).")
                else:
                    print(f"  -> Stale 5-20% ({stale_pct:.1f}%), insufficient data for comparison.")
            else:
                print(f"  -> Stale >20% ({stale_pct:.1f}%). FIX NEEDED regardless of performance.")

        # -- Summary CSV ----------------------------------------------

        summary_path = output_dir / "e3_fill_timing_summary.csv"
        summary_fields = ["session", "time_bucket", "unique_fills", "fill_pct",
                          "n_outcomes", "avg_r", "total_r", "wr", "sharpe"]

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        print(f"\n\nSummary CSV: {summary_path}")

        # -- Overall summary ------------------------------------------

        total = len(unique_fills)
        total_fresh = sum(1 for f in unique_fills if f["is_fresh"])
        total_stale = total - total_fresh

        print(f"\n{'=' * 80}")
        print(f"OVERALL: {total} unique E3 fills, "
              f"{total_fresh} fresh ({total_fresh/total*100:.1f}%), "
              f"{total_stale} stale ({total_stale/total*100:.1f}%)")

        all_fresh_r = [f["pnl_r"] for f in all_fills if f["is_fresh"] and f["pnl_r"] is not None]
        all_stale_r = [f["pnl_r"] for f in all_fills if not f["is_fresh"] and f["pnl_r"] is not None]
        fm = compute_metrics(all_fresh_r)
        sm = compute_metrics(all_stale_r)
        print(f"  Fresh avgR={fm['avg_r']}, Stale avgR={sm['avg_r']}")
        if fm["avg_r"] is not None and sm["avg_r"] is not None:
            delta = sm["avg_r"] - fm["avg_r"]
            print(f"  Delta (stale - fresh) = {delta:+.4f}R")
            if delta > 0.1:
                print(f"  STALE FILLS INFLATE PERFORMANCE -- time cap recommended")
            elif delta < -0.1:
                print(f"  Stale fills DILUTE performance -- removing improves metrics")
            else:
                print(f"  Marginal difference -- review per-session data")

        # RR2.0 focus
        rr2_fills = [f for f in all_fills if abs(f["rr_target"] - 2.0) < 0.01]
        if rr2_fills:
            print(f"\n  RR2.0 focus:")
            rr2_fresh = compute_metrics([f["pnl_r"] for f in rr2_fills if f["is_fresh"] and f["pnl_r"] is not None])
            rr2_stale = compute_metrics([f["pnl_r"] for f in rr2_fills if not f["is_fresh"] and f["pnl_r"] is not None])
            print(f"    Fresh: N={rr2_fresh['n']}, avgR={rr2_fresh['avg_r']}, WR={rr2_fresh['wr']}")
            print(f"    Stale: N={rr2_stale['n']}, avgR={rr2_stale['avg_r']}, WR={rr2_stale['wr']}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="E3 fill timing audit")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH,
                        help="Path to gold.db")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD_MIN,
                        help=f"Minutes threshold for fresh/stale split (default: {DEFAULT_THRESHOLD_MIN})")
    args = parser.parse_args()

    run_audit(args.db_path, threshold_min=args.threshold)


if __name__ == "__main__":
    main()
