#!/usr/bin/env python3
"""
Hypothesis test: validate new filter ideas against existing orb_outcomes data.

Tests 5 hypotheses derived from the ORB Size Deep Dive (Feb 2026):
  1. G3 filter for MNQ (catches edges the current G4+ grid misses)
  2. Band filter for MES (G3 with 12pt upper cap)
  3. Percentage-based ORB filter (normalize across instruments)
  4. ORB/ATR ratio filter (volatility-adjusted size)
  5. Direction filter (LONG-only Asia, SHORT-only US sessions)

All tests use existing orb_outcomes + daily_features -- no pipeline rebuild needed.
Run: python scripts/tools/hypothesis_test.py [--db-path PATH]
"""

import argparse
import os
import sys
from pathlib import Path

# --- path setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip install duckdb")
    sys.exit(1)

# Cost models (round-trip in points)
FRICTION = {
    "MGC": {"rt_points": 0.84, "tick_value": 1.00},
    "MES": {"rt_points": 1.68, "tick_value": 1.25},
    "MNQ": {"rt_points": 1.37, "tick_value": 2.00},
    "MCL": {"rt_points": 0.30, "tick_value": 1.00},
}

# Typical price levels (for percentage calculation when daily_close missing)
TYPICAL_PRICE = {
    "MGC": 2900.0,
    "MES": 6000.0,
    "MNQ": 21000.0,
    "MCL": 70.0,
}


def get_db_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()
    if args.db_path:
        return Path(args.db_path)
    env = os.environ.get("DUCKDB_PATH")
    if env:
        return Path(env)
    return PROJECT_ROOT / "gold.db"


def fmt_r(val):
    """Format R value with sign."""
    if val is None:
        return "  N/A "
    return f"{val:+.4f}"


def print_header(title):
    width = 80
    print("")
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    print(f"\n  --- {title} ---")


def print_row(label, n, avg_r, total_r, wr=None):
    """Print a formatted result row."""
    wr_str = f"  WR={wr:.1f}%" if wr is not None else ""
    print(f"    {label:<40s}  N={n:>5d}  avgR={fmt_r(avg_r)}  totR={total_r:>+8.1f}{wr_str}")


# =========================================================================
# HYPOTHESIS 1: G3 filter for MNQ
# =========================================================================
def test_h1_g3_for_mnq(con):
    print_header("HYPOTHESIS 1: G3 (>=3pt) filter for MNQ")
    print("  Current grid uses G4+ minimum. Deep dive showed MNQ profitable from G3+.")
    print("  Question: Does G3 add REAL new edge, or is it just noise?")

    sessions = ["0900", "1000", "1100", "1800"]
    for sess in sessions:
        print_subheader(f"MNQ {sess}")

        for gate_min in [0, 2, 3, 4, 5, 6]:
            gate_label = f"G{gate_min}+" if gate_min > 0 else "ALL"
            row = con.execute("""
                SELECT COUNT(*) as n,
                       AVG(pnl_r) as avg_r,
                       SUM(pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                JOIN daily_features f ON o.trading_day = f.trading_day
                    AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                WHERE o.symbol = 'MNQ'
                  AND o.orb_label = ?
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1'
                  AND o.rr_target = 2.0
                  AND o.confirm_bars = 2
                  AND (? = 0 OR f.orb_{sess}_size >= ?)
            """.format(sess=sess), [sess, gate_min, gate_min]).fetchone()

            if row and row[0] > 0:
                print_row(gate_label, row[0], row[1], row[2], row[3])
            else:
                print(f"    {gate_label:<40s}  N=    0  (no data)")

        # Show the G3-ONLY band (3-4pt) -- is this marginal or real?
        row = con.execute(f"""
            SELECT COUNT(*) as n,
                   AVG(pnl_r) as avg_r,
                   SUM(pnl_r) as total_r,
                   100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
            FROM orb_outcomes o
            JOIN daily_features f ON o.trading_day = f.trading_day
                AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
            WHERE o.symbol = 'MNQ'
              AND o.orb_label = ?
              AND o.outcome IN ('win', 'loss')
              AND o.entry_model = 'E1'
              AND o.rr_target = 2.0
              AND o.confirm_bars = 2
              AND f.orb_{sess}_size >= 3 AND f.orb_{sess}_size < 4
        """, [sess]).fetchone()

        if row and row[0] > 0:
            print_row("BAND 3-4pt ONLY (new trades)", row[0], row[1], row[2], row[3])
        else:
            print(f"    {'BAND 3-4pt ONLY':<40s}  N=    0  (no data)")


# =========================================================================
# HYPOTHESIS 2: Band filter for MES (G3 with 12pt cap)
# =========================================================================
def test_h2_band_filter_mes(con):
    print_header("HYPOTHESIS 2: Band filter for MES (cap at 12pt)")
    print("  Deep dive showed MES 1000 at 12pt+ crashes to -0.56 avg R.")
    print("  Question: Does capping at 12pt rescue MES edges?")

    sessions = ["0900", "1000"]
    for sess in sessions:
        print_subheader(f"MES {sess}")

        # Standard gates
        for gate_min in [3, 4, 5, 6]:
            row = con.execute(f"""
                SELECT COUNT(*) as n, AVG(pnl_r) as avg_r, SUM(pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                JOIN daily_features f ON o.trading_day = f.trading_day
                    AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                WHERE o.symbol = 'MES' AND o.orb_label = ?
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                  AND f.orb_{sess}_size >= ?
            """, [sess, gate_min]).fetchone()
            print_row(f"G{gate_min}+ (no cap)", row[0], row[1], row[2], row[3])

        # Band filters with 12pt cap
        for gate_min in [3, 4, 5, 6]:
            row = con.execute(f"""
                SELECT COUNT(*) as n, AVG(pnl_r) as avg_r, SUM(pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                JOIN daily_features f ON o.trading_day = f.trading_day
                    AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                WHERE o.symbol = 'MES' AND o.orb_label = ?
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                  AND f.orb_{sess}_size >= ? AND f.orb_{sess}_size < 12
            """, [sess, gate_min]).fetchone()
            print_row(f"BAND G{gate_min} to L12 (capped)", row[0], row[1], row[2], row[3])

        # Show what 12pt+ looks like alone
        row = con.execute(f"""
            SELECT COUNT(*) as n, AVG(pnl_r) as avg_r, SUM(pnl_r) as total_r,
                   100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
            FROM orb_outcomes o
            JOIN daily_features f ON o.trading_day = f.trading_day
                AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
            WHERE o.symbol = 'MES' AND o.orb_label = ?
              AND o.outcome IN ('win', 'loss')
              AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
              AND f.orb_{sess}_size >= 12
        """, [sess]).fetchone()
        if row and row[0] > 0:
            print_row("TOXIC ZONE: 12pt+ only", row[0], row[1], row[2], row[3])


# =========================================================================
# HYPOTHESIS 3: Percentage-based ORB filter
# =========================================================================
def test_h3_percentage_filter(con):
    print_header("HYPOTHESIS 3: Percentage-based ORB filter")
    print("  Instead of fixed points, use ORB size as % of price.")
    print("  5pt on MGC (price ~2900) = 0.17%. 5pt on MNQ (~21000) = 0.024%.")
    print("  Question: Does a universal % threshold work across instruments?")

    instruments = ["MGC", "MNQ", "MES"]
    pct_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    for sym in instruments:
        print_subheader(f"{sym} -- 0900 session")

        for pct in pct_thresholds:
            # Use (orb_high + orb_low) / 2 as price proxy for the ORB period
            row = con.execute(f"""
                SELECT COUNT(*) as n, AVG(o.pnl_r) as avg_r, SUM(o.pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                JOIN daily_features f ON o.trading_day = f.trading_day
                    AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                WHERE o.symbol = '{sym}' AND o.orb_label = '0900'
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                  AND f.orb_0900_size IS NOT NULL
                  AND (f.orb_0900_high + f.orb_0900_low) > 0
                  AND (f.orb_0900_size / ((f.orb_0900_high + f.orb_0900_low) / 2.0)) * 100.0 >= {pct}
            """).fetchone()
            if row and row[0] > 0:
                # Also show what point threshold this corresponds to
                avg_pts = con.execute(f"""
                    SELECT AVG(f.orb_0900_size) as avg_size
                    FROM daily_features f
                    WHERE f.symbol = '{sym}'
                      AND f.orb_0900_size IS NOT NULL
                      AND (f.orb_0900_high + f.orb_0900_low) > 0
                      AND (f.orb_0900_size / ((f.orb_0900_high + f.orb_0900_low) / 2.0)) * 100.0 >= {pct}
                """).fetchone()
                equiv = f"(~{avg_pts[0]:.1f}pt avg)" if avg_pts and avg_pts[0] else ""
                print_row(f">= {pct:.2f}% of price {equiv}", row[0], row[1], row[2], row[3])
            else:
                print(f"    >= {pct:.2f}% of price{'':>25s}  N=    0")

    # Cross-instrument comparison at best threshold
    print_subheader("CROSS-INSTRUMENT: Best universal % threshold")
    for pct in [0.10, 0.15, 0.20]:
        print(f"\n    At >= {pct:.2f}%:")
        for sym in instruments:
            for sess in ["0900", "1000"]:
                row = con.execute(f"""
                    SELECT COUNT(*) as n, AVG(o.pnl_r) as avg_r, SUM(o.pnl_r) as total_r
                    FROM orb_outcomes o
                    JOIN daily_features f ON o.trading_day = f.trading_day
                        AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                    WHERE o.symbol = '{sym}' AND o.orb_label = '{sess}'
                      AND o.outcome IN ('win', 'loss')
                      AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                      AND f.orb_{sess}_size IS NOT NULL
                      AND (f.orb_{sess}_high + f.orb_{sess}_low) > 0
                      AND (f.orb_{sess}_size / ((f.orb_{sess}_high + f.orb_{sess}_low) / 2.0)) * 100.0 >= {pct}
                """).fetchone()
                if row and row[0] > 0:
                    print(f"      {sym} {sess}: N={row[0]:>4d}  avgR={fmt_r(row[1])}  totR={row[2]:>+8.1f}")


# =========================================================================
# HYPOTHESIS 4: ORB/ATR ratio filter
# =========================================================================
def test_h4_orb_atr_ratio(con):
    print_header("HYPOTHESIS 4: ORB size relative to ATR(20)")
    print("  ORB >= 0.5x ATR means the opening range is unusually large")
    print("  for current volatility. Adapts automatically to regime.")
    print("  Question: Does ATR-normalized ORB filter beat fixed points?")

    instruments = ["MGC", "MNQ", "MES"]
    atr_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80]

    for sym in instruments:
        print_subheader(f"{sym} -- 0900 session")

        for ratio in atr_thresholds:
            row = con.execute(f"""
                SELECT COUNT(*) as n, AVG(o.pnl_r) as avg_r, SUM(o.pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                JOIN daily_features f ON o.trading_day = f.trading_day
                    AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
                WHERE o.symbol = '{sym}' AND o.orb_label = '0900'
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                  AND f.atr_20 IS NOT NULL AND f.atr_20 > 0
                  AND f.orb_0900_size IS NOT NULL
                  AND (f.orb_0900_size / f.atr_20) >= {ratio}
            """).fetchone()
            if row and row[0] > 0:
                print_row(f"ORB >= {ratio:.2f}x ATR(20)", row[0], row[1], row[2], row[3])
            else:
                print(f"    ORB >= {ratio:.2f}x ATR(20){'':>22s}  N=    0")

        # Compare to best fixed-point gate for reference
        best_gate = {"MGC": 5, "MNQ": 3, "MES": 3}[sym]
        row = con.execute(f"""
            SELECT COUNT(*) as n, AVG(o.pnl_r) as avg_r, SUM(o.pnl_r) as total_r,
                   100.0 * SUM(CASE WHEN o.outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
            FROM orb_outcomes o
            JOIN daily_features f ON o.trading_day = f.trading_day
                AND o.symbol = f.symbol AND o.orb_minutes = f.orb_minutes
            WHERE o.symbol = '{sym}' AND o.orb_label = '0900'
              AND o.outcome IN ('win', 'loss')
              AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
              AND f.orb_0900_size >= {best_gate}
        """).fetchone()
        if row and row[0] > 0:
            print_row(f"[BASELINE] G{best_gate}+ fixed points", row[0], row[1], row[2], row[3])


# =========================================================================
# HYPOTHESIS 5: Direction filter (LONG-only / SHORT-only)
# =========================================================================
def test_h5_direction_filter(con):
    print_header("HYPOTHESIS 5: Direction filter by time-of-day")
    print("  Deep dive showed: Asia sessions favor LONGS, US sessions favor SHORTS.")
    print("  Question: Does filtering by direction improve edge?")

    instruments = ["MGC", "MNQ", "MES"]

    for sym in instruments:
        print_subheader(f"{sym}")

        # Test combinations
        tests = [
            ("0900", "long",  "0900 LONG-ONLY (Asia bias)"),
            ("0900", "short", "0900 SHORT-ONLY"),
            ("0900", None,    "0900 BOTH (baseline)"),
            ("1000", "long",  "1000 LONG-ONLY"),
            ("1000", "short", "1000 SHORT-ONLY (US bias)"),
            ("1000", None,    "1000 BOTH (baseline)"),
            ("1800", "long",  "1800 LONG-ONLY"),
            ("1800", "short", "1800 SHORT-ONLY (London close bias)"),
            ("1800", None,    "1800 BOTH (baseline)"),
        ]

        best_gate = {"MGC": 5, "MNQ": 3, "MES": 3}[sym]

        for sess, direction, label in tests:
            dir_clause = f"AND o.orb_label = '{sess}'" if sess else ""
            dir_filter = ""
            if direction:
                dir_filter = f"""
                    AND EXISTS (
                        SELECT 1 FROM daily_features f2
                        WHERE f2.trading_day = o.trading_day
                          AND f2.symbol = o.symbol
                          AND f2.orb_minutes = o.orb_minutes
                          AND f2.orb_{sess}_break_dir = '{direction}'
                    )
                """

            size_filter = f"""
                AND EXISTS (
                    SELECT 1 FROM daily_features f3
                    WHERE f3.trading_day = o.trading_day
                      AND f3.symbol = o.symbol
                      AND f3.orb_minutes = o.orb_minutes
                      AND f3.orb_{sess}_size >= {best_gate}
                )
            """

            row = con.execute(f"""
                SELECT COUNT(*) as n, AVG(pnl_r) as avg_r, SUM(pnl_r) as total_r,
                       100.0 * SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) / COUNT(*) as wr
                FROM orb_outcomes o
                WHERE o.symbol = '{sym}'
                  AND o.orb_label = '{sess}'
                  AND o.outcome IN ('win', 'loss')
                  AND o.entry_model = 'E1' AND o.rr_target = 2.0 AND o.confirm_bars = 2
                  {dir_filter}
                  {size_filter}
            """).fetchone()

            if row and row[0] > 0:
                print_row(f"{label} (G{best_gate}+)", row[0], row[1], row[2], row[3])
            else:
                print(f"    {label:<40s}  N=    0")


# =========================================================================
# VERDICT
# =========================================================================
def print_verdict():
    print_header("INTERPRETATION GUIDE")
    print("""
  How to read these results:

  1. G3 for MNQ (H1):
     - Look at the BAND 3-4pt rows. If they're positive with N >= 30,
       G3 adds real new tradeable edge (not just padding from G4+).
     - If BAND 3-4pt is negative, G3 is noise. Stick with G4+.

  2. Band filter MES (H2):
     - Compare "G4+ no cap" vs "BAND G4 to L12 capped".
     - If capped version has HIGHER avg R, the 12pt+ zone is confirmed toxic.
     - If total R also improves, the cap is capturing real edge improvement.

  3. Percentage filter (H3):
     - Look for a % threshold where ALL instruments show positive avg R.
     - If 0.15% works for MGC, MNQ, AND MES, we have a universal filter.
     - Compare N and avg R to the best fixed-point baseline.

  4. ORB/ATR ratio (H4):
     - Compare to [BASELINE] row. If any ATR ratio beats the baseline
       on BOTH avg R and total R, it's a better filter.
     - ATR-based adapts to regime -- a 5pt ORB in low-vol is huge,
       in high-vol it's nothing.

  5. Direction filter (H5):
     - Compare LONG-ONLY and SHORT-ONLY to BOTH baseline.
     - If LONG-ONLY 0900 has higher avg R AND reasonable N,
       direction filtering adds edge.
     - CAUTION: Halving N by direction can kill statistical significance.
       Only useful if avg R improvement > 50% with N >= 50.

  NEXT STEPS:
  - Hypotheses that PASS: add to config.py, re-run discovery
  - Hypotheses that FAIL: document in RESEARCH_ARCHIVE.md, move on
  - Hypotheses that are AMBIGUOUS: need more data or different test
""")


def main():
    db_path = get_db_path()
    print(f"Database: {db_path}")

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    # Verify we have multi-instrument data
    row = con.execute("""
        SELECT symbol, COUNT(DISTINCT trading_day) as days
        FROM orb_outcomes
        WHERE outcome IN ('win', 'loss')
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()
    print("\nData coverage:")
    for r in row:
        print(f"  {r[0]}: {r[1]} trading days with outcomes")
    print("")

    test_h1_g3_for_mnq(con)
    test_h2_band_filter_mes(con)
    test_h3_percentage_filter(con)
    test_h4_orb_atr_ratio(con)
    test_h5_direction_filter(con)
    print_verdict()

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
