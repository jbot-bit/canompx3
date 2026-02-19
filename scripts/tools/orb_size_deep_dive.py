"""
ORB Size Deep Dive -- map optimal ORB size thresholds per session per instrument.

We know ORB size IS the edge. This script figures out:
  1. What's the optimal size threshold for each session on each instrument?
  2. Where does the breakeven point land? (below X points = house wins)
  3. Does the optimal threshold differ by instrument? (friction hypothesis)
  4. Cumulative R curve by ORB size bucket -- where does the money come from?
  5. Friction-adjusted breakeven -- theoretical minimum ORB for each instrument

Usage:
    python scripts/tools/orb_size_deep_dive.py
    python scripts/tools/orb_size_deep_dive.py --db C:/db/gold.db
"""

import sys
import os
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "gold.db"

# Cost models (RT friction in points)
FRICTION = {
    "MGC": {"rt_dollars": 8.40, "tick_value": 1.00, "rt_points": 0.84},
    "MES": {"rt_dollars": 2.10, "tick_value": 1.25, "rt_points": 1.68},
    "MNQ": {"rt_dollars": 2.74, "tick_value": 2.00, "rt_points": 1.37},
    "MCL": {"rt_dollars": 3.00, "tick_value": 1.00, "rt_points": 0.30},
}


def get_db_path():
    for i, arg in enumerate(sys.argv):
        if arg == "--db" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    env = os.environ.get("DUCKDB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB


def print_table(headers, rows, col_widths=None):
    if not rows:
        print("  No data.")
        return
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i] if row[i] is not None else "")))
            col_widths.append(min(max_w + 2, 22))
    header_str = "  "
    for h, w in zip(headers, col_widths):
        header_str += str(h).ljust(w)
    print(header_str)
    print("  " + "-" * sum(col_widths))
    for row in rows:
        row_str = "  "
        for val, w in zip(row, col_widths):
            row_str += str(val if val is not None else "-").ljust(w)
        print(row_str)


def section_1_size_heatmap(con):
    """ORB size buckets x session x instrument -- where does the money live?"""
    print()
    print("  ================================================================")
    print("  SECTION 1: ORB SIZE HEATMAP -- Where Does the Money Live?")
    print("  ================================================================")
    print()

    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    for sym in symbols:
        sessions = [r[0] for r in con.execute(
            "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ? ORDER BY orb_label",
            [sym]
        ).fetchall()]

        print(f"\n  --- {sym} ---")
        print(f"  (Aggregated across all CB/RR/EM combos, win+loss only)\n")

        for sess in sessions:
            size_col = f"orb_{sess}_size"
            try:
                rows = con.execute(f"""
                    SELECT
                        CASE
                            WHEN d.{size_col} < 2 THEN '< 2pt'
                            WHEN d.{size_col} < 4 THEN '2-4pt'
                            WHEN d.{size_col} < 6 THEN '4-6pt'
                            WHEN d.{size_col} < 8 THEN '6-8pt'
                            WHEN d.{size_col} < 12 THEN '8-12pt'
                            ELSE '12pt+'
                        END as size_bucket,
                        COUNT(DISTINCT o.trading_day) as days,
                        COUNT(*) as trades,
                        ROUND(AVG(o.pnl_r), 4) as avg_r,
                        ROUND(SUM(o.pnl_r), 1) as total_r,
                        ROUND(AVG(CASE WHEN o.outcome='win' THEN 100.0 ELSE 0.0 END), 1) as wr
                    FROM orb_outcomes o
                    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                    WHERE o.symbol = ? AND o.orb_label = ?
                      AND o.outcome IN ('win', 'loss')
                      AND d.{size_col} IS NOT NULL
                    GROUP BY size_bucket
                    ORDER BY MIN(d.{size_col})
                """, [sym, sess]).fetchall()
            except Exception:
                continue

            if not rows or all(r[2] == 0 for r in rows):
                continue

            print(f"  {sess}:")
            print_table(
                ["Size", "Days", "Trades", "Avg R", "Total R", "WR%"],
                rows
            )
            print()


def section_2_breakeven_finder(con):
    """Find the exact breakeven ORB size for each session/instrument."""
    print()
    print("  ================================================================")
    print("  SECTION 2: BREAKEVEN ORB SIZE -- Below X, House Wins")
    print("  ================================================================")
    print()
    print("  For each session, finds the ORB size where avg R crosses from")
    print("  negative to positive. Below this = losing money. Above = edge.")
    print()

    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    summary_rows = []

    for sym in symbols:
        sessions = [r[0] for r in con.execute(
            "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ? ORDER BY orb_label",
            [sym]
        ).fetchall()]

        for sess in sessions:
            size_col = f"orb_{sess}_size"
            try:
                # Test each integer threshold from 1 to 15
                for threshold in range(1, 16):
                    row = con.execute(f"""
                        SELECT
                            COUNT(*) as n,
                            ROUND(AVG(o.pnl_r), 4) as avg_r,
                            ROUND(SUM(o.pnl_r), 1) as total_r
                        FROM orb_outcomes o
                        JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                        WHERE o.symbol = ? AND o.orb_label = ?
                          AND o.outcome IN ('win', 'loss')
                          AND d.{size_col} >= ?
                    """, [sym, sess, float(threshold)]).fetchone()

                    if row and row[0] and row[0] >= 20 and row[1] and row[1] > 0:
                        # Also get the "below threshold" stats
                        below = con.execute(f"""
                            SELECT COUNT(*) as n, ROUND(AVG(o.pnl_r), 4) as avg_r
                            FROM orb_outcomes o
                            JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                            WHERE o.symbol = ? AND o.orb_label = ?
                              AND o.outcome IN ('win', 'loss')
                              AND d.{size_col} < ?
                        """, [sym, sess, float(threshold)]).fetchone()

                        below_n = below[0] if below else 0
                        below_r = below[1] if below and below[1] else 0

                        summary_rows.append((
                            sym, sess,
                            f">={threshold}pt",
                            row[0], f"{row[1]:+.4f}",
                            below_n, f"{below_r:+.4f}" if below_r else "-"
                        ))
                        break
                else:
                    # Never went positive
                    summary_rows.append((sym, sess, "NONE", "-", "-", "-", "Always negative"))
            except Exception:
                continue

    print_table(
        ["Symbol", "Session", "Breakeven", "N(above)", "AvgR(above)", "N(below)", "AvgR(below)"],
        summary_rows
    )


def section_3_friction_theory(con):
    """Test the friction hypothesis: instruments with higher friction need bigger ORBs."""
    print()
    print("  ================================================================")
    print("  SECTION 3: FRICTION HYPOTHESIS")
    print("  ================================================================")
    print()
    print("  Theory: Higher friction (cost as % of risk) requires bigger ORBs.")
    print("  MGC has highest friction -> needs biggest ORBs.")
    print("  MNQ has lowest friction -> should profit from smaller ORBs.")
    print()

    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    rows = []
    for sym in symbols:
        fric = FRICTION.get(sym, {})
        rt_points = fric.get("rt_points", "?")

        # Get avg R at different size thresholds across ALL sessions
        for gate in [2, 3, 4, 5, 6, 8]:
            # Use all sessions that have the size column
            sessions = [r[0] for r in con.execute(
                "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ? ORDER BY orb_label",
                [sym]
            ).fetchall()]

            total_n = 0
            total_pnl = 0.0

            for sess in sessions:
                size_col = f"orb_{sess}_size"
                try:
                    row = con.execute(f"""
                        SELECT COUNT(*) as n, SUM(o.pnl_r) as total_r
                        FROM orb_outcomes o
                        JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                        WHERE o.symbol = ? AND o.orb_label = ?
                          AND o.outcome IN ('win', 'loss')
                          AND d.{size_col} >= ?
                    """, [sym, sess, float(gate)]).fetchone()
                    if row and row[0]:
                        total_n += row[0]
                        total_pnl += row[1] if row[1] else 0
                except Exception:
                    continue

            avg_r = round(total_pnl / total_n, 4) if total_n > 0 else 0
            fric_pct = round(float(rt_points) / gate * 100, 1) if isinstance(rt_points, (int, float)) else "?"

            rows.append((
                sym, f"G{gate}+", total_n,
                f"{avg_r:+.4f}" if total_n > 0 else "-",
                f"{fric_pct}%" if isinstance(fric_pct, float) else "?"
            ))

    print_table(
        ["Symbol", "Gate", "Total N", "Avg R", "Friction %"],
        rows
    )

    print()
    print("  READING THIS TABLE:")
    print("  - Look at where Avg R crosses from negative to positive for each symbol.")
    print("  - Compare that crossing point to the friction %.")
    print("  - If friction hypothesis is correct: MGC crosses later (needs G5+),")
    print("    MNQ crosses earlier (works from G3+ or G4+).")


def section_4_optimal_gate_per_session(con):
    """For each session on each instrument, find the gate that maximizes total R."""
    print()
    print("  ================================================================")
    print("  SECTION 4: OPTIMAL GATE PER SESSION")
    print("  ================================================================")
    print()
    print("  For each session, which size gate maximizes TOTAL R?")
    print("  (Total R = sum of all trade R-multiples. Balances edge size vs frequency.)")
    print()

    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    best_rows = []

    for sym in symbols:
        sessions = [r[0] for r in con.execute(
            "SELECT DISTINCT orb_label FROM orb_outcomes WHERE symbol = ? ORDER BY orb_label",
            [sym]
        ).fetchall()]

        for sess in sessions:
            size_col = f"orb_{sess}_size"
            best_gate = None
            best_total_r = -999
            best_n = 0
            best_avg_r = 0
            gate_results = []

            for gate in [0, 2, 3, 4, 5, 6, 8, 10]:
                try:
                    if gate == 0:
                        # No filter
                        row = con.execute(f"""
                            SELECT COUNT(*) as n,
                                   ROUND(AVG(o.pnl_r), 4) as avg_r,
                                   ROUND(SUM(o.pnl_r), 1) as total_r
                            FROM orb_outcomes o
                            WHERE o.symbol = ? AND o.orb_label = ?
                              AND o.outcome IN ('win', 'loss')
                        """, [sym, sess]).fetchone()
                    else:
                        row = con.execute(f"""
                            SELECT COUNT(*) as n,
                                   ROUND(AVG(o.pnl_r), 4) as avg_r,
                                   ROUND(SUM(o.pnl_r), 1) as total_r
                            FROM orb_outcomes o
                            JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                            WHERE o.symbol = ? AND o.orb_label = ?
                              AND o.outcome IN ('win', 'loss')
                              AND d.{size_col} >= ?
                        """, [sym, sess, float(gate)]).fetchone()

                    if row and row[0] and row[0] >= 10:
                        n = row[0]
                        avg_r = row[1] if row[1] else 0
                        total_r = row[2] if row[2] else 0
                        gate_label = "NONE" if gate == 0 else f"G{gate}+"
                        gate_results.append((gate_label, n, avg_r, total_r))

                        if total_r > best_total_r:
                            best_total_r = total_r
                            best_gate = gate_label
                            best_n = n
                            best_avg_r = avg_r
                except Exception:
                    continue

            if best_gate and best_total_r > 0:
                best_rows.append((
                    sym, sess, best_gate,
                    best_n, f"{best_avg_r:+.4f}", f"{best_total_r:+.1f}"
                ))

    print_table(
        ["Symbol", "Session", "Best Gate", "N", "Avg R", "Total R"],
        best_rows
    )

    print()
    print("  NOTE: 'Best Gate' maximizes Total R (edge * frequency).")
    print("  A tighter gate (G6+ vs G4+) may have higher avg R but fewer trades.")
    print("  Total R balances both. Use this to set default gates per session.")


def section_5_size_vs_direction(con):
    """Does ORB size affect long vs short differently?"""
    print()
    print("  ================================================================")
    print("  SECTION 5: SIZE x DIRECTION -- Does Size Affect Longs/Shorts Differently?")
    print("  ================================================================")
    print()

    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM orb_outcomes ORDER BY symbol"
    ).fetchall()]

    for sym in symbols:
        # Pick the most active sessions
        sessions = [r[0] for r in con.execute("""
            SELECT orb_label FROM orb_outcomes
            WHERE symbol = ? AND outcome IN ('win', 'loss')
            GROUP BY orb_label
            HAVING COUNT(*) >= 100
            ORDER BY COUNT(*) DESC
            LIMIT 4
        """, [sym]).fetchall()]

        if not sessions:
            continue

        print(f"\n  --- {sym} (top sessions by trade count) ---\n")

        for sess in sessions:
            size_col = f"orb_{sess}_size"
            dir_col = f"orb_{sess}_break_dir"
            try:
                rows = con.execute(f"""
                    SELECT
                        CASE WHEN d.{size_col} >= 5.0 THEN 'LARGE' ELSE 'SMALL' END as sz,
                        UPPER(d.{dir_col}) as dir,
                        COUNT(*) as n,
                        ROUND(AVG(o.pnl_r), 4) as avg_r,
                        ROUND(SUM(o.pnl_r), 1) as total_r
                    FROM orb_outcomes o
                    JOIN daily_features d ON o.trading_day = d.trading_day AND o.symbol = d.symbol
                    WHERE o.symbol = ? AND o.orb_label = ?
                      AND o.outcome IN ('win', 'loss')
                      AND d.{dir_col} IS NOT NULL
                      AND d.{size_col} IS NOT NULL
                    GROUP BY sz, dir
                    ORDER BY sz, dir
                """, [sym, sess]).fetchall()
            except Exception:
                continue

            if not rows:
                continue

            print(f"  {sess}:")
            print_table(["Size", "Dir", "N", "Avg R", "Total R"], rows)
            print()


def section_6_verdict(con):
    """Summarize the findings."""
    print()
    print("  ================================================================")
    print("  VERDICT: What Did We Learn?")
    print("  ================================================================")
    print()
    print("  This script maps the ORB size landscape. Key questions answered:")
    print()
    print("  1. WHERE does each session's edge start? (Section 2 breakeven)")
    print("  2. Does friction predict the breakeven? (Section 3)")
    print("  3. What gate maximizes total R per session? (Section 4)")
    print("  4. Does size affect longs/shorts differently? (Section 5)")
    print()
    print("  USE THIS DATA TO:")
    print("  - Set instrument-specific default gates (not one-size-fits-all)")
    print("  - Decide which sessions to keep/drop per instrument")
    print("  - Understand whether percentage-based ORB sizing would auto-adapt")
    print("    (e.g., 0.15% of price rather than fixed 5pt)")
    print()
    print("  CAVEATS:")
    print("  - All results are in-sample. Walk-forward needed for deployment.")
    print("  - Outcome data may be incomplete (check rebuild status).")
    print("  - Size buckets with N < 30 are REGIME-class at best.")
    print()


def main():
    db_path = get_db_path()
    if not db_path.exists():
        print(f"\n  Database not found: {db_path}")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        print()
        print("  +=========================================================+")
        print("  |           ORB SIZE DEEP DIVE                             |")
        print("  |  Mapping the real edge: optimal thresholds everywhere    |")
        print("  +=========================================================+")
        print()
        print(f"  DB: {db_path}")

        section_1_size_heatmap(con)
        section_2_breakeven_finder(con)
        section_3_friction_theory(con)
        section_4_optimal_gate_per_session(con)
        section_5_size_vs_direction(con)
        section_6_verdict(con)

    finally:
        con.close()


if __name__ == "__main__":
    main()
