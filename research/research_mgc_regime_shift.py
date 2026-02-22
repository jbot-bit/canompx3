"""
MGC structural regime shift analysis.

Tests: ORB sizes went from 1.4 pts (2024) to 22.6 pts (2026 YTD).
ATR expanded 31->105. Do G4/G6/G8 filters still make sense?
How does this affect validated strategies?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "mgc_regime_shift_findings.md")


def run():
    con = duckdb.connect(str(DB), read_only=True)
    lines = []
    lines.append("# MGC Structural Regime Shift Analysis")
    lines.append(f"**Date:** 2026-02-21")
    lines.append(f"**Script:** research/research_mgc_regime_shift.py\n")

    # ── Quarterly ORB size and ATR trends ───────────────────────
    lines.append("=" * 70)
    lines.append("PART 1: QUARTERLY ORB SIZE & ATR TRENDS")
    lines.append("=" * 70 + "\n")

    rows = con.execute("""
        SELECT
            EXTRACT(YEAR FROM trading_day) as yr,
            EXTRACT(QUARTER FROM trading_day) as qtr,
            AVG(atr_20) as avg_atr,
            AVG(orb_1000_size) as avg_orb_1000,
            AVG(orb_0900_size) as avg_orb_0900,
            AVG(orb_1800_size) as avg_orb_1800,
            COUNT(*) as n_days,
            AVG(CASE WHEN orb_1000_size IS NOT NULL AND atr_20 > 0
                THEN orb_1000_size / atr_20 END) as avg_orb_atr_ratio
        FROM daily_features
        WHERE symbol = 'MGC' AND orb_minutes = 5 AND atr_20 IS NOT NULL
        GROUP BY yr, qtr
        ORDER BY yr, qtr
    """).fetchall()

    lines.append(f"  {'Qtr':>8} {'N':>4} {'ATR':>7} {'ORB1000':>8} {'ORB0900':>8} {'ORB1800':>8} {'ORB/ATR':>8}")
    lines.append(f"  {'-'*8} {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for yr, qtr, atr, orb1k, orb9, orb18, n, ratio in rows:
        yr_q = f"{int(yr)}Q{int(qtr)}"
        lines.append(f"  {yr_q:>8} {n:>4} {atr:>7.1f} {orb1k or 0:>8.1f} {orb9 or 0:>8.1f} "
                     f"{orb18 or 0:>8.1f} {ratio or 0:>8.3f}")
    lines.append("")

    # ── G filter pass rates by year ─────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 2: G FILTER PASS RATES BY YEAR (1000 session)")
    lines.append("=" * 70 + "\n")

    rows = con.execute("""
        SELECT
            EXTRACT(YEAR FROM trading_day) as yr,
            COUNT(*) as total,
            COUNT(CASE WHEN orb_1000_size >= 4 THEN 1 END) as g4_pass,
            COUNT(CASE WHEN orb_1000_size >= 6 THEN 1 END) as g6_pass,
            COUNT(CASE WHEN orb_1000_size >= 8 THEN 1 END) as g8_pass,
            AVG(orb_1000_size) as avg_orb
        FROM daily_features
        WHERE symbol = 'MGC' AND orb_minutes = 5 AND orb_1000_size IS NOT NULL
        GROUP BY yr
        ORDER BY yr
    """).fetchall()

    lines.append(f"  {'Year':>6} {'N':>5} {'AvgORB':>7} {'G4%':>6} {'G6%':>6} {'G8%':>6}")
    lines.append(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for yr, n, g4, g6, g8, avg in rows:
        lines.append(f"  {int(yr):>6} {n:>5} {avg:>7.1f} "
                     f"{g4/n*100 if n else 0:>5.1f}% {g6/n*100 if n else 0:>5.1f}% "
                     f"{g8/n*100 if n else 0:>5.1f}%")
    lines.append("")

    # ── Same for 0900 session ───────────────────────────────────
    lines.append("  --- 0900 session ---")
    rows = con.execute("""
        SELECT
            EXTRACT(YEAR FROM trading_day) as yr,
            COUNT(*) as total,
            COUNT(CASE WHEN orb_0900_size >= 4 THEN 1 END) as g4_pass,
            COUNT(CASE WHEN orb_0900_size >= 6 THEN 1 END) as g6_pass,
            COUNT(CASE WHEN orb_0900_size >= 8 THEN 1 END) as g8_pass,
            AVG(orb_0900_size) as avg_orb
        FROM daily_features
        WHERE symbol = 'MGC' AND orb_minutes = 5 AND orb_0900_size IS NOT NULL
        GROUP BY yr
        ORDER BY yr
    """).fetchall()

    lines.append(f"  {'Year':>6} {'N':>5} {'AvgORB':>7} {'G4%':>6} {'G6%':>6} {'G8%':>6}")
    lines.append(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
    for yr, n, g4, g6, g8, avg in rows:
        lines.append(f"  {int(yr):>6} {n:>5} {avg:>7.1f} "
                     f"{g4/n*100 if n else 0:>5.1f}% {g6/n*100 if n else 0:>5.1f}% "
                     f"{g8/n*100 if n else 0:>5.1f}%")
    lines.append("")

    # ── Performance by ATR regime ───────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 3: MGC PERFORMANCE BY ATR QUARTILE")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for sess in ["0900", "1000"]:
            rows = con.execute(f"""
                SELECT d.atr_20, o.pnl_r,
                       EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = 'MGC'
                  AND o.orb_label = ?
                  AND o.entry_model = ?
                  AND o.rr_target = 2.0
                  AND d.atr_20 IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [sess, em]).fetchall()

            if len(rows) < 100:
                continue

            atrs = [r[0] for r in rows]
            p25 = np.percentile(atrs, 25)
            p50 = np.percentile(atrs, 50)
            p75 = np.percentile(atrs, 75)

            lines.append(f"  MGC {sess} {em} RR2.0 (ATR quartiles: P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}):")

            for label, lo, hi in [("Q1 (low ATR)", 0, p25), ("Q2", p25, p50),
                                    ("Q3", p50, p75), ("Q4 (high ATR)", p75, 999)]:
                subset = [r[1] for r in rows if lo <= r[0] < hi]
                if len(subset) >= 10:
                    avg = np.mean(subset)
                    wr = np.mean([1 if x > 0 else 0 for x in subset])
                    lines.append(f"    {label:>16}: N={len(subset):>5}, avgR={avg:+.3f}, WR={wr:.1%}")

            # High ATR (>75) vs Low ATR (<40) t-test
            high = [r[1] for r in rows if r[0] >= 75]
            low = [r[1] for r in rows if r[0] < 40]
            if len(high) >= 20 and len(low) >= 20:
                _, p = stats.ttest_ind(high, low, equal_var=False)
                lines.append(f"    ATR>=75 vs ATR<40: high N={len(high)}, avgR={np.mean(high):+.3f} | "
                             f"low N={len(low)}, avgR={np.mean(low):+.3f} | p={p:.4f}")
            lines.append("")

    # ── Year-by-year performance (is 2025/2026 structurally different?) ──
    lines.append("=" * 70)
    lines.append("PART 4: YEAR-BY-YEAR MGC PERFORMANCE (regime shift timing)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for sess in ["0900", "1000", "1100"]:
            rows = con.execute(f"""
                SELECT o.pnl_r, EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                WHERE o.symbol = 'MGC'
                  AND o.orb_label = ?
                  AND o.entry_model = ?
                  AND o.rr_target = 2.0
                  AND o.pnl_r IS NOT NULL
            """, [sess, em]).fetchall()

            if not rows:
                continue

            lines.append(f"  MGC {sess} {em} RR2.0:")
            years = sorted(set(int(r[1]) for r in rows))
            for yr in years:
                yr_data = [r[0] for r in rows if int(r[1]) == yr]
                if yr_data:
                    avg = np.mean(yr_data)
                    wr = np.mean([1 if x > 0 else 0 for x in yr_data])
                    sign = "[+]" if avg > 0 else "[-]"
                    lines.append(f"    {yr}: N={len(yr_data):>5}, avgR={avg:+.3f}, WR={wr:.1%} {sign}")
            lines.append("")

    # ── Pre vs post regime shift ────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 5: PRE vs POST REGIME SHIFT (cut at 2025-01-01)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for sess in ["0900", "1000", "1100"]:
            rows = con.execute(f"""
                SELECT o.pnl_r,
                       CASE WHEN o.trading_day >= '2025-01-01' THEN 'post' ELSE 'pre' END as era
                FROM orb_outcomes o
                WHERE o.symbol = 'MGC'
                  AND o.orb_label = ?
                  AND o.entry_model = ?
                  AND o.rr_target = 2.0
                  AND o.pnl_r IS NOT NULL
            """, [sess, em]).fetchall()

            pre = [r[0] for r in rows if r[1] == "pre"]
            post = [r[0] for r in rows if r[1] == "post"]

            if len(pre) < 30 or len(post) < 30:
                continue

            avg_pre = np.mean(pre)
            avg_post = np.mean(post)
            wr_pre = np.mean([1 if x > 0 else 0 for x in pre])
            wr_post = np.mean([1 if x > 0 else 0 for x in post])
            _, p = stats.ttest_ind(pre, post, equal_var=False)

            lines.append(f"  MGC {sess} {em} RR2.0:")
            lines.append(f"    Pre-2025:  N={len(pre):>5}, avgR={avg_pre:+.3f}, WR={wr_pre:.1%}")
            lines.append(f"    Post-2025: N={len(post):>5}, avgR={avg_post:+.3f}, WR={wr_post:.1%}")
            lines.append(f"    Delta: {avg_post - avg_pre:+.3f}R, WR_delta: {(wr_post - wr_pre)*100:+.1f}pp, p={p:.4f}")
            lines.append("")

    # ── G filter effectiveness in high-ATR regime ───────────────
    lines.append("=" * 70)
    lines.append("PART 6: G FILTER EFFECTIVENESS IN HIGH-ATR REGIME (2025+)")
    lines.append("=" * 70 + "\n")

    for sess in ["0900", "1000"]:
        size_col = f"orb_{sess}_size"
        for em in ["E0", "E1"]:
            rows = con.execute(f"""
                SELECT d.{size_col} as orb_size, o.pnl_r
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = 'MGC'
                  AND o.orb_label = ?
                  AND o.entry_model = ?
                  AND o.rr_target = 2.0
                  AND d.{size_col} IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
                  AND o.trading_day >= '2025-01-01'
            """, [sess, em]).fetchall()

            if len(rows) < 30:
                continue

            lines.append(f"  MGC {sess} {em} RR2.0 (2025+ only):")
            for g_name, g_thresh in [("ALL", 0), ("G4+", 4), ("G6+", 6), ("G8+", 8),
                                      ("G10+", 10), ("G15+", 15), ("G20+", 20)]:
                subset = [r[1] for r in rows if r[0] >= g_thresh]
                if len(subset) >= 10:
                    avg = np.mean(subset)
                    wr = np.mean([1 if x > 0 else 0 for x in subset])
                    lines.append(f"    {g_name:>5}: N={len(subset):>5}, avgR={avg:+.3f}, WR={wr:.1%}")
            lines.append("")

    # ── Verdict ─────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")

    lines.append("### Key questions:")
    lines.append("  1. Has MGC ATR structurally shifted? -> see Part 1")
    lines.append("  2. Are G4/G6/G8 now meaningless? -> see Parts 2, 6")
    lines.append("  3. Is MGC performance structurally better now? -> see Parts 4, 5")
    lines.append("  4. Should filters be ATR-normalized? -> see Part 6")
    lines.append("")
    lines.append("### CAVEATS")
    lines.append("  - 2026 YTD is only ~7 weeks of data")
    lines.append("  - Gold price at all-time highs drives ATR mechanically")
    lines.append("  - Pre-2025 MGC data uses GC source (same price, different contract)")
    lines.append("  - G filters are in POINTS, not ATR-multiples")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    run()
