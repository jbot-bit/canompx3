"""
Overnight PDH/PDL take as 0900 directional signal.

Tests: When overnight price takes PDH, does MNQ 0900 WR rise to 56.4%?
When only PDL taken, does performance degrade?

Requires computing prev_day_high/low from bars data, then checking
if overnight bars breach those levels before 0900 session.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "pdh_pdl_signal_findings.md")


def bh_fdr(pvals, q=0.10):
    n = len(pvals)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        rank = np.where(sorted_idx == sorted_idx[i])[0][0] + 1
        adjusted[sorted_idx[i]] = min(
            sorted_p[i] * n / rank,
            adjusted[sorted_idx[i + 1]] if i + 1 < n else 1.0
        )
    return np.clip(adjusted, 0, 1).tolist()


def run():
    con = duckdb.connect(str(DB), read_only=True)
    lines = []
    lines.append("# Overnight PDH/PDL Take Signal Validation")
    lines.append(f"**Date:** 2026-02-21")
    lines.append(f"**Script:** research/research_pdh_pdl_signal.py\n")

    # Check what columns are available for prev day high/low
    lines.append("=" * 70)
    lines.append("PART 0: DATA AVAILABILITY CHECK")
    lines.append("=" * 70 + "\n")

    # Check if prev_day_high/low exist in daily_features
    cols = con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'daily_features'
          AND column_name LIKE '%prev%'
        ORDER BY column_name
    """).fetchall()
    lines.append(f"  daily_features prev columns: {[c[0] for c in cols]}")

    # Check for daily_high/low
    cols2 = con.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'daily_features'
          AND column_name LIKE '%daily%'
        ORDER BY column_name
    """).fetchall()
    lines.append(f"  daily_features daily columns: {[c[0] for c in cols2]}")

    # Check bars_1m for overnight computation feasibility
    bar_count = con.execute("""
        SELECT symbol, COUNT(*) as n
        FROM bars_1m
        GROUP BY symbol
    """).fetchall()
    lines.append(f"  bars_1m row counts: {[(r[0], r[1]) for r in bar_count]}")
    lines.append("")

    # We need to compute PDH/PDL from daily_features or bars.
    # daily_features has daily_high, daily_low, daily_open, daily_close.
    # We can use LAG() to get prev day high/low.
    # Then check if overnight bars (before 0900) take PDH/PDL.

    # Strategy: Use daily_features for prev_day_high/low via LAG(),
    # then check if the daily_open > prev_day_high (gap above PDH)
    # or if orb_0900_high > prev_day_high (pre-0900 price took PDH).
    # But we can't precisely check overnight bars without bars_1m.

    # Simpler approach: Use gap direction + prev day range.
    # If gap_open > prev_day_high -> PDH taken overnight (gap above).
    # If gap_open < prev_day_low -> PDL taken overnight.
    # Otherwise -> no take.

    lines.append("=" * 70)
    lines.append("PART 1: PDH/PDL TAKE VIA GAP (gap_open vs prev_day_high/low)")
    lines.append("=" * 70 + "\n")

    lines.append("  Method: Use daily_features columns + LAG() for prev_day_high/low.")
    lines.append("  PDH_taken: today's daily_open > prev_day_high (gapped above PDH)")
    lines.append("  PDL_taken: today's daily_open < prev_day_low (gapped below PDL)")
    lines.append("  Neither: daily_open between prev_day_low and prev_day_high")
    lines.append("  Note: This only captures GAP-based PDH/PDL take, not intra-overnight take.")
    lines.append("")

    all_tests = []

    for sym in ["MNQ", "MGC", "MES"]:
        for sess in ["0900", "1000"]:
            for em in ["E0", "E1"]:
                rows = con.execute(f"""
                    WITH prev AS (
                        SELECT trading_day, symbol,
                               daily_high, daily_low, daily_open,
                               LAG(daily_high) OVER (
                                   PARTITION BY symbol ORDER BY trading_day
                               ) as prev_high,
                               LAG(daily_low) OVER (
                                   PARTITION BY symbol ORDER BY trading_day
                               ) as prev_low
                        FROM daily_features
                        WHERE symbol = ? AND orb_minutes = 5
                    )
                    SELECT
                        CASE
                            WHEN p.daily_open > p.prev_high THEN 'PDH_taken'
                            WHEN p.daily_open < p.prev_low THEN 'PDL_taken'
                            ELSE 'neither'
                        END as pdh_status,
                        o.pnl_r,
                        EXTRACT(YEAR FROM o.trading_day) as yr
                    FROM orb_outcomes o
                    JOIN prev p
                      ON o.trading_day = p.trading_day
                      AND o.symbol = p.symbol
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND o.pnl_r IS NOT NULL
                      AND p.prev_high IS NOT NULL
                      AND p.prev_low IS NOT NULL
                """, [sym, sym, sess, em]).fetchall()

                if not rows:
                    continue

                pdh = [r[1] for r in rows if r[0] == "PDH_taken"]
                pdl = [r[1] for r in rows if r[0] == "PDL_taken"]
                neither = [r[1] for r in rows if r[0] == "neither"]

                label = f"{sym}_{sess}_{em}_RR2.0"

                lines.append(f"  {label}:")
                for name, data in [("PDH_taken", pdh), ("PDL_taken", pdl), ("Neither", neither)]:
                    if len(data) >= 5:
                        avg = np.mean(data)
                        wr = np.mean([1 if x > 0 else 0 for x in data])
                        lines.append(f"    {name:>12}: N={len(data):>5}, avgR={avg:+.3f}, WR={wr:.1%}")

                # T-test: PDH_taken vs rest
                rest = pdl + neither
                if len(pdh) >= 10 and len(rest) >= 10:
                    _, p = stats.ttest_ind(pdh, rest, equal_var=False)
                    avg_pdh = np.mean(pdh)
                    avg_rest = np.mean(rest)
                    wr_pdh = np.mean([1 if x > 0 else 0 for x in pdh])
                    wr_rest = np.mean([1 if x > 0 else 0 for x in rest])
                    delta = avg_pdh - avg_rest
                    all_tests.append((label + "_PDH", p, len(pdh), avg_pdh, avg_rest,
                                      delta, wr_pdh, wr_rest))
                    lines.append(f"    PDH vs rest: delta={delta:+.3f}R, "
                                 f"WR_delta={(wr_pdh-wr_rest)*100:+.1f}pp, p={p:.4f}")

                # T-test: PDL_taken vs rest
                rest2 = pdh + neither
                if len(pdl) >= 10 and len(rest2) >= 10:
                    _, p = stats.ttest_ind(pdl, rest2, equal_var=False)
                    avg_pdl = np.mean(pdl)
                    avg_rest2 = np.mean(rest2)
                    wr_pdl = np.mean([1 if x > 0 else 0 for x in pdl])
                    wr_rest2 = np.mean([1 if x > 0 else 0 for x in rest2])
                    delta = avg_pdl - avg_rest2
                    all_tests.append((label + "_PDL", p, len(pdl), avg_pdl, avg_rest2,
                                      delta, wr_pdl, wr_rest2))
                    lines.append(f"    PDL vs rest: delta={delta:+.3f}R, "
                                 f"WR_delta={(wr_pdl-wr_rest2)*100:+.1f}pp, p={p:.4f}")
                lines.append("")

    # ── Year-by-year for MNQ 0900 ──────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 2: YEAR-BY-YEAR (MNQ 0900, PDH_taken)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        rows = con.execute(f"""
            WITH prev AS (
                SELECT trading_day, symbol, daily_open,
                       LAG(daily_high) OVER (PARTITION BY symbol ORDER BY trading_day) as prev_high,
                       LAG(daily_low) OVER (PARTITION BY symbol ORDER BY trading_day) as prev_low
                FROM daily_features
                WHERE symbol = 'MNQ' AND orb_minutes = 5
            )
            SELECT
                CASE
                    WHEN p.daily_open > p.prev_high THEN 'PDH_taken'
                    WHEN p.daily_open < p.prev_low THEN 'PDL_taken'
                    ELSE 'neither'
                END as pdh_status,
                o.pnl_r,
                EXTRACT(YEAR FROM o.trading_day) as yr
            FROM orb_outcomes o
            JOIN prev p ON o.trading_day = p.trading_day AND o.symbol = p.symbol
            WHERE o.symbol = 'MNQ' AND o.orb_label = '0900'
              AND o.entry_model = ? AND o.rr_target = 2.0
              AND o.pnl_r IS NOT NULL AND p.prev_high IS NOT NULL
        """, [em]).fetchall()

        pdh_rows = [(r[1], int(r[2])) for r in rows if r[0] == "PDH_taken"]
        if len(pdh_rows) < 10:
            lines.append(f"  MNQ 0900 {em}: PDH_taken N={len(pdh_rows)} (too small)")
            continue

        lines.append(f"  MNQ 0900 {em} RR2.0 PDH_taken (N={len(pdh_rows)}):")
        years = sorted(set(y for _, y in pdh_rows))
        for yr in years:
            yr_data = [r for r, y in pdh_rows if y == yr]
            if yr_data:
                avg = np.mean(yr_data)
                wr = np.mean([1 if x > 0 else 0 for x in yr_data])
                sign = "[+]" if avg > 0 else "[-]"
                lines.append(f"    {yr}: N={len(yr_data):>4}, avgR={avg:+.3f}, WR={wr:.1%} {sign}")
        lines.append("")

    # ── BH FDR ──────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("BH FDR CORRECTION")
    lines.append("=" * 70 + "\n")

    lines.append(f"  Total tests: {len(all_tests)}")
    if all_tests:
        raw_pvals = [t[1] for t in all_tests]
        adj_pvals = bh_fdr(raw_pvals)

        survivors = [(all_tests[i], adj_pvals[i]) for i in range(len(all_tests)) if adj_pvals[i] < 0.10]
        survivors.sort(key=lambda x: x[1])

        lines.append(f"  BH survivors (q=0.10): {len(survivors)}")
        for (label, raw_p, n, avg_b, avg_a, delta, wr_b, wr_a), adj_p in survivors:
            direction = "HELPS" if delta > 0 else "HURTS"
            lines.append(f"    {label}: raw_p={raw_p:.4f}, p_bh={adj_p:.4f}, "
                         f"N={n}, delta={delta:+.3f}R, WR_delta={(wr_b-wr_a)*100:+.1f}pp, {direction}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")
    lines.append("### CAVEATS")
    lines.append("  - 'PDH taken' here means GAPPED above prev day high (daily_open > prev_high)")
    lines.append("  - Does NOT capture intra-overnight PDH take (would need bars_1m analysis)")
    lines.append("  - Gap-above-PDH is a relatively rare event, so N may be small")
    lines.append("  - MNQ only ~2 years of data")
    lines.append("  - Not DST-split")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    run()
