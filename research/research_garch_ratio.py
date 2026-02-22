"""
GARCH/ATR ratio mispricing filter validation.

Tests: When GARCH undershoots realized vol (garch_atr_ratio < 0.58),
does MNQ 0900 win rate hit 60.7% with +0.26R?

Also tests cross-instrument, cross-session, threshold sensitivity, year-by-year.
BH FDR correction across all tests.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
from scipy import stats
import numpy as np
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "garch_ratio_findings.md")


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


def size_label(n):
    if n >= 200: return "CORE"
    if n >= 100: return "PRELIMINARY"
    if n >= 30: return "REGIME"
    return "INVALID"


def run():
    con = duckdb.connect(str(DB), read_only=True)
    lines = []
    lines.append("# GARCH/ATR Ratio Mispricing Filter Validation")
    lines.append(f"**Date:** 2026-02-21")
    lines.append(f"**Script:** research/research_garch_ratio.py\n")

    # ── Claim check ─────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 0: DIRECT CLAIM CHECK -- MNQ 0900, garch_atr_ratio < 0.58")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for rr in [1.0, 1.5, 2.0]:
            rows = con.execute("""
                SELECT d.garch_atr_ratio, o.pnl_r,
                       EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = 'MNQ'
                  AND o.orb_label = '0900'
                  AND o.entry_model = ?
                  AND o.rr_target = ?
                  AND d.garch_atr_ratio IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [em, rr]).fetchall()

            if not rows:
                lines.append(f"  MNQ 0900 {em} RR{rr}: NO DATA")
                continue

            below = [r[1] for r in rows if r[0] < 0.58]
            above = [r[1] for r in rows if r[0] >= 0.58]

            if len(below) < 5 or len(above) < 5:
                lines.append(f"  MNQ 0900 {em} RR{rr}: below_0.58 N={len(below)} (too small)")
                continue

            avg_b = np.mean(below)
            avg_a = np.mean(above)
            wr_b = np.mean([1 if x > 0 else 0 for x in below])
            wr_a = np.mean([1 if x > 0 else 0 for x in above])
            _, p = stats.ttest_ind(below, above, equal_var=False)

            lines.append(f"  MNQ 0900 {em} RR{rr}:")
            lines.append(f"    GARCH<0.58: N={len(below):>4}, avgR={avg_b:+.3f}, WR={wr_b:.1%}")
            lines.append(f"    GARCH>=0.58: N={len(above):>4}, avgR={avg_a:+.3f}, WR={wr_a:.1%}")
            lines.append(f"    Delta: {avg_b - avg_a:+.3f}R, WR delta: {(wr_b - wr_a)*100:+.1f}pp, "
                         f"p={p:.4f}, {size_label(len(below))}")
            lines.append("")

    # ── Cross-instrument x session ──────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 1: GARCH<0.58 vs >=0.58 (all instruments x sessions x entry models)")
    lines.append("=" * 70 + "\n")

    all_tests = []

    for sym in ["MGC", "MES", "MNQ"]:
        for sess in ["0900", "1000", "1800"]:
            for em in ["E0", "E1"]:
                rows = con.execute(f"""
                    SELECT d.garch_atr_ratio, o.pnl_r
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day
                      AND o.symbol = d.symbol
                      AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND d.garch_atr_ratio IS NOT NULL
                      AND d.orb_minutes = 5
                      AND o.pnl_r IS NOT NULL
                """, [sym, sess, em]).fetchall()

                if not rows:
                    continue

                below = [r[1] for r in rows if r[0] < 0.58]
                above = [r[1] for r in rows if r[0] >= 0.58]

                if len(below) < 10 or len(above) < 10:
                    continue

                avg_b = np.mean(below)
                avg_a = np.mean(above)
                wr_b = np.mean([1 if x > 0 else 0 for x in below])
                wr_a = np.mean([1 if x > 0 else 0 for x in above])
                _, p = stats.ttest_ind(below, above, equal_var=False)

                label = f"{sym}_{sess}_{em}_RR2.0"
                all_tests.append((label, p, len(below), avg_b, avg_a, avg_b - avg_a, wr_b, wr_a))

                lines.append(f"  {label}:")
                lines.append(f"    GARCH<0.58: N={len(below):>4}, avgR={avg_b:+.3f}, WR={wr_b:.1%}")
                lines.append(f"    GARCH>=0.58: N={len(above):>4}, avgR={avg_a:+.3f}, WR={wr_a:.1%}")
                lines.append(f"    Delta: {avg_b - avg_a:+.3f}R, WR_delta: {(wr_b - wr_a)*100:+.1f}pp, "
                             f"p={p:.4f}, {size_label(len(below))}")
                lines.append("")

    # ── Threshold sensitivity ───────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 2: THRESHOLD SENSITIVITY (MNQ 0900)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        rows = con.execute("""
            SELECT d.garch_atr_ratio, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
              AND o.symbol = d.symbol
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = 'MNQ'
              AND o.orb_label = '0900'
              AND o.entry_model = ?
              AND o.rr_target = 2.0
              AND d.garch_atr_ratio IS NOT NULL
              AND d.orb_minutes = 5
              AND o.pnl_r IS NOT NULL
        """, [em]).fetchall()

        if not rows:
            continue

        lines.append(f"  MNQ 0900 {em} RR2.0:")
        for thresh in [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
            below = [r[1] for r in rows if r[0] < thresh]
            above = [r[1] for r in rows if r[0] >= thresh]
            if len(below) >= 10 and len(above) >= 10:
                avg_b = np.mean(below)
                avg_a = np.mean(above)
                wr_b = np.mean([1 if x > 0 else 0 for x in below])
                wr_a = np.mean([1 if x > 0 else 0 for x in above])
                _, p = stats.ttest_ind(below, above, equal_var=False)
                lines.append(f"    <{thresh:.2f}: N={len(below):>4}, avgR={avg_b:+.3f}, WR={wr_b:.1%} | "
                             f">={thresh:.2f}: N={len(above):>4}, avgR={avg_a:+.3f}, WR={wr_a:.1%} | "
                             f"delta={avg_b - avg_a:+.3f}, p={p:.4f}")
        lines.append("")

    # ── Year-by-year for MNQ 0900 ──────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 3: YEAR-BY-YEAR (MNQ 0900, GARCH<0.58)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for rr in [1.0, 2.0]:
            rows = con.execute("""
                SELECT d.garch_atr_ratio, o.pnl_r,
                       EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = 'MNQ'
                  AND o.orb_label = '0900'
                  AND o.entry_model = ?
                  AND o.rr_target = ?
                  AND d.garch_atr_ratio IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [em, rr]).fetchall()

            below = [(r[1], int(r[2])) for r in rows if r[0] < 0.58]
            if len(below) < 5:
                continue

            lines.append(f"  MNQ 0900 {em} RR{rr} GARCH<0.58 (N={len(below)}):")
            years = sorted(set(y for _, y in below))
            yrs_pos = 0
            for yr in years:
                yr_data = [r for r, y in below if y == yr]
                if yr_data:
                    avg = np.mean(yr_data)
                    wr = np.mean([1 if x > 0 else 0 for x in yr_data])
                    sign = "[+]" if avg > 0 else "[-]"
                    if avg > 0: yrs_pos += 1
                    lines.append(f"    {yr}: N={len(yr_data):>4}, avgR={avg:+.3f}, WR={wr:.1%} {sign}")
            lines.append(f"    Years positive: {yrs_pos}/{len(years)}")
            lines.append("")

    # ── Redundancy vs ATR velocity ──────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 4: REDUNDANCY -- GARCH RATIO vs ATR VELOCITY REGIME")
    lines.append("=" * 70 + "\n")

    for sym in ["MGC", "MES", "MNQ"]:
        rows = con.execute("""
            SELECT d.garch_atr_ratio, d.atr_vel_regime, o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
              AND o.symbol = d.symbol
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = ?
              AND o.orb_label = '0900'
              AND o.entry_model = 'E0'
              AND o.rr_target = 2.0
              AND d.garch_atr_ratio IS NOT NULL
              AND d.atr_vel_regime IS NOT NULL
              AND d.orb_minutes = 5
              AND o.pnl_r IS NOT NULL
        """, [sym]).fetchall()

        if not rows:
            continue

        lines.append(f"  {sym} 0900 E0 RR2.0:")
        for vel in ["Expanding", "Stable", "Contracting"]:
            for garch_bin in ["GARCH<0.58", "GARCH>=0.58"]:
                thresh = 0.58
                if garch_bin == "GARCH<0.58":
                    subset = [r[2] for r in rows if r[1] == vel and r[0] < thresh]
                else:
                    subset = [r[2] for r in rows if r[1] == vel and r[0] >= thresh]
                if len(subset) >= 5:
                    avg = np.mean(subset)
                    wr = np.mean([1 if x > 0 else 0 for x in subset])
                    lines.append(f"    {vel:>12} x {garch_bin:>12}: N={len(subset):>4}, "
                                 f"avgR={avg:+.3f}, WR={wr:.1%}")
        lines.append("")

    # ── Correlation between GARCH ratio and ATR vel ─────────────
    lines.append("  CORRELATION: garch_atr_ratio vs atr_vel_ratio:")
    for sym in ["MGC", "MES", "MNQ"]:
        rows = con.execute("""
            SELECT garch_atr_ratio, atr_vel_ratio
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
              AND garch_atr_ratio IS NOT NULL
              AND atr_vel_ratio IS NOT NULL
        """, [sym]).fetchall()
        if len(rows) >= 10:
            r, p = stats.pearsonr([r[0] for r in rows], [r[1] for r in rows])
            lines.append(f"    {sym}: r={r:.3f}, p={p:.4f}, N={len(rows)}")
    lines.append("")

    # ── BH FDR ──────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("BH FDR CORRECTION (Part 1 tests)")
    lines.append("=" * 70 + "\n")

    lines.append(f"  Total tests: {len(all_tests)}")
    if all_tests:
        raw_pvals = [t[1] for t in all_tests]
        adj_pvals = bh_fdr(raw_pvals)

        survivors = [(all_tests[i], adj_pvals[i]) for i in range(len(all_tests)) if adj_pvals[i] < 0.10]
        survivors.sort(key=lambda x: x[1])

        lines.append(f"  BH survivors (q=0.10): {len(survivors)}")
        for (label, raw_p, n, avg_b, avg_a, delta, wr_b, wr_a), adj_p in survivors:
            direction = "LOW GARCH HELPS" if delta > 0 else "LOW GARCH HURTS"
            lines.append(f"    {label}: raw_p={raw_p:.4f}, p_bh={adj_p:.4f}, "
                         f"N_below={n}, delta={delta:+.3f}R, {direction}")
    lines.append("")

    # ── Verdict ─────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")

    lines.append("### MNQ 0900 GARCH<0.58 Claim Check")
    lines.append("  Claim: WR=60.7%, avgR=+0.26R when garch_atr_ratio < 0.58")
    lines.append("  [Results above show actual values]")
    lines.append("")
    lines.append("### CAVEATS")
    lines.append("  - MNQ has only ~332 rows with GARCH data (2+ year warm-up eats early data)")
    lines.append("  - GARCH<0.58 produces N~66 for MNQ -- REGIME size at best")
    lines.append("  - Previous research (research_garch_vs_atr.py) already found ATR wins 6/8 sessions")
    lines.append("  - 0.58 threshold appears cherry-picked from a continuous distribution")
    lines.append("  - 0900 session NOT DST-split")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    run()
