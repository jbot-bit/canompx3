"""
Compressed ORB filter validation.

Tests whether ORB compression (orb_size/ATR z-score) predicts breakout outcomes.
Specifically: does MNQ 0900 compressed ORB genuinely boost win rate by 13pp?

Methodology:
- Join orb_outcomes x daily_features on (trading_day, symbol, orb_minutes)
- Split by compression_tier (Compressed / Neutral / Expanded)
- Welch t-test: Compressed vs non-Compressed
- Year-by-year consistency
- Cross-instrument (MGC, MES, MNQ) x cross-session (0900, 1000, 1800)
- BH FDR correction across all tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
from scipy import stats
import numpy as np
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "compressed_orb_findings.md")


def bh_fdr(pvals, q=0.10):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[sorted_idx]
    adjusted = np.zeros(n)
    # Work backwards
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
    lines.append("# Compressed ORB Filter Validation")
    lines.append(f"**Date:** 2026-02-21")
    lines.append(f"**Script:** research/research_compressed_orb.py\n")

    # ── Coverage check ──────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 0: DATA COVERAGE")
    lines.append("=" * 70 + "\n")

    for sym in ["MGC", "MES", "MNQ"]:
        for sess in ["0900", "1000", "1800"]:
            col = f"orb_{sess}_compression_tier"
            row = con.execute(f"""
                SELECT COUNT(*) as total,
                       COUNT({col}) as has_tier,
                       COUNT(CASE WHEN {col}='Compressed' THEN 1 END) as n_compressed,
                       COUNT(CASE WHEN {col}='Neutral' THEN 1 END) as n_neutral,
                       COUNT(CASE WHEN {col}='Expanded' THEN 1 END) as n_expanded
                FROM daily_features
                WHERE symbol=? AND orb_minutes=5
            """, [sym]).fetchone()
            lines.append(f"  {sym} {sess}: total={row[0]}, has_tier={row[1]}, "
                         f"Compressed={row[2]}, Neutral={row[3]}, Expanded={row[4]}")
    lines.append("")

    # ── Main analysis ───────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 1: COMPRESSION TIER vs OUTCOME (all instruments x sessions)")
    lines.append("=" * 70 + "\n")

    all_tests = []  # (label, p_value, N, avgR_compressed, avgR_other, delta, wr_comp, wr_other)

    for sym in ["MGC", "MES", "MNQ"]:
        for sess in ["0900", "1000", "1800"]:
            tier_col = f"orb_{sess}_compression_tier"

            # Get R-values by tier
            rows = con.execute(f"""
                SELECT d.{tier_col} as tier,
                       o.pnl_r,
                       EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ?
                  AND o.orb_label = ?
                  AND o.entry_model = 'E1'
                  AND o.rr_target = 2.0
                  AND d.{tier_col} IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [sym, sess]).fetchall()

            if not rows:
                continue

            compressed = [r[1] for r in rows if r[0] == "Compressed"]
            neutral = [r[1] for r in rows if r[0] == "Neutral"]
            expanded = [r[1] for r in rows if r[0] == "Expanded"]
            non_compressed = neutral + expanded

            if len(compressed) < 10 or len(non_compressed) < 10:
                continue

            # T-test: compressed vs non-compressed
            t_stat, p_val = stats.ttest_ind(compressed, non_compressed, equal_var=False)
            avg_c = np.mean(compressed)
            avg_nc = np.mean(non_compressed)
            wr_c = np.mean([1 if x > 0 else 0 for x in compressed])
            wr_nc = np.mean([1 if x > 0 else 0 for x in non_compressed])

            label = f"{sym}_{sess}_E1_RR2.0"
            all_tests.append((label, p_val, len(compressed), avg_c, avg_nc,
                              avg_c - avg_nc, wr_c, wr_nc))

            size_label = ("CORE" if len(compressed) >= 200 else
                          "PRELIMINARY" if len(compressed) >= 100 else
                          "REGIME" if len(compressed) >= 30 else "INVALID")

            lines.append(f"  {label}:")
            lines.append(f"    Compressed: N={len(compressed):>4}, avgR={avg_c:+.3f}, WR={wr_c:.1%}")
            lines.append(f"    Non-Comp:   N={len(non_compressed):>4}, avgR={avg_nc:+.3f}, WR={wr_nc:.1%}")
            lines.append(f"    Delta: {avg_c - avg_nc:+.3f}R, WR delta: {(wr_c - wr_nc)*100:+.1f}pp, "
                         f"p={p_val:.4f}, {size_label}")

            # Breakdown by tier
            for tier_name, tier_data in [("Compressed", compressed),
                                          ("Neutral", neutral),
                                          ("Expanded", expanded)]:
                if len(tier_data) >= 5:
                    lines.append(f"      {tier_name:>12}: N={len(tier_data):>4}, "
                                 f"avgR={np.mean(tier_data):+.3f}, "
                                 f"WR={np.mean([1 if x > 0 else 0 for x in tier_data]):.1%}")
            lines.append("")

    # ── Also test E0 (the stronger entry model) ────────────────────
    lines.append("=" * 70)
    lines.append("PART 1b: SAME AT E0 (strongest entry model)")
    lines.append("=" * 70 + "\n")

    for sym in ["MGC", "MES", "MNQ"]:
        for sess in ["0900", "1000", "1800"]:
            tier_col = f"orb_{sess}_compression_tier"

            rows = con.execute(f"""
                SELECT d.{tier_col} as tier,
                       o.pnl_r,
                       EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ?
                  AND o.orb_label = ?
                  AND o.entry_model = 'E0'
                  AND o.rr_target = 2.0
                  AND d.{tier_col} IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [sym, sess]).fetchall()

            if not rows:
                continue

            compressed = [r[1] for r in rows if r[0] == "Compressed"]
            non_compressed = [r[1] for r in rows if r[0] in ("Neutral", "Expanded")]

            if len(compressed) < 10 or len(non_compressed) < 10:
                continue

            t_stat, p_val = stats.ttest_ind(compressed, non_compressed, equal_var=False)
            avg_c = np.mean(compressed)
            avg_nc = np.mean(non_compressed)
            wr_c = np.mean([1 if x > 0 else 0 for x in compressed])
            wr_nc = np.mean([1 if x > 0 else 0 for x in non_compressed])

            label = f"{sym}_{sess}_E0_RR2.0"
            all_tests.append((label, p_val, len(compressed), avg_c, avg_nc,
                              avg_c - avg_nc, wr_c, wr_nc))

            size_label = ("CORE" if len(compressed) >= 200 else
                          "PRELIMINARY" if len(compressed) >= 100 else
                          "REGIME" if len(compressed) >= 30 else "INVALID")

            lines.append(f"  {label}:")
            lines.append(f"    Compressed: N={len(compressed):>4}, avgR={avg_c:+.3f}, WR={wr_c:.1%}")
            lines.append(f"    Non-Comp:   N={len(non_compressed):>4}, avgR={avg_nc:+.3f}, WR={wr_nc:.1%}")
            lines.append(f"    Delta: {avg_c - avg_nc:+.3f}R, WR delta: {(wr_c - wr_nc)*100:+.1f}pp, "
                         f"p={p_val:.4f}, {size_label}")
            lines.append("")

    # ── Year-by-year for top results ────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 2: YEAR-BY-YEAR CONSISTENCY (MNQ 0900 focus)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        for rr in [1.0, 1.5, 2.0]:
            rows = con.execute(f"""
                SELECT d.orb_0900_compression_tier as tier,
                       o.pnl_r,
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
                  AND d.orb_0900_compression_tier IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [em, rr]).fetchall()

            if not rows:
                continue

            compressed = [(r[1], int(r[2])) for r in rows if r[0] == "Compressed"]
            non_comp = [(r[1], int(r[2])) for r in rows if r[0] in ("Neutral", "Expanded")]

            if len(compressed) < 10:
                continue

            lines.append(f"  MNQ 0900 {em} RR{rr} Compressed (N={len(compressed)}):")
            years = sorted(set(y for _, y in compressed))
            yrs_positive = 0
            for yr in years:
                yr_data = [r for r, y in compressed if y == yr]
                avg = np.mean(yr_data) if yr_data else 0
                wr = np.mean([1 if x > 0 else 0 for x in yr_data]) if yr_data else 0
                sign = "[+]" if avg > 0 else "[-]"
                if avg > 0:
                    yrs_positive += 1
                lines.append(f"    {yr}: N={len(yr_data):>4}, avgR={avg:+.3f}, WR={wr:.1%} {sign}")
            lines.append(f"    Years positive: {yrs_positive}/{len(years)}")
            lines.append("")

    # ── Redundancy check vs ATR velocity ────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 3: REDUNDANCY CHECK -- COMPRESSION vs ATR VELOCITY")
    lines.append("=" * 70 + "\n")

    for sym in ["MGC", "MES", "MNQ"]:
        for sess in ["0900", "1000"]:
            tier_col = f"orb_{sess}_compression_tier"
            rows = con.execute(f"""
                SELECT d.{tier_col} as comp_tier,
                       d.atr_vel_regime as vel_regime,
                       o.pnl_r
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ?
                  AND o.orb_label = ?
                  AND o.entry_model = 'E0'
                  AND o.rr_target = 2.0
                  AND d.{tier_col} IS NOT NULL
                  AND d.atr_vel_regime IS NOT NULL
                  AND d.orb_minutes = 5
                  AND o.pnl_r IS NOT NULL
            """, [sym, sess]).fetchall()

            if not rows:
                continue

            lines.append(f"  {sym} {sess} E0 RR2.0 -- 2x2 matrix:")
            for vel in ["Expanding", "Stable", "Contracting"]:
                for comp in ["Compressed", "Neutral", "Expanded"]:
                    subset = [r[2] for r in rows if r[1] == vel and r[0] == comp]
                    if len(subset) >= 5:
                        avg = np.mean(subset)
                        wr = np.mean([1 if x > 0 else 0 for x in subset])
                        lines.append(f"    {vel:>12} x {comp:>12}: N={len(subset):>4}, "
                                     f"avgR={avg:+.3f}, WR={wr:.1%}")
            lines.append("")

    # ── Threshold sensitivity ───────────────────────────────────────
    lines.append("=" * 70)
    lines.append("PART 4: Z-SCORE THRESHOLD SENSITIVITY (MNQ 0900)")
    lines.append("=" * 70 + "\n")

    for em in ["E0", "E1"]:
        rows = con.execute(f"""
            SELECT d.orb_0900_compression_z as z,
                   o.pnl_r
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.trading_day = d.trading_day
              AND o.symbol = d.symbol
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = 'MNQ'
              AND o.orb_label = '0900'
              AND o.entry_model = ?
              AND o.rr_target = 2.0
              AND d.orb_0900_compression_z IS NOT NULL
              AND d.orb_minutes = 5
              AND o.pnl_r IS NOT NULL
        """, [em]).fetchall()

        if not rows:
            continue

        lines.append(f"  MNQ 0900 {em} RR2.0 -- z-score thresholds:")
        for z_thresh in [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
            below = [r[1] for r in rows if r[0] <= z_thresh]
            above = [r[1] for r in rows if r[0] > z_thresh]
            if len(below) >= 10 and len(above) >= 10:
                avg_b = np.mean(below)
                avg_a = np.mean(above)
                wr_b = np.mean([1 if x > 0 else 0 for x in below])
                wr_a = np.mean([1 if x > 0 else 0 for x in above])
                _, p = stats.ttest_ind(below, above, equal_var=False)
                lines.append(f"    z<={z_thresh:+.2f}: N={len(below):>4}, avgR={avg_b:+.3f}, WR={wr_b:.1%} | "
                             f"z>{z_thresh:+.2f}: N={len(above):>4}, avgR={avg_a:+.3f}, WR={wr_a:.1%} | "
                             f"delta={avg_b - avg_a:+.3f}, p={p:.4f}")
        lines.append("")

    # ── BH FDR ──────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("BH FDR CORRECTION (all tests pooled)")
    lines.append("=" * 70 + "\n")

    lines.append(f"  Total tests: {len(all_tests)}")
    raw_pvals = [t[1] for t in all_tests]
    adj_pvals = bh_fdr(raw_pvals)

    survivors = [(all_tests[i], adj_pvals[i]) for i in range(len(all_tests)) if adj_pvals[i] < 0.10]
    survivors.sort(key=lambda x: x[1])

    lines.append(f"  BH survivors (q=0.10): {len(survivors)}")
    for (label, raw_p, n, avg_c, avg_nc, delta, wr_c, wr_nc), adj_p in survivors:
        lines.append(f"    {label}: raw_p={raw_p:.4f}, p_bh={adj_p:.4f}, "
                     f"N_comp={n}, delta={delta:+.3f}R, WR_comp={wr_c:.1%}, WR_other={wr_nc:.1%}")

    # ── Honest summary ──────────────────────────────────────────────
    lines.append("\n" + "=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")

    # Check if MNQ 0900 compressed shows the claimed +13pp
    mnq_0900_tests = [t for t in all_tests if "MNQ_0900" in t[0]]
    if mnq_0900_tests:
        lines.append("### MNQ 0900 Compressed ORB Claim Check")
        for label, raw_p, n, avg_c, avg_nc, delta, wr_c, wr_nc in mnq_0900_tests:
            lines.append(f"  {label}: WR_comp={wr_c:.1%}, WR_other={wr_nc:.1%}, "
                         f"WR_delta={((wr_c - wr_nc)*100):+.1f}pp, p={raw_p:.4f}")
        lines.append("")

    lines.append("### VERDICT")
    if survivors:
        lines.append("  BH-significant compression effects found:")
        for (label, raw_p, n, avg_c, avg_nc, delta, wr_c, wr_nc), adj_p in survivors:
            direction = "POSITIVE (compressed helps)" if delta > 0 else "NEGATIVE (compressed hurts)"
            lines.append(f"    {label}: {direction}, delta={delta:+.3f}R, p_bh={adj_p:.4f}")
    else:
        lines.append("  NO compression effects survived BH FDR correction.")
    lines.append("")

    lines.append("### CAVEATS")
    lines.append("  - MNQ only has ~2 years of data -- any MNQ finding is PRELIMINARY at best")
    lines.append("  - Compression tier z=-0.5 threshold is a design choice, not optimized")
    lines.append("  - 0900 session NOT DST-split in this analysis")
    lines.append("  - orb_0900_compression_tier compares vs prior 20 days of SAME instrument only")
    lines.append("  - This is NOT the cross-instrument 'all_narrow' concordance from deep research")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")

    con.close()


if __name__ == "__main__":
    run()
