"""
ATR Velocity Contraction Gate — quantified Sharpe improvement.

Validates the claim: "A simple ATR velocity gate improves Sharpe by 0.2-0.3."

The ATRVelocityFilter skips trades when BOTH:
  1. atr_vel_regime == 'Contracting' (ATR < 95% of 5-day avg)
  2. compression_tier in ('Neutral', 'Compressed') (NOT Expanded)

This script measures the EXACT lift from applying this gate:
- Sharpe with/without the gate per instrument/session/entry_model
- Year-by-year consistency of the skipped days being negative
- Cross-instrument BH FDR confirmation
- Quantify days/trades removed and their toxicity
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
from scipy import stats
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "atr_velocity_gate_findings.md")


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


def annualized_sharpe(returns, trading_days_per_year=252):
    """Annualized Sharpe from daily R returns."""
    if len(returns) < 10:
        return None
    arr = np.array(returns)
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=1)
    if sigma == 0:
        return None
    return (mu / sigma) * np.sqrt(trading_days_per_year)


def run():
    con = duckdb.connect(str(DB), read_only=True)
    lines = []
    lines.append("# ATR Velocity Contraction Gate — Quantified Improvement")
    lines.append(f"**Date:** 2026-02-22")
    lines.append(f"**Script:** research/research_atr_velocity_gate.py\n")

    # ── PART 0: How many days does the gate remove? ──────────────
    lines.append("=" * 70)
    lines.append("PART 0: GATE REMOVAL RATES")
    lines.append("=" * 70 + "\n")

    lines.append("  Gate logic: skip when atr_vel_regime='Contracting' AND")
    lines.append("  orb_{session}_compression_tier IN ('Neutral','Compressed')\n")

    for sym in ["MNQ", "MGC", "MES"]:
        for sess in ["0900", "1000"]:
            comp_col = f"orb_{sess}_compression_tier"
            row = con.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN d.atr_vel_regime = 'Contracting'
                               AND d.{comp_col} IN ('Neutral','Compressed')
                          THEN 1 END) as skipped,
                    COUNT(CASE WHEN d.atr_vel_regime = 'Contracting'
                          THEN 1 END) as contracting_total
                FROM orb_outcomes o
                JOIN daily_features d
                  ON o.trading_day = d.trading_day
                  AND o.symbol = d.symbol
                  AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ?
                  AND o.orb_label = ?
                  AND o.entry_model = 'E0'
                  AND o.rr_target = 2.0
                  AND o.pnl_r IS NOT NULL
                  AND d.atr_vel_regime IS NOT NULL
                  AND d.{comp_col} IS NOT NULL
            """, [sym, sess]).fetchone()

            total, skipped, contracting = row
            skip_pct = skipped / total * 100 if total else 0
            lines.append(f"  {sym} {sess}: {skipped}/{total} days skipped ({skip_pct:.1f}%), "
                         f"contracting total={contracting}")
    lines.append("")

    # ── PART 1: Sharpe WITH vs WITHOUT gate ───────────────────────
    lines.append("=" * 70)
    lines.append("PART 1: SHARPE LIFT FROM ATR VELOCITY GATE")
    lines.append("=" * 70 + "\n")

    lines.append("  Comparing: ALL trades vs GATED trades (skipping Contracting+Neutral/Compressed)")
    lines.append("  Sharpe = annualized (mean/std * sqrt(252)) on daily R returns\n")

    all_tests = []

    for sym in ["MNQ", "MGC", "MES"]:
        for sess in ["0900", "1000"]:
            comp_col = f"orb_{sess}_compression_tier"
            for em in ["E0", "E1"]:
                rows = con.execute(f"""
                    SELECT o.pnl_r,
                           d.atr_vel_regime,
                           d.{comp_col} as comp_tier,
                           EXTRACT(YEAR FROM o.trading_day) as yr
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day
                      AND o.symbol = d.symbol
                      AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND o.pnl_r IS NOT NULL
                      AND d.atr_vel_regime IS NOT NULL
                      AND d.{comp_col} IS NOT NULL
                """, [sym, sess, em]).fetchall()

                if len(rows) < 50:
                    continue

                all_r = [r[0] for r in rows]
                # Skipped = Contracting AND (Neutral or Compressed)
                skipped = [r[0] for r in rows
                           if r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed')]
                # Gated = everything NOT skipped
                gated = [r[0] for r in rows
                         if not (r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed'))]

                if len(skipped) < 5 or len(gated) < 30:
                    continue

                label = f"{sym}_{sess}_{em}_RR2.0"

                sharpe_all = annualized_sharpe(all_r)
                sharpe_gated = annualized_sharpe(gated)
                avg_all = np.mean(all_r)
                avg_gated = np.mean(gated)
                avg_skipped = np.mean(skipped)
                wr_all = np.mean([1 if x > 0 else 0 for x in all_r])
                wr_gated = np.mean([1 if x > 0 else 0 for x in gated])
                wr_skipped = np.mean([1 if x > 0 else 0 for x in skipped])

                sharpe_lift = (sharpe_gated - sharpe_all) if sharpe_all and sharpe_gated else None

                lines.append(f"  {label}:")
                lines.append(f"    ALL:     N={len(all_r):>5}, avgR={avg_all:+.3f}, WR={wr_all:.1%}"
                             f", Sharpe={sharpe_all:+.2f}" if sharpe_all else
                             f"    ALL:     N={len(all_r):>5}, avgR={avg_all:+.3f}, WR={wr_all:.1%}")
                lines.append(f"    GATED:   N={len(gated):>5}, avgR={avg_gated:+.3f}, WR={wr_gated:.1%}"
                             f", Sharpe={sharpe_gated:+.2f}" if sharpe_gated else
                             f"    GATED:   N={len(gated):>5}, avgR={avg_gated:+.3f}, WR={wr_gated:.1%}")
                lines.append(f"    SKIPPED: N={len(skipped):>5}, avgR={avg_skipped:+.3f}, WR={wr_skipped:.1%}")
                if sharpe_lift is not None:
                    lines.append(f"    Sharpe lift: {sharpe_lift:+.3f}")

                # T-test: skipped vs gated (are skipped days significantly worse?)
                if len(skipped) >= 10:
                    _, p = stats.ttest_ind(skipped, gated, equal_var=False)
                    delta = avg_skipped - avg_gated
                    all_tests.append((label, p, len(skipped), avg_skipped, avg_gated,
                                      delta, wr_skipped, wr_gated,
                                      sharpe_all, sharpe_gated, sharpe_lift))
                    lines.append(f"    Skipped vs Gated: delta={delta:+.3f}R, p={p:.4f}")
                lines.append("")

    # ── PART 2: Year-by-year skipped-day toxicity ─────────────────
    lines.append("=" * 70)
    lines.append("PART 2: YEAR-BY-YEAR SKIPPED-DAY TOXICITY")
    lines.append("=" * 70 + "\n")

    for sym in ["MNQ", "MGC", "MES"]:
        for sess in ["0900", "1000"]:
            comp_col = f"orb_{sess}_compression_tier"
            for em in ["E0", "E1"]:
                rows = con.execute(f"""
                    SELECT o.pnl_r,
                           d.atr_vel_regime,
                           d.{comp_col} as comp_tier,
                           EXTRACT(YEAR FROM o.trading_day) as yr
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day
                      AND o.symbol = d.symbol
                      AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND o.pnl_r IS NOT NULL
                      AND d.atr_vel_regime IS NOT NULL
                      AND d.{comp_col} IS NOT NULL
                """, [sym, sess, em]).fetchall()

                skipped = [(r[0], int(r[3])) for r in rows
                           if r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed')]

                if len(skipped) < 10:
                    continue

                label = f"{sym}_{sess}_{em}_RR2.0"
                years = sorted(set(y for _, y in skipped))
                neg_years = 0
                yr_details = []
                for yr in years:
                    yr_data = [r for r, y in skipped if y == yr]
                    if len(yr_data) >= 3:
                        avg = np.mean(yr_data)
                        wr = np.mean([1 if x > 0 else 0 for x in yr_data])
                        sign = "[-]" if avg < 0 else "[+]"
                        if avg < 0:
                            neg_years += 1
                        yr_details.append(f"    {yr}: N={len(yr_data):>3}, avgR={avg:+.3f}, WR={wr:.1%} {sign}")

                if yr_details:
                    total_years = len(yr_details)
                    lines.append(f"  {label} skipped-day R by year ({neg_years}/{total_years} negative):")
                    for d in yr_details:
                        lines.append(d)
                    lines.append("")

    # ── PART 3: What if we ONLY skip Contracting (ignore compression)? ──
    lines.append("=" * 70)
    lines.append("PART 3: SIMPLER GATE — JUST 'Contracting' (no compression check)")
    lines.append("=" * 70 + "\n")

    lines.append("  Tests whether the compression tier check adds value")
    lines.append("  or if 'Contracting' alone is sufficient.\n")

    for sym in ["MNQ", "MGC", "MES"]:
        for sess in ["0900", "1000"]:
            comp_col = f"orb_{sess}_compression_tier"
            for em in ["E0", "E1"]:
                rows = con.execute(f"""
                    SELECT o.pnl_r,
                           d.atr_vel_regime,
                           d.{comp_col} as comp_tier
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day
                      AND o.symbol = d.symbol
                      AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND o.pnl_r IS NOT NULL
                      AND d.atr_vel_regime IS NOT NULL
                      AND d.{comp_col} IS NOT NULL
                """, [sym, sess, em]).fetchall()

                if len(rows) < 50:
                    continue

                # Simple gate: skip ALL contracting days
                simple_skipped = [r[0] for r in rows if r[1] == 'Contracting']
                simple_gated = [r[0] for r in rows if r[1] != 'Contracting']

                # Current gate: skip Contracting + (Neutral/Compressed) only
                current_skipped = [r[0] for r in rows
                                   if r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed')]
                # Contracting + Expanded (kept by current gate)
                expanded_contracting = [r[0] for r in rows
                                        if r[1] == 'Contracting' and r[2] == 'Expanded']

                if len(simple_skipped) < 10 or len(current_skipped) < 5:
                    continue

                label = f"{sym}_{sess}_{em}"
                avg_simple_skip = np.mean(simple_skipped)
                avg_current_skip = np.mean(current_skipped)
                avg_expanded_keep = np.mean(expanded_contracting) if expanded_contracting else None

                sharpe_simple = annualized_sharpe(simple_gated)
                sharpe_all = annualized_sharpe([r[0] for r in rows])

                lines.append(f"  {label}:")
                lines.append(f"    ALL Contracting:  N={len(simple_skipped):>4}, avgR={avg_simple_skip:+.3f}")
                lines.append(f"    Contr+Neut/Comp:  N={len(current_skipped):>4}, avgR={avg_current_skip:+.3f}")
                if expanded_contracting:
                    lines.append(f"    Contr+Expanded:   N={len(expanded_contracting):>4}, avgR={avg_expanded_keep:+.3f}")
                lines.append(f"    Simple gate Sharpe: {sharpe_simple:+.2f}" if sharpe_simple else "")
                lines.append("")

    # ── PART 4: RR1.0 vs RR2.0 sensitivity ───────────────────────
    lines.append("=" * 70)
    lines.append("PART 4: RR SENSITIVITY (does the gate work at RR1.0 too?)")
    lines.append("=" * 70 + "\n")

    for rr in [1.0, 2.0]:
        for sym in ["MNQ", "MGC", "MES"]:
            for sess in ["0900", "1000"]:
                comp_col = f"orb_{sess}_compression_tier"
                rows = con.execute(f"""
                    SELECT o.pnl_r,
                           d.atr_vel_regime,
                           d.{comp_col} as comp_tier
                    FROM orb_outcomes o
                    JOIN daily_features d
                      ON o.trading_day = d.trading_day
                      AND o.symbol = d.symbol
                      AND o.orb_minutes = d.orb_minutes
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = 'E0'
                      AND o.rr_target = ?
                      AND o.pnl_r IS NOT NULL
                      AND d.atr_vel_regime IS NOT NULL
                      AND d.{comp_col} IS NOT NULL
                """, [sym, sess, rr]).fetchall()

                if len(rows) < 50:
                    continue

                skipped = [r[0] for r in rows
                           if r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed')]
                gated = [r[0] for r in rows
                         if not (r[1] == 'Contracting' and r[2] in ('Neutral', 'Compressed'))]

                if len(skipped) < 5:
                    continue

                avg_skip = np.mean(skipped)
                avg_gate = np.mean(gated)
                avg_all = np.mean([r[0] for r in rows])
                s_all = annualized_sharpe([r[0] for r in rows])
                s_gate = annualized_sharpe(gated)

                label = f"{sym}_{sess}_E0_RR{rr:.0f}"
                lines.append(f"  {label}: skip N={len(skipped)}, avgR_skip={avg_skip:+.3f}, "
                             f"Sharpe_all={s_all:+.2f}, Sharpe_gated={s_gate:+.2f}, "
                             f"lift={s_gate - s_all:+.3f}" if s_all and s_gate else
                             f"  {label}: skip N={len(skipped)}, avgR_skip={avg_skip:+.3f}")
    lines.append("")

    # ── BH FDR ────────────────────────────────────────────────────
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
        for (label, raw_p, n, avg_skip, avg_gate, delta, wr_skip, wr_gate,
             s_all, s_gate, s_lift), adj_p in survivors:
            direction = "TOXIC (skip helps)" if delta < 0 else "BENEFICIAL (skip hurts)"
            lift_str = f", Sharpe_lift={s_lift:+.3f}" if s_lift is not None else ""
            lines.append(f"    {label}: raw_p={raw_p:.4f}, p_bh={adj_p:.4f}, "
                         f"N_skip={n}, delta={delta:+.3f}R, "
                         f"WR_skip={wr_skip:.1%} vs WR_gate={wr_gate:.1%}"
                         f"{lift_str}, {direction}")
    lines.append("")

    # ── HONEST SUMMARY ────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")

    lines.append("### Claim: 'ATR velocity gate improves Sharpe by 0.2-0.3'")
    lines.append("  See Part 1 for actual Sharpe lifts per instrument/session.\n")

    lines.append("### Key questions answered:")
    lines.append("  1. How many days does the gate remove? -> Part 0")
    lines.append("  2. Actual Sharpe improvement? -> Part 1")
    lines.append("  3. Year-by-year consistency of skipped toxicity? -> Part 2")
    lines.append("  4. Does compression check add value? -> Part 3")
    lines.append("  5. Does it work at RR1.0? -> Part 4")
    lines.append("  6. BH FDR survived? -> BH section\n")

    lines.append("### CAVEATS")
    lines.append("  - ATR velocity gate is already live in paper_trader.py")
    lines.append("  - MNQ only ~2 years of data")
    lines.append("  - Gate was designed from same data (IS/OOS split not available)")
    lines.append("  - Sharpe lift depends on base strategy performance")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    run()
