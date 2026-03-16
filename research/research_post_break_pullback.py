#!/usr/bin/env python3
"""Post-break pullback analysis: does bar+1 or bar+2 closing back inside ORB predict failure?

Hypothesis: If within 2 bars after breakout, a bar closes back inside the ORB range,
the trade is more likely to fail. Should we cancel/exit?

Definition:
  - LONG break: "pullback" = bar close < ORB high (price fell back below breakout level)
  - SHORT break: "pullback" = bar close > ORB low (price rose back above breakdown level)

Scope: All 4 active instruments, all sessions, E1+E2, CB1, O5/O15/O30.
Uses bulk JOINs for performance instead of per-trade queries.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns number of survivors."""
    n = len(pvals)
    if n == 0:
        return 0
    sorted_pvals = sorted(enumerate(pvals), key=lambda x: x[1])
    survivors = 0
    for rank, (_, pv) in enumerate(sorted_pvals, 1):
        if pv <= q * rank / n:
            survivors = rank
    return survivors


def run_t_test(group_a, group_b):
    """Welch t-test. Returns (t_stat, p_val)."""
    if len(group_a) < 5 or len(group_b) < 5:
        return 0.0, 1.0
    return stats.ttest_ind(group_a, group_b, equal_var=False)


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    inst_tuple = str(tuple(instruments))

    cols = [r[0] for r in con.execute("DESCRIBE daily_features").fetchall()]
    sessions = sorted(
        set(
            c.replace("orb_", "").replace("_break_ts", "")
            for c in cols
            if c.endswith("_break_ts")
        )
    )

    print("Building bulk dataset (one query per session)...")

    # Collect all trades with pullback classification
    all_rows = []  # (inst, session, orb_min, pnl_r, pullback_flag)

    for session in sessions:
        col_high = f"orb_{session}_high"
        col_low = f"orb_{session}_low"
        col_dir = f"orb_{session}_break_dir"
        col_ts = f"orb_{session}_break_ts"

        if col_ts not in cols or col_high not in cols:
            continue

        for orb_min in [5, 15, 30]:
            # Bulk query: join trades with their next 2 bars in one shot
            # Use LATERAL or window, but simplest: get all trades + break_ts,
            # then join bars_5m with a range condition
            query = f"""
                WITH trades AS (
                    SELECT
                        df.trading_day,
                        df.symbol,
                        df."{col_high}" as orb_high,
                        df."{col_low}" as orb_low,
                        df."{col_dir}" as break_dir,
                        df."{col_ts}" as break_ts,
                        oo.pnl_r
                    FROM daily_features df
                    JOIN orb_outcomes oo
                        ON df.trading_day = oo.trading_day
                        AND df.symbol = oo.symbol
                        AND df.orb_minutes = oo.orb_minutes
                        AND oo.orb_label = '{session}'
                    WHERE df.orb_minutes = {orb_min}
                        AND df."{col_dir}" IS NOT NULL
                        AND df."{col_ts}" IS NOT NULL
                        AND oo.pnl_r IS NOT NULL
                        AND oo.entry_model IN ('E1', 'E2')
                        AND oo.confirm_bars = 1
                        AND df.symbol IN {inst_tuple}
                ),
                bars_after AS (
                    SELECT
                        t.trading_day,
                        t.symbol,
                        t.orb_high,
                        t.orb_low,
                        t.break_dir,
                        t.pnl_r,
                        b.close as bar_close,
                        ROW_NUMBER() OVER (
                            PARTITION BY t.trading_day, t.symbol
                            ORDER BY b.ts_utc
                        ) as bar_num
                    FROM trades t
                    JOIN bars_5m b
                        ON t.symbol = b.symbol
                        AND b.ts_utc > t.break_ts
                        AND b.ts_utc <= t.break_ts + INTERVAL '15 minutes'
                )
                SELECT
                    symbol,
                    trading_day,
                    orb_high,
                    orb_low,
                    break_dir,
                    pnl_r,
                    -- Check if ANY of the next 2 bars closed back inside ORB
                    MAX(CASE
                        WHEN break_dir = 'long' AND bar_close < orb_high THEN 1
                        WHEN break_dir = 'short' AND bar_close > orb_low THEN 1
                        ELSE 0
                    END) as pullback_flag
                FROM bars_after
                WHERE bar_num <= 2
                GROUP BY symbol, trading_day, orb_high, orb_low, break_dir, pnl_r
            """

            try:
                rows = con.execute(query).fetchall()
            except Exception as e:
                print(f"  SKIP {session} O{orb_min}: {e}")
                continue

            for row in rows:
                sym, td, orb_h, orb_l, bdir, pnl_r, pb_flag = row
                all_rows.append((sym, session, orb_min, pnl_r, pb_flag))

        print(f"  {session} done ({len(all_rows):,} total rows so far)")

    con.close()

    print(f"\nTotal trade-observations: {len(all_rows):,}")
    print()

    # === ANALYSIS ===
    all_pb = [r[3] for r in all_rows if r[4] == 1]
    all_npb = [r[3] for r in all_rows if r[4] == 0]

    print("=" * 85)
    print("POST-BREAK PULLBACK: bar+1/+2 closing inside ORB => failure signal?")
    print("=" * 85)
    print()
    print("DEFINITION: 'Pullback' = within 2 bars after break, >= 1 bar CLOSES")
    print("  back inside ORB (LONG: close < ORB_high, SHORT: close > ORB_low)")
    print("SCOPE: MGC/MNQ/MES/M2K, all sessions, E1+E2, CB1, O5/O15/O30")
    print()

    # === UNIVERSAL ===
    n_pb = len(all_pb)
    n_npb = len(all_npb)
    if n_pb > 30 and n_npb > 30:
        avg_pb = np.mean(all_pb)
        avg_npb = np.mean(all_npb)
        t_stat, p_val = run_t_test(all_npb, all_pb)
        pct = n_pb / (n_pb + n_npb) * 100
        print("=== UNIVERSAL POOLED ===")
        print(f"  Pullback:    N={n_pb:,} ({pct:.1f}%)  avgR = {avg_pb:+.4f}")
        print(f"  No-pullback: N={n_npb:,} ({100-pct:.1f}%)  avgR = {avg_npb:+.4f}")
        print(f"  Delta (stay - pullback): {avg_npb - avg_pb:+.4f}")
        print(f"  Welch t-test: t={t_stat:+.3f}, p={p_val:.6f}")
        if p_val < 0.05:
            print("  ** SIGNIFICANT at p<0.05 **")
        else:
            print(f"  NOT significant (p={p_val:.4f})")

    # === PER-INSTRUMENT ===
    print()
    print("=== PER-INSTRUMENT POOLED ===")
    for inst in instruments:
        pb = [r[3] for r in all_rows if r[0] == inst and r[4] == 1]
        npb = [r[3] for r in all_rows if r[0] == inst and r[4] == 0]
        if len(pb) < 30 or len(npb) < 30:
            print(f"  {inst}: N_pb={len(pb)}, N_stay={len(npb)} -- insufficient")
            continue
        avg_pb = np.mean(pb)
        avg_npb = np.mean(npb)
        t_stat, p_val = run_t_test(npb, pb)
        pct = len(pb) / (len(pb) + len(npb)) * 100
        sig = " *" if p_val < 0.05 else ""
        print(
            f"  {inst}: pb={len(pb):,} ({pct:.0f}%) avgR={avg_pb:+.3f}"
            f" | stay={len(npb):,} avgR={avg_npb:+.3f}"
            f" | delta={avg_npb-avg_pb:+.3f} | t={t_stat:+.2f} p={p_val:.4f}{sig}"
        )

    # === PER-SESSION ===
    print()
    print("=== PER-SESSION POOLED (all instruments, all apertures) ===")
    sess_results = []
    for session in sessions:
        pb = [r[3] for r in all_rows if r[1] == session and r[4] == 1]
        npb = [r[3] for r in all_rows if r[1] == session and r[4] == 0]
        if len(pb) < 20 or len(npb) < 20:
            continue
        avg_pb = np.mean(pb)
        avg_npb = np.mean(npb)
        t_stat, p_val = run_t_test(npb, pb)
        pct = len(pb) / (len(pb) + len(npb)) * 100
        sess_results.append((session, len(pb), len(npb), avg_pb, avg_npb, t_stat, p_val, pct))

    sess_results.sort(key=lambda x: x[6])
    for s, npb_n, nnpb, apb, anpb, t, p, pct in sess_results:
        sig = " *" if p < 0.05 else ""
        print(
            f"  {s:>18}: pb={npb_n:,} ({pct:.0f}%)"
            f" avgR={apb:+.3f} | stay={nnpb:,} avgR={anpb:+.3f}"
            f" | delta={anpb-apb:+.3f} | t={t:+.2f} p={p:.4f}{sig}"
        )

    # === DETAIL GRID ===
    print()
    print("=== TOP 20 INSTRUMENT x SESSION x APERTURE (by p-value) ===")
    detail_results = []
    for inst in instruments:
        for session in sessions:
            for orb_min in [5, 15, 30]:
                pb = [r[3] for r in all_rows if r[0] == inst and r[1] == session and r[2] == orb_min and r[4] == 1]
                npb = [r[3] for r in all_rows if r[0] == inst and r[1] == session and r[2] == orb_min and r[4] == 0]
                if len(pb) < 15 or len(npb) < 15:
                    continue
                avg_pb = np.mean(pb)
                avg_npb = np.mean(npb)
                t_stat, p_val = run_t_test(npb, pb)
                pct = len(pb) / (len(pb) + len(npb)) * 100
                detail_results.append(
                    (inst, session, orb_min, len(pb), len(npb), avg_pb, avg_npb, t_stat, p_val, pct)
                )

    detail_results.sort(key=lambda x: x[8])
    header = (
        f"{'Inst':>4} {'Session':>18} {'ORB':>3}"
        f" {'N_pb':>5} {'N_stay':>6} {'%pb':>4}"
        f" {'avgR_pb':>8} {'avgR_stay':>9} {'delta':>7}"
        f" {'t':>6} {'p':>8}"
    )
    print(header)
    print("-" * 100)
    for inst, sess, om, npb_n, nnpb, apb, anpb, t, p, pct in detail_results[:20]:
        sig_mark = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(
            f"{inst:>4} {sess:>18} {om:>3}m"
            f" {npb_n:>5} {nnpb:>6} {pct:>3.0f}%"
            f" {apb:>+8.3f} {anpb:>+9.3f} {anpb-apb:>+7.3f}"
            f" {t:>+6.2f} {p:>8.4f} {sig_mark}"
        )

    # === BH FDR ===
    print()
    n_tests = len(detail_results)
    print(f"=== BH FDR CORRECTION ({n_tests} tests at q=0.05) ===")
    if n_tests > 0:
        pvals = [r[8] for r in detail_results]
        n_raw = sum(1 for p in pvals if p < 0.05)
        n_surv = bh_fdr(pvals, q=0.05)
        print(f"  Raw p<0.05: {n_raw} / {n_tests}")
        print(f"  BH FDR survivors: {n_surv} / {n_tests}")

        if n_surv > 0:
            print()
            print("  BH FDR SURVIVORS:")
            sorted_detail = sorted(detail_results, key=lambda x: x[8])
            for rank, (inst, sess, om, npb_n, nnpb, apb, anpb, _t, p, _pct) in enumerate(
                sorted_detail, 1
            ):
                threshold = 0.05 * rank / n_tests
                if p <= threshold:
                    direction = "PULLBACK WORSE" if anpb > apb else "PULLBACK BETTER"
                    print(
                        f"    {inst} {sess} O{om}: pb={npb_n} avgR={apb:+.3f}"
                        f" | stay={nnpb} avgR={anpb:+.3f}"
                        f" | p={p:.4f} p_bh={threshold:.4f} | {direction}"
                    )

    # === WIN RATE ===
    print()
    print("=== WIN RATE COMPARISON (universal) ===")
    pb_wins = sum(1 for x in all_pb if x > 0)
    npb_wins = sum(1 for x in all_npb if x > 0)
    if n_pb > 0 and n_npb > 0:
        wr_pb = pb_wins / n_pb * 100
        wr_npb = npb_wins / n_npb * 100
        pb_losses = n_pb - pb_wins
        npb_losses = n_npb - npb_wins
        odds, fisher_p = stats.fisher_exact(
            [[pb_wins, pb_losses], [npb_wins, npb_losses]]
        )
        print(f"  Pullback WR:    {wr_pb:.1f}% ({pb_wins}/{n_pb})")
        print(f"  No-pullback WR: {wr_npb:.1f}% ({npb_wins}/{n_npb})")
        print(f"  Fisher exact: p={fisher_p:.6f}")

    # === VERDICT ===
    print()
    print("=" * 85)
    print("VERDICT")
    print("=" * 85)
    if n_pb > 30 and n_npb > 30:
        avg_pb = np.mean(all_pb)
        avg_npb = np.mean(all_npb)
        _, p_univ = run_t_test(all_npb, all_pb)
        n_surv = bh_fdr([r[8] for r in detail_results], q=0.05) if detail_results else 0
        if p_univ < 0.05 and avg_npb > avg_pb:
            print("SIGNAL DETECTED: Post-break pullback predicts worse outcomes universally.")
            if n_surv > 0:
                print(f"  {n_surv} BH FDR survivors -- check which instrument/session combos are actionable.")
            else:
                print("  BUT zero BH FDR survivors in the detail grid -- universal only.")
                print("  This means the effect is real but diffuse (not concentrated in specific combos).")
        elif p_univ < 0.05 and avg_npb < avg_pb:
            print("REVERSE SIGNAL: Pullback trades actually OUTPERFORM (mean-reversion after break).")
        else:
            print("NO SIGNAL: Post-break pullback does NOT predict trade outcome.")
            print("The ORB stop is already your protection. Adding a cancel rule")
            print("would reduce N without improving expectancy.")


if __name__ == "__main__":
    main()
