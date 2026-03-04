"""
Nested ORB Stacking Research: 5m vs 30m Alignment
===================================================
Phase 1 (Option A): Uses FINALIZED 30m ORB for classification.
This is descriptive/characterization only — the 30m ORB is not known at 5m
break time, so classification is look-ahead for early breaks.

Hypothesis:
  H0: 5m ORB break outcomes are independent of whether the break is contained
      within or exceeds the 30m ORB range.
  H1: Multi-timeframe alignment (5m break touching/exceeding 30m boundary)
      produces different outcomes than contained breaks.

Classification:
  For longs:  "aligned" if orb_5m_high >= orb_30m_high  (5m range touches/exceeds 30m on break side)
              "contained" otherwise
  For shorts: "aligned" if orb_5m_low <= orb_30m_low
              "contained" otherwise

Tests:
  1. Welch's t-test: aligned vs contained pnl_r
  2. One-sample t-test per group vs zero
  3. Spearman correlation: 30m/5m ORB size ratio vs pnl_r
  4. Direction sub-analysis: longs-only, shorts-only
  5. BH FDR correction across ALL tests
  6. Year-by-year stability for any p < 0.05

LOOK-AHEAD WARNING: Phase 1 uses finalized 30m ORB. A 5m break can occur at
minute 6 but the 30m ORB isn't final until minute 30. All results are
descriptive/characterization only.

Output: research/output/nested_orb_stacking_findings.md
"""

import sys
from pathlib import Path

# Windows line buffering
if sys.platform == "win32":
    sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH

DB_PATH = str(GOLD_DB_PATH)
OUTPUT = Path(__file__).resolve().parent / "output" / "nested_orb_stacking_findings.md"

# ── Config ────────────────────────────────────────────────────────────────────

ENTRY_MODELS = ["E1", "E2"]
RR_TARGETS = [2.0, 2.5, 3.0]
CONFIRM_BARS = [1, 2]  # CB1 and CB2
MIN_GROUP_N = 30        # minimum N per group for t-test
BH_Q = 0.10             # FDR threshold


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(arr):
    """Remove NaN from array."""
    a = np.array(arr, dtype=float)
    return a[~np.isnan(a)]


def welch_t(group_a, group_b):
    """Welch's t-test. Returns (t_stat, p_val, cohen_d)."""
    a, b = clean(group_a), clean(group_b)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan)
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.nanmean(a) - np.nanmean(b)) / pooled_std if pooled_std > 0 else 0
    return (t_stat, p_val, d)


def one_sample_t(arr):
    """One-sample t-test vs 0. Returns (t_stat, p_val)."""
    a = clean(arr)
    if len(a) < 5:
        return (np.nan, np.nan)
    t_stat, p_val = stats.ttest_1samp(a, 0)
    return (t_stat, p_val)


def spearman_corr(x, y):
    """Spearman rank correlation. Returns (rho, p_val)."""
    x_clean, y_clean = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
    x_clean, y_clean = x_clean[mask], y_clean[mask]
    if len(x_clean) < 10:
        return (np.nan, np.nan)
    rho, p_val = stats.spearmanr(x_clean, y_clean)
    return (rho, p_val)


def bh_fdr(p_values, q=0.10):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result


def fmt(p):
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def sample_label(n):
    if n < 30:
        return "INVALID"
    if n < 100:
        return "REGIME"
    if n < 200:
        return "PRELIMINARY"
    if n < 500:
        return "CORE"
    return "HIGH-CONFIDENCE"


# ── Report Collector ──────────────────────────────────────────────────────────

report_lines = []

def report(line=""):
    print(line)
    report_lines.append(line)


# ── Main Logic ────────────────────────────────────────────────────────────────

def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    report("=" * 72)
    report("NESTED ORB STACKING: 5m vs 30m ALIGNMENT (Phase 1 — Descriptive)")
    report("=" * 72)
    report("")
    report("LOOK-AHEAD WARNING: Classification uses finalized 30m ORB.")
    report("A 5m break can occur at minute 6, but 30m ORB is not known until")
    report("minute 30. ALL results are descriptive/characterization only.")
    report("")

    instruments = ACTIVE_ORB_INSTRUMENTS
    report(f"Instruments: {instruments}")
    report(f"Entry models: {ENTRY_MODELS}")
    report(f"RR targets: {RR_TARGETS}")
    report(f"Confirm bars: {CONFIRM_BARS}")
    report(f"Filter: G4 (orb size >= 4 points)")
    report(f"BH FDR q = {BH_Q}")
    report("")

    all_tests = []         # (label, p_val, test_dict) for BH pooling
    alignment_stats = []   # Summary stats per instrument/session

    for instrument in instruments:
        enabled_sessions = get_enabled_sessions(instrument)

        for session in enabled_sessions:
            report("-" * 72)
            report(f"  {instrument} | {session}")
            report("-" * 72)

            # ── Step 1: Self-join daily_features to get 5m and 30m ORB data ──
            # Join on (trading_day, symbol) across orb_minutes = 5 and 30
            sql = f"""
            SELECT
                d5.trading_day,
                d5.symbol,
                d5.orb_{session}_high    AS orb_5m_high,
                d5.orb_{session}_low     AS orb_5m_low,
                d5.orb_{session}_size    AS orb_5m_size,
                d5.orb_{session}_break_dir AS break_dir,
                d30.orb_{session}_high   AS orb_30m_high,
                d30.orb_{session}_low    AS orb_30m_low,
                d30.orb_{session}_size   AS orb_30m_size,
                EXTRACT(YEAR FROM d5.trading_day) AS year
            FROM daily_features d5
            JOIN daily_features d30
              ON d5.trading_day = d30.trading_day
              AND d5.symbol = d30.symbol
            WHERE d5.orb_minutes = 5
              AND d30.orb_minutes = 30
              AND d5.symbol = '{instrument}'
              AND d5.orb_{session}_high IS NOT NULL
              AND d5.orb_{session}_low IS NOT NULL
              AND d5.orb_{session}_break_dir IS NOT NULL
              AND d5.orb_{session}_size IS NOT NULL
              AND d30.orb_{session}_high IS NOT NULL
              AND d30.orb_{session}_low IS NOT NULL
              AND d30.orb_{session}_size IS NOT NULL
              AND d5.orb_{session}_size >= 4
            ORDER BY d5.trading_day
            """

            try:
                df = con.execute(sql).fetchdf()
            except Exception as e:
                report(f"  ERROR querying: {e}")
                continue

            if len(df) < MIN_GROUP_N:
                report(f"  Only {len(df)} days with both 5m+30m ORB data (G4+). SKIP.")
                report("")
                continue

            # ── Step 2: Classify aligned vs contained ──
            # For longs: aligned if 5m high >= 30m high (5m range touches/exceeds 30m on break side)
            # For shorts: aligned if 5m low <= 30m low
            conditions_long = (df["break_dir"] == "long") & (df["orb_5m_high"] >= df["orb_30m_high"])
            conditions_short = (df["break_dir"] == "short") & (df["orb_5m_low"] <= df["orb_30m_low"])
            df["alignment"] = "contained"
            df.loc[conditions_long | conditions_short, "alignment"] = "aligned"

            n_aligned = (df["alignment"] == "aligned").sum()
            n_contained = (df["alignment"] == "contained").sum()
            n_total = len(df)

            # Edge case: how often is 5m ORB == 30m ORB (no expansion)?
            same_high = (df["orb_5m_high"] == df["orb_30m_high"]).sum()
            same_low = (df["orb_5m_low"] == df["orb_30m_low"]).sum()
            same_both = ((df["orb_5m_high"] == df["orb_30m_high"]) &
                         (df["orb_5m_low"] == df["orb_30m_low"])).sum()

            report(f"  Days (G4+): {n_total}")
            report(f"  Aligned: {n_aligned} ({100*n_aligned/n_total:.1f}%)")
            report(f"  Contained: {n_contained} ({100*n_contained/n_total:.1f}%)")
            report(f"  5m==30m (high): {same_high} ({100*same_high/n_total:.1f}%)")
            report(f"  5m==30m (low):  {same_low} ({100*same_low/n_total:.1f}%)")
            report(f"  5m==30m (both): {same_both} ({100*same_both/n_total:.1f}%)")

            # ORB size ratio stats
            df["size_ratio_30_5"] = df["orb_30m_size"] / df["orb_5m_size"]
            ratio_med = df["size_ratio_30_5"].median()
            ratio_mean = df["size_ratio_30_5"].mean()
            report(f"  30m/5m size ratio: mean={ratio_mean:.2f}, median={ratio_med:.2f}")

            alignment_stats.append({
                "instrument": instrument,
                "session": session,
                "n_total": n_total,
                "n_aligned": n_aligned,
                "n_contained": n_contained,
                "pct_aligned": 100 * n_aligned / n_total if n_total > 0 else 0,
                "same_both_pct": 100 * same_both / n_total if n_total > 0 else 0,
                "ratio_mean": ratio_mean,
                "ratio_med": ratio_med,
            })

            # ── Step 3: Join with orb_outcomes (orb_minutes=5) ──
            for em in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    for cb in CONFIRM_BARS:
                        outcome_sql = f"""
                        SELECT trading_day, pnl_r, outcome
                        FROM orb_outcomes
                        WHERE symbol = '{instrument}'
                          AND orb_label = '{session}'
                          AND orb_minutes = 5
                          AND entry_model = '{em}'
                          AND rr_target = {rr}
                          AND confirm_bars = {cb}
                          AND outcome IN ('win', 'loss')
                          AND pnl_r IS NOT NULL
                        """
                        df_out = con.execute(outcome_sql).fetchdf()
                        if len(df_out) == 0:
                            continue

                        merged = df.merge(df_out, on="trading_day", how="inner")
                        if len(merged) < MIN_GROUP_N:
                            continue

                        aligned_r = clean(merged[merged["alignment"] == "aligned"]["pnl_r"].values)
                        contained_r = clean(merged[merged["alignment"] == "contained"]["pnl_r"].values)

                        combo_tag = f"{instrument}|{session}|{em}|RR{rr}|CB{cb}"

                        # ── Test A: Welch's t-test aligned vs contained ──
                        if len(aligned_r) >= MIN_GROUP_N and len(contained_r) >= MIN_GROUP_N:
                            t_stat, p_val, d = welch_t(aligned_r, contained_r)
                            if not np.isnan(p_val):
                                all_tests.append({
                                    "label": f"WELCH|{combo_tag}",
                                    "p": p_val,
                                    "instrument": instrument,
                                    "session": session,
                                    "em": em,
                                    "rr": rr,
                                    "cb": cb,
                                    "test_type": "welch",
                                    "n_aligned": len(aligned_r),
                                    "n_contained": len(contained_r),
                                    "avg_aligned": np.nanmean(aligned_r),
                                    "avg_contained": np.nanmean(contained_r),
                                    "delta": np.nanmean(aligned_r) - np.nanmean(contained_r),
                                    "d": d,
                                    "t_stat": t_stat,
                                })

                        # ── Test B: One-sample t-test aligned vs zero ──
                        if len(aligned_r) >= MIN_GROUP_N:
                            t1, p1 = one_sample_t(aligned_r)
                            if not np.isnan(p1):
                                all_tests.append({
                                    "label": f"1SAMP_ALN|{combo_tag}",
                                    "p": p1,
                                    "instrument": instrument,
                                    "session": session,
                                    "em": em,
                                    "rr": rr,
                                    "cb": cb,
                                    "test_type": "one_sample_aligned",
                                    "n_aligned": len(aligned_r),
                                    "n_contained": 0,
                                    "avg_aligned": np.nanmean(aligned_r),
                                    "avg_contained": np.nan,
                                    "delta": np.nanmean(aligned_r),
                                    "d": np.nan,
                                    "t_stat": t1,
                                })

                        # ── Test C: One-sample t-test contained vs zero ──
                        if len(contained_r) >= MIN_GROUP_N:
                            t2, p2 = one_sample_t(contained_r)
                            if not np.isnan(p2):
                                all_tests.append({
                                    "label": f"1SAMP_CON|{combo_tag}",
                                    "p": p2,
                                    "instrument": instrument,
                                    "session": session,
                                    "em": em,
                                    "rr": rr,
                                    "cb": cb,
                                    "test_type": "one_sample_contained",
                                    "n_aligned": 0,
                                    "n_contained": len(contained_r),
                                    "avg_aligned": np.nan,
                                    "avg_contained": np.nanmean(contained_r),
                                    "delta": np.nanmean(contained_r),
                                    "d": np.nan,
                                    "t_stat": t2,
                                })

                        # ── Test D: Spearman correlation 30m/5m size ratio vs pnl_r ──
                        valid_ratio = merged.dropna(subset=["size_ratio_30_5", "pnl_r"])
                        if len(valid_ratio) >= MIN_GROUP_N:
                            rho, p_sp = spearman_corr(
                                valid_ratio["size_ratio_30_5"].values,
                                valid_ratio["pnl_r"].values,
                            )
                            if not np.isnan(p_sp):
                                all_tests.append({
                                    "label": f"SPEARMAN|{combo_tag}",
                                    "p": p_sp,
                                    "instrument": instrument,
                                    "session": session,
                                    "em": em,
                                    "rr": rr,
                                    "cb": cb,
                                    "test_type": "spearman",
                                    "n_aligned": len(valid_ratio),
                                    "n_contained": 0,
                                    "avg_aligned": rho,  # reuse field for rho
                                    "avg_contained": np.nan,
                                    "delta": rho,
                                    "d": np.nan,
                                    "t_stat": np.nan,
                                })

            # ── Direction Sub-Analysis (longs-only, shorts-only) ──
            # Use representative combo: E2 RR2.0 CB1
            rep_sql = f"""
            SELECT trading_day, pnl_r, outcome
            FROM orb_outcomes
            WHERE symbol = '{instrument}'
              AND orb_label = '{session}'
              AND orb_minutes = 5
              AND entry_model = 'E2'
              AND rr_target = 2.0
              AND confirm_bars = 1
              AND outcome IN ('win', 'loss')
              AND pnl_r IS NOT NULL
            """
            df_rep = con.execute(rep_sql).fetchdf()
            if len(df_rep) > 0:
                merged_rep = df.merge(df_rep, on="trading_day", how="inner")
                for direction in ["long", "short"]:
                    dir_sub = merged_rep[merged_rep["break_dir"] == direction]
                    dir_aln = clean(dir_sub[dir_sub["alignment"] == "aligned"]["pnl_r"].values)
                    dir_con = clean(dir_sub[dir_sub["alignment"] == "contained"]["pnl_r"].values)

                    if len(dir_aln) >= MIN_GROUP_N and len(dir_con) >= MIN_GROUP_N:
                        t_d, p_d, d_d = welch_t(dir_aln, dir_con)
                        combo_dir_tag = f"{instrument}|{session}|E2|RR2.0|CB1|{direction.upper()}"
                        if not np.isnan(p_d):
                            all_tests.append({
                                "label": f"WELCH_DIR|{combo_dir_tag}",
                                "p": p_d,
                                "instrument": instrument,
                                "session": session,
                                "em": "E2",
                                "rr": 2.0,
                                "cb": 1,
                                "test_type": f"welch_dir_{direction}",
                                "n_aligned": len(dir_aln),
                                "n_contained": len(dir_con),
                                "avg_aligned": np.nanmean(dir_aln),
                                "avg_contained": np.nanmean(dir_con),
                                "delta": np.nanmean(dir_aln) - np.nanmean(dir_con),
                                "d": d_d,
                                "t_stat": t_d,
                            })

            report("")

    # ══════════════════════════════════════════════════════════════════════════
    # ALIGNMENT SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    report("=" * 72)
    report("ALIGNMENT SUMMARY (G4+ days with both 5m and 30m ORB)")
    report("=" * 72)
    report(f"  {'Instr':<6} {'Session':<18} {'Total':>6} {'Aligned':>8} {'%Aln':>6} "
           f"{'5m==30m':>8} {'Ratio':>6}")
    report("  " + "-" * 68)
    for s in alignment_stats:
        report(f"  {s['instrument']:<6} {s['session']:<18} {s['n_total']:>6} "
               f"{s['n_aligned']:>8} {s['pct_aligned']:>5.1f}% "
               f"{s['same_both_pct']:>7.1f}% {s['ratio_med']:>6.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # RAW RESULTS (top 30 by p-value)
    # ══════════════════════════════════════════════════════════════════════════
    report("")
    report("=" * 72)
    report(f"RAW RESULTS — TOP 30 (by p-value, {len(all_tests)} total tests)")
    report("=" * 72)

    all_tests.sort(key=lambda x: x["p"] if not np.isnan(x["p"]) else 999)

    report(f"  {'Type':<12} {'Label':<45} {'N_a':>5} {'N_c':>5} "
           f"{'avg_a':>7} {'avg_c':>7} {'delta':>7} {'p':>8}")
    report("  " + "-" * 100)

    for t in all_tests[:30]:
        n_a = t.get("n_aligned", 0)
        n_c = t.get("n_contained", 0)
        avg_a = t.get("avg_aligned", np.nan)
        avg_c = t.get("avg_contained", np.nan)
        avg_a_s = f"{avg_a:>+7.3f}" if not np.isnan(avg_a) else "   N/A "
        avg_c_s = f"{avg_c:>+7.3f}" if not np.isnan(avg_c) else "   N/A "
        delta_s = f"{t['delta']:>+7.3f}" if not np.isnan(t["delta"]) else "   N/A "
        ttype = t["test_type"][:12]
        label_short = t["label"][:45]
        report(f"  {ttype:<12} {label_short:<45} {n_a:>5} {n_c:>5} "
               f"{avg_a_s} {avg_c_s} {delta_s} {fmt(t['p']):>8}")

    # ══════════════════════════════════════════════════════════════════════════
    # BH FDR CORRECTION
    # ══════════════════════════════════════════════════════════════════════════
    report("")
    report("=" * 72)
    report(f"BH FDR CORRECTION (q={BH_Q}, {len(all_tests)} tests)")
    report("=" * 72)

    valid_tests = [t for t in all_tests if not np.isnan(t["p"])]
    survivors = []
    if len(valid_tests) == 0:
        report("  No valid tests to correct.")
    else:
        p_vals = [t["p"] for t in valid_tests]
        labels = [t["label"] for t in valid_tests]
        adjusted = bh_fdr(p_vals, q=BH_Q)

        survivors = [(l, p, adj, t) for l, p, adj, t
                     in zip(labels, p_vals, adjusted, valid_tests)
                     if adj < BH_Q]

        report(f"  Total tests:  {len(valid_tests)}")
        report(f"  BH survivors (q={BH_Q}): {len(survivors)}")

        # Separate survivors by type
        welch_survivors = [(l, p, a, t) for l, p, a, t in survivors
                           if t["test_type"] == "welch"]
        dir_survivors = [(l, p, a, t) for l, p, a, t in survivors
                         if t["test_type"].startswith("welch_dir")]
        onesamp_survivors = [(l, p, a, t) for l, p, a, t in survivors
                             if t["test_type"].startswith("one_sample")]
        spearman_survivors = [(l, p, a, t) for l, p, a, t in survivors
                              if t["test_type"] == "spearman"]

        report(f"    Welch (aligned vs contained): {len(welch_survivors)}")
        report(f"    Welch direction:              {len(dir_survivors)}")
        report(f"    One-sample vs zero:           {len(onesamp_survivors)}")
        report(f"    Spearman correlation:         {len(spearman_survivors)}")

        if welch_survivors or dir_survivors:
            report("")
            report("  --- BH SURVIVORS: Welch tests (PRIMARY — aligned vs contained) ---")
            report(f"  {'Label':<50} {'raw_p':>8} {'p_bh':>8} {'delta':>7}")
            report("  " + "-" * 80)
            for label, raw_p, adj_p, t in sorted(welch_survivors + dir_survivors,
                                                  key=lambda x: x[2]):
                delta_s = f"{t['delta']:>+7.3f}" if not np.isnan(t["delta"]) else "   N/A "
                report(f"  {label:<50} {fmt(raw_p):>8} {fmt(adj_p):>8} {delta_s}")

        if spearman_survivors:
            report("")
            report(f"  --- BH SURVIVORS: Spearman (top 20 of {len(spearman_survivors)}) ---")
            report(f"  {'Label':<50} {'raw_p':>8} {'p_bh':>8} {'rho':>7}")
            report("  " + "-" * 80)
            for label, raw_p, adj_p, t in sorted(spearman_survivors,
                                                  key=lambda x: x[2])[:20]:
                rho_s = f"{t['delta']:>+7.3f}" if not np.isnan(t["delta"]) else "   N/A "
                report(f"  {label:<50} {fmt(raw_p):>8} {fmt(adj_p):>8} {rho_s}")

        if onesamp_survivors:
            report("")
            report(f"  --- BH SURVIVORS: One-sample vs zero (top 20 of {len(onesamp_survivors)}) ---")
            report(f"  {'Label':<50} {'raw_p':>8} {'p_bh':>8} {'avgR':>7}")
            report("  " + "-" * 80)
            for label, raw_p, adj_p, t in sorted(onesamp_survivors,
                                                  key=lambda x: x[2])[:20]:
                delta_s = f"{t['delta']:>+7.3f}" if not np.isnan(t["delta"]) else "   N/A "
                report(f"  {label:<50} {fmt(raw_p):>8} {fmt(adj_p):>8} {delta_s}")

        if not survivors:
            report("  ** NO TESTS SURVIVED BH FDR **")

        # ── Year-by-year stability for any raw p < 0.05 ──
        report("")
        report("=" * 72)
        report("YEAR-BY-YEAR STABILITY (raw p < 0.05 Welch tests only)")
        report("=" * 72)

        raw_notable = [t for t in valid_tests
                       if t["p"] < 0.05 and t["test_type"] == "welch"]

        if not raw_notable:
            report("  No Welch tests with raw p < 0.05.")
        else:
            for t in raw_notable:
                inst = t["instrument"]
                sess = t["session"]
                em = t["em"]
                rr = t["rr"]
                cb = t["cb"]

                # Re-query year-by-year
                yy_sql = f"""
                SELECT o.trading_day,
                       EXTRACT(YEAR FROM o.trading_day) AS year,
                       o.pnl_r
                FROM orb_outcomes o
                WHERE o.symbol = '{inst}'
                  AND o.orb_label = '{sess}'
                  AND o.orb_minutes = 5
                  AND o.entry_model = '{em}'
                  AND o.rr_target = {rr}
                  AND o.confirm_bars = {cb}
                  AND o.outcome IN ('win', 'loss')
                  AND o.pnl_r IS NOT NULL
                """
                df_yy_out = con.execute(yy_sql).fetchdf()

                # Need to re-classify each row
                df_yy_class = con.execute(f"""
                SELECT
                    d5.trading_day,
                    d5.orb_{sess}_high    AS orb_5m_high,
                    d5.orb_{sess}_low     AS orb_5m_low,
                    d5.orb_{sess}_break_dir AS break_dir,
                    d30.orb_{sess}_high   AS orb_30m_high,
                    d30.orb_{sess}_low    AS orb_30m_low
                FROM daily_features d5
                JOIN daily_features d30
                  ON d5.trading_day = d30.trading_day AND d5.symbol = d30.symbol
                WHERE d5.orb_minutes = 5 AND d30.orb_minutes = 30
                  AND d5.symbol = '{inst}'
                  AND d5.orb_{sess}_high IS NOT NULL
                  AND d5.orb_{sess}_break_dir IS NOT NULL
                  AND d5.orb_{sess}_size >= 4
                  AND d30.orb_{sess}_high IS NOT NULL
                """).fetchdf()

                cond_l = (df_yy_class["break_dir"] == "long") & (df_yy_class["orb_5m_high"] >= df_yy_class["orb_30m_high"])
                cond_s = (df_yy_class["break_dir"] == "short") & (df_yy_class["orb_5m_low"] <= df_yy_class["orb_30m_low"])
                df_yy_class["alignment"] = "contained"
                df_yy_class.loc[cond_l | cond_s, "alignment"] = "aligned"

                merged_yy = df_yy_out.merge(
                    df_yy_class[["trading_day", "alignment"]],
                    on="trading_day", how="inner"
                )

                report(f"\n  {t['label']} (raw p={fmt(t['p'])}, delta={t['delta']:+.3f}):")

                years = sorted(merged_yy["year"].unique())
                pos_years = 0
                total_years = 0
                for y in years:
                    yr_data = merged_yy[merged_yy["year"] == y]
                    yr_aln = clean(yr_data[yr_data["alignment"] == "aligned"]["pnl_r"].values)
                    yr_con = clean(yr_data[yr_data["alignment"] == "contained"]["pnl_r"].values)
                    if len(yr_aln) >= 5 and len(yr_con) >= 5:
                        yr_delta = np.nanmean(yr_aln) - np.nanmean(yr_con)
                        total_years += 1
                        same_sign = (yr_delta > 0) == (t["delta"] > 0)
                        if same_sign:
                            pos_years += 1
                        marker = "+" if same_sign else "-"
                        report(f"    {int(y)}: N_a={len(yr_aln):>4}, N_c={len(yr_con):>4}, "
                               f"avg_a={np.nanmean(yr_aln):>+7.3f}, avg_c={np.nanmean(yr_con):>+7.3f}, "
                               f"delta={yr_delta:>+7.3f} [{marker}]")
                    else:
                        report(f"    {int(y)}: N_a={len(yr_aln):>4}, N_c={len(yr_con):>4} "
                               f"(too few for split)")
                if total_years > 0:
                    report(f"    Years consistent: {pos_years}/{total_years} "
                           f"({100*pos_years/total_years:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # PROXY CHECK: Is alignment just an ORB size proxy?
    # ══════════════════════════════════════════════════════════════════════════
    report("")
    report("=" * 72)
    report("PROXY CHECK: Is 'aligned' just a proxy for larger 5m ORB?")
    report("=" * 72)

    for s in alignment_stats:
        inst = s["instrument"]
        sess = s["session"]
        if s["n_total"] < MIN_GROUP_N:
            continue

        proxy_sql = f"""
        SELECT
            d5.trading_day,
            d5.orb_{sess}_size AS orb_5m_size,
            d5.orb_{sess}_high AS orb_5m_high,
            d5.orb_{sess}_low AS orb_5m_low,
            d5.orb_{sess}_break_dir AS break_dir,
            d30.orb_{sess}_high AS orb_30m_high,
            d30.orb_{sess}_low AS orb_30m_low
        FROM daily_features d5
        JOIN daily_features d30
          ON d5.trading_day = d30.trading_day AND d5.symbol = d30.symbol
        WHERE d5.orb_minutes = 5 AND d30.orb_minutes = 30
          AND d5.symbol = '{inst}'
          AND d5.orb_{sess}_high IS NOT NULL
          AND d5.orb_{sess}_break_dir IS NOT NULL
          AND d5.orb_{sess}_size >= 4
          AND d30.orb_{sess}_high IS NOT NULL
        """
        df_proxy = con.execute(proxy_sql).fetchdf()

        cond_l = (df_proxy["break_dir"] == "long") & (df_proxy["orb_5m_high"] >= df_proxy["orb_30m_high"])
        cond_s = (df_proxy["break_dir"] == "short") & (df_proxy["orb_5m_low"] <= df_proxy["orb_30m_low"])
        df_proxy["alignment"] = "contained"
        df_proxy.loc[cond_l | cond_s, "alignment"] = "aligned"

        aln_sizes = clean(df_proxy[df_proxy["alignment"] == "aligned"]["orb_5m_size"].values)
        con_sizes = clean(df_proxy[df_proxy["alignment"] == "contained"]["orb_5m_size"].values)
        if len(aln_sizes) >= 5 and len(con_sizes) >= 5:
            report(f"  {inst} {sess}: aligned avg_5m_size={np.mean(aln_sizes):.2f}, "
                   f"contained avg_5m_size={np.mean(con_sizes):.2f}")
        else:
            report(f"  {inst} {sess}: insufficient data for proxy check")

    # ══════════════════════════════════════════════════════════════════════════
    # SPEARMAN RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    report("")
    report("=" * 72)
    report("SPEARMAN CORRELATION: 30m/5m size ratio vs pnl_r (top 15)")
    report("=" * 72)

    spearman_tests = [t for t in all_tests if t["test_type"] == "spearman"]
    spearman_tests.sort(key=lambda x: x["p"] if not np.isnan(x["p"]) else 999)

    if spearman_tests:
        report(f"  {'Label':<45} {'N':>5} {'rho':>7} {'p':>8}")
        report("  " + "-" * 70)
        for t in spearman_tests[:15]:
            report(f"  {t['label']:<45} {t['n_aligned']:>5} "
                   f"{t['avg_aligned']:>+7.3f} {fmt(t['p']):>8}")
    else:
        report("  No Spearman tests computed.")

    # ══════════════════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════════════════
    report("")
    report("=" * 72)
    report("HONEST SUMMARY")
    report("=" * 72)

    n_bh_survivors = len(survivors)
    if n_bh_survivors > 0:
        report(f"  {n_bh_survivors} test(s) survived BH FDR at q={BH_Q}.")
        report("  Check year-by-year consistency before drawing conclusions.")
    else:
        report("  ** NO TESTS SURVIVED BH FDR **")
        report("  5m/30m alignment does NOT predict outcomes after multiple")
        report("  comparison correction. The null hypothesis stands.")

    report("")
    report("  CAVEATS:")
    report("  1. Phase 1 uses FINALIZED 30m ORB — look-ahead for early breaks")
    report("  2. G4 filter only (expand later if signal found)")
    report("  3. MNQ has only ~2 years of data — any MNQ finding is REGIME at best")
    report("  4. Alignment classification may proxy for ORB size (check proxy table)")
    report("  5. Even if BH survivors exist, need Phase 2 (bars_1m reconstruction)")
    report("     to confirm with honest real-time 30m ORB estimate")

    # Write findings file
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("# Nested ORB Stacking Findings (Phase 1 — Descriptive)\n\n")
        f.write("**Date:** 2026-03-01\n")
        f.write("**Script:** research/research_nested_orb_stacking.py\n\n")
        f.write("**LOOK-AHEAD WARNING:** Classification uses finalized 30m ORB. ")
        f.write("All results are descriptive/characterization only.\n\n")
        for line in report_lines:
            f.write(line + "\n")

    report(f"\nFindings written to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    main()
