"""
Session correlation and allocation analysis.

Tests: Are 0900 and CME_OPEN 48% correlated (redundant)?
Are 1000 and 1100 anti-correlated? What's the optimal session portfolio?

Computes pairwise session return correlations, identifies redundancies,
and tests portfolio overlay stability.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
from scipy import stats
import numpy as np
from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
OUTPUT = os.path.join(os.path.dirname(__file__), "output", "session_correlation_findings.md")


def run():
    con = duckdb.connect(str(DB), read_only=True)
    lines = []
    lines.append("# Session Correlation & Allocation Analysis")
    lines.append(f"**Date:** 2026-02-21")
    lines.append(f"**Script:** research/research_session_correlation.py\n")

    # Get all session labels per instrument
    sessions_by_sym = {}
    for sym in ["MGC", "MES", "MNQ"]:
        sess_rows = con.execute("""
            SELECT DISTINCT orb_label FROM orb_outcomes
            WHERE symbol = ? AND entry_model = 'E0' AND rr_target = 2.0
              AND pnl_r IS NOT NULL
            ORDER BY orb_label
        """, [sym]).fetchall()
        sessions_by_sym[sym] = [r[0] for r in sess_rows]
        lines.append(f"  {sym} sessions: {', '.join(sessions_by_sym[sym])}")
    lines.append("")

    # ── Pairwise correlations per instrument ────────────────────
    for sym in ["MGC", "MES", "MNQ"]:
        for em in ["E0"]:
            lines.append("=" * 70)
            lines.append(f"{sym} {em} RR2.0 -- SESSION PAIRWISE CORRELATIONS")
            lines.append("=" * 70 + "\n")

            sessions = sessions_by_sym[sym]

            # Build daily return vectors per session
            session_returns = {}
            for sess in sessions:
                rows = con.execute("""
                    SELECT o.trading_day, o.pnl_r
                    FROM orb_outcomes o
                    WHERE o.symbol = ?
                      AND o.orb_label = ?
                      AND o.entry_model = ?
                      AND o.rr_target = 2.0
                      AND o.pnl_r IS NOT NULL
                    ORDER BY o.trading_day
                """, [sym, sess, em]).fetchall()
                session_returns[sess] = {str(r[0]): r[1] for r in rows}

            # Compute pairwise correlations on overlapping dates
            header = f"  {'':>16}"
            for s in sessions:
                header += f" {s:>10}"
            lines.append(header)

            corr_matrix = {}
            for i, s1 in enumerate(sessions):
                row_str = f"  {s1:>16}"
                for j, s2 in enumerate(sessions):
                    common_dates = set(session_returns[s1].keys()) & set(session_returns[s2].keys())
                    if len(common_dates) < 30:
                        row_str += f" {'n/a':>10}"
                        continue
                    r1 = [session_returns[s1][d] for d in sorted(common_dates)]
                    r2 = [session_returns[s2][d] for d in sorted(common_dates)]
                    r, p = stats.pearsonr(r1, r2)
                    corr_matrix[(s1, s2)] = (r, p, len(common_dates))
                    row_str += f" {r:>10.3f}"
                lines.append(row_str)
            lines.append("")

            # Top redundancies and diversifiers
            pairs = []
            for (s1, s2), (r, p, n) in corr_matrix.items():
                if s1 < s2:
                    pairs.append((s1, s2, r, p, n))
            pairs.sort(key=lambda x: -abs(x[2]))

            lines.append(f"  TOP CORRELATED PAIRS (redundancy risk):")
            for s1, s2, r, p, n in pairs[:5]:
                label = "REDUNDANT" if r > 0.3 else "MODERATE" if r > 0.1 else "INDEPENDENT"
                lines.append(f"    {s1:>12} x {s2:<12}: r={r:+.3f}, p={p:.4f}, N={n}, {label}")
            lines.append("")

            lines.append(f"  MOST DIVERSIFYING PAIRS (negative/near-zero correlation):")
            pairs_by_div = sorted(pairs, key=lambda x: x[2])
            for s1, s2, r, p, n in pairs_by_div[:5]:
                label = "ANTI-CORR" if r < -0.05 else "INDEPENDENT" if r < 0.1 else "WEAK"
                lines.append(f"    {s1:>12} x {s2:<12}: r={r:+.3f}, p={p:.4f}, N={n}, {label}")
            lines.append("")

    # ── Portfolio overlay: session combo stability ──────────────
    lines.append("=" * 70)
    lines.append("PORTFOLIO OVERLAY: SESSION COMBO STABILITY")
    lines.append("=" * 70 + "\n")

    for sym in ["MGC", "MES", "MNQ"]:
        lines.append(f"  --- {sym} E0 RR2.0 ---")
        sessions = sessions_by_sym[sym]

        # Individual session stats
        sess_stats = {}
        for sess in sessions:
            rows = con.execute("""
                SELECT o.pnl_r, EXTRACT(YEAR FROM o.trading_day) as yr
                FROM orb_outcomes o
                WHERE o.symbol = ?
                  AND o.orb_label = ?
                  AND o.entry_model = 'E0'
                  AND o.rr_target = 2.0
                  AND o.pnl_r IS NOT NULL
                ORDER BY o.trading_day
            """, [sym, sess]).fetchall()

            if len(rows) < 30:
                continue

            pnl = [r[0] for r in rows]
            avg = np.mean(pnl)
            std = np.std(pnl, ddof=1)
            sharpe = avg / std * np.sqrt(252) if std > 0 else 0
            wr = np.mean([1 if x > 0 else 0 for x in pnl])
            n = len(pnl)

            # Year-by-year
            years = sorted(set(int(r[1]) for r in rows))
            yr_sharpes = []
            for yr in years:
                yr_pnl = [r[0] for r in rows if int(r[1]) == yr]
                if len(yr_pnl) >= 20:
                    yr_avg = np.mean(yr_pnl)
                    yr_std = np.std(yr_pnl, ddof=1)
                    yr_sharpe = yr_avg / yr_std * np.sqrt(252) if yr_std > 0 else 0
                    yr_sharpes.append((yr, yr_sharpe, len(yr_pnl)))

            sess_stats[sess] = {
                "avg": avg, "std": std, "sharpe": sharpe,
                "wr": wr, "n": n, "yr_sharpes": yr_sharpes
            }

            yr_str = ", ".join(f"{yr}:{s:.2f}" for yr, s, _ in yr_sharpes)
            lines.append(f"    {sess:>16}: N={n:>5}, avgR={avg:+.3f}, WR={wr:.1%}, "
                         f"Sharpe={sharpe:.2f}  [{yr_str}]")
        lines.append("")

        # Test selected combos
        if len(sess_stats) >= 2:
            lines.append(f"  COMBO ANALYSIS (equal-weight daily average):")
            # Generate interesting combos
            combos = []
            sessions_with_data = list(sess_stats.keys())

            # All pairs
            for i, s1 in enumerate(sessions_with_data):
                for s2 in sessions_with_data[i+1:]:
                    combos.append((s1, s2))

            # Key triples from claim
            for triple in [("0900", "1100", "CME_OPEN"),
                           ("0900", "1000", "1100"),
                           ("1000", "1100", "1800"),
                           ("0900", "1000", "CME_OPEN")]:
                if all(s in sessions_with_data for s in triple):
                    combos.append(triple)

            for combo in combos:
                # Get overlapping dates
                sess_data = {}
                for sess in combo:
                    rows = con.execute("""
                        SELECT o.trading_day, o.pnl_r
                        FROM orb_outcomes o
                        WHERE o.symbol = ?
                          AND o.orb_label = ?
                          AND o.entry_model = 'E0'
                          AND o.rr_target = 2.0
                          AND o.pnl_r IS NOT NULL
                    """, [sym, sess]).fetchall()
                    sess_data[sess] = {str(r[0]): r[1] for r in rows}

                common = set.intersection(*[set(d.keys()) for d in sess_data.values()])
                if len(common) < 30:
                    continue

                # Equal-weight daily average
                daily_avg = []
                daily_years = []
                for d in sorted(common):
                    avg_r = np.mean([sess_data[s][d] for s in combo])
                    daily_avg.append(avg_r)
                    daily_years.append(int(d[:4]))

                combo_avg = np.mean(daily_avg)
                combo_std = np.std(daily_avg, ddof=1)
                combo_sharpe = combo_avg / combo_std * np.sqrt(252) if combo_std > 0 else 0
                combo_wr = np.mean([1 if x > 0 else 0 for x in daily_avg])

                # Year-by-year Sharpe
                yr_sharpes = []
                for yr in sorted(set(daily_years)):
                    yr_data = [daily_avg[i] for i in range(len(daily_avg)) if daily_years[i] == yr]
                    if len(yr_data) >= 20:
                        yr_avg = np.mean(yr_data)
                        yr_std = np.std(yr_data, ddof=1)
                        yr_s = yr_avg / yr_std * np.sqrt(252) if yr_std > 0 else 0
                        yr_sharpes.append((yr, yr_s))

                combo_label = "+".join(combo)
                yr_str = ", ".join(f"{yr}:{s:.2f}" for yr, s in yr_sharpes)
                lines.append(f"    {combo_label:>35}: N={len(common):>5}, avgR={combo_avg:+.3f}, "
                             f"WR={combo_wr:.1%}, Sharpe={combo_sharpe:.2f}  [{yr_str}]")

            lines.append("")

    # ── Specific claim checks ──────────────────────────────────
    lines.append("=" * 70)
    lines.append("CLAIM CHECKS")
    lines.append("=" * 70 + "\n")

    # MNQ 0900 vs CME_OPEN correlation
    for sym in ["MNQ", "MGC", "MES"]:
        for s1, s2 in [("0900", "CME_OPEN"), ("1000", "1100"), ("0900", "1100")]:
            key = (s1, s2)
            # Re-query if needed
            r1_data = con.execute("""
                SELECT o.trading_day, o.pnl_r FROM orb_outcomes o
                WHERE o.symbol=? AND o.orb_label=? AND o.entry_model='E0'
                  AND o.rr_target=2.0 AND o.pnl_r IS NOT NULL
            """, [sym, s1]).fetchall()
            r2_data = con.execute("""
                SELECT o.trading_day, o.pnl_r FROM orb_outcomes o
                WHERE o.symbol=? AND o.orb_label=? AND o.entry_model='E0'
                  AND o.rr_target=2.0 AND o.pnl_r IS NOT NULL
            """, [sym, s2]).fetchall()

            d1 = {str(r[0]): r[1] for r in r1_data}
            d2 = {str(r[0]): r[1] for r in r2_data}
            common = sorted(set(d1.keys()) & set(d2.keys()))

            if len(common) < 30:
                continue

            v1 = [d1[d] for d in common]
            v2 = [d2[d] for d in common]
            r, p = stats.pearsonr(v1, v2)
            lines.append(f"  {sym} {s1} vs {s2}: r={r:+.3f}, p={p:.4f}, N={len(common)}")

    lines.append("")

    # ── Verdict ─────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("HONEST SUMMARY")
    lines.append("=" * 70 + "\n")

    lines.append("### Claims to check:")
    lines.append("  1. 'MNQ 0900 and CME_OPEN are 48% correlated' -> see above")
    lines.append("  2. '1000 vs 1100 are anti-correlated (r=-0.07)' -> see above")
    lines.append("  3. 'Combined 0900+1100+CME_OPEN held Sharpe ~2.0' -> see combo analysis")
    lines.append("")
    lines.append("### CAVEATS")
    lines.append("  - Sharpe computed on raw pnl_r (no position sizing or capital allocation)")
    lines.append("  - 'Equal weight daily average' assumes simultaneous 1-lot per session")
    lines.append("  - Correlation computed on all overlapping dates, not rolling windows")
    lines.append("  - MNQ only ~2 years of data")
    lines.append("  - CME_OPEN is a dynamic session (DST-dependent timing)")

    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved to {OUTPUT}")
    con.close()


if __name__ == "__main__":
    run()
