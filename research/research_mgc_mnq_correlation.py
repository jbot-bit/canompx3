#!/usr/bin/env python3
"""
MGC/MNQ Correlation & Portfolio Research.

Tests whether the "safe-haven gold vs risk-on Nasdaq" inverse relationship
holds in this project's actual data — and whether it's stable enough to
inform portfolio construction or trade sheet context.

Three blocks:
  1. Daily Returns Correlation — close-to-close, rolling stability, year-by-year
  2. Session Direction Concordance — same-session break direction alignment
  3. Portfolio P&L Correlation — live pnl_r on same-day trade pairs

Honest framing:
  - "Low correlation between gold and equities" = baseline fact, not a discovery.
  - A discovery would be a session-level directional signal surviving BH FDR.
  - Lead-lag MGC→MNQ already tested Feb 2026: NO SIGNAL (p=0.189–0.936). Not repeated.

Usage:
    python research/research_mgc_mnq_correlation.py
    python research/research_mgc_mnq_correlation.py --db-path C:/db/gold.db
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

# ── Config ────────────────────────────────────────────────────────────────────

SHARED_SESSIONS = [
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "EUROPE_FLOW",
    "LONDON_METALS",
    "NYSE_OPEN",
    "US_DATA_830",
    "US_DATA_1000",
    "CME_PRECLOSE",
]

MIN_SAMPLE = 30
BH_Q = 0.10  # FDR threshold


# ── BH FDR ───────────────────────────────────────────────────────────────────


def bh_fdr(p_values: list[float], q: float = BH_Q) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    p_adj = [1.0] * n
    prev = 1.0
    for rank, (orig_i, p) in enumerate(reversed(indexed), 1):
        adjusted = min(prev, p * n / (n - rank + 1))
        p_adj[orig_i] = adjusted
        prev = adjusted
    return p_adj


# ── Data loading ──────────────────────────────────────────────────────────────


def load_daily(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Daily close + ATR for MGC and MNQ, one row per (trading_day, symbol)."""
    df = con.execute("""
        SELECT trading_day, symbol, daily_close, atr_20
        FROM daily_features
        WHERE symbol IN ('MGC', 'MNQ')
          AND orb_minutes = 5
          AND daily_close IS NOT NULL
        ORDER BY trading_day, symbol
    """).fetchdf()
    return df


def load_session_dirs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Break direction per session for MGC and MNQ."""
    # Build SELECT for each shared session's break_dir column
    selects = ", ".join(
        f"orb_{s}_break_dir AS {s}" for s in SHARED_SESSIONS
    )
    df = con.execute(f"""
        SELECT trading_day, symbol, {selects}
        FROM daily_features
        WHERE symbol IN ('MGC', 'MNQ')
          AND orb_minutes = 5
        ORDER BY trading_day, symbol
    """).fetchdf()
    return df


def load_portfolio_pnl(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """orb_outcomes pnl_r for MGC and MNQ, E2, orb_minutes=5, rr=1.0, cb=1.

    Fixed configuration (rr_target=1.0, confirm_bars=1) to avoid synthetic
    averages across different RR variants where wins at rr=3.0 produce +3.0R
    vs +1.0R at rr=1.0. Using a single spec keeps pnl_r values real.
    """
    df = con.execute("""
        SELECT trading_day, symbol, orb_label, pnl_r
        FROM orb_outcomes
        WHERE symbol IN ('MGC', 'MNQ')
          AND entry_model = 'E2'
          AND orb_minutes = 5
          AND rr_target = 1.0
          AND confirm_bars = 1
          AND pnl_r IS NOT NULL
        ORDER BY trading_day, symbol, orb_label
    """).fetchdf()
    return df


# ── Block 1: Daily Returns Correlation ───────────────────────────────────────


def block1_daily_returns(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 70)
    print("BLOCK 1: DAILY RETURNS CORRELATION (MGC vs MNQ)")
    print("=" * 70)

    df = load_daily(con)

    # Pivot to wide: one row per trading_day
    wide = df.pivot(index="trading_day", columns="symbol", values="daily_close").dropna()
    if len(wide) < 100:
        print(f"  WARNING: Only {len(wide)} shared days — insufficient data.")
        return

    wide = wide.sort_index()
    mgc = wide["MGC"]
    mnq = wide["MNQ"]

    # Pct returns
    mgc_ret = mgc.pct_change().dropna()
    mnq_ret = mnq.pct_change().dropna()
    common_idx = mgc_ret.index.intersection(mnq_ret.index)
    mgc_ret = mgc_ret.loc[common_idx]
    mnq_ret = mnq_ret.loc[common_idx]

    N = len(mgc_ret)
    r, p = stats.pearsonr(mgc_ret, mnq_ret)

    print(f"\nFull period: {common_idx.min()} to {common_idx.max()}  N={N} days")
    print(f"  Pearson r = {r:+.4f}   p = {p:.4f}")
    print(f"  Interpretation: ", end="")
    if p < 0.01:
        if r < -0.1:
            print(f"SIGNIFICANT negative correlation (r={r:.3f}). Modest inverse relationship.")
        elif r > 0.1:
            print(f"SIGNIFICANT positive correlation (r={r:.3f}). Move TOGETHER, not opposite.")
        else:
            print(f"Statistically significant but near-zero (r={r:.3f}). Negligible relationship.")
    else:
        print(f"NOT significant (p={p:.3f}). Cannot claim directional relationship.")

    # Year-by-year breakdown
    print("\n  Year-by-year Pearson r:")
    ret_df = pd.DataFrame({"MGC": mgc_ret, "MNQ": mnq_ret})
    ret_df.index = pd.to_datetime(ret_df.index)
    for yr in sorted(ret_df.index.year.unique()):
        sub = ret_df[ret_df.index.year == yr].dropna()
        if len(sub) < 30:
            print(f"    {yr}: N={len(sub)} — too few, skip")
            continue
        yr_r, yr_p = stats.pearsonr(sub["MGC"], sub["MNQ"])
        sig = "**" if yr_p < 0.01 else ("*" if yr_p < 0.05 else "")
        print(f"    {yr}: N={len(sub):3d}  r={yr_r:+.3f}  p={yr_p:.4f} {sig}")

    # Rolling 60-day stability
    print("\n  Rolling 60-day correlation (min_periods=30):")
    ret_df2 = pd.DataFrame({"MGC": mgc_ret, "MNQ": mnq_ret})
    rolling_corr = ret_df2["MGC"].rolling(60, min_periods=30).corr(ret_df2["MNQ"])
    rolling_corr = rolling_corr.dropna()
    if len(rolling_corr) > 0:
        print(f"    Min: {rolling_corr.min():+.3f}   Max: {rolling_corr.max():+.3f}   "
              f"Mean: {rolling_corr.mean():+.3f}   Std: {rolling_corr.std():.3f}")
        pct_negative = (rolling_corr < 0).mean() * 100
        pct_positive = (rolling_corr > 0).mean() * 100
        print(f"    {pct_negative:.0f}% of windows negative  |  {pct_positive:.0f}% positive")

    # ATR regime split (high vs low volatility in MGC)
    print("\n  ATR regime split (MGC atr_20 above/below median):")
    atr_df = df[df["symbol"] == "MGC"][["trading_day", "atr_20"]].set_index("trading_day")
    atr_median = atr_df["atr_20"].median()
    ret_atr = ret_df2.join(atr_df.rename(columns={"atr_20": "mgc_atr"}), how="left")
    for regime, label in [(True, "HIGH-ATR"), (False, "LOW-ATR")]:
        sub = ret_atr[ret_atr["mgc_atr"] > atr_median if regime
                      else ret_atr["mgc_atr"] <= atr_median].dropna(subset=["MGC", "MNQ"])
        if len(sub) < 30:
            continue
        rr, rp = stats.pearsonr(sub["MGC"], sub["MNQ"])
        print(f"    {label} (ATR {'>' if regime else '<='} {atr_median:.0f}): "
              f"N={len(sub)}  r={rr:+.3f}  p={rp:.4f}")

    # Honest verdict
    print("\n  VERDICT:")
    if p < 0.01 and r < -0.1:
        print(f"  Negative correlation confirmed but r={r:.3f} — weak to moderate.")
        print("  Rolling window shows instability — this relationship shifts by regime.")
        print("  Classification: MARKET STRUCTURE OBSERVATION (not a trading signal)")
    elif p < 0.01 and r > 0.0:
        print(f"  WARNING: Positive correlation (r={r:.3f}) — gold and NNQ moved TOGETHER")
        print("  in this sample. The 'natural hedge' narrative is NOT supported by data.")
    else:
        print("  Near-zero or unstable correlation. Neither consistently inverse nor positive.")
        print("  The 'natural hedge' narrative is oversimplified for this time period.")


# ── Block 2: Session Direction Concordance ────────────────────────────────────


def block2_session_concordance(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 70)
    print("BLOCK 2: SESSION DIRECTION CONCORDANCE (same session, same day)")
    print("=" * 70)
    print("Question: Do MGC and MNQ break the SAME or OPPOSITE direction more than base rate (two-sided)?")
    print("Method: Binomial test vs 50% base rate, BH FDR across all sessions\n")

    df = load_session_dirs(con)

    # Pivot to wide: one row per (trading_day, session_direction)
    mgc_df = df[df["symbol"] == "MGC"].set_index("trading_day")
    mnq_df = df[df["symbol"] == "MNQ"].set_index("trading_day")

    all_results = []

    for session in SHARED_SESSIONS:
        if session not in mgc_df.columns or session not in mnq_df.columns:
            continue

        mgc_dir = mgc_df[session]
        mnq_dir = mnq_df[session]

        # Align on common days
        common = mgc_dir.index.intersection(mnq_dir.index)
        mgc_dir = mgc_dir.loc[common]
        mnq_dir = mnq_dir.loc[common]

        # Only days where BOTH broke
        both_broke = (mgc_dir.notna()) & (mnq_dir.notna())
        mgc_broke = mgc_dir[both_broke]
        mnq_broke = mnq_dir[both_broke]

        n_both = both_broke.sum()
        if n_both < MIN_SAMPLE:
            all_results.append({
                "session": session, "n_both_broke": n_both,
                "n_opposite": None, "n_concordant": None,
                "pct_opposite": None, "p_raw": None, "p_bh": None,
                "note": f"N={n_both} < {MIN_SAMPLE} — skip"
            })
            continue

        # Concordant = same direction, Opposite = different direction
        concordant = ((mgc_broke == "long") & (mnq_broke == "long")) | \
                     ((mgc_broke == "short") & (mnq_broke == "short"))
        opposite = ((mgc_broke == "long") & (mnq_broke == "short")) | \
                   ((mgc_broke == "short") & (mnq_broke == "long"))

        n_concordant = concordant.sum()
        n_opposite = opposite.sum()
        n_total = n_concordant + n_opposite

        if n_total < MIN_SAMPLE:
            all_results.append({
                "session": session, "n_both_broke": n_both,
                "n_opposite": n_opposite, "n_concordant": n_concordant,
                "pct_opposite": None, "p_raw": None, "p_bh": None,
                "note": f"N_valid={n_total} < {MIN_SAMPLE} — skip"
            })
            continue

        pct_opposite = n_opposite / n_total

        # Fisher exact: is opposite > 50%?
        # Two-sided: tests both "opposite > 50%" and "concordant > 50%".
        # One-sided "greater" would miss sessions with strong concordance signal.
        p_raw = stats.binomtest(n_opposite, n_total, p=0.5, alternative="two-sided").pvalue

        all_results.append({
            "session": session,
            "n_both_broke": n_both,
            "n_opposite": n_opposite,
            "n_concordant": n_concordant,
            "pct_opposite": pct_opposite,
            "p_raw": p_raw,
            "p_bh": None,
            "note": ""
        })

    # Apply BH FDR
    valid = [r for r in all_results if r["p_raw"] is not None]
    if valid:
        p_vals = [r["p_raw"] for r in valid]
        p_adj = bh_fdr(p_vals, q=BH_Q)
        for r, padj in zip(valid, p_adj):
            r["p_bh"] = padj

    # Print results
    bh_survivors = []
    for r in all_results:
        note = r.get("note", "")
        if note:
            print(f"  {r['session']:20s}: {note}")
            continue
        sig = "** BH-SIG **" if r["p_bh"] < BH_Q else ""
        print(f"  {r['session']:20s}: N={r['n_both_broke']:4d} | "
              f"opposite={r['pct_opposite']:.1%} ({r['n_opposite']}/{r['n_opposite']+r['n_concordant']}) | "
              f"p_raw={r['p_raw']:.4f}  p_bh={r['p_bh']:.4f}  {sig}")
        if r["p_bh"] < BH_Q:
            bh_survivors.append(r)

    print(f"\n  BH survivors: {len(bh_survivors)} / {len(valid)} sessions tested")

    if bh_survivors:
        print("\n  SURVIVING FINDINGS:")
        for r in bh_survivors:
            direction = "opposite" if r["pct_opposite"] > 0.5 else "concordant"
            pct_display = r["pct_opposite"] if r["pct_opposite"] > 0.5 else 1 - r["pct_opposite"]
            print(f"    {r['session']}: {pct_display:.1%} {direction} breaks "
                  f"(N={r['n_both_broke']}, p_bh={r['p_bh']:.4f})")
        print("\n  MECHANISM CHECK: Does this have a structural reason?")
        print("  CONCORDANT (same direction > 50%): macro trend effect - shared directional response")
        print("  to data releases or Asian/European opens. Both assets move with the macro flow.")
        print("  OPPOSITE (opposite direction > 50%): flight-to-safety - risk-off/risk-on split.")
        print("  Check year-by-year below to confirm stability.")

        # Year-by-year for survivors
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        for r in bh_survivors:
            session = r["session"]
            print(f"\n  Year-by-year: {session}")
            mgc_d = df[df["symbol"] == "MGC"].set_index("trading_day")[session]
            mnq_d = df[df["symbol"] == "MNQ"].set_index("trading_day")[session]
            for yr in sorted(df["trading_day"].dt.year.unique()):
                mgc_yr = mgc_d[mgc_d.index.year == yr]
                mnq_yr = mnq_d[mnq_d.index.year == yr]
                common = mgc_yr.index.intersection(mnq_yr.index)
                if len(common) == 0:
                    continue
                mb = mgc_yr.loc[common]
                nb = mnq_yr.loc[common]
                both = (mb.notna()) & (nb.notna())
                n_opp = (((mb == "long") & (nb == "short")) |
                         ((mb == "short") & (nb == "long")))[both].sum()
                n_con = (((mb == "long") & (nb == "long")) |
                         ((mb == "short") & (nb == "short")))[both].sum()
                n_t = n_opp + n_con
                if n_t < 10:
                    continue
                print(f"    {yr}: N={n_t:3d}  opposite={n_opp/n_t:.1%}")
    else:
        print("\n  VERDICT: NO sessions show significant directional opposition.")
        print("  MGC and MNQ break in opposite directions no more than chance.")
        print("  The 'natural hedge from direction' narrative is NOT supported.")


# ── Block 3: Portfolio P&L Correlation ───────────────────────────────────────


def block3_portfolio_pnl(con: duckdb.DuckDBPyConnection) -> None:
    print("\n" + "=" * 70)
    print("BLOCK 3: PORTFOLIO P&L CORRELATION (live trade pnl_r)")
    print("=" * 70)
    print("Question: On days both MGC and MNQ have trades, do they")
    print("          win/lose together or independently?\n")

    df = load_portfolio_pnl(con)

    # For each session, compute cross-instrument pnl correlation
    results = []
    for session in SHARED_SESSIONS:
        sess_df = df[df["orb_label"] == session].copy()
        if sess_df.empty:
            continue

        pivot = sess_df.pivot_table(
            index="trading_day", columns="symbol", values="pnl_r"
        )
        if "MGC" not in pivot.columns or "MNQ" not in pivot.columns:
            continue

        paired = pivot[["MGC", "MNQ"]].dropna()
        N = len(paired)
        if N < MIN_SAMPLE:
            print(f"  {session:20s}: N={N} shared trade days — too few")
            continue

        r, p = stats.pearsonr(paired["MGC"], paired["MNQ"])

        # Co-loss: both lose on same day
        co_loss = ((paired["MGC"] < 0) & (paired["MNQ"] < 0)).mean()
        co_win = ((paired["MGC"] > 0) & (paired["MNQ"] > 0)).mean()

        # Simulate portfolio: equal weight, long MGC + long MNQ
        portfolio_r = (paired["MGC"] + paired["MNQ"]) / 2
        # Per-trade Sharpe (relative comparison only — not annualized, sessions
        # don't fire every day so sqrt(252) overstates vs daily strategies)
        port_sharpe = (portfolio_r.mean() / portfolio_r.std()
                       if portfolio_r.std() > 0 else np.nan)
        mgc_sharpe = (paired["MGC"].mean() / paired["MGC"].std()
                      if paired["MGC"].std() > 0 else np.nan)
        mnq_sharpe = (paired["MNQ"].mean() / paired["MNQ"].std()
                      if paired["MNQ"].std() > 0 else np.nan)

        results.append({
            "session": session, "N": N, "r": r, "p": p, "p_bh": None,
            "co_loss": co_loss, "co_win": co_win,
            "port_sharpe": port_sharpe, "mgc_sharpe": mgc_sharpe,
            "mnq_sharpe": mnq_sharpe
        })

    if not results:
        print("  No sessions with sufficient shared trade days.")
        return

    # Apply BH FDR before printing (consistent with Block 2)
    p_vals = [r["p"] for r in results]
    p_adj = bh_fdr(p_vals)
    for res, pa in zip(results, p_adj):
        res["p_bh"] = pa

    for res in results:
        sig = " ** BH-SIG" if res["p_bh"] < BH_Q else ""
        print(f"  {res['session']:20s}: N={res['N']:4d}  r={res['r']:+.3f}  "
              f"p_raw={res['p']:.4f}  p_bh={res['p_bh']:.4f}{sig}  "
              f"co-loss={res['co_loss']:.1%}  co-win={res['co_win']:.1%}")
        print(f"    Sharpe (per-trade, relative): "
              f"MGC={res['mgc_sharpe']:.2f}  MNQ={res['mnq_sharpe']:.2f}  "
              f"Combined={res['port_sharpe']:.2f}")

    # Overall assessment
    avg_r = np.mean([r["r"] for r in results])
    print(f"\n  Average cross-instrument pnl_r correlation: {avg_r:+.3f}")

    if avg_r > 0.3:
        print("  WARNING: Moderate positive P&L correlation — shared drawdown risk.")
        print("  Long MGC + Long MNQ does NOT diversify well — they tend to lose together.")
    elif avg_r < -0.1:
        print("  Modest negative P&L correlation — some natural hedge benefit.")
    else:
        print("  Near-zero P&L correlation — instruments trade largely independently.")
        print("  This is GOOD for portfolio construction: diversification without hedge.")

    bh_sig = [r for r in results if r["p_bh"] < BH_Q]
    print(f"\n  BH FDR survivors: {len(bh_sig)} / {len(results)}")
    for r in bh_sig:
        print(f"    {r['session']}: r={r['r']:+.3f}, p_bh={r['p_bh']:.4f}")


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary() -> None:
    print("\n" + "=" * 70)
    print("SUMMARY & TRADE SHEET IMPLICATIONS")
    print("=" * 70)
    print(f"""
Results above tell you one of three things:

A) NEGATIVE correlation confirmed + stable + survives FDR:
   - 'Long MGC / Short MNQ' has empirical basis as a diversified position.
   - Add to Market Context panel: 'Instruments show X% opposition rate at SESSION.'

B) Near-zero or unstable correlation:
   - Instruments are INDEPENDENT, not naturally hedged.
   - This is GOOD (uncorrelated edges = true diversification).
   - Market Context: 'Instruments trade independently - no cross-hedge benefit,
                       no shared drawdown risk at session level.'

C) POSITIVE correlation:
   - Long both = concentrated risk. They lose together.
   - Market Context: 'Both instruments move together at SESSION - size down
                       if holding both simultaneously.'

Trade sheet panel: ONLY add findings that survived BH FDR (p_bh < {BH_Q:.2f}).
Grade PRELIMINARY (N=100-199) or HIGH-CONFIDENCE (N=500+) per RESEARCH_RULES.md.
""")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MGC/MNQ correlation research")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()

    print("MGC/MNQ Correlation Research")
    print(f"DB: {args.db_path}")
    print(f"Shared sessions tested: {', '.join(SHARED_SESSIONS)}")

    con = duckdb.connect(str(args.db_path), read_only=True)
    try:
        block1_daily_returns(con)
        block2_session_concordance(con)
        block3_portfolio_pnl(con)
        print_summary()
    finally:
        con.close()

    print("\nDone. Write findings to research/output/mgc_mnq_correlation_findings.md")
    print(f"Only BH-surviving findings (p_bh < {BH_Q:.2f}) go to the trade sheet panel.")


if __name__ == "__main__":
    main()
