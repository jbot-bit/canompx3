#!/usr/bin/env python3
"""Intuition/logic next test on A0 (fixed hypothesis set, no brute-force).

A0 base:
- M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN
- E0 / CB1 / RR3.0
- same direction + no-lookahead

Logical hypotheses tested:
1) Mid-week event-risk suppression (exclude Wed / Wed+Thu)
2) Leader quality (fast/continuation)
3) Overstretch avoidance (size_atr cap)
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"


def stats(s: pd.Series, years: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "sig_per_year": np.nan, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    y = max(1, int(years.nunique()))
    return {
        "n": int(len(s)),
        "sig_per_year": float(len(s) / y),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
    }


def main() -> int:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
           df_l.orb_US_EQUITY_OPEN_break_dir AS l_dir,
           df_l.orb_US_EQUITY_OPEN_break_ts  AS l_ts,
           df_l.orb_US_EQUITY_OPEN_break_delay_min AS l_delay,
           df_l.orb_US_EQUITY_OPEN_break_bar_continues AS l_cont,
           df_l.orb_US_EQUITY_OPEN_size AS l_size,
           df_l.atr_20 AS l_atr
    FROM orb_outcomes o
    JOIN daily_features df_f ON df_f.symbol=o.symbol AND df_f.trading_day=o.trading_day AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l ON df_l.symbol='M6E' AND df_l.trading_day=o.trading_day AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='MES' AND o.orb_label='US_EQUITY_OPEN'
      AND o.entry_model='E0' AND o.confirm_bars=1 AND o.rr_target=3.0
      AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows.")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["dow"] = df["trading_day"].dt.dayofweek
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    base = (
        df["f_dir"].isin(["long", "short"]) &
        df["l_dir"].isin(["long", "short"]) &
        (df["f_dir"] == df["l_dir"]) &
        df["l_ts"].notna() &
        (df["l_ts"] <= df["entry_ts"])
    )

    d = df[base].copy()
    if d.empty:
        print("No base rows.")
        return 0

    d["l_size_atr"] = np.where((d["l_atr"].notna()) & (d["l_atr"] > 0), d["l_size"] / d["l_atr"], np.nan)
    q80 = d["l_size_atr"].quantile(0.80)

    variants = {
        "base": pd.Series(True, index=d.index),
        "exclude_wed": d["dow"] != 2,
        "exclude_wed_thu": ~d["dow"].isin([2, 3]),
        "mon_tue_fri": d["dow"].isin([0, 1, 4]),
        "leader_fast30": d["l_delay"].notna() & (d["l_delay"] <= 30),
        "leader_cont": d["l_cont"] == True,
        "leader_fast30_ex_wed": (d["l_delay"].notna() & (d["l_delay"] <= 30) & (d["dow"] != 2)),
        "nonstretch_ex_wed": (d["l_size_atr"].notna() & (d["l_size_atr"] <= q80) & (d["dow"] != 2)),
    }

    base_stats = stats(d["pnl_r"], d["year"])

    rows = []
    for name, m in variants.items():
        on = d.loc[m, "pnl_r"]
        st = stats(on, d.loc[m, "year"]) if name != "base" else base_stats

        # OOS delta vs base (2025)
        te = d[d["year"] == 2025]
        test_delta = np.nan
        if not te.empty:
            mt = m.loc[te.index]
            te_on = te.loc[mt, "pnl_r"]
            if len(te_on) >= 15:
                test_delta = float(te_on.mean() - te["pnl_r"].mean())

        rows.append(
            {
                "variant": name,
                "n": st["n"],
                "signals_per_year": st["sig_per_year"],
                "avg_r": st["avg_r"],
                "wr": st["wr"],
                "delta_avg_vs_base": float(st["avg_r"] - base_stats["avg_r"]) if name != "base" else 0.0,
                "delta_test2025_vs_base": test_delta if name != "base" else 0.0,
            }
        )

    out = pd.DataFrame(rows).sort_values(["avg_r", "delta_avg_vs_base"], ascending=False)

    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "a0_intuition_logic_next.csv"
    p_md = out_dir / "a0_intuition_logic_next.md"

    out.to_csv(p_csv, index=False)

    lines = ["# A0 Intuition/Logic Next", "", "Fixed hypothesis set (no brute-force).", ""]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant}: N={r.n}, sig/yr={r.signals_per_year:.1f}, avgR={r.avg_r:+.4f}, Δavg={r.delta_avg_vs_base:+.4f}, test2025Δ={r.delta_test2025_vs_base:+.4f}"
        )

    # next hypothesis from this result
    lines.append("")
    lines.append("## Next hypothesis from this result")
    lines.append("If mid-week suppression variants dominate, hypothesis is event-regime contamination.")
    lines.append("Next test should use explicit macro-calendar tags (FOMC/CPI/NFP days) instead of weekday proxy.")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
