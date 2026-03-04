#!/usr/bin/env python3
"""Refine M1 with exactly 3 fixed variants (fast decision).

M1 base setup:
- Leader: MES 0900
- Follower: MES 1000
- Strategy: E0 / CB2 / RR2.5
- No-lookahead: leader_break_ts <= follower entry_ts
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"


def stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
    }


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    q = """
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_ts,
      df_f.orb_1000_break_dir AS f_dir,
      df_l.orb_0900_break_dir AS l_dir,
      df_l.orb_0900_break_ts  AS l_ts,
      df_l.orb_0900_break_delay_min AS l_delay,
      df_l.orb_0900_break_bar_continues AS l_cont,
      df_l.orb_0900_size AS l_size,
      df_l.atr_20 AS l_atr
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON df_f.symbol=o.symbol
     AND df_f.trading_day=o.trading_day
     AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l
      ON df_l.symbol='MES'
     AND df_l.trading_day=o.trading_day
     AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='MES'
      AND o.orb_label='1000'
      AND o.entry_model='E0'
      AND o.confirm_bars=2
      AND o.rr_target=2.5
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows.")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    valid = (
        df["f_dir"].isin(["long", "short"]) &
        df["l_dir"].isin(["long", "short"]) &
        (df["l_ts"].notna()) &
        (df["l_ts"] <= df["entry_ts"])
    )
    df = df[valid].copy()
    if df.empty:
        print("No valid no-lookahead rows.")
        return 0

    same = (df["l_dir"] == df["f_dir"])
    fast = df["l_delay"].notna() & (df["l_delay"] <= 30)
    size_atr = np.where((df["l_atr"].notna()) & (df["l_atr"] > 0), df["l_size"] / df["l_atr"], np.nan)
    q60 = pd.Series(size_atr).quantile(0.60)

    variants = {
        "V1_same_fast": same & fast,
        "V2_same_fast_cont": same & fast & (df["l_cont"] == True),
        "V3_same_fast_size60": same & fast & pd.Series(size_atr).notna() & (pd.Series(size_atr) >= q60),
    }

    rows = []
    for name, m in variants.items():
        on = df.loc[m, "pnl_r"]
        off = df.loc[~m, "pnl_r"]
        if len(on) < 30 or len(off) < 30:
            continue

        s_on = stats(on)
        s_off = stats(off)

        # OOS (2025)
        tr = df[df["year"] <= 2024]
        te = df[df["year"] == 2025]
        mtr = m.loc[tr.index]
        mte = m.loc[te.index]
        tr_on = tr.loc[mtr, "pnl_r"]
        tr_off = tr.loc[~mtr, "pnl_r"]
        te_on = te.loc[mte, "pnl_r"]
        te_off = te.loc[~mte, "pnl_r"]

        train_uplift = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 30 and len(tr_off) >= 30 else np.nan
        test_uplift = float(te_on.mean() - te_off.mean()) if len(te_on) >= 20 and len(te_off) >= 20 else np.nan

        years_pos, years_total = 0, 0
        for _, gy in df.groupby("year"):
            my = m.loc[gy.index]
            oy = gy.loc[my, "pnl_r"]
            fy = gy.loc[~my, "pnl_r"]
            if len(oy) < 20 or len(fy) < 20:
                continue
            years_total += 1
            if oy.mean() - fy.mean() > 0:
                years_pos += 1

        rows.append({
            "variant": name,
            "n_on": s_on["n"],
            "signals_per_year": s_on["n"] / max(1, df["year"].nunique()),
            "avg_on": s_on["avg_r"],
            "avg_off": s_off["avg_r"],
            "uplift": s_on["avg_r"] - s_off["avg_r"],
            "wr_on": s_on["wr"],
            "wr_off": s_off["wr"],
            "years_pos": years_pos,
            "years_total": years_total,
            "years_pos_ratio": (years_pos / years_total) if years_total else np.nan,
            "train_uplift": train_uplift,
            "test2025_uplift": test_uplift,
            "n_test_on": int(len(te_on)),
        })

    out = pd.DataFrame(rows).sort_values(["avg_on", "uplift"], ascending=False)

    def verdict(r):
        if (
            r["avg_on"] >= 0.08 and r["uplift"] >= 0.15 and
            pd.notna(r["test2025_uplift"]) and r["test2025_uplift"] > 0 and
            r["n_on"] >= 70
        ):
            return "PROMOTE"
        if r["avg_on"] > 0 and r["uplift"] > 0.08 and r["n_on"] >= 40:
            return "WATCH"
        return "KILL"

    if not out.empty:
        out["result"] = out.apply(verdict, axis=1)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "m1_refine_three_variants.csv"
    p_md = out_dir / "m1_refine_three_variants.md"

    if out.empty:
        p_md.write_text("# M1 refine\n\nNo variants met minimum sample.", encoding="utf-8")
        print("No variants met minimum sample.")
        return 0

    out.to_csv(p_csv, index=False)
    lines = ["# M1 Refine (3 Variants)", ""]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant} => {r.result}: avg_on={r.avg_on:+.4f}, uplift={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, test2025Δ={r.test2025_uplift:+.4f}, N_on={r.n_on}"
        )
    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
