#!/usr/bin/env python3
"""Next hypothesis: A0 directional/tempo asymmetry overlays.

Base candidate A0:
- Leader: M6E_US_EQUITY_OPEN
- Follower: MES_US_EQUITY_OPEN
- Strategy: E0 / CB1 / RR3.0
- Base condition: same-direction + no-lookahead

Hypothesis:
- A0 may improve materially by side-specific and speed-specific gates.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

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
      df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
      df_f.orb_US_EQUITY_OPEN_break_delay_min AS f_delay,
      df_f.orb_US_EQUITY_OPEN_break_bar_continues AS f_cont,
      df_f.orb_US_EQUITY_OPEN_size AS f_size,
      df_f.atr_20 AS f_atr,
      df_l.orb_US_EQUITY_OPEN_break_dir AS l_dir,
      df_l.orb_US_EQUITY_OPEN_break_ts  AS l_ts
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON df_f.symbol=o.symbol
     AND df_f.trading_day=o.trading_day
     AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l
      ON df_l.symbol='M6E'
     AND df_l.trading_day=o.trading_day
     AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='MES'
      AND o.orb_label='US_EQUITY_OPEN'
      AND o.entry_model='E0'
      AND o.confirm_bars=1
      AND o.rr_target=3.0
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

    same = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )

    size_atr = np.where((df["f_atr"].notna()) & (df["f_atr"] > 0), df["f_size"] / df["f_atr"], np.nan)
    q60 = pd.Series(size_atr).quantile(0.60)

    variants = {
        "base_same_dir": same,
        "same_long_only": same & (df["f_dir"] == "long"),
        "same_short_only": same & (df["f_dir"] == "short"),
        "same_fast15": same & df["f_delay"].notna() & (df["f_delay"] <= 15),
        "same_fast30": same & df["f_delay"].notna() & (df["f_delay"] <= 30),
        "same_cont": same & (df["f_cont"] == True),
        "same_short_fast15": same & (df["f_dir"] == "short") & df["f_delay"].notna() & (df["f_delay"] <= 15),
        "same_long_fast15": same & (df["f_dir"] == "long") & df["f_delay"].notna() & (df["f_delay"] <= 15),
        "same_fast15_size60": same & df["f_delay"].notna() & (df["f_delay"] <= 15) & pd.Series(size_atr).notna() & (pd.Series(size_atr) >= q60),
    }

    rows = []
    for name, m in variants.items():
        on = df.loc[m, "pnl_r"]
        off = df.loc[~m, "pnl_r"]
        if len(on) < 40 or len(off) < 40:
            continue

        s_on = stats(on)
        s_off = stats(off)

        # yearly uplift sign count
        yp, yt = 0, 0
        for _, gy in df.groupby("year"):
            my = m.loc[gy.index]
            oy = gy.loc[my, "pnl_r"]
            fy = gy.loc[~my, "pnl_r"]
            if len(oy) < 20 or len(fy) < 20:
                continue
            yt += 1
            if oy.mean() - fy.mean() > 0:
                yp += 1

        # OOS 2025
        tr = df[df["year"] <= 2024]
        te = df[df["year"] == 2025]
        mtr = m.loc[tr.index]
        mte = m.loc[te.index]
        tr_on = tr.loc[mtr, "pnl_r"]
        tr_off = tr.loc[~mtr, "pnl_r"]
        te_on = te.loc[mte, "pnl_r"]
        te_off = te.loc[~mte, "pnl_r"]

        tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 40 and len(tr_off) >= 40 else np.nan
        te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 30 and len(te_off) >= 30 else np.nan

        rows.append(
            {
                "variant": name,
                "n_on": s_on["n"],
                "on_rate": s_on["n"] / len(df),
                "avg_on": s_on["avg_r"],
                "avg_off": s_off["avg_r"],
                "uplift": s_on["avg_r"] - s_off["avg_r"],
                "wr_on": s_on["wr"],
                "wr_off": s_off["wr"],
                "years_pos": yp,
                "years_total": yt,
                "train_uplift": tr_up,
                "test2025_uplift": te_up,
            }
        )

    out = pd.DataFrame(rows).sort_values(["avg_on", "uplift"], ascending=False)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "a0_asymmetry_hypothesis.csv"
    p_md = out_dir / "a0_asymmetry_hypothesis.md"

    if out.empty:
        p_md.write_text("# A0 asymmetry hypothesis\n\nNo rows met thresholds.", encoding="utf-8")
        print("No rows met thresholds.")
        return 0

    out.to_csv(p_csv, index=False)

    lines = ["# A0 Asymmetry Hypothesis", ""]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.variant}: N_on={r.n_on}, avg_on={r.avg_on:+.4f}, avg_off={r.avg_off:+.4f}, Δ={r.uplift:+.4f}, years+={r.years_pos}/{r.years_total}, test2025Δ={r.test2025_uplift:+.4f}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
