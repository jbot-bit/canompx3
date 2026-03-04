#!/usr/bin/env python3
"""Loose, gain-first scan around A0 family.

Leader fixed: M6E_US_EQUITY_OPEN
Follower fixed: MES_US_EQUITY_OPEN
No-lookahead mandatory.

Scans strategy grid + directional/tempo variants with looser frequency constraints,
so numbers can surface high-payoff pockets.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    return {"n": int(len(s)), "wr": float((s > 0).mean()), "avg_r": float(s.mean()), "total_r": float(s.sum())}


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    q = """
    SELECT
      o.trading_day,
      o.entry_model,
      o.confirm_bars,
      o.rr_target,
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
      AND o.entry_model IN ('E0','E1','E3')
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

    rows = []
    for (em, cb, rr), g in df.groupby(["entry_model", "confirm_bars", "rr_target"]):
        if len(g) < 300:
            continue

        sg = same.loc[g.index]
        variants = {
            "same_dir": sg,
            "same_short_only": sg & (g["f_dir"] == "short"),
            "same_long_only": sg & (g["f_dir"] == "long"),
            "same_fast15": sg & g["f_delay"].notna() & (g["f_delay"] <= 15),
            "same_fast30": sg & g["f_delay"].notna() & (g["f_delay"] <= 30),
            "same_short_fast15": sg & (g["f_dir"] == "short") & g["f_delay"].notna() & (g["f_delay"] <= 15),
            "same_long_fast15": sg & (g["f_dir"] == "long") & g["f_delay"].notna() & (g["f_delay"] <= 15),
            "same_fast15_size60": sg & g["f_delay"].notna() & (g["f_delay"] <= 15) & pd.Series(size_atr, index=df.index).loc[g.index].notna() & (pd.Series(size_atr, index=df.index).loc[g.index] >= q60),
        }

        for vname, vm in variants.items():
            on = g.loc[vm, "pnl_r"]
            off = g.loc[~vm, "pnl_r"]
            if len(on) < 40 or len(off) < 40:
                continue

            s_on = _stats(on)
            s_off = _stats(off)
            uplift = s_on["avg_r"] - s_off["avg_r"]

            # OOS 2025
            tr = g[g["year"] <= 2024]
            te = g[g["year"] == 2025]
            vtr = vm.loc[tr.index]
            vte = vm.loc[te.index]
            tr_on = tr.loc[vtr, "pnl_r"]
            tr_off = tr.loc[~vtr, "pnl_r"]
            te_on = te.loc[vte, "pnl_r"]
            te_off = te.loc[~vte, "pnl_r"]

            tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 40 and len(tr_off) >= 40 else np.nan
            te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 30 and len(te_off) >= 30 else np.nan

            # yearly positive ratio
            yp, yt = 0, 0
            for _, gy in g.groupby("year"):
                my = vm.loc[gy.index]
                oy = gy.loc[my, "pnl_r"]
                fy = gy.loc[~my, "pnl_r"]
                if len(oy) < 20 or len(fy) < 20:
                    continue
                yt += 1
                if oy.mean() - fy.mean() > 0:
                    yp += 1

            years_cov = max(1, g["year"].nunique())
            rows.append(
                {
                    "entry_model": em,
                    "confirm_bars": int(cb),
                    "rr_target": float(rr),
                    "variant": vname,
                    "n_on": s_on["n"],
                    "signals_per_year": s_on["n"] / years_cov,
                    "avg_on": s_on["avg_r"],
                    "avg_off": s_off["avg_r"],
                    "uplift": uplift,
                    "wr_on": s_on["wr"],
                    "wr_off": s_off["wr"],
                    "years_pos": yp,
                    "years_total": yt,
                    "years_pos_ratio": (yp / yt) if yt else np.nan,
                    "train_uplift": tr_up,
                    "test2025_uplift": te_up,
                    "n_test_on": int(len(te_on)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No rows after thresholds.")
        return 0

    out = out.sort_values(["avg_on", "uplift"], ascending=False)

    # gain-first shortlist
    top_gain = out[
        (out["avg_on"] >= 0.12)
        & (out["uplift"] >= 0.20)
        & (out["years_total"] >= 3)
        & (out["years_pos_ratio"] >= 0.6)
        & (out["test2025_uplift"].fillna(-999) >= 0)
    ].copy().sort_values(["avg_on", "uplift"], ascending=False)

    # balanced shortlist (near your tradability needs)
    balanced = top_gain[top_gain["signals_per_year"] >= 70].copy().sort_values(["avg_on", "uplift"], ascending=False)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "a0_family_loose_all.csv"
    p_top = out_dir / "a0_family_loose_top_gain.csv"
    p_bal = out_dir / "a0_family_loose_balanced.csv"
    p_md = out_dir / "a0_family_loose_notes.md"

    out.to_csv(p_all, index=False)
    top_gain.to_csv(p_top, index=False)
    balanced.to_csv(p_bal, index=False)

    lines = [
        "# A0 Family Loose Scan",
        "",
        f"All rows: {len(out)}",
        f"Top gain rows: {len(top_gain)}",
        f"Balanced rows (>=70/yr): {len(balanced)}",
        "",
        "## Top gain",
    ]
    for r in top_gain.head(20).itertuples(index=False):
        lines.append(
            f"- {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} {r.variant}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, testΔ={r.test2025_uplift:+.4f}"
        )

    lines.append("")
    lines.append("## Balanced")
    if balanced.empty:
        lines.append("- None")
    else:
        for r in balanced.head(20).itertuples(index=False):
            lines.append(
                f"- {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} {r.variant}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, testΔ={r.test2025_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_top}")
    print(f"Saved: {p_bal}")
    print(f"Saved: {p_md}")
    print("\nTop gain:")
    print(top_gain.head(20).to_string(index=False))
    print("\nBalanced:")
    print(balanced.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
