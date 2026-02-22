#!/usr/bin/env python3
"""Build deployment presets for KEEP strategies: base vs fast15 vs vol60 vs both.

Outputs:
- research/output/shinies_overlay_stack_presets.csv
- research/output/shinies_overlay_stack_presets.md
"""

from __future__ import annotations

from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
REG_PATH = PROJECT_ROOT / "research" / "output" / "shinies_registry.csv"


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def parse_tag(tag: str):
    if not isinstance(tag, str) or "_" not in tag:
        return None
    a, b = tag.split("_", 1)
    if not a or not b:
        return None
    if re.fullmatch(r"[A-Za-z0-9]+", a) is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", b) is None:
        return None
    return a, b


def load_df(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    fsym, fsess = parse_tag(str(row["follower"]))
    fs = safe_label(fsess)

    leader_tag = str(row["leader"])
    if str(row.get("id", "")) == "B2" or leader_tag.endswith("fast_le_15"):
        leader = None
    else:
        leader = parse_tag(leader_tag)

    if leader is not None:
        lsym, lsess = leader
        ls = safe_label(lsess)
        q = f"""
        SELECT
          o.trading_day,o.pnl_r,o.entry_ts,
          d_f.orb_{fs}_break_dir AS f_dir,
          d_f.orb_{fs}_break_delay_min AS f_delay,
          d_f.orb_{fs}_break_bar_continues AS f_cont,
          d_f.orb_{fs}_size AS f_size,
          d_f.orb_{fs}_volume AS f_vol,
          d_f.orb_{fs}_break_bar_volume AS f_bvol,
          d_f.atr_20 AS f_atr,
          d_l.orb_{ls}_break_dir AS l_dir,
          d_l.orb_{ls}_break_ts  AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        JOIN daily_features d_l ON d_l.symbol='{lsym}' AND d_l.trading_day=o.trading_day AND d_l.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}'
          AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """
    else:
        q = f"""
        SELECT
          o.trading_day,o.pnl_r,o.entry_ts,
          d_f.orb_{fs}_break_dir AS f_dir,
          d_f.orb_{fs}_break_delay_min AS f_delay,
          d_f.orb_{fs}_break_bar_continues AS f_cont,
          d_f.orb_{fs}_size AS f_size,
          d_f.orb_{fs}_volume AS f_vol,
          d_f.orb_{fs}_break_bar_volume AS f_bvol,
          d_f.atr_20 AS f_atr,
          NULL::VARCHAR AS l_dir,
          NULL::TIMESTAMPTZ AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}'
          AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    if "l_ts" in df.columns:
        df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)
    return df


def base_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    if str(row.get("id", "")) == "B2" or str(row["leader"]).endswith("fast_le_15"):
        return df["f_delay"].notna() & (df["f_delay"] <= 15)

    return (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )


def summarize(series: pd.Series, years: pd.Series) -> dict:
    if series.empty:
        return {"n": 0, "sigyr": np.nan, "wr": np.nan, "avg": np.nan}
    yrs = max(1, int(years.nunique()))
    return {
        "n": int(len(series)),
        "sigyr": float(len(series) / yrs),
        "wr": float((series > 0).mean()),
        "avg": float(series.mean()),
    }


def main() -> int:
    reg = pd.read_csv(REG_PATH)
    reg = reg[reg["status"] == "KEEP"].copy()

    con = duckdb.connect(DB_PATH, read_only=True)
    rows = []

    for _, r in reg.iterrows():
        df = load_df(con, r)
        if df.empty:
            continue

        bm = base_mask(df, r)
        dfb = df[bm].copy()
        if len(dfb) < 80:
            continue

        vol_q60 = dfb["f_vol_imp"].quantile(0.60)

        masks = {
            "base": pd.Series(True, index=dfb.index),
            "base_plus_fast15": dfb["f_delay"].notna() & (dfb["f_delay"] <= 15),
            "base_plus_vol60": dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vol_q60),
            "base_plus_both": (dfb["f_delay"].notna() & (dfb["f_delay"] <= 15) & dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vol_q60)),
        }

        base_stats = summarize(dfb["pnl_r"], dfb["year"])
        for name, m in masks.items():
            on = dfb.loc[m, "pnl_r"]
            st = summarize(on, dfb.loc[m, "year"])
            # OOS delta vs base for 2025
            te = dfb[dfb["year"] == 2025]
            te_delta = np.nan
            if not te.empty:
                mt = m.loc[te.index]
                te_on = te.loc[mt, "pnl_r"]
                if len(te_on) >= 15:
                    te_delta = float(te_on.mean() - te["pnl_r"].mean())

            rows.append({
                "strategy_id": r["id"],
                "strategy": f"{r['leader']} -> {r['follower']} {r['entry_model']}/CB{int(r['confirm_bars'])}/RR{float(r['rr_target'])}",
                "preset": name,
                "n": st["n"],
                "signals_per_year": st["sigyr"],
                "wr": st["wr"],
                "avg_r": st["avg"],
                "delta_avg_vs_base": float(st["avg"] - base_stats["avg"]) if name != "base" else 0.0,
                "delta_test2025_vs_base": te_delta if name != "base" else 0.0,
            })

    con.close()

    out = pd.DataFrame(rows)
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "shinies_overlay_stack_presets.csv"
    p_md = out_dir / "shinies_overlay_stack_presets.md"

    if out.empty:
        p_md.write_text("# Overlay stack presets\n\nNo rows.", encoding="utf-8")
        print("No rows")
        return 0

    out.to_csv(p_csv, index=False)

    lines = ["# Overlay Stack Presets", "", "Per strategy: base vs +fast15 vs +vol60 vs +both", ""]
    for sid, g in out.groupby("strategy_id"):
        lines.append(f"## {sid}")
        gg = g.sort_values("preset")
        for r in gg.itertuples(index=False):
            lines.append(
                f"- {r.preset}: N={r.n}, sig/yr={r.signals_per_year:.1f}, avgR={r.avg_r:+.4f}, Δavg={r.delta_avg_vs_base:+.4f}, test2025Δ={r.delta_test2025_vs_base:+.4f}"
            )
        best = gg[gg["preset"] != "base"].sort_values(["delta_avg_vs_base", "delta_test2025_vs_base"], ascending=False).head(1)
        if not best.empty:
            b = best.iloc[0]
            lines.append(f"  -> Recommended preset: {b['preset']}")
        lines.append("")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
