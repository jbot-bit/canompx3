#!/usr/bin/env python3
"""Quick challenger scan: asymmetric + strength-gated lead-lag filters.

Focus on current top pairs and test alternatives to plain same-direction logic.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"

PAIRS = [
    # (name, leader_symbol, leader_session, follower_symbol, follower_session, em, cb, rr)
    ("P1", "M6E", "US_EQUITY_OPEN", "M2K", "US_POST_EQUITY", "E1", 5, 1.5),
    ("P2", "MES", "US_DATA_OPEN", "M2K", "US_DATA_OPEN", "E0", 1, 1.5),
]


def stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan}
    return {"n": int(len(s)), "wr": float((s > 0).mean()), "avg_r": float(s.mean())}


def eval_pair(con: duckdb.DuckDBPyConnection, spec: tuple) -> pd.DataFrame:
    name, lsym, lsess, fsym, fsess, em, cb, rr = spec

    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_ts,
      df_f.orb_{fsess}_break_dir AS f_dir,
      df_l.orb_{lsess}_break_dir AS l_dir,
      df_l.orb_{lsess}_break_ts  AS l_ts,
      df_l.orb_{lsess}_size      AS l_size,
      df_l.orb_{lsess}_break_delay_min AS l_delay
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON df_f.symbol=o.symbol
     AND df_f.trading_day=o.trading_day
     AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l
      ON df_l.symbol='{lsym}'
     AND df_l.trading_day=o.trading_day
     AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='{fsym}'
      AND o.orb_label='{fsess}'
      AND o.entry_model='{em}'
      AND o.confirm_bars={cb}
      AND o.rr_target={rr}
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        return pd.DataFrame()

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    valid = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )

    size_q75 = df.loc[df["l_size"].notna(), "l_size"].quantile(0.75)

    conds = {
        "same_dir": valid & (df["l_dir"] == df["f_dir"]),
        "opp_dir": valid & (df["l_dir"] != df["f_dir"]),
        "same_dir_strong": valid & (df["l_dir"] == df["f_dir"]) & df["l_delay"].notna() & (df["l_delay"] <= 15) & df["l_size"].notna() & (df["l_size"] >= size_q75),
        "opp_dir_strong": valid & (df["l_dir"] != df["f_dir"]) & df["l_delay"].notna() & (df["l_delay"] <= 15) & df["l_size"].notna() & (df["l_size"] >= size_q75),
        "same_long_only": valid & (df["l_dir"] == "long") & (df["f_dir"] == "long"),
        "same_short_only": valid & (df["l_dir"] == "short") & (df["f_dir"] == "short"),
    }

    base = df["pnl_r"]
    rows = []

    # OOS year (prefer 2025 if present)
    oos_year = 2025 if 2025 in df["year"].unique() else int(df["year"].max())
    tr = df[df["year"] < oos_year]
    te = df[df["year"] == oos_year]

    for cname, c in conds.items():
        on = df.loc[c, "pnl_r"]
        off = df.loc[~c, "pnl_r"]
        if len(on) < 50 or len(off) < 50:
            continue

        b = stats(base)
        s_on = stats(on)
        s_off = stats(off)

        # yearly uplift count
        yp, yt = 0, 0
        for y, gy in df.groupby("year"):
            cy = c.loc[gy.index]
            oy = gy.loc[cy, "pnl_r"]
            fy = gy.loc[~cy, "pnl_r"]
            if len(oy) < 20 or len(fy) < 20:
                continue
            yt += 1
            if oy.mean() - fy.mean() > 0:
                yp += 1

        # OOS uplift
        ctr = c.loc[tr.index]
        cte = c.loc[te.index]
        tr_on = tr.loc[ctr, "pnl_r"]
        tr_off = tr.loc[~ctr, "pnl_r"]
        te_on = te.loc[cte, "pnl_r"]
        te_off = te.loc[~cte, "pnl_r"]

        tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 30 and len(tr_off) >= 30 else np.nan
        te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 20 and len(te_off) >= 20 else np.nan

        rows.append(
            {
                "pair": name,
                "leader": f"{lsym}_{lsess}",
                "follower": f"{fsym}_{fsess}",
                "strategy": f"{em}/CB{cb}/RR{rr}",
                "condition": cname,
                "n_base": b["n"],
                "n_on": s_on["n"],
                "avg_base": b["avg_r"],
                "avg_on": s_on["avg_r"],
                "avg_off": s_off["avg_r"],
                "uplift_on_off": float(s_on["avg_r"] - s_off["avg_r"]),
                "wr_on": s_on["wr"],
                "wr_off": s_off["wr"],
                "years_pos": yp,
                "years_total": yt,
                "oos_year": oos_year,
                "train_uplift": tr_up,
                "test_uplift": te_up,
                "n_test_on": int(len(te_on)),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)
    parts = []
    for spec in PAIRS:
        p = eval_pair(con, spec)
        if not p.empty:
            parts.append(p)
    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "lead_lag_asym_strength_summary.csv"
    p_md = out_dir / "lead_lag_asym_strength_notes.md"

    if not parts:
        p_md.write_text("# Asym/Strength scan\n\nNo rows met thresholds.", encoding="utf-8")
        print("No rows met thresholds.")
        return 0

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["uplift_on_off", "avg_on"], ascending=False)
    df.to_csv(p_csv, index=False)

    lines = ["# Lead-Lag Asymmetric + Strength Scan", ""]
    for r in df.head(20).itertuples(index=False):
        lines.append(
            f"- {r.pair} {r.condition}: avg_on={r.avg_on:+.4f}, avg_off={r.avg_off:+.4f}, Δ={r.uplift_on_off:+.4f}, N_on={r.n_on}, years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
        )
    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
