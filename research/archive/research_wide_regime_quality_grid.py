#!/usr/bin/env python3
"""Wide regime+quality filter grid (breadth-first, high-frequency oriented).

Goal:
- Search beyond lead-lag with practical zero-lookahead filters.
- Keep only candidates meeting common-ground frequency and quality gates.

Slice:
- Symbols: M2K, MES, M6E
- Models: E0/E1
- RR: 1.5, 2.0, 2.5, 3.0
- orb_minutes=5
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"

SYMBOLS = ["M2K", "MES", "M6E"]
RR_LIST = [1.5, 2.0, 2.5, 3.0]

# Hard gates
MIN_BASE = 500
MIN_ON = 150
MIN_OFF = 150
MIN_SIGNALS_PER_YEAR = 150
MIN_AVG_ON = 0.0
MIN_UPLIFT = 0.10
MIN_YEARS_TOTAL = 3
MIN_YEARS_POS_RATIO = 0.6


def make_case(stem: str, labels: list[str]) -> str:
    parts = ["CASE o.orb_label"]
    for lbl in labels:
        parts.append(f" WHEN '{lbl}' THEN d.orb_{lbl}_{stem}")
    parts.append(" ELSE NULL END")
    return "\n".join(parts)


def load_data() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)

    labels_df = con.execute(
        """
        SELECT DISTINCT orb_label
        FROM orb_outcomes
        WHERE orb_minutes=5
          AND symbol IN ('M2K','MES','M6E')
        ORDER BY 1
        """
    ).fetchdf()
    labels = labels_df["orb_label"].tolist()

    c_break_dir = make_case("break_dir", labels)
    c_delay = make_case("break_delay_min", labels)
    c_cont = make_case("break_bar_continues", labels)
    c_size = make_case("size", labels)
    c_vol = make_case("volume", labels)
    c_bvol = make_case("break_bar_volume", labels)

    rr_csv = ",".join(str(x) for x in RR_LIST)

    q = f"""
    SELECT
      o.symbol,
      o.trading_day,
      o.orb_label,
      o.entry_model,
      o.confirm_bars,
      o.rr_target,
      o.pnl_r,
      d.atr_20,
      {c_break_dir} AS break_dir,
      {c_delay}     AS break_delay,
      {c_cont}      AS break_cont,
      {c_size}      AS orb_size,
      {c_vol}       AS orb_volume,
      {c_bvol}      AS break_bar_volume
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.symbol=o.symbol
     AND d.trading_day=o.trading_day
     AND d.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol IN ('M2K','MES','M6E')
      AND o.entry_model IN ('E0','E1')
      AND o.rr_target IN ({rr_csv})
      AND o.pnl_r IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year

    # Derived zero-lookahead proxies
    df["size_atr"] = np.where((df["atr_20"].notna()) & (df["atr_20"] > 0), df["orb_size"] / df["atr_20"], np.nan)
    df["vol_imp"] = np.where((df["orb_volume"].notna()) & (df["orb_volume"] > 0), df["break_bar_volume"] / (df["orb_volume"] / 5.0), np.nan)

    return df


def add_conditions(df: pd.DataFrame) -> pd.DataFrame:
    gss = df.groupby(["symbol", "orb_label"], dropna=False)

    df["size_q60"] = gss["size_atr"].transform(lambda s: s.quantile(0.60))
    df["vol_q60"] = gss["vol_imp"].transform(lambda s: s.quantile(0.60))

    gs = df.groupby(["symbol"], dropna=False)
    df["atr_q30"] = gs["atr_20"].transform(lambda s: s.quantile(0.30))
    df["atr_q70"] = gs["atr_20"].transform(lambda s: s.quantile(0.70))

    c_cont = (df["break_cont"] == True)
    c_delay30 = df["break_delay"].notna() & (df["break_delay"] <= 30)
    c_delay15 = df["break_delay"].notna() & (df["break_delay"] <= 15)
    c_size = df["size_atr"].notna() & (df["size_atr"] >= df["size_q60"])
    c_vol = df["vol_imp"].notna() & (df["vol_imp"] >= df["vol_q60"])
    c_atr_mid = df["atr_20"].notna() & (df["atr_20"] >= df["atr_q30"]) & (df["atr_20"] <= df["atr_q70"])
    c_atr_low = df["atr_20"].notna() & (df["atr_20"] < df["atr_q30"])

    df["cond_cont"] = c_cont
    df["cond_delay30"] = c_delay30
    df["cond_delay15"] = c_delay15
    df["cond_size"] = c_size
    df["cond_vol"] = c_vol
    df["cond_atr_mid"] = c_atr_mid
    df["cond_atr_low"] = c_atr_low

    df["F1_cont_delay30"] = c_cont & c_delay30
    df["F2_cont_delay15"] = c_cont & c_delay15
    df["F3_cont_delay30_size"] = c_cont & c_delay30 & c_size
    df["F4_cont_delay30_vol"] = c_cont & c_delay30 & c_vol
    df["F5_midatr_cont_delay30"] = c_atr_mid & c_cont & c_delay30
    df["F6_lowatr_cont_delay30"] = c_atr_low & c_cont & c_delay30
    df["F7_delay30_size_vol"] = c_delay30 & c_size & c_vol
    df["F8_delay15_size"] = c_delay15 & c_size

    return df


def yearly_pos_ratio(g: pd.DataFrame, mask: pd.Series) -> tuple[int, int]:
    yp, yt = 0, 0
    for _, gy in g.groupby("year"):
        my = mask.loc[gy.index]
        on = gy.loc[my, "pnl_r"]
        off = gy.loc[~my, "pnl_r"]
        if len(on) < 50 or len(off) < 50:
            continue
        yt += 1
        if on.mean() - off.mean() > 0:
            yp += 1
    return yp, yt


def oos_uplift(g: pd.DataFrame, mask: pd.Series) -> tuple[float, float, int]:
    # prefer 2025 holdout
    if 2025 in g["year"].unique():
        test_year = 2025
    else:
        test_year = int(g["year"].max())

    tr = g[g["year"] < test_year]
    te = g[g["year"] == test_year]
    if tr.empty or te.empty:
        return np.nan, np.nan, 0

    mtr = mask.loc[tr.index]
    mte = mask.loc[te.index]
    tr_on = tr.loc[mtr, "pnl_r"]
    tr_off = tr.loc[~mtr, "pnl_r"]
    te_on = te.loc[mte, "pnl_r"]
    te_off = te.loc[~mte, "pnl_r"]

    tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 100 and len(tr_off) >= 100 else np.nan
    te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 50 and len(te_off) >= 50 else np.nan
    return tr_up, te_up, int(len(te_on))


def scan(df: pd.DataFrame) -> pd.DataFrame:
    cond_cols = [c for c in df.columns if c.startswith("F")]
    rows = []

    for key, g in df.groupby(["symbol", "orb_label", "entry_model", "confirm_bars", "rr_target"]):
        n_base = len(g)
        if n_base < MIN_BASE:
            continue
        years_n = max(1, g["year"].nunique())

        for cname in cond_cols:
            m = g[cname]
            on = g.loc[m, "pnl_r"]
            off = g.loc[~m, "pnl_r"]
            n_on, n_off = len(on), len(off)
            if n_on < MIN_ON or n_off < MIN_OFF:
                continue

            avg_on = float(on.mean())
            avg_off = float(off.mean())
            uplift = avg_on - avg_off
            sig_yr = n_on / years_n

            yp, yt = yearly_pos_ratio(g, m)
            tr_up, te_up, n_test_on = oos_uplift(g, m)

            rows.append(
                {
                    "symbol": key[0],
                    "session": key[1],
                    "entry_model": key[2],
                    "confirm_bars": int(key[3]),
                    "rr_target": float(key[4]),
                    "filter": cname,
                    "n_base": n_base,
                    "n_on": n_on,
                    "signals_per_year": sig_yr,
                    "avg_on": avg_on,
                    "avg_off": avg_off,
                    "uplift": uplift,
                    "wr_on": float((on > 0).mean()),
                    "wr_off": float((off > 0).mean()),
                    "years_pos": yp,
                    "years_total": yt,
                    "years_pos_ratio": (yp / yt) if yt else np.nan,
                    "train_uplift": tr_up,
                    "test_uplift": te_up,
                    "n_test_on": n_test_on,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    df = load_data()
    if df.empty:
        print("No data rows.")
        return 0

    df = add_conditions(df)
    all_rows = scan(df)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_all = out_dir / "wide_regime_quality_all.csv"
    p_short = out_dir / "wide_regime_quality_shortlist.csv"
    p_md = out_dir / "wide_regime_quality_notes.md"

    if all_rows.empty:
        p_md.write_text("# Wide regime-quality grid\n\nNo rows met base thresholds.", encoding="utf-8")
        print("No rows met base thresholds.")
        return 0

    all_rows = all_rows.sort_values(["avg_on", "uplift"], ascending=False)
    all_rows.to_csv(p_all, index=False)

    short = all_rows[
        (all_rows["signals_per_year"] >= MIN_SIGNALS_PER_YEAR)
        & (all_rows["avg_on"] >= MIN_AVG_ON)
        & (all_rows["uplift"] >= MIN_UPLIFT)
        & (all_rows["years_total"] >= MIN_YEARS_TOTAL)
        & (all_rows["years_pos_ratio"] >= MIN_YEARS_POS_RATIO)
        & (all_rows["test_uplift"].fillna(-999) >= 0)
    ].copy().sort_values(["avg_on", "uplift", "signals_per_year"], ascending=False)
    short.to_csv(p_short, index=False)

    lines = [
        "# Wide Regime+Quality Grid",
        "",
        f"Total rows: {len(all_rows)}",
        f"Shortlist rows: {len(short)}",
        "",
        "## Top shortlist",
    ]

    if short.empty:
        lines.append("- None met hard gates.")
    else:
        for r in short.head(20).itertuples(index=False):
            lines.append(
                f"- {r.symbol} {r.session} {r.entry_model}/CB{r.confirm_bars}/RR{r.rr_target} | {r.filter}: avg_on={r.avg_on:+.4f}, Δ={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, years+={r.years_pos}/{r.years_total}, testΔ={r.test_uplift:+.4f}"
            )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_all}")
    print(f"Saved: {p_short}")
    print(f"Saved: {p_md}")
    print("\nTop shortlist:")
    print(short.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
