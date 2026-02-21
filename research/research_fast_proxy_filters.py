#!/usr/bin/env python3
"""Fast proxy-filter scan (all symbols/sessions) with quick OOS check.

Purpose:
- Rapidly test practical, non-lookahead proxy filters on existing ORB outcomes.
- Keep scope broad (MGC/MES/MNQ; 0900/1000/1100), then rank by uplift + stability.

Default target strategy slice (editable via args): E1 / CB2 / RR2.5.

Outputs:
- research/output/fast_proxy_filter_summary.csv
- research/output/fast_proxy_filter_yearly.csv
- research/output/fast_proxy_filter_oos.csv
- research/output/fast_proxy_filter_notes.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _session_value(row: pd.Series, stem: str):
    return row[f"orb_{row['orb_label']}_{stem}"]


def _max_dd(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    c = s.cumsum()
    peak = c.cummax()
    dd = peak - c
    return float(dd.max())


def load_data(db_path: str, entry_model: str, confirm_bars: int, rr_target: float) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    q = f"""
    SELECT o.symbol,o.trading_day,o.orb_label,o.entry_model,o.confirm_bars,o.rr_target,o.pnl_r,
           d.orb_0900_size,d.orb_0900_break_delay_min,d.orb_0900_break_bar_continues,
           d.orb_1000_size,d.orb_1000_break_delay_min,d.orb_1000_break_bar_continues,
           d.orb_1100_size,d.orb_1100_break_delay_min,d.orb_1100_break_bar_continues
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.symbol=d.symbol AND o.trading_day=d.trading_day AND o.orb_minutes=d.orb_minutes
    WHERE o.orb_minutes=5
      AND o.pnl_r IS NOT NULL
      AND o.symbol IN ('MGC','MES','MNQ')
      AND o.orb_label IN ('0900','1000','1100')
      AND o.entry_model = '{entry_model}'
      AND o.confirm_bars = {confirm_bars}
      AND o.rr_target = {rr_target}
    """
    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        return df

    for stem in ["size", "break_delay_min", "break_bar_continues"]:
        df[stem] = df.apply(lambda r: _session_value(r, stem), axis=1)

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year

    # Exploratory global quartiles by symbol/session (fast check)
    df["size_q75"] = df.groupby(["symbol", "orb_label"])["size"].transform(lambda s: s.quantile(0.75))

    return df


def add_filters(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "one_way_proxy": (
            (df["break_bar_continues"] == True)
            & df["break_delay_min"].notna()
            & (df["break_delay_min"] <= 15)
            & df["size"].notna()
            & (df["size"] >= df["size_q75"])
        ),
        "fast_break_only": (
            df["break_delay_min"].notna() & (df["break_delay_min"] <= 15)
        ),
        "continue_only": (
            df["break_bar_continues"] == True
        ),
    }


def summarize(df: pd.DataFrame, filters: dict[str, pd.Series], min_on: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    yearly_rows = []

    for filt_name, mask in filters.items():
        for (sym, sess), g in df.groupby(["symbol", "orb_label"]):
            base = g
            on = g[mask.loc[g.index]]

            if len(base) < 200 or len(on) < min_on:
                continue

            base_avg = float(base["pnl_r"].mean())
            on_avg = float(on["pnl_r"].mean())

            rows.append({
                "filter": filt_name,
                "symbol": sym,
                "session": sess,
                "n_base": len(base),
                "n_on": len(on),
                "on_rate": len(on) / len(base),
                "avg_r_base": base_avg,
                "avg_r_on": on_avg,
                "uplift_avg_r": on_avg - base_avg,
                "wr_base": float((base["pnl_r"] > 0).mean()),
                "wr_on": float((on["pnl_r"] > 0).mean()),
                "wr_uplift": float((on["pnl_r"] > 0).mean() - (base["pnl_r"] > 0).mean()),
                "dd_base": _max_dd(base["pnl_r"]),
                "dd_on": _max_dd(on["pnl_r"]),
            })

            for y, gy in on.groupby("year"):
                base_y = base[base["year"] == y]
                if base_y.empty:
                    continue
                yearly_rows.append({
                    "filter": filt_name,
                    "symbol": sym,
                    "session": sess,
                    "year": int(y),
                    "n_on": len(gy),
                    "avg_r_on": float(gy["pnl_r"].mean()),
                    "avg_r_base": float(base_y["pnl_r"].mean()),
                    "uplift": float(gy["pnl_r"].mean() - base_y["pnl_r"].mean()),
                })

    s = pd.DataFrame(rows)
    y = pd.DataFrame(yearly_rows)
    if not s.empty:
        s = s.sort_values(["uplift_avg_r", "wr_uplift"], ascending=False)
    return s, y


def oos_check(df: pd.DataFrame, filters: dict[str, pd.Series], min_on: int) -> pd.DataFrame:
    rows = []

    years = sorted(df["year"].dropna().unique().tolist())
    if len(years) < 2:
        return pd.DataFrame()

    # Prefer latest "full-ish" year for OOS (avoid tiny partial current year)
    counts = df.groupby("year").size().to_dict()
    eligible = [y for y in years if counts.get(y, 0) >= 500]
    split_year = max(eligible) if eligible else years[-1]

    train = df[df["year"] < split_year].copy()
    test = df[df["year"] == split_year].copy()

    for filt_name, mask_all in filters.items():
        mask_train = mask_all.loc[train.index]
        mask_test = mask_all.loc[test.index]

        for (sym, sess), gtr in train.groupby(["symbol", "orb_label"]):
            gte = test[(test["symbol"] == sym) & (test["orb_label"] == sess)]
            if gte.empty:
                continue

            tr_on = gtr[mask_train.loc[gtr.index]]
            te_on = gte[mask_test.loc[gte.index]]

            if len(tr_on) < min_on or len(te_on) < max(10, min_on // 3):
                continue

            tr_off = gtr[~mask_train.loc[gtr.index]]
            te_off = gte[~mask_test.loc[gte.index]]
            if len(tr_off) == 0 or len(te_off) == 0:
                continue

            rows.append({
                "filter": filt_name,
                "symbol": sym,
                "session": sess,
                "train_years": f"{min(years)}-{split_year-1}",
                "test_year": int(split_year),
                "n_train_on": len(tr_on),
                "n_test_on": len(te_on),
                "train_uplift": float(tr_on["pnl_r"].mean() - tr_off["pnl_r"].mean()) if len(tr_off) else np.nan,
                "test_uplift": float(te_on["pnl_r"].mean() - te_off["pnl_r"].mean()) if len(te_off) else np.nan,
                "train_avg_on": float(tr_on["pnl_r"].mean()),
                "test_avg_on": float(te_on["pnl_r"].mean()),
                "test_wr_on": float((te_on["pnl_r"] > 0).mean()),
            })

    o = pd.DataFrame(rows)
    if not o.empty:
        o = o.sort_values(["test_uplift", "train_uplift"], ascending=False)
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast proxy filter scan")
    ap.add_argument("--db-path", default="gold.db")
    ap.add_argument("--entry-model", default="E1")
    ap.add_argument("--confirm-bars", type=int, default=2)
    ap.add_argument("--rr-target", type=float, default=2.5)
    ap.add_argument("--min-on", type=int, default=80)
    args = ap.parse_args()

    df = load_data(args.db_path, args.entry_model, args.confirm_bars, args.rr_target)
    if df.empty:
        print("No rows for requested strategy slice.")
        return 0

    filters = add_filters(df)
    s, y = summarize(df, filters, min_on=args.min_on)
    o = oos_check(df, filters, min_on=args.min_on)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_s = out_dir / "fast_proxy_filter_summary.csv"
    p_y = out_dir / "fast_proxy_filter_yearly.csv"
    p_o = out_dir / "fast_proxy_filter_oos.csv"
    p_m = out_dir / "fast_proxy_filter_notes.md"

    if not s.empty:
        s.to_csv(p_s, index=False)
    if not y.empty:
        y.to_csv(p_y, index=False)
    if not o.empty:
        o.to_csv(p_o, index=False)

    lines = []
    lines.append("# Fast Proxy Filter Scan")
    lines.append("")
    lines.append(f"- Slice: {args.entry_model} / CB{args.confirm_bars} / RR{args.rr_target}")
    lines.append(f"- min_on: {args.min_on}")
    lines.append("")

    if s.empty:
        lines.append("No summary rows met thresholds.")
    else:
        lines.append("## Top summary rows")
        for r in s.head(12).itertuples(index=False):
            lines.append(
                f"- {r.filter} | {r.symbol} {r.session}: N={r.n_on}/{r.n_base}, "
                f"avgR {r.avg_r_base:+.4f}->{r.avg_r_on:+.4f} (Δ={r.uplift_avg_r:+.4f}), "
                f"WR {r.wr_base:.1%}->{r.wr_on:.1%}"
            )

    if not o.empty:
        lines.append("")
        lines.append("## Quick OOS (last-year holdout)")
        for r in o.head(10).itertuples(index=False):
            lines.append(
                f"- {r.filter} | {r.symbol} {r.session}: train Δ={r.train_uplift:+.4f}, "
                f"test Δ={r.test_uplift:+.4f}, n_test_on={r.n_test_on}"
            )

    p_m.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_s}")
    print(f"Saved: {p_y}")
    print(f"Saved: {p_o}")
    print(f"Saved: {p_m}")

    if not s.empty:
        print("\nTop summary:")
        print(s.head(20).to_string(index=False))
    if not o.empty:
        print("\nTop OOS:")
        print(o.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
