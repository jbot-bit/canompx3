#!/usr/bin/env python3
"""Fast lead-lag scan (cross-instrument, same-day session sequence).

Goal: quick edge checks, not exhaustive research.

Method:
- Follower trade universe from orb_outcomes (default E1/CB2/RR2.5, orb_minutes=5)
- Leader signal from daily_features break direction at an earlier session.
- Condition: leader break direction == follower break direction.
- Compare follower expectancy with condition ON vs baseline.
- Quick OOS: last full-ish year holdout.

Outputs:
- research/output/fast_lead_lag_summary.csv
- research/output/fast_lead_lag_oos.csv
- research/output/fast_lead_lag_notes.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SESS_ORDER = ["0900", "1000", "1100", "1800", "2300", "0030"]
SESS_IDX = {s: i for i, s in enumerate(SESS_ORDER)}
SYMS = ["MGC", "MES", "MNQ"]


def load_daily_features(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    q = """
    SELECT trading_day, symbol, orb_minutes,
           orb_0900_break_dir, orb_1000_break_dir, orb_1100_break_dir,
           orb_1800_break_dir, orb_2300_break_dir, orb_0030_break_dir
    FROM daily_features
    WHERE symbol IN ('MGC','MES','MNQ')
      AND orb_minutes = 5
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def load_outcomes(db_path: str, entry_model: str, confirm_bars: int, rr_target: float) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    q = f"""
    SELECT trading_day, symbol, orb_label, pnl_r
    FROM orb_outcomes
    WHERE orb_minutes = 5
      AND symbol IN ('MGC','MES','MNQ')
      AND orb_label IN ('0900','1000','1100','1800','2300','0030')
      AND pnl_r IS NOT NULL
      AND entry_model = '{entry_model}'
      AND confirm_bars = {confirm_bars}
      AND rr_target = {rr_target}
    """
    df = con.execute(q).fetchdf()
    con.close()
    return df


def get_break_col(sess: str) -> str:
    return f"orb_{sess}_break_dir"


def build_pairs() -> list[tuple[str, str, str, str]]:
    pairs = []
    for lsym in SYMS:
        for fsym in SYMS:
            for ls in SESS_ORDER:
                for fs in SESS_ORDER:
                    if SESS_IDX[ls] >= SESS_IDX[fs]:
                        continue
                    pairs.append((lsym, ls, fsym, fs))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast cross-instrument lead-lag scan")
    ap.add_argument("--db-path", default="gold.db")
    ap.add_argument("--entry-model", default="E1")
    ap.add_argument("--confirm-bars", type=int, default=2)
    ap.add_argument("--rr-target", type=float, default=2.5)
    ap.add_argument("--min-cond", type=int, default=80)
    args = ap.parse_args()

    df_feat = load_daily_features(args.db_path)
    df_out = load_outcomes(args.db_path, args.entry_model, args.confirm_bars, args.rr_target)

    if df_feat.empty or df_out.empty:
        print("No data rows for scan.")
        return 0

    # Build quick lookup tables for break_dir
    feat = df_feat[["trading_day", "symbol"] + [get_break_col(s) for s in SESS_ORDER]].copy()
    feat["trading_day"] = pd.to_datetime(feat["trading_day"])
    df_out["trading_day"] = pd.to_datetime(df_out["trading_day"])
    df_out["year"] = df_out["trading_day"].dt.year

    # Map follower break dir onto outcomes
    out = df_out.copy()
    out = out.merge(feat, on=["trading_day", "symbol"], how="left", suffixes=("", "_feat"))

    rows = []
    oos_rows = []

    years = sorted(out["year"].dropna().unique().tolist())
    if years:
        year_counts = out.groupby("year").size().to_dict()
        eligible = [y for y in years if year_counts.get(y, 0) >= 500]
        split_year = max(eligible) if eligible else years[-1]
    else:
        split_year = None

    for lsym, lsess, fsym, fsess in build_pairs():
        g = out[(out["symbol"] == fsym) & (out["orb_label"] == fsess)].copy()
        if len(g) < 200:
            continue

        # leader break series for same trading_day
        leader = feat[feat["symbol"] == lsym][["trading_day", get_break_col(lsess)]].rename(
            columns={get_break_col(lsess): "leader_dir"}
        )
        g = g.merge(leader, on="trading_day", how="left")

        # follower break dir from its own session col
        g["follower_dir"] = g[get_break_col(fsess)]

        cond = (
            g["leader_dir"].isin(["long", "short"])
            & g["follower_dir"].isin(["long", "short"])
            & (g["leader_dir"] == g["follower_dir"])
        )

        on = g[cond]
        off = g[~cond]
        if len(on) < args.min_cond or len(off) < args.min_cond:
            continue

        avg_base = float(g["pnl_r"].mean())
        avg_on = float(on["pnl_r"].mean())
        avg_off = float(off["pnl_r"].mean())

        rows.append(
            {
                "leader": f"{lsym}_{lsess}",
                "follower": f"{fsym}_{fsess}",
                "n_base": len(g),
                "n_on": len(on),
                "on_rate": len(on) / len(g),
                "avg_r_base": avg_base,
                "avg_r_on": avg_on,
                "avg_r_off": avg_off,
                "uplift_on_vs_base": avg_on - avg_base,
                "uplift_on_vs_off": avg_on - avg_off,
                "wr_base": float((g["pnl_r"] > 0).mean()),
                "wr_on": float((on["pnl_r"] > 0).mean()),
                "wr_off": float((off["pnl_r"] > 0).mean()),
            }
        )

        if split_year is not None:
            tr = g[g["year"] < split_year]
            te = g[g["year"] == split_year]
            if len(tr) == 0 or len(te) == 0:
                continue

            tr_cond = cond.loc[tr.index]
            te_cond = cond.loc[te.index]
            tr_on = tr[tr_cond]
            tr_off = tr[~tr_cond]
            te_on = te[te_cond]
            te_off = te[~te_cond]

            if len(tr_on) < args.min_cond or len(tr_off) < args.min_cond:
                continue
            if len(te_on) < max(20, args.min_cond // 3) or len(te_off) < max(20, args.min_cond // 3):
                continue

            oos_rows.append(
                {
                    "leader": f"{lsym}_{lsess}",
                    "follower": f"{fsym}_{fsess}",
                    "train_years": f"{min(years)}-{split_year-1}",
                    "test_year": int(split_year),
                    "n_train_on": len(tr_on),
                    "n_test_on": len(te_on),
                    "train_uplift": float(tr_on["pnl_r"].mean() - tr_off["pnl_r"].mean()),
                    "test_uplift": float(te_on["pnl_r"].mean() - te_off["pnl_r"].mean()),
                    "test_avg_on": float(te_on["pnl_r"].mean()),
                    "test_wr_on": float((te_on["pnl_r"] > 0).mean()),
                }
            )

    s = pd.DataFrame(rows)
    o = pd.DataFrame(oos_rows)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_s = out_dir / "fast_lead_lag_summary.csv"
    p_o = out_dir / "fast_lead_lag_oos.csv"
    p_m = out_dir / "fast_lead_lag_notes.md"

    if not s.empty:
        s = s.sort_values(["uplift_on_vs_off", "wr_on"], ascending=False)
        s.to_csv(p_s, index=False)
    if not o.empty:
        o = o.sort_values(["test_uplift", "train_uplift"], ascending=False)
        o.to_csv(p_o, index=False)

    lines = [
        "# Fast Lead-Lag Scan",
        "",
        f"- Slice: {args.entry_model}/CB{args.confirm_bars}/RR{args.rr_target}",
        f"- min_cond: {args.min_cond}",
        "- Condition: leader break direction == follower break direction",
        "",
    ]

    if s.empty:
        lines.append("No summary rows met thresholds.")
    else:
        lines.append("## Top summary")
        for r in s.head(12).itertuples(index=False):
            lines.append(
                f"- {r.leader} -> {r.follower}: N_on={r.n_on}/{r.n_base}, "
                f"avgR on/off {r.avg_r_on:+.4f}/{r.avg_r_off:+.4f}, "
                f"Δ(on-off)={r.uplift_on_vs_off:+.4f}, WR on/off {r.wr_on:.1%}/{r.wr_off:.1%}"
            )

    if not o.empty:
        lines.append("")
        lines.append("## Quick OOS")
        for r in o.head(12).itertuples(index=False):
            lines.append(
                f"- {r.leader} -> {r.follower}: train Δ={r.train_uplift:+.4f}, test Δ={r.test_uplift:+.4f}, n_test_on={r.n_test_on}"
            )

    p_m.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_s}")
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
