#!/usr/bin/env python3
"""OOS validation for MNQ 0900 Dalton acceptance filter (no-lookahead).

Focuses the one surviving candidate from anchor-level uplift scan.

Outputs:
- research/output/dalton_mnq0900_oos_summary.csv
- research/output/dalton_mnq0900_oos_monthly.csv
- research/output/dalton_mnq0900_oos.md
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.research_dalton_filter_uplift import build_filter_flags


def max_dd(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    c = s.cumsum()
    p = c.cummax()
    d = p - c
    return float(d.max())


def stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": 0.0, "avg_r": 0.0, "total_r": 0.0, "max_dd": 0.0}
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
        "max_dd": max_dd(s),
    }


def main() -> int:
    con = duckdb.connect("gold.db", read_only=True)

    flags = build_filter_flags(con)
    outcomes = con.execute(
        """
        SELECT symbol, trading_day, orb_label AS anchor, pnl_r, entry_ts
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND pnl_r IS NOT NULL
          AND entry_ts IS NOT NULL
          AND symbol = 'MNQ'
          AND orb_label = '0900'
        """
    ).fetchdf()
    con.close()

    outcomes["entry_ts"] = pd.to_datetime(outcomes["entry_ts"], utc=True)
    df = outcomes.merge(flags, on=["symbol", "trading_day", "anchor"], how="inner")
    df = df[df["entry_ts"] >= df["gate_ts"]].copy()  # no-lookahead only

    if df.empty:
        print("No rows after no-lookahead gate.")
        return 0

    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    df["month"] = pd.to_datetime(df["trading_day"]).dt.to_period("M").astype(str)

    years = sorted(df["year"].unique().tolist())

    # Year holdout: first year train, remaining years test.
    train_year = years[0]
    train = df[df["year"] == train_year].copy()
    test = df[df["year"] > train_year].copy()

    train_on = train[train["dalton_accept"] == 1]["pnl_r"]
    train_off = train[train["dalton_accept"] == 0]["pnl_r"]
    test_on = test[test["dalton_accept"] == 1]["pnl_r"]
    test_off = test[test["dalton_accept"] == 0]["pnl_r"]

    train_uplift = float(train_on.mean() - train_off.mean()) if (len(train_on) and len(train_off)) else float("nan")
    test_uplift = float(test_on.mean() - test_off.mean()) if (len(test_on) and len(test_off)) else float("nan")

    # Monthly table
    m = (
        df.groupby(["month", "dalton_accept"], as_index=False)
        .agg(n=("pnl_r", "count"), avg_r=("pnl_r", "mean"), total_r=("pnl_r", "sum"), wr=("pnl_r", lambda s: (s > 0).mean()))
        .sort_values(["month", "dalton_accept"])
    )

    # Wide monthly uplift ON vs OFF
    mw = m.pivot_table(index="month", columns="dalton_accept", values="avg_r", aggfunc="first").reset_index()
    if 0 in mw.columns and 1 in mw.columns:
        mw["uplift_on_vs_off"] = mw[1] - mw[0]
    else:
        mw["uplift_on_vs_off"] = float("nan")

    # Save outputs
    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_sum = out_dir / "dalton_mnq0900_oos_summary.csv"
    p_mon = out_dir / "dalton_mnq0900_oos_monthly.csv"
    p_md = out_dir / "dalton_mnq0900_oos.md"

    summary_rows = []
    summary_rows.append({"segment": "all_on", **stats(df[df["dalton_accept"] == 1]["pnl_r"])})
    summary_rows.append({"segment": "all_off", **stats(df[df["dalton_accept"] == 0]["pnl_r"])})
    summary_rows.append({"segment": f"train_{train_year}_on", **stats(train_on)})
    summary_rows.append({"segment": f"train_{train_year}_off", **stats(train_off)})
    if not test.empty:
        test_label = f"test_{years[1]}_{years[-1]}"
        summary_rows.append({"segment": f"{test_label}_on", **stats(test_on)})
        summary_rows.append({"segment": f"{test_label}_off", **stats(test_off)})

    s = pd.DataFrame(summary_rows)
    s.to_csv(p_sum, index=False)
    mw.to_csv(p_mon, index=False)

    lines = []
    lines.append("# MNQ 0900 Dalton Filter OOS Check")
    lines.append("")
    lines.append("No-lookahead enforced: entry_ts >= A/B gate_ts.")
    lines.append("")
    lines.append(f"- Years in sample: {years}")
    lines.append(f"- Train year: {train_year}")
    lines.append(f"- Train uplift (ON-OFF avgR): {train_uplift:+.4f}")
    lines.append(f"- Test uplift (ON-OFF avgR): {test_uplift:+.4f}")
    lines.append("")

    all_on = stats(df[df["dalton_accept"] == 1]["pnl_r"])
    all_off = stats(df[df["dalton_accept"] == 0]["pnl_r"])
    lines.append("## Aggregate")
    lines.append(
        f"- ON: N={all_on['n']}, WR={all_on['wr']:.1%}, avgR={all_on['avg_r']:+.4f}, totalR={all_on['total_r']:+.2f}, maxDD={all_on['max_dd']:.2f}"
    )
    lines.append(
        f"- OFF: N={all_off['n']}, WR={all_off['wr']:.1%}, avgR={all_off['avg_r']:+.4f}, totalR={all_off['total_r']:+.2f}, maxDD={all_off['max_dd']:.2f}"
    )

    lines.append("")
    lines.append("## Verdict guide")
    lines.append("- KEEP if test uplift positive with adequate N and stable monthly behavior.")
    lines.append("- WATCH if positive but sparse/volatile.")
    lines.append("- KILL if test uplift flips negative.")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_mon}")
    print(f"Saved: {p_md}")
    print("\nSummary:")
    print(s.to_string(index=False))
    print("\nMonthly uplift:")
    print(mw.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
