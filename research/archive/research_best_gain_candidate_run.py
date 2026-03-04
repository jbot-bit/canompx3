#!/usr/bin/env python3
"""Run deep check on the current best gain-first lead-lag candidate.

Candidate chosen from oldstyle round2 top-gain table:
- Leader: M6E_US_EQUITY_OPEN
- Follower: MES_US_EQUITY_OPEN
- Strategy: E0 / CB1 / RR3.0

Condition ON:
- leader_break_dir == follower_break_dir
- leader_break_ts <= follower entry_ts  (no-lookahead)
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"


def stats(s: pd.Series):
    if s.empty:
        return {"n": 0, "wr": 0.0, "avg_r": 0.0, "total_r": 0.0}
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

    cond = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )

    base = df["pnl_r"]
    on = df.loc[cond, "pnl_r"]
    off = df.loc[~cond, "pnl_r"]

    s_base = stats(base)
    s_on = stats(on)
    s_off = stats(off)

    # yearly detail
    yearly_rows = []
    for y, g in df.groupby("year"):
        cy = cond.loc[g.index]
        oy = g.loc[cy, "pnl_r"]
        fy = g.loc[~cy, "pnl_r"]
        if len(oy) == 0 or len(fy) == 0:
            continue
        yearly_rows.append(
            {
                "year": int(y),
                "n_on": int(len(oy)),
                "n_off": int(len(fy)),
                "avg_on": float(oy.mean()),
                "avg_off": float(fy.mean()),
                "uplift": float(oy.mean() - fy.mean()),
            }
        )

    yearly = pd.DataFrame(yearly_rows)

    # OOS: train <=2024, test=2025
    tr = df[df["year"] <= 2024]
    te = df[df["year"] == 2025]
    ctr = cond.loc[tr.index]
    cte = cond.loc[te.index]
    tr_on = tr.loc[ctr, "pnl_r"]
    tr_off = tr.loc[~ctr, "pnl_r"]
    te_on = te.loc[cte, "pnl_r"]
    te_off = te.loc[~cte, "pnl_r"]

    train_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) and len(tr_off) else float("nan")
    test_up = float(te_on.mean() - te_off.mean()) if len(te_on) and len(te_off) else float("nan")

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_sum = out_dir / "best_gain_candidate_summary.csv"
    p_year = out_dir / "best_gain_candidate_yearly.csv"
    p_md = out_dir / "best_gain_candidate_report.md"

    summary = pd.DataFrame(
        [
            {"bucket": "baseline", **s_base},
            {"bucket": "on", **s_on},
            {"bucket": "off", **s_off},
            {
                "bucket": "uplift",
                "n": int(s_on["n"]),
                "wr": float(s_on["wr"] - s_off["wr"]),
                "avg_r": float(s_on["avg_r"] - s_off["avg_r"]),
                "total_r": float(s_on["total_r"] - s_off["total_r"]),
            },
        ]
    )
    summary.to_csv(p_sum, index=False)
    yearly.to_csv(p_year, index=False)

    lines = [
        "# Best Gain Candidate Run",
        "",
        "Candidate: M6E_US_EQUITY_OPEN -> MES_US_EQUITY_OPEN (E0/CB1/RR3.0)",
        "No-lookahead enforced: leader_break_ts <= follower entry_ts",
        "",
        f"Baseline: N={s_base['n']}, WR={s_base['wr']:.1%}, avgR={s_base['avg_r']:+.4f}",
        f"ON:       N={s_on['n']}, WR={s_on['wr']:.1%}, avgR={s_on['avg_r']:+.4f}",
        f"OFF:      N={s_off['n']}, WR={s_off['wr']:.1%}, avgR={s_off['avg_r']:+.4f}",
        f"Uplift ON-OFF avgR: {(s_on['avg_r'] - s_off['avg_r']):+.4f}",
        "",
        "## OOS",
        f"Train uplift (<=2024): {train_up:+.4f}",
        f"Test uplift (2025):    {test_up:+.4f}",
        f"N test ON/OFF: {len(te_on)}/{len(te_off)}",
    ]

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_year}")
    print(f"Saved: {p_md}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nYearly:")
    print(yearly.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
