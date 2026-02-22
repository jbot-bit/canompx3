#!/usr/bin/env python3
"""A0 final validation pack (strict one-way, no-threshold leakage).

A0 definition:
- Leader: M6E_US_EQUITY_OPEN
- Follower: MES_US_EQUITY_OPEN
- Strategy: E0 / CB1 / RR3.0
- Base condition: same direction + no-lookahead (leader_ts <= follower entry_ts)
- Overlay preset: base_plus_both = follower fast15 + follower vol_imp >= train q60

This script performs expanding yearly walk-forward where train-derived q60 is applied to next-year test.
No test-year information is used to set thresholds.

Outputs:
- research/output/a0_final_validation_splits.csv
- research/output/a0_final_validation_summary.csv
- research/output/a0_final_validation_report.md
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"


def load_a0() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    q = """
    SELECT o.trading_day,o.pnl_r,o.entry_ts,
           df_f.orb_US_EQUITY_OPEN_break_dir AS f_dir,
           df_f.orb_US_EQUITY_OPEN_break_delay_min AS f_delay,
           df_f.orb_US_EQUITY_OPEN_break_bar_continues AS f_cont,
           df_f.orb_US_EQUITY_OPEN_size AS f_size,
           df_f.orb_US_EQUITY_OPEN_volume AS f_vol,
           df_f.orb_US_EQUITY_OPEN_break_bar_volume AS f_bvol,
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
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)
    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)

    # frozen base condition
    base = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )

    return df[base].copy().sort_values(["trading_day", "entry_ts"])


def max_dd(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    c = series.cumsum()
    p = c.cummax()
    d = p - c
    return float(d.max())


def main() -> int:
    df = load_a0()
    out_dir = ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_splits = out_dir / "a0_final_validation_splits.csv"
    p_summary = out_dir / "a0_final_validation_summary.csv"
    p_report = out_dir / "a0_final_validation_report.md"

    if df.empty:
        p_report.write_text("# A0 final validation\n\nNo rows.", encoding="utf-8")
        print("No rows.")
        return 0

    years = sorted(int(y) for y in df["year"].unique())
    split_rows = []

    test_selected = []
    test_base = []

    for y in years:
        tr = df[df["year"] < y].copy()
        te = df[df["year"] == y].copy()
        if tr.empty or te.empty:
            continue

        # train-only threshold (no leakage)
        q60 = tr["f_vol_imp"].quantile(0.60)

        tr_mask = tr["f_delay"].notna() & (tr["f_delay"] <= 15) & tr["f_vol_imp"].notna() & (tr["f_vol_imp"] >= q60)
        te_mask = te["f_delay"].notna() & (te["f_delay"] <= 15) & te["f_vol_imp"].notna() & (te["f_vol_imp"] >= q60)

        tr_on = tr.loc[tr_mask, "pnl_r"]
        te_on = te.loc[te_mask, "pnl_r"]
        tr_base = tr["pnl_r"]
        te_base_year = te["pnl_r"]

        if len(te_on) == 0:
            continue

        split_rows.append(
            {
                "test_year": y,
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "train_q60_vol_imp": float(q60) if pd.notna(q60) else np.nan,
                "train_n_on": int(len(tr_on)),
                "test_n_on": int(len(te_on)),
                "train_avg_base": float(tr_base.mean()),
                "train_avg_on": float(tr_on.mean()) if len(tr_on) else np.nan,
                "train_uplift": float(tr_on.mean() - tr_base.mean()) if len(tr_on) else np.nan,
                "test_avg_base": float(te_base_year.mean()),
                "test_avg_on": float(te_on.mean()),
                "test_uplift": float(te_on.mean() - te_base_year.mean()),
                "test_wr_on": float((te_on > 0).mean()),
            }
        )

        test_selected.append(te_on)
        test_base.append(te_base_year)

    splits = pd.DataFrame(split_rows)
    if splits.empty:
        p_report.write_text("# A0 final validation\n\nNo valid walk-forward splits.", encoding="utf-8")
        print("No valid walk-forward splits.")
        return 0

    all_sel = pd.concat(test_selected, ignore_index=True)
    all_base = pd.concat(test_base, ignore_index=True)

    wf_years = len(splits)
    pos_avg_ratio = float((splits["test_avg_on"] > 0).mean())
    pos_uplift_ratio = float((splits["test_uplift"] > 0).mean())
    median_test_avg = float(splits["test_avg_on"].median())
    worst_test_avg = float(splits["test_avg_on"].min())

    avg_sel = float(all_sel.mean())
    avg_base = float(all_base.mean())
    uplift = avg_sel - avg_base
    wr_sel = float((all_sel > 0).mean())

    dd_sel = max_dd(all_sel)
    dd_base = max_dd(all_base)

    # stress
    slip005 = avg_sel - 0.05
    slip010 = avg_sel - 0.10
    trim5 = float(all_sel.sort_values().iloc[: max(1, len(all_sel)-max(1, int(round(len(all_sel)*0.05))))].mean())

    # strict verdict
    promote = (
        wf_years >= 3
        and pos_avg_ratio >= 0.67
        and pos_uplift_ratio >= 0.67
        and median_test_avg > 0
        and worst_test_avg > -0.10
        and slip005 > 0
        and trim5 > 0
    )

    verdict = "PROMOTE" if promote else "KILL"

    summary = pd.DataFrame([
        {
            "strategy": "A0 M6E_USEO -> MES_USEO E0/CB1/RR3.0 base_plus_both",
            "wf_years": wf_years,
            "n_test_selected_total": int(len(all_sel)),
            "avg_selected": avg_sel,
            "avg_base": avg_base,
            "uplift": uplift,
            "wr_selected": wr_sel,
            "pos_avg_ratio": pos_avg_ratio,
            "pos_uplift_ratio": pos_uplift_ratio,
            "median_test_avg": median_test_avg,
            "worst_test_avg": worst_test_avg,
            "dd_selected": dd_sel,
            "dd_base": dd_base,
            "stress_slip_0_05": slip005,
            "stress_slip_0_10": slip010,
            "stress_trim_top5_avg": trim5,
            "verdict": verdict,
        }
    ])

    splits.to_csv(p_splits, index=False)
    summary.to_csv(p_summary, index=False)

    lines = [
        "# A0 Final Validation Pack",
        "",
        "Walk-forward was run with train-only q60 threshold per split (no test leakage).",
        "",
        f"WF years: {wf_years}",
        f"Test selected N: {len(all_sel)}",
        f"avg_selected: {avg_sel:+.4f}",
        f"avg_base: {avg_base:+.4f}",
        f"uplift: {uplift:+.4f}",
        f"wr_selected: {wr_sel:.1%}",
        f"pos_avg_ratio: {pos_avg_ratio:.2f}",
        f"pos_uplift_ratio: {pos_uplift_ratio:.2f}",
        f"median_test_avg: {median_test_avg:+.4f}",
        f"worst_test_avg: {worst_test_avg:+.4f}",
        f"stress_slip_0_05: {slip005:+.4f}",
        f"stress_trim_top5_avg: {trim5:+.4f}",
        "",
        f"Verdict: {verdict}",
    ]

    p_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_splits}")
    print(f"Saved: {p_summary}")
    print(f"Saved: {p_report}")
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
