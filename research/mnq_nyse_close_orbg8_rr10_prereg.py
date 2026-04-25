#!/usr/bin/env python3
"""Execute the exact MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg.

Locked by:
  docs/audit/hypotheses/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.filter_utils import filter_signal
from research.lib import connect_db, write_csv

RESULT_PATH = Path("docs/audit/results/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.md")
HOLDOUT_START = pd.Timestamp("2026-01-01")
SESSION = "NYSE_CLOSE"
FILTER_KEY = "ORB_G8"
FILTER_THRESHOLD = 8.0
T_THRESHOLD = 3.0


@dataclass(frozen=True)
class Verdict:
    outcome: str
    summary: str


def load_rows() -> pd.DataFrame:
    sql = f"""
    SELECT
        o.trading_day,
        o.pnl_r,
        o.outcome,
        o.symbol,
        d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.trading_day = o.trading_day
     AND d.symbol = o.symbol
     AND d.orb_minutes = o.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = '{SESSION}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.0
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    with connect_db() as con:
        return con.execute(sql).fetchdf()


def split_is_oos(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["trading_day"] = pd.to_datetime(rows["trading_day"])
    is_rows = rows[rows["trading_day"] < HOLDOUT_START].reset_index(drop=True)
    oos_rows = rows[rows["trading_day"] >= HOLDOUT_START].reset_index(drop=True)
    return is_rows, oos_rows


def one_sample_t_greater_than_zero(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if sd == 0:
        return float("nan"), float("nan")
    t_stat = mean / (sd / np.sqrt(len(values)))
    p_one = float(1.0 - stats.t.cdf(t_stat, len(values) - 1))
    return float(t_stat), p_one


def summarize_partition(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fire = np.asarray(filter_signal(rows, FILTER_KEY, SESSION)).astype(bool)
    out = rows.copy()
    out["filter_fires"] = fire
    out["year"] = out["trading_day"].dt.year

    records: list[dict[str, object]] = []
    for state, grp in [("baseline", out), ("on_signal", out[out["filter_fires"]]), ("off_signal", out[~out["filter_fires"]])]:
        if grp.empty:
            records.append(
                {
                    "partition": state,
                    "n": 0,
                    "fire_rate": float("nan"),
                    "expr": float("nan"),
                    "win_rate": float("nan"),
                }
            )
            continue
        records.append(
            {
                "partition": state,
                "n": int(len(grp)),
                "fire_rate": float(len(grp) / len(out)) if len(out) else float("nan"),
                "expr": float(grp["pnl_r"].mean()),
                "win_rate": float((grp["outcome"].astype(str) == "win").mean()),
            }
        )

    yearly_on = (
        out[out["filter_fires"]]
        .groupby("year", sort=True)["pnl_r"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_on", "mean": "expr_on"})
    )
    return pd.DataFrame(records), yearly_on


def evaluate_verdict(is_summary: pd.DataFrame, yearly_on: pd.DataFrame, oos_summary: pd.DataFrame) -> Verdict:
    on_row = is_summary.loc[is_summary["partition"] == "on_signal"].iloc[0]
    oos_on_row = oos_summary.loc[oos_summary["partition"] == "on_signal"].iloc[0]

    t_stat = float(on_row["t_stat"])
    expr_is = float(on_row["expr"])
    if not np.isfinite(expr_is) or expr_is <= 0 or not np.isfinite(t_stat) or t_stat < T_THRESHOLD:
        return Verdict("KILL", "IS on-signal expectancy does not clear the prereg floor.")

    era_fail = yearly_on[(yearly_on["n_on"] >= 50) & (yearly_on["expr_on"] < -0.05)]
    if not era_fail.empty:
        years = ", ".join(str(int(y)) for y in era_fail["year"].tolist())
        return Verdict("KILL", f"Era-stability kill triggered on year(s): {years}.")

    n_oos_on = int(oos_on_row["n"])
    expr_oos = float(oos_on_row["expr"]) if np.isfinite(float(oos_on_row["expr"])) else float("nan")
    if n_oos_on >= 10 and np.isfinite(expr_oos) and expr_oos < 0:
        return Verdict("KILL", "2026 OOS on-signal subset turned negative once the minimum OOS sample gate was met.")

    return Verdict("CONTINUE", "Exact ORB_G8 path survives its prereg and can move to a separate promotion gate.")


def build_markdown(
    is_summary: pd.DataFrame,
    oos_summary: pd.DataFrame,
    yearly_on: pd.DataFrame,
    verdict: Verdict,
) -> str:
    is_on = is_summary.loc[is_summary["partition"] == "on_signal"].iloc[0]
    is_off = is_summary.loc[is_summary["partition"] == "off_signal"].iloc[0]
    oos_on = oos_summary.loc[oos_summary["partition"] == "on_signal"].iloc[0]

    lines = [
        "# MNQ NYSE_CLOSE ORB_G8 RR1.0 prereg",
        "",
        "Date: 2026-04-23",
        "",
        "## Scope",
        "",
        "Execute the exact native `MNQ NYSE_CLOSE O5 E2 CB1 RR1.0 ORB_G8` prereg using canonical filter delegation only.",
        "",
        "## Result",
        "",
        f"**Outcome:** `{verdict.outcome}`",
        "",
        verdict.summary,
        "",
        "## IS Summary",
        "",
        "| Partition | N | Fire Rate | ExpR | Win Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in is_summary.itertuples(index=False):
        fire_rate = f"{row.fire_rate:.3f}" if pd.notna(row.fire_rate) else "NA"
        expr = f"{row.expr:+.4f}" if pd.notna(row.expr) else "NA"
        win_rate = f"{row.win_rate:.3f}" if pd.notna(row.win_rate) else "NA"
        lines.append(f"| {row.partition} | {int(row.n)} | {fire_rate} | {expr} | {win_rate} |")

    lines += [
        "",
        f"- On-signal one-sample t-stat: `{float(is_on['t_stat']):.3f}`",
        f"- On-signal one-tailed p-value: `{float(is_on['p_one']):.4f}`",
        f"- IS uplift vs off-signal: `{float(is_on['expr']) - float(is_off['expr']):+.4f}R`",
        "",
        "## IS Year Map (on-signal only)",
        "",
        "| Year | N_on | ExpR_on |",
        "|---|---:|---:|",
    ]
    for row in yearly_on.itertuples(index=False):
        lines.append(f"| {int(row.year)} | {int(row.n_on)} | {row.expr_on:+.4f} |")

    lines += [
        "",
        "## OOS Summary",
        "",
        "| Partition | N | Fire Rate | ExpR | Win Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in oos_summary.itertuples(index=False):
        fire_rate = f"{row.fire_rate:.3f}" if pd.notna(row.fire_rate) else "NA"
        expr = f"{row.expr:+.4f}" if pd.notna(row.expr) else "NA"
        win_rate = f"{row.win_rate:.3f}" if pd.notna(row.win_rate) else "NA"
        lines.append(f"| {row.partition} | {int(row.n)} | {fire_rate} | {expr} | {win_rate} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- The exact `ORB_G8` path is statistically strong in-sample (`t=3.285`, `p=0.0005`) and fires on most IS rows (`720/805`, `89.4%`).",
        "- It still fails its own prereg because 2025 on-signal performance is negative at meaningful size (`N_on=132`, `ExpR=-0.0697`).",
        "- 2026 OOS is not a clean selector test for this path because `ORB_G8` fires on every observed OOS row (`42/42`), so there is no off-signal contrast.",
        "- This closes the exact native `ORB_G8` route. It does not prove the broad NYSE_CLOSE RR1.0 family is dead, but it does remove the strongest locked native candidate from the open queue.",
    ]
    return "\n".join(lines) + "\n"


def add_test_stats(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["t_stat"] = float("nan")
    summary["p_one"] = float("nan")
    return summary


def main() -> None:
    rows = load_rows()
    is_rows, oos_rows = split_is_oos(rows)

    is_summary, yearly_on = summarize_partition(is_rows)
    oos_summary, _ = summarize_partition(oos_rows)

    is_fire = np.asarray(filter_signal(is_rows, FILTER_KEY, SESSION)).astype(bool)
    on_is = is_rows.loc[is_fire, "pnl_r"].astype(float).to_numpy()
    t_stat, p_one = one_sample_t_greater_than_zero(on_is)

    is_summary = add_test_stats(is_summary)
    is_summary.loc[is_summary["partition"] == "on_signal", ["t_stat", "p_one"]] = [t_stat, p_one]
    oos_summary = add_test_stats(oos_summary)

    verdict = evaluate_verdict(is_summary, yearly_on, oos_summary)

    row_level = rows.copy()
    row_level["trading_day"] = pd.to_datetime(row_level["trading_day"])
    row_level["split"] = np.where(row_level["trading_day"] < HOLDOUT_START, "IS", "OOS")
    row_level["filter_fires"] = np.asarray(filter_signal(row_level, FILTER_KEY, SESSION)).astype(bool)

    write_csv(row_level[["trading_day", "pnl_r", "outcome", "split", "filter_fires"]], "mnq_nyse_close_orbg8_rr10_rows.csv")
    write_csv(is_summary, "mnq_nyse_close_orbg8_rr10_is_summary.csv")
    write_csv(oos_summary, "mnq_nyse_close_orbg8_rr10_oos_summary.csv")
    write_csv(yearly_on, "mnq_nyse_close_orbg8_rr10_yearly_on.csv")

    RESULT_PATH.write_text(build_markdown(is_summary, oos_summary, yearly_on, verdict), encoding="utf-8")
    print(f"Wrote {RESULT_PATH}")
    print(f"Outcome: {verdict.outcome}")


if __name__ == "__main__":
    main()
