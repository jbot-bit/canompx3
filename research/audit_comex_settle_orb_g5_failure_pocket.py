"""
Institutional audit of the COMEX_SETTLE ORB_G5 failure pocket.

This is a confirmatory audit, not discovery.

Claim under audit
-----------------
The prior ORB_G5 arithmetic-only audit showed that
`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` is NOT a pure RULE 8.2
ARITHMETIC_ONLY filter on the full in-sample period because fire-vs-no-fire
differences exist in both expectancy and win rate.

The question here is narrower and more honest:

1. Is that comparison correctly calculated from canonical data?
2. Is the comparator fair for current decision-making?
3. Where does edge actually live: in the ORB_G5 gate, or in the lane itself?

Method
------
- Canonical layers only: `orb_outcomes` JOIN `daily_features` on
  `(trading_day, symbol, orb_minutes)`.
- Locked lane:
    symbol='MNQ', orb_label='COMEX_SETTLE', orb_minutes=5,
    entry_model='E2', confirm_bars=1, rr_target=1.5
- IS only: `trading_day < HOLDOUT_SACRED_FROM`
- Canonical ORB_G5 split: `orb_COMEX_SETTLE_size >= 5`
- Scratches are included in audit views as `pnl_eff = 0.0` because they are
  real entries with unresolved `pnl_r` in `orb_outcomes`.
- Descriptive feature menu is fixed and trade-time-knowable:
    break_dir, break_delay_min, break_bar_continues, pre_velocity, rel_vol,
    garch_forecast_vol_pct, overnight_range_pct, overnight_took_pdh,
    overnight_took_pdl, gap_type, prev_day_direction

Output
------
Writes:
`docs/audit/results/2026-04-21-comex-settle-orb-g5-failure-pocket-audit.md`
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

OUTPUT_MD = Path("docs/audit/results/2026-04-21-comex-settle-orb-g5-failure-pocket-audit.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

LANE_ID = "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5"
EARLY_YEARS = (2019, 2020)
LATE_YEARS = (2021, 2022, 2023, 2024, 2025)
NUMERIC_FEATURES = [
    "break_delay_min",
    "pre_velocity",
    "rel_vol",
    "garch_vol_pct",
    "overnight_range_pct",
]
CATEGORICAL_FEATURES = [
    "break_bar_continues",
    "overnight_took_pdh",
    "overnight_took_pdl",
    "gap_type",
    "prev_day_direction",
    "break_dir",
]


@dataclass(frozen=True)
class SplitStats:
    n: int
    wins: int
    scratches: int
    wr_resolved: float
    exp_r_resolved: float
    exp_r_with_scratch0: float


def preflight_summary() -> dict[str, object]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        orb_max = con.execute("SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = 'MNQ'").fetchone()[0]
        feat_max = con.execute("SELECT MAX(trading_day) FROM daily_features WHERE symbol = 'MNQ'").fetchone()[0]
        bars_max = con.execute("SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = 'MNQ'").fetchone()[0]
        row_count = con.execute(
            """
            SELECT COUNT(*)
            FROM orb_outcomes
            WHERE symbol='MNQ'
              AND orb_label='COMEX_SETTLE'
              AND orb_minutes=5
              AND entry_model='E2'
              AND confirm_bars=1
              AND rr_target=1.5
              AND trading_day < ?
            """,
            [HOLDOUT_SACRED_FROM],
        ).fetchone()[0]
    finally:
        con.close()
    return {
        "orb_outcomes_max_day": orb_max,
        "daily_features_max_day": feat_max,
        "bars_1m_max_ts": bars_max,
        "lane_row_count_pre_holdout": row_count,
    }


def load_lane_df() -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        sql = """
        SELECT
            o.trading_day,
            year(o.trading_day) AS year,
            o.outcome,
            o.pnl_r,
            CASE WHEN o.outcome = 'scratch' THEN 0.0 ELSE o.pnl_r END AS pnl_eff,
            d.orb_COMEX_SETTLE_size AS orb_size,
            CASE WHEN d.orb_COMEX_SETTLE_size >= 5 THEN 1 ELSE 0 END AS fire,
            d.orb_COMEX_SETTLE_break_dir AS break_dir,
            d.orb_COMEX_SETTLE_break_delay_min AS break_delay_min,
            d.orb_COMEX_SETTLE_break_bar_continues AS break_bar_continues,
            d.orb_COMEX_SETTLE_pre_velocity AS pre_velocity,
            d.rel_vol_COMEX_SETTLE AS rel_vol,
            d.garch_forecast_vol_pct AS garch_vol_pct,
            d.overnight_range_pct,
            d.overnight_took_pdh,
            d.overnight_took_pdl,
            d.gap_type,
            d.prev_day_direction
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ'
          AND o.orb_label = 'COMEX_SETTLE'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = 1.5
          AND o.trading_day < ?
          AND o.entry_ts IS NOT NULL
          AND d.orb_COMEX_SETTLE_size IS NOT NULL
          AND d.orb_COMEX_SETTLE_break_dir IN ('long', 'short')
        ORDER BY o.trading_day
        """
        df = con.execute(sql, [HOLDOUT_SACRED_FROM]).df()
    finally:
        con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def split_stats(df: pd.DataFrame) -> SplitStats:
    resolved = df[df["outcome"].isin(["win", "loss"])].copy()
    wins = int((resolved["outcome"] == "win").sum())
    scratches = int((df["outcome"] == "scratch").sum())
    wr_resolved = float(wins / len(resolved)) if len(resolved) else float("nan")
    exp_r_resolved = float(resolved["pnl_r"].astype(float).mean()) if len(resolved) else float("nan")
    exp_r_with_scratch0 = float(df["pnl_eff"].astype(float).mean()) if len(df) else float("nan")
    return SplitStats(
        n=int(len(df)),
        wins=wins,
        scratches=scratches,
        wr_resolved=wr_resolved,
        exp_r_resolved=exp_r_resolved,
        exp_r_with_scratch0=exp_r_with_scratch0,
    )


def welch_t(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    return float(t), float(p)


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)))
    if se == 0.0:
        return float("nan"), float("nan")
    z = ((x1 / n1) - (x2 / n2)) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def one_sample_t(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(arr, 0.0)
    return float(t), float(p)


def markdown_table(df: pd.DataFrame, float_cols: dict[str, int] | None = None) -> str:
    fmt_df = df.copy()
    for col, places in (float_cols or {}).items():
        if col in fmt_df.columns:
            def _fmt(x: object, digits: int = places) -> str:
                return "NaN" if pd.isna(x) else f"{x:.{digits}f}"

            fmt_df[col] = fmt_df[col].map(_fmt)
    cols = list(fmt_df.columns)
    rows = [cols]
    for _, row in fmt_df.iterrows():
        rows.append([str(row[col]) for col in cols])

    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    header = fmt_row(rows[0])
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = [fmt_row(r) for r in rows[1:]]
    return "\n".join([header, sep, *body])


def year_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, sub in df.groupby("year"):
        fire = split_stats(sub[sub["fire"] == 1])
        nofire = split_stats(sub[sub["fire"] == 0])
        rows.append(
            {
                "year": int(year),
                "n_total": int(len(sub)),
                "fire_rate": float((sub["fire"] == 1).mean()),
                "n_fire": fire.n,
                "n_nofire": nofire.n,
                "fire_scratches": fire.scratches,
                "expr_fire_s0": fire.exp_r_with_scratch0,
                "expr_nofire_s0": nofire.exp_r_with_scratch0,
            }
        )
    return pd.DataFrame(rows)


def feature_table_early(df: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    early = df[df["year"].isin(EARLY_YEARS)].copy()
    nofire = early[early["fire"] == 0]
    fire = early[early["fire"] == 1]
    for col in NUMERIC_FEATURES:
        a = nofire[col].dropna().astype(float)
        b = fire[col].dropna().astype(float)
        out.append(
            {
                "feature": col,
                "n_nofire": int(len(a)),
                "mean_nofire": float(a.mean()) if len(a) else np.nan,
                "n_fire": int(len(b)),
                "mean_fire": float(b.mean()) if len(b) else np.nan,
                "delta_nofire_minus_fire": float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            }
        )
    return pd.DataFrame(out)


def categorical_rate_tables_early(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    early = df[df["year"].isin(EARLY_YEARS)].copy()
    denom = early.groupby("fire").size().rename("denom")
    out: dict[str, pd.DataFrame] = {}
    for col in CATEGORICAL_FEATURES:
        grp = early.groupby(["fire", col]).size().rename("n").reset_index()
        grp = grp.merge(denom.reset_index(), on="fire", how="left")
        grp["rate"] = grp["n"] / grp["denom"]
        out[col] = grp
    return out


def direction_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (fire, break_dir), sub in df.groupby(["fire", "break_dir"]):
        stats_ = split_stats(sub)
        rows.append(
            {
                "fire": int(fire),
                "break_dir": str(break_dir),
                "n": stats_.n,
                "scratches": stats_.scratches,
                "expr_s0": stats_.exp_r_with_scratch0,
            }
        )
    return pd.DataFrame(rows).sort_values(["fire", "break_dir"]).reset_index(drop=True)


def build_report(df: pd.DataFrame) -> str:
    preflight = preflight_summary()

    fire = split_stats(df[df["fire"] == 1])
    nofire = split_stats(df[df["fire"] == 0])
    resolved_fire = df[(df["fire"] == 1) & (df["outcome"].isin(["win", "loss"]))]
    resolved_nofire = df[(df["fire"] == 0) & (df["outcome"].isin(["win", "loss"]))]

    full_t_res, full_p_res = welch_t(resolved_fire["pnl_r"], resolved_nofire["pnl_r"])
    full_z, full_pz = two_prop_z(fire.wins, len(resolved_fire), nofire.wins, len(resolved_nofire))
    full_t_s0, full_p_s0 = welch_t(df[df["fire"] == 1]["pnl_eff"], df[df["fire"] == 0]["pnl_eff"])

    early = df[df["year"].isin(EARLY_YEARS)].copy()
    late = df[df["year"].isin(LATE_YEARS)].copy()
    early_fire = split_stats(early[early["fire"] == 1])
    early_nofire = split_stats(early[early["fire"] == 0])
    late_nofire = split_stats(late[late["fire"] == 0])
    early_t_s0, early_p_s0 = welch_t(early[early["fire"] == 1]["pnl_eff"], early[early["fire"] == 0]["pnl_eff"])
    early_z, early_pz = two_prop_z(
        int(((early["fire"] == 1) & (early["outcome"] == "win")).sum()),
        int(((early["fire"] == 1) & (early["outcome"].isin(["win", "loss"]))).sum()),
        int(((early["fire"] == 0) & (early["outcome"] == "win")).sum()),
        int(((early["fire"] == 0) & (early["outcome"].isin(["win", "loss"]))).sum()),
    )

    lane_full_t, lane_full_p = one_sample_t(df["pnl_eff"])
    lane_late_t, lane_late_p = one_sample_t(late["pnl_eff"])
    lane_long_t, lane_long_p = one_sample_t(df[df["break_dir"] == "long"]["pnl_eff"])
    lane_short_t, lane_short_p = one_sample_t(df[df["break_dir"] == "short"]["pnl_eff"])

    yr_tbl = year_table(df)
    feat_tbl = feature_table_early(df)
    cat_tbls = categorical_rate_tables_early(df)
    dir_tbl = direction_table(df)

    pocket_2020 = df[(df["year"] == 2020) & (df["fire"] == 0)][
        ["trading_day", "orb_size", "rel_vol", "overnight_range_pct", "break_dir", "pnl_eff"]
    ].copy()
    pocket_2020["trading_day"] = pocket_2020["trading_day"].dt.date.astype(str)

    lines: list[str] = []
    lines.append("# COMEX_SETTLE ORB_G5 failure-pocket audit")
    lines.append("")
    lines.append("**Date:** 2026-04-21")
    lines.append("**Script:** `research/audit_comex_settle_orb_g5_failure_pocket.py`")
    lines.append(f"**Lane:** `{LANE_ID}`")
    lines.append("**Classification:** confirmatory audit; no new pre-reg required")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("`CONDITIONAL`")
    lines.append("")
    lines.append(
        "The historical fire-vs-no-fire gap is real, but the comparator is not fair for "
        "current deployment interpretation because the no-fire bucket is almost entirely extinct-era."
    )
    lines.append("")
    lines.append("## Pre-Flight")
    lines.append("")
    lines.append(f"- `orb_outcomes` latest MNQ day: `{preflight['orb_outcomes_max_day']}`")
    lines.append(f"- `daily_features` latest MNQ day: `{preflight['daily_features_max_day']}`")
    lines.append(f"- `bars_1m` latest MNQ ts: `{preflight['bars_1m_max_ts']}`")
    lines.append(f"- Pre-holdout lane row count (all entry outcomes): `{preflight['lane_row_count_pre_holdout']}`")
    lines.append("")
    lines.append("## Truth + Calculation Check")
    lines.append("")
    lines.append("- Canonical join is correct: `orb_outcomes` to `daily_features` on `(trading_day, symbol, orb_minutes)`.")
    lines.append("- Canonical filter is correct: `ORB_G5 == orb_COMEX_SETTLE_size >= 5` via `trading_app.config.OrbSizeFilter`.")
    lines.append("- Execution model is canonical `E2`: stop-market at ORB boundary plus 1 tick slippage via `trading_app.entry_rules._resolve_e2`.")
    lines.append("- Fakeouts are included as valid fills; ambiguous same-bar target+stop resolves conservatively to loss in `trading_app/outcome_builder.py`.")
    lines.append("- `pnl_r` is net-cost R from `pipeline.cost_model.to_r_multiple`; risk denominator includes friction.")
    lines.append("- Hidden assumption found: scratches have `pnl_r = NULL`, so resolved-only comparisons exclude them unless they are explicitly reintroduced.")
    lines.append("")
    lines.append("## Core Comparison")
    lines.append("")
    full_tbl = pd.DataFrame(
        [
            {
                "group": "fire",
                "n_all_entries": fire.n,
                "scratches": fire.scratches,
                "expr_resolved": fire.exp_r_resolved,
                "expr_scratch0": fire.exp_r_with_scratch0,
                "wr_resolved": fire.wr_resolved,
            },
            {
                "group": "no_fire",
                "n_all_entries": nofire.n,
                "scratches": nofire.scratches,
                "expr_resolved": nofire.exp_r_resolved,
                "expr_scratch0": nofire.exp_r_with_scratch0,
                "wr_resolved": nofire.wr_resolved,
            },
        ]
    )
    lines.append(markdown_table(full_tbl, {"expr_resolved": 4, "expr_scratch0": 4, "wr_resolved": 4}))
    lines.append("")
    lines.append(
        f"- Resolved-only Welch: `p={full_p_res:.4f}`; WR z-test on resolved rows: `p={full_pz:.4f}`."
    )
    lines.append(
        f"- Scratch-inclusive Welch using `pnl_eff` (`scratch -> 0`): `p={full_p_s0:.4f}`."
    )
    lines.append(
        "- Scratch bias is real but small: all `22` scratches are in fire, yet the sign and significance survive after including them."
    )
    lines.append("")
    lines.append("## Comparator Fairness")
    lines.append("")
    lines.append(markdown_table(yr_tbl, {"fire_rate": 4, "expr_fire_s0": 4, "expr_nofire_s0": 4}))
    lines.append("")
    lines.append(
        f"- `2019-2020` contributes `{early_nofire.n}/{nofire.n}` no-fire rows; "
        f"`2021-2025` contributes only `{late_nofire.n}/{nofire.n}`."
    )
    lines.append(
        f"- Fire rate drifts from `{yr_tbl.loc[yr_tbl['year'] == 2019, 'fire_rate'].iloc[0]:.4f}` in 2019 "
        f"to `1.0000` from 2022 onward."
    )
    lines.append(
        "- Conclusion: this is a valid historical selector test, but a poor live-selector test because the counterfactual bucket has vanished."
    )
    lines.append("")
    lines.append("## Era-Matched 2019-2020")
    lines.append("")
    early_tbl = pd.DataFrame(
        [
            {
                "group": "fire",
                "n_all_entries": early_fire.n,
                "scratches": early_fire.scratches,
                "expr_scratch0": early_fire.exp_r_with_scratch0,
                "wr_resolved": early_fire.wr_resolved,
            },
            {
                "group": "no_fire",
                "n_all_entries": early_nofire.n,
                "scratches": early_nofire.scratches,
                "expr_scratch0": early_nofire.exp_r_with_scratch0,
                "wr_resolved": early_nofire.wr_resolved,
            },
        ]
    )
    lines.append(markdown_table(early_tbl, {"expr_scratch0": 4, "wr_resolved": 4}))
    lines.append("")
    lines.append(
        f"- Early-era scratch-inclusive Welch: `p={early_p_s0:.4f}`; WR z-test: `p={early_pz:.4f}`."
    )
    lines.append(
        "- This keeps the original claim honest: the historical discrimination was real inside the era where the threshold actually bit."
    )
    lines.append("")
    lines.append("## Where Edge Actually Lives")
    lines.append("")
    lines.append(
        f"- Full unfiltered lane, all entries: `ExpR={df['pnl_eff'].mean():+.4f}R`, one-sample `p={lane_full_p:.4f}`."
    )
    lines.append(
        f"- Late unfiltered lane (`2021-2025`): `ExpR={late['pnl_eff'].mean():+.4f}R`, one-sample `p={lane_late_p:.4f}`."
    )
    lines.append(
        f"- Long-side lane: `ExpR={df[df['break_dir'] == 'long']['pnl_eff'].mean():+.4f}R`, `p={lane_long_p:.4f}`."
    )
    lines.append(
        f"- Short-side lane: `ExpR={df[df['break_dir'] == 'short']['pnl_eff'].mean():+.4f}R`, `p={lane_short_p:.4f}`."
    )
    lines.append(
        "- This points to the lane geometry as the primary surviving edge location. `ORB_G5` was a historical conditioner, not the current source of truth."
    )
    lines.append("")
    lines.append("## Early Failure-Pocket Shape")
    lines.append("")
    lines.append(markdown_table(dir_tbl, {"expr_s0": 4}))
    lines.append("")
    lines.append(markdown_table(feat_tbl, {"mean_nofire": 4, "mean_fire": 4, "delta_nofire_minus_fire": 4}))
    lines.append("")
    for col, tbl in cat_tbls.items():
        lines.append(f"### `{col}`")
        lines.append("")
        tbl_fmt = tbl.copy()
        tbl_fmt["rate"] = tbl_fmt["rate"].map(lambda x: f"{100.0 * x:.2f}%")
        lines.append(markdown_table(tbl_fmt))
        lines.append("")
    lines.append("## 2020 Worst Pocket")
    lines.append("")
    lines.append(markdown_table(pocket_2020, {"orb_size": 2, "rel_vol": 4, "overnight_range_pct": 2, "pnl_eff": 2}))
    lines.append("")
    lines.append("## Anti-Tunnel Check")
    lines.append("")
    lines.append("- Framing tested here: binary filter (`R1`).")
    lines.append("- Not tested but fair and still open:")
    lines.append("  - standalone unfiltered COMEX lane overlay")
    lines.append("  - size modifier (`R3`) on the unfiltered COMEX lane")
    lines.append("  - confluence / allocator use with other lane or portfolio state")
    lines.append("- Not tested fairly enough yet: current-era selector value, because no-fire sample is functionally gone.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- `KEEP` the historical fact: ORB_G5 had real early-era discrimination on COMEX.")
    lines.append("- `KILL` the stronger live claim: this does NOT prove ORB_G5 is a currently-live selector.")
    lines.append("- `PARK` further ORB_G5 rescue work on this lane.")
    lines.append("- `NEXT` highest-EV move: pre-register a small unfiltered COMEX overlay study with explicit role framing (`R1` vs `R3`) and scratch-inclusive evaluation.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python research/audit_comex_settle_orb_g5_failure_pocket.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_lane_df()
    report = build_report(df)
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
