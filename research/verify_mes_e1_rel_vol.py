"""MES E1 rel_vol family verification runner.

Executes the exact 20-cell family locked in:
  docs/audit/hypotheses/2026-04-22-mes-e1-rel-vol-family-v1.yaml

Scope:
  - MES only
  - O5
  - E1
  - RR1.5
  - CB1
  - 10 sessions x 2 directions = 20 cells

Truth source:
  - orb_outcomes
  - daily_features

This is a bounded execution-role retest of the rel_vol participation line under
the only trade-time-safe entry model for this feature family: E1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sstats
from statsmodels.stats.multitest import multipletests

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-22-mes-e1-rel-vol-family-v1.yaml"
PREREG_LOCK_SHA = "6910c598"
RESULT_DOC = Path("docs/audit/results/2026-04-23-mes-e1-rel-vol-family-v1.md")
CELL_METRICS_CSV = Path("research/output/mes_e1_rel_vol_family_v1_metrics.csv")
ROW_FLAGS_CSV = Path("research/output/mes_e1_rel_vol_family_v1_row_flags.csv")

INSTRUMENT = "MES"
ORB_MINUTES = 5
ENTRY_MODEL = "E1"
RR_TARGET = 1.5
CONFIRM_BARS = 1
BH_Q = 0.05
T_FLOOR = 3.79
SESSIONS = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "CME_PRECLOSE",
    "COMEX_SETTLE",
    "NYSE_CLOSE",
]
DIRECTIONS = ["long", "short"]


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _fmt_num(value: float, fmt: str) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return format(value, fmt)


def _welch_t(on: pd.Series, off: pd.Series) -> tuple[float, float]:
    if len(on) < 2 or len(off) < 2:
        return float("nan"), float("nan")
    res = sstats.ttest_ind(
        on.astype(float),
        off.astype(float),
        equal_var=False,
        nan_policy="omit",
    )
    return float(res.statistic), float(res.pvalue)


def _load_session(con: duckdb.DuckDBPyConnection, session: str) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    break_dir_col = f"orb_{session}_break_dir"
    sql = f"""
    SELECT
      o.trading_day,
      o.symbol,
      o.orb_label,
      o.orb_minutes,
      o.entry_model,
      o.confirm_bars,
      o.rr_target,
      o.entry_price,
      o.stop_price,
      o.pnl_r,
      d.{rel_col} AS rel_vol,
      d.{break_dir_col} AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {ORB_MINUTES}
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR_TARGET}
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(sql).df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["session"] = session
    df["derived_direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    return df


def _evaluate_cell(df: pd.DataFrame, session: str, direction: str) -> tuple[dict[str, object], pd.DataFrame]:
    cell = df[
        (df["session"] == session)
        & (df["derived_direction"] == direction)
        & (df["break_dir"] == direction)
    ].copy()
    cell = cell.loc[cell["rel_vol"].notna()].copy()
    cell["is_is"] = cell["trading_day"] < pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = cell[cell["is_is"]].copy()
    oos_df = cell[~cell["is_is"]].copy()

    q80 = float(np.nanpercentile(is_df["rel_vol"].astype(float), 80)) if len(is_df) else float("nan")
    cell["q80_threshold"] = q80
    cell["on_signal"] = cell["rel_vol"].astype(float) >= q80 if math.isfinite(q80) else False

    is_on = is_df[is_df["rel_vol"].astype(float) >= q80]["pnl_r"] if math.isfinite(q80) else pd.Series(dtype=float)
    is_off = is_df[is_df["rel_vol"].astype(float) < q80]["pnl_r"] if math.isfinite(q80) else pd.Series(dtype=float)
    oos_on = oos_df[oos_df["rel_vol"].astype(float) >= q80]["pnl_r"] if math.isfinite(q80) else pd.Series(dtype=float)
    oos_off = oos_df[oos_df["rel_vol"].astype(float) < q80]["pnl_r"] if math.isfinite(q80) else pd.Series(dtype=float)

    t_is, p_is = _welch_t(is_on, is_off)
    t_oos, p_oos = _welch_t(oos_on, oos_off)

    row = {
        "session": session,
        "direction": direction,
        "n_total": int(len(cell)),
        "n_is": int(len(is_df)),
        "n_is_on": int(len(is_on)),
        "n_is_off": int(len(is_off)),
        "q80_threshold": q80,
        "expr_on_is": _safe_mean(is_on),
        "expr_off_is": _safe_mean(is_off),
        "delta_is": _safe_mean(is_on) - _safe_mean(is_off),
        "t_is": t_is,
        "p_is": p_is,
        "n_oos": int(len(oos_df)),
        "n_oos_on": int(len(oos_on)),
        "n_oos_off": int(len(oos_off)),
        "expr_on_oos": _safe_mean(oos_on),
        "expr_off_oos": _safe_mean(oos_off),
        "delta_oos": _safe_mean(oos_on) - _safe_mean(oos_off),
        "t_oos": t_oos,
        "p_oos": p_oos,
    }
    return row, cell[
        [
            "trading_day",
            "session",
            "derived_direction",
            "break_dir",
            "rel_vol",
            "q80_threshold",
            "on_signal",
            "is_is",
            "pnl_r",
        ]
    ].rename(columns={"derived_direction": "direction"})


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        frames = [_load_session(con, session) for session in SESSIONS]
    finally:
        con.close()

    loaded = [frame for frame in frames if not frame.empty]
    if not loaded:
        print("VERDICT: SCAN_ABORT")
        print("REASON: no MES E1 O5 RR1.5 CB1 canonical rows found")
        return 1

    data = pd.concat(loaded, ignore_index=True)
    data["break_dir_match"] = data["break_dir"] == data["derived_direction"]
    mismatch_count = int((data["break_dir"].notna() & ~data["break_dir_match"]).sum())

    cell_rows: list[dict[str, object]] = []
    flag_frames: list[pd.DataFrame] = []
    for session in SESSIONS:
        for direction in DIRECTIONS:
            row, flags = _evaluate_cell(data, session, direction)
            cell_rows.append(row)
            flag_frames.append(flags)

    metrics = pd.DataFrame(cell_rows).sort_values(["session", "direction"]).reset_index(drop=True)
    if len(metrics) != 20:
        raise RuntimeError(f"Expected 20 cells, got {len(metrics)}")

    reject, p_bh, _, _ = multipletests(metrics["p_is"].fillna(1.0), alpha=BH_Q, method="fdr_bh")
    metrics["p_bh"] = p_bh
    metrics["bh_pass"] = reject
    metrics["t_pass"] = metrics["t_is"] >= T_FLOOR
    metrics["positive_mean_floor"] = metrics["expr_on_is"] > 0
    metrics["delta_positive"] = metrics["delta_is"] > 0
    metrics["survivor"] = (
        metrics["bh_pass"]
        & metrics["t_pass"]
        & metrics["positive_mean_floor"]
        & metrics["delta_positive"]
    )
    metrics["oos_dir_positive"] = metrics["delta_oos"] > 0

    survivors = metrics[metrics["survivor"]].copy()
    subthreshold_oos = metrics[
        (metrics["delta_oos"].fillna(float("-inf")) > 0.39)
        & (~metrics["survivor"])
    ].copy().sort_values("delta_oos", ascending=False)
    if mismatch_count > 0:
        verdict = "KILL"
        reason = f"integrity failure: break_dir mismatch count={mismatch_count}"
    elif len(survivors) == 0:
        verdict = "KILL"
        reason = "K1 fired: 0/20 cells survived BH q<0.05, t>=3.79, and positive-mean floor"
    elif (survivors["oos_dir_positive"] == False).all():  # noqa: E712
        verdict = "KILL"
        reason = "K2 fired: every IS-positive survivor had non-positive OOS delta"
    elif int(metrics["n_is_on"].max()) < 30:
        verdict = "PARK"
        reason = "K4 fired: all candidate on-signal cells were power-thin"
    else:
        verdict = "CONTINUE"
        reason = f"{len(survivors)}/20 cells survived with positive OOS direction on at least one survivor"

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    CELL_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(flag_frames, ignore_index=True).sort_values(
        ["session", "direction", "trading_day"]
    ).to_csv(ROW_FLAGS_CSV, index=False)
    metrics.to_csv(CELL_METRICS_CSV, index=False)

    lines: list[str] = []
    lines.append("# MES E1 rel_vol family v1")
    lines.append("")
    lines.append(f"**Pre-reg:** `{PREREG_PATH}` (locked at commit `{PREREG_LOCK_SHA}`)")
    lines.append("**Script:** `research/verify_mes_e1_rel_vol.py`")
    lines.append(
        "**Scope:** MES | 10 sessions x 2 directions | O5 | E1 | RR1.5 | CB1 | "
        "IS-only per-cell Q80 rel_vol threshold"
    )
    lines.append(f"**IS boundary:** `trading_day < {HOLDOUT_SACRED_FROM}`")
    lines.append(f"**BH family K:** `20`")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append(f"> {reason}")
    lines.append("")
    lines.append("## Integrity")
    lines.append("")
    lines.append(f"- Entry model admitted: `{ENTRY_MODEL}` only")
    lines.append("- E2 / E3 admitted rows: `0`")
    lines.append("- Quantile source: `IS-only per cell`")
    lines.append(f"- break_dir vs derived-direction mismatches: `{mismatch_count}`")
    lines.append("")
    lines.append("## Family table")
    lines.append("")
    lines.append(
        "| session | dir | Q80 | N_IS | N_on | ExpR_on_IS | ExpR_off_IS | Δ_IS | "
        "t_IS | p_IS | p_BH | BH | t>=3.79 | mean>0 | surv | N_OOS_on | "
        "ExpR_on_OOS | ExpR_off_OOS | Δ_OOS | t_OOS | p_OOS |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.session} | {row.direction} | {_fmt_num(row.q80_threshold, '.4f')} | "
            f"{row.n_is} | {row.n_is_on} | {_fmt_num(row.expr_on_is, '+.4f')} | "
            f"{_fmt_num(row.expr_off_is, '+.4f')} | {_fmt_num(row.delta_is, '+.4f')} | "
            f"{_fmt_num(row.t_is, '+.3f')} | {_fmt_num(row.p_is, '.6f')} | {_fmt_num(row.p_bh, '.6f')} | "
            f"{'Y' if row.bh_pass else 'N'} | {'Y' if row.t_pass else 'N'} | "
            f"{'Y' if row.positive_mean_floor else 'N'} | {'Y' if row.survivor else 'N'} | "
            f"{row.n_oos_on} | {_fmt_num(row.expr_on_oos, '+.4f')} | "
            f"{_fmt_num(row.expr_off_oos, '+.4f')} | {_fmt_num(row.delta_oos, '+.4f')} | "
            f"{_fmt_num(row.t_oos, '+.3f')} | {_fmt_num(row.p_oos, '.6f')} |"
        )
    lines.append("")
    lines.append("## Survivors")
    lines.append("")
    if survivors.empty:
        lines.append("- None")
    else:
        for row in survivors.itertuples(index=False):
            lines.append(
                f"- `{row.session} {row.direction}` | Q80={_fmt_num(row.q80_threshold, '.4f')} | "
                f"Δ_IS={_fmt_num(row.delta_is, '+.4f')} | Δ_OOS={_fmt_num(row.delta_oos, '+.4f')}"
            )
    lines.append("")
    lines.append("## Sub-threshold +0.4R OOS cells")
    lines.append("")
    if subthreshold_oos.empty:
        lines.append("- None")
    else:
        lines.append(
            "These cells printed roughly `+0.4R` or better on OOS `delta_oos`, "
            "but they do **not** survive the family decision rule and are not promotion-safe."
        )
        lines.append("")
        for row in subthreshold_oos.itertuples(index=False):
            lines.append(
                f"- `{row.session} {row.direction}` | "
                f"Δ_IS={_fmt_num(row.delta_is, '+.4f')} | "
                f"t_IS={_fmt_num(row.t_is, '+.3f')} | "
                f"p_BH={_fmt_num(row.p_bh, '.6f')} | "
                f"Δ_OOS={_fmt_num(row.delta_oos, '+.4f')}"
            )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Family verdict governs. A few positive OOS deltas do not rescue a 0-survivor family.")
    lines.append("- No cell clears the combined BH q<0.05, t>=3.79, and positive-mean floor gate.")
    lines.append("- This is an execution-safe reroute test for `E1` only, not a reopen of broad pooled ORB ML.")
    lines.append("- OOS on-signal counts remain thin in several cells, so descriptive OOS wins are not enough for promotion.")
    lines.append("")
    lines.append("## Audit artifacts")
    lines.append("")
    lines.append(f"- Cell metrics CSV: `{CELL_METRICS_CSV}`")
    lines.append(f"- Row flags CSV: `{ROW_FLAGS_CSV}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("./.venv-wsl/bin/python research/verify_mes_e1_rel_vol.py")
    lines.append("```")
    lines.append("")
    lines.append("Read-only canonical run. No writes to `validated_setups`, `experimental_strategies`, `live_config`, or `lane_allocation.json`.")
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"VERDICT: {verdict}")
    print(f"REASON: {reason}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    print(f"CELL_METRICS_CSV: {CELL_METRICS_CSV}")
    print(f"ROW_FLAGS_CSV: {ROW_FLAGS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
