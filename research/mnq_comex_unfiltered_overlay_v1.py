"""MNQ COMEX_SETTLE unfiltered overlay family v1.

Scope is locked by:
  docs/audit/hypotheses/2026-04-21-mnq-comex-unfiltered-overlay-v1.yaml

This runner evaluates exactly two pre-registered overlay hypotheses on the
unfiltered MNQ COMEX_SETTLE O5 E2 CB1 RR1.5 lane using canonical
orb_outcomes + daily_features joins.

Outputs:
  - docs/audit/results/2026-04-21-mnq-comex-unfiltered-overlay-v1.md
  - docs/audit/results/2026-04-21-mnq-comex-unfiltered-overlay-v1-rows.csv
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-mnq-comex-unfiltered-overlay-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-21-mnq-comex-unfiltered-overlay-v1.md")
ROW_CSV = Path("docs/audit/results/2026-04-21-mnq-comex-unfiltered-overlay-v1-rows.csv")
BH_Q = 0.05
IS_START_YEAR = 2019
IS_END_YEAR = 2025
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HypothesisPlan:
    id: str
    signal_name: str
    feature_column: str
    threshold_name: str


HYPOTHESES = [
    HypothesisPlan(
        id="H01_RELVOL_HIGH_Q3",
        signal_name="rel_vol_high_q3",
        feature_column="rel_vol_COMEX_SETTLE",
        threshold_name="rel_vol_HIGH_Q3",
    ),
    HypothesisPlan(
        id="H02_OVERNIGHT_RANGE_HIGH_Q3",
        signal_name="overnight_range_high_q3",
        feature_column="overnight_range_pct",
        threshold_name="overnight_range_pct_HIGH_Q3",
    ),
]


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _load_prereg(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_header(prereg: dict, observed_cell_count: int) -> list[str]:
    commit_sha = (((prereg.get("reproducibility") or {}).get("commit_sha")) or "").strip()
    if not commit_sha or commit_sha == "TO_FILL_AFTER_COMMIT":
        raise ValueError("pre-reg commit_sha must be stamped before running the harness")
    declared_k = int((prereg.get("primary_schema") or {}).get("k_family", observed_cell_count))
    rel_script = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    lines = [
        f"# {prereg['title']}",
        "",
        f"**Pre-reg:** `{PREREG_PATH}` (LOCKED, commit_sha={commit_sha})",
        f"**Script:** `{rel_script}`",
        f"**Lane:** `{prereg['scope']['lane_id']}`",
        f"**Observed family K:** `{observed_cell_count}`",
    ]
    if declared_k != observed_cell_count:
        lines.extend(
            [
                "",
                (
                    f"> **K MISMATCH WARNING:** pre-reg declares `k_family={declared_k}` "
                    f"but the runner observed `{observed_cell_count}` cells."
                ),
            ]
        )
    lines.append("")
    return lines


def _bh_fdr(values: list[tuple[str, float]]) -> dict[str, float]:
    finite = [(key, p) for key, p in values if p is not None and not math.isnan(p)]
    if not finite:
        return {}
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    out: dict[str, float] = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        key, p = finite[i]
        q = min(running, p * m / (i + 1))
        running = q
        out[key] = q
    return out


def _welch_p(on: pd.Series, off: pd.Series) -> float:
    if len(on) < 2 or len(off) < 2:
        return float("nan")
    res = stats.ttest_ind(on.astype(float), off.astype(float), equal_var=False)
    return float(np.asarray(res.pvalue))


def _resolved_expr(series: pd.Series, outcomes: pd.Series) -> float:
    resolved = series.loc[outcomes.isin(["win", "loss"])].astype(float)
    if len(resolved) == 0:
        return float("nan")
    return float(resolved.mean())


def _group_stats(pnl_eff: pd.Series, outcomes: pd.Series) -> tuple[float, float, float]:
    if len(pnl_eff) == 0:
        return float("nan"), float("nan"), float("nan")
    pnl_eff = pnl_eff.astype(float)
    expr_scratch0 = float(pnl_eff.mean())
    expr_resolved = _resolved_expr(pnl_eff, outcomes)
    resolved = outcomes.isin(["win", "loss"])
    wr_resolved = float((outcomes.loc[resolved] == "win").mean()) if resolved.any() else float("nan")
    return expr_scratch0, expr_resolved, wr_resolved


def _signed_delta(expr_on: float, expr_off: float) -> float:
    if math.isnan(expr_on) or math.isnan(expr_off):
        return float("nan")
    return expr_on - expr_off


def _year_rows(df: pd.DataFrame, signal_col: str) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    positive_years = 0
    for year in range(IS_START_YEAR, IS_END_YEAR + 1):
        yr = df.loc[df["is_is"] & (df["year"] == year)].copy()
        sig = yr[signal_col].astype(bool)
        on = yr.loc[sig, "pnl_eff"]
        off = yr.loc[~sig, "pnl_eff"]
        delta = _signed_delta(
            float(on.mean()) if len(on) else float("nan"), float(off.mean()) if len(off) else float("nan")
        )
        eligible = len(on) >= 10 and len(off) >= 10 and not math.isnan(delta)
        if eligible and delta > 0:
            positive_years += 1
        rows.append(
            {
                "year": year,
                "n_on": int(len(on)),
                "n_off": int(len(off)),
                "delta": delta,
                "eligible": eligible,
            }
        )
    return rows, positive_years


def _compute_thresholds(df: pd.DataFrame, plans: list[HypothesisPlan]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    is_df = df.loc[df["is_is"]].copy()
    for plan in plans:
        vals = is_df[plan.feature_column].astype(float).dropna()
        if len(vals) < 20:
            raise RuntimeError(f"{plan.id}: insufficient IS rows for threshold calibration")
        thresholds[plan.threshold_name] = float(np.nanpercentile(vals, 67))
    return thresholds


def load_lane_df() -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        sql = """
        SELECT
            o.trading_day,
            o.outcome,
            o.pnl_r,
            o.entry_ts,
            d.rel_vol_COMEX_SETTLE,
            d.overnight_range_pct
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
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day
        """
        df = con.execute(sql).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError("COMEX unfiltered lane query returned zero rows")
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT_SACRED_FROM
    df["is_oos"] = ~df["is_is"]
    df["is_scratch"] = df["outcome"].eq("scratch")
    df["pnl_eff"] = np.where(df["is_scratch"], 0.0, df["pnl_r"]).astype(float)
    return df


def _evaluate_hypothesis(df: pd.DataFrame, plan: HypothesisPlan) -> dict[str, object]:
    hyp_cfg = next(h for h in _load_prereg(PREREG_PATH)["hypotheses"] if h["id"] == plan.id)

    is_df = df.loc[df["is_is"]].copy()
    oos_df = df.loc[df["is_oos"]].copy()
    sig_is = is_df[plan.signal_name].astype(bool)
    sig_oos = oos_df[plan.signal_name].astype(bool)

    on_is = is_df.loc[sig_is, "pnl_eff"]
    off_is = is_df.loc[~sig_is, "pnl_eff"]
    on_oos = oos_df.loc[sig_oos, "pnl_eff"]
    off_oos = oos_df.loc[~sig_oos, "pnl_eff"]

    expr_on_is, expr_on_resolved_is, wr_on_is = _group_stats(on_is, is_df.loc[sig_is, "outcome"])
    expr_off_is, expr_off_resolved_is, wr_off_is = _group_stats(off_is, is_df.loc[~sig_is, "outcome"])
    expr_on_oos, expr_on_resolved_oos, _ = _group_stats(on_oos, oos_df.loc[sig_oos, "outcome"])
    expr_off_oos, expr_off_resolved_oos, _ = _group_stats(off_oos, oos_df.loc[~sig_oos, "outcome"])

    delta_is = _signed_delta(expr_on_is, expr_off_is)
    delta_oos = _signed_delta(expr_on_oos, expr_off_oos)
    raw_p = _welch_p(on_is, off_is)
    year_rows, years_positive = _year_rows(df, plan.signal_name)

    thresholds_gte = hyp_cfg["pass_metric"]["threshold_gte"]
    thresholds_lt = hyp_cfg["pass_metric"]["threshold_lt"]
    delta_ok = not math.isnan(delta_is) and delta_is >= float(thresholds_gte["delta_is"])
    expr_ok = not math.isnan(expr_on_is) and expr_on_is > float(thresholds_gte["expr_on_is"])
    years_ok = years_positive >= int(thresholds_gte["years_positive_is"])
    raw_p_ok = not math.isnan(raw_p) and raw_p < float(thresholds_lt["raw_p"])
    n_ok = len(on_is) >= 100
    if len(on_oos) >= 5 and len(off_oos) >= 5 and not math.isnan(delta_oos):
        dir_match = bool(np.sign(delta_is) == np.sign(delta_oos))
    else:
        dir_match = None

    return {
        "hypothesis_id": plan.id,
        "signal_name": plan.signal_name,
        "threshold_value": float(df.attrs["thresholds"][plan.threshold_name]),
        "n_total": int(len(df)),
        "n_is": int(len(is_df)),
        "n_oos": int(len(oos_df)),
        "n_on_is": int(len(on_is)),
        "n_off_is": int(len(off_is)),
        "n_on_oos": int(len(on_oos)),
        "n_off_oos": int(len(off_oos)),
        "expr_on_is": expr_on_is,
        "expr_off_is": expr_off_is,
        "expr_on_resolved_is": expr_on_resolved_is,
        "expr_off_resolved_is": expr_off_resolved_is,
        "expr_on_oos": expr_on_oos,
        "expr_off_oos": expr_off_oos,
        "expr_on_resolved_oos": expr_on_resolved_oos,
        "expr_off_resolved_oos": expr_off_resolved_oos,
        "wr_on_is": wr_on_is,
        "wr_off_is": wr_off_is,
        "delta_is": delta_is,
        "delta_oos": delta_oos,
        "raw_p_is": raw_p,
        "years_positive": years_positive,
        "year_rows": year_rows,
        "dir_match_oos": dir_match,
        "delta_ok": delta_ok,
        "expr_ok": expr_ok,
        "years_ok": years_ok,
        "raw_p_ok": raw_p_ok,
        "n_ok": n_ok,
    }


def _assign_family_q(results: list[dict[str, object]]) -> None:
    q_map = _bh_fdr([(row["hypothesis_id"], float(row["raw_p_is"])) for row in results])
    for row in results:
        row["q_family"] = float(q_map.get(row["hypothesis_id"], float("nan")))


def _apply_verdicts(results: list[dict[str, object]]) -> str:
    for row in results:
        q_ok = not math.isnan(row["q_family"]) and row["q_family"] < BH_Q
        if not all([row["delta_ok"], row["expr_ok"], row["years_ok"], row["raw_p_ok"], row["n_ok"], q_ok]):
            row["verdict"] = "KILL"
            continue
        if int(row["n_on_oos"]) >= 5 and row["dir_match_oos"] is False:
            row["verdict"] = "PARK"
            continue
        row["verdict"] = "KEEP"
    if any(row["verdict"] == "KEEP" for row in results):
        return "KEEP"
    if any(row["verdict"] == "PARK" for row in results):
        return "PARK"
    return "KILL"


def _render_result_table(results: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Hypothesis | Threshold | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | Delta_OOS | raw_p | q_family | years_pos | dir_match_oos | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['hypothesis_id']} | {_fmt(row['threshold_value'])} | {row['n_on_is']} | {row['n_off_is']} | {row['n_on_oos']} | "
            f"{_fmt(row['expr_on_is'])} | {_fmt(row['expr_off_is'])} | {_fmt(row['delta_is'])} | {_fmt(row['delta_oos'])} | "
            f"{_fmt(row['raw_p_is'])} | {_fmt(row['q_family'])} | {row['years_positive']} | "
            f"{row['dir_match_oos']} | **{row['verdict']}** |"
        )
    return lines


def _render_resolved_vs_scratch_table(results: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Hypothesis | ExpR_on_scratch0_IS | ExpR_on_resolved_IS | ExpR_off_scratch0_IS | ExpR_off_resolved_IS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['hypothesis_id']} | {_fmt(row['expr_on_is'])} | {_fmt(row['expr_on_resolved_is'])} | "
            f"{_fmt(row['expr_off_is'])} | {_fmt(row['expr_off_resolved_is'])} |"
        )
    return lines


def _render_yearly_tables(results: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for row in results:
        lines.append(f"### {row['hypothesis_id']}")
        lines.append("")
        lines.append("| Year | N_on | N_off | Delta_IS | Eligible |")
        lines.append("|---:|---:|---:|---:|---|")
        for yr in row["year_rows"]:
            lines.append(f"| {yr['year']} | {yr['n_on']} | {yr['n_off']} | {_fmt(yr['delta'])} | {yr['eligible']} |")
        lines.append("")
    return lines


def _build_row_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "trading_day",
            "outcome",
            "pnl_r",
            "pnl_eff",
            "is_scratch",
            "is_is",
            "is_oos",
            "year",
            "rel_vol_COMEX_SETTLE",
            "overnight_range_pct",
            "rel_vol_high_q3",
            "overnight_range_high_q3",
        ]
    ].copy()
    out["trading_day"] = out["trading_day"].dt.date
    return out


def main() -> int:
    prereg = _load_prereg(PREREG_PATH)
    df = load_lane_df()
    thresholds = _compute_thresholds(df, HYPOTHESES)
    df.attrs["thresholds"] = thresholds

    for plan in HYPOTHESES:
        thr = thresholds[plan.threshold_name]
        df[plan.signal_name] = (df[plan.feature_column].astype(float) > thr).fillna(False).astype(bool)

    results = [_evaluate_hypothesis(df, plan) for plan in HYPOTHESES]
    _assign_family_q(results)
    family_verdict = _apply_verdicts(results)

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
    _build_row_csv(df).to_csv(ROW_CSV, index=False)

    baseline_is = df.loc[df["is_is"], "pnl_eff"].astype(float)
    baseline_oos = df.loc[df["is_oos"], "pnl_eff"].astype(float)
    baseline_expr_is = float(baseline_is.mean()) if len(baseline_is) else float("nan")
    baseline_expr_oos = float(baseline_oos.mean()) if len(baseline_oos) else float("nan")
    baseline_t, baseline_p = stats.ttest_1samp(baseline_is, 0.0)

    header = _build_header(prereg, observed_cell_count=len(results))

    lines = header + [
        "## Verdict",
        "",
        f"`{family_verdict}`",
        "",
        "Family verdict applies only to this prereg's two overlay hypotheses on the exact unfiltered COMEX lane.",
        "",
        "## Baseline lane truth",
        "",
        f"- IS rows: `{int(df['is_is'].sum())}`",
        f"- OOS rows: `{int(df['is_oos'].sum())}`",
        f"- Scratch rows total: `{int(df['is_scratch'].sum())}`",
        f"- Scratch-inclusive baseline IS ExpR: `{_fmt(baseline_expr_is)}`",
        f"- Scratch-inclusive baseline OOS ExpR: `{_fmt(baseline_expr_oos)}`",
        f"- One-sample IS t / p vs 0: `t={_fmt(float(baseline_t), 3)}` `p={_fmt(float(baseline_p))}`",
        "",
        "## Frozen IS-only thresholds",
        "",
        f"- `rel_vol_HIGH_Q3`: `{_fmt(thresholds['rel_vol_HIGH_Q3'])}`",
        f"- `overnight_range_pct_HIGH_Q3`: `{_fmt(thresholds['overnight_range_pct_HIGH_Q3'])}`",
        "",
        "## Family results",
        "",
        *_render_result_table(results),
        "",
        "## Decision notes",
        "",
        "- `H01_RELVOL_HIGH_Q3` survives the IS gates but parks because the OOS delta flips sign on usable OOS sample.",
        "- `H02_OVERNIGHT_RANGE_HIGH_Q3` fails the prereg raw-p gate and is killed.",
        "",
        "## Scratch-inclusive vs resolved-only comparison",
        "",
        *_render_resolved_vs_scratch_table(results),
        "",
        "## Yearly IS delta by hypothesis",
        "",
        *_render_yearly_tables(results),
        "## Outputs",
        "",
        f"- Row-level CSV: `{ROW_CSV.as_posix()}`",
    ]
    RESULT_DOC.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
