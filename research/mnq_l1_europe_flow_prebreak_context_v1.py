"""MNQ EUROPE_FLOW unfiltered pre-break context family v1.

Scope is locked by:
  docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml

# e2-lookahead-policy: tainted
# rel_vol_EUROPE_FLOW is used as a predictor/filter signal on E2 entries. On E2, ~41% of
# trades have entry_ts < break_ts (range-touch fires before close-outside-ORB), making
# break-bar volume (rel_vol numerator) post-entry on that subset. All findings for E2 lanes
# are unreliable and must not be cited. Clean re-derivation with ovn_range_pct required.
# Registry: docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md

This runner evaluates exactly two pre-registered overlay hypotheses on the
unfiltered MNQ EUROPE_FLOW O5 E2 CB1 RR1.5 lane using canonical
orb_outcomes + daily_features joins.

Outputs:
  - docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md
  - docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg-rows.csv
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

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md")
ROW_CSV = Path("docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg-rows.csv")
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
        id="H01_PRE_VELOCITY_HIGH_Q3",
        signal_name="pre_velocity_high_q3",
        feature_column="orb_EUROPE_FLOW_pre_velocity",
        threshold_name="pre_velocity_HIGH_Q3",
    ),
    HypothesisPlan(
        id="H02_RELVOL_HIGH_Q3",
        signal_name="rel_vol_high_q3",
        feature_column="rel_vol_EUROPE_FLOW",
        threshold_name="rel_vol_HIGH_Q3",
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


def _year_rows(df: pd.DataFrame, signal_col: str) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    positive_years = 0
    for year in range(IS_START_YEAR, IS_END_YEAR + 1):
        yr = df.loc[df["is_is"] & (df["year"] == year)].copy()
        sig = yr[signal_col].astype(bool)
        on = yr.loc[sig, "pnl_r"]
        off = yr.loc[~sig, "pnl_r"]
        delta = float(on.mean() - off.mean()) if len(on) and len(off) else float("nan")
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
            o.pnl_r,
            d.orb_EUROPE_FLOW_pre_velocity,
            d.rel_vol_EUROPE_FLOW
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ'
          AND o.orb_label = 'EUROPE_FLOW'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = 1.5
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
        """
        df = con.execute(sql).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError("L1 EUROPE_FLOW unfiltered lane query returned zero rows")
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["is_is"] = df["trading_day"].dt.date < HOLDOUT_SACRED_FROM
    df["is_oos"] = ~df["is_is"]
    return df


def _evaluate_hypothesis(df: pd.DataFrame, plan: HypothesisPlan, prereg: dict) -> dict[str, object]:
    hyp_cfg = next(h for h in prereg["hypotheses"] if h["id"] == plan.id)

    is_df = df.loc[df["is_is"]].copy()
    oos_df = df.loc[df["is_oos"]].copy()
    sig_is = is_df[plan.signal_name].astype(bool)
    sig_oos = oos_df[plan.signal_name].astype(bool)

    on_is = is_df.loc[sig_is, "pnl_r"].astype(float)
    off_is = is_df.loc[~sig_is, "pnl_r"].astype(float)
    on_oos = oos_df.loc[sig_oos, "pnl_r"].astype(float)
    off_oos = oos_df.loc[~sig_oos, "pnl_r"].astype(float)

    expr_on_is = float(on_is.mean()) if len(on_is) else float("nan")
    expr_off_is = float(off_is.mean()) if len(off_is) else float("nan")
    expr_on_oos = float(on_oos.mean()) if len(on_oos) else float("nan")
    expr_off_oos = float(off_oos.mean()) if len(off_oos) else float("nan")
    delta_is = float(expr_on_is - expr_off_is) if len(on_is) and len(off_is) else float("nan")
    delta_oos = float(expr_on_oos - expr_off_oos) if len(on_oos) and len(off_oos) else float("nan")
    raw_p = _welch_p(on_is, off_is)
    fire_rate = float(df[plan.signal_name].mean())
    year_rows, years_positive = _year_rows(df, plan.signal_name)

    thresholds_gte = hyp_cfg["pass_metric"]["threshold_gte"]
    thresholds_lt = hyp_cfg["pass_metric"]["threshold_lt"]
    delta_ok = not math.isnan(delta_is) and delta_is >= float(thresholds_gte["delta_is"])
    expr_ok = not math.isnan(expr_on_is) and expr_on_is > float(thresholds_gte["expr_on_is"])
    years_ok = years_positive >= int(thresholds_gte["years_positive_is"])
    raw_p_ok = not math.isnan(raw_p) and raw_p < float(thresholds_lt["raw_p"])
    n_ok = len(on_is) >= 100
    operating_range_ok = 0.05 <= fire_rate <= 0.95
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
        "expr_on_oos": expr_on_oos,
        "expr_off_oos": expr_off_oos,
        "delta_is": delta_is,
        "delta_oos": delta_oos,
        "raw_p_is": raw_p,
        "fire_rate": fire_rate,
        "years_positive": years_positive,
        "year_rows": year_rows,
        "dir_match_oos": dir_match,
        "delta_ok": delta_ok,
        "expr_ok": expr_ok,
        "years_ok": years_ok,
        "raw_p_ok": raw_p_ok,
        "n_ok": n_ok,
        "operating_range_ok": operating_range_ok,
    }


def _assign_family_q(results: list[dict[str, object]]) -> None:
    q_map = _bh_fdr([(row["hypothesis_id"], float(row["raw_p_is"])) for row in results])
    for row in results:
        row["q_family"] = float(q_map.get(row["hypothesis_id"], float("nan")))


def _apply_verdicts(results: list[dict[str, object]]) -> str:
    for row in results:
        q_ok = not math.isnan(row["q_family"]) and row["q_family"] < BH_Q
        if not all(
            [
                row["delta_ok"],
                row["expr_ok"],
                row["years_ok"],
                row["raw_p_ok"],
                row["n_ok"],
                row["operating_range_ok"],
                q_ok,
            ]
        ):
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
        "| Hypothesis | Threshold | Fire% | N_on_IS | N_off_IS | N_on_OOS | ExpR_on_IS | ExpR_off_IS | Delta_IS | Delta_OOS | raw_p | q_family | years_pos | dir_match_oos | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['hypothesis_id']} | {_fmt(row['threshold_value'])} | {_fmt(row['fire_rate'])} | "
            f"{row['n_on_is']} | {row['n_off_is']} | {row['n_on_oos']} | {_fmt(row['expr_on_is'])} | "
            f"{_fmt(row['expr_off_is'])} | {_fmt(row['delta_is'])} | {_fmt(row['delta_oos'])} | "
            f"{_fmt(row['raw_p_is'])} | {_fmt(row['q_family'])} | {row['years_positive']} | "
            f"{row['dir_match_oos']} | **{row['verdict']}** |"
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
            "pnl_r",
            "is_is",
            "is_oos",
            "year",
            "orb_EUROPE_FLOW_pre_velocity",
            "rel_vol_EUROPE_FLOW",
            "pre_velocity_high_q3",
            "rel_vol_high_q3",
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

    results = [_evaluate_hypothesis(df, plan, prereg) for plan in HYPOTHESES]
    _assign_family_q(results)
    family_verdict = _apply_verdicts(results)

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
    _build_row_csv(df).to_csv(ROW_CSV, index=False)

    baseline_is = df.loc[df["is_is"], "pnl_r"].astype(float)
    baseline_oos = df.loc[df["is_oos"], "pnl_r"].astype(float)
    baseline_expr_is = float(baseline_is.mean()) if len(baseline_is) else float("nan")
    baseline_expr_oos = float(baseline_oos.mean()) if len(baseline_oos) else float("nan")
    baseline_t, baseline_p = stats.ttest_1samp(baseline_is, 0.0)

    header = _build_header(prereg, observed_cell_count=len(results))
    lines = header + [
        "## Verdict",
        "",
        f"`{family_verdict}`",
        "",
        "Family verdict applies only to this prereg's two overlay hypotheses on the exact unfiltered L1 lane.",
        "",
        "## Baseline lane truth",
        "",
        f"- IS rows: `{int(df['is_is'].sum())}`",
        f"- OOS rows: `{int(df['is_oos'].sum())}`",
        f"- Unfiltered baseline IS ExpR: `{_fmt(baseline_expr_is)}`",
        f"- Unfiltered baseline OOS ExpR: `{_fmt(baseline_expr_oos)}`",
        f"- One-sample IS t / p vs 0: `t={_fmt(float(baseline_t), 3)}` `p={_fmt(float(baseline_p))}`",
        "",
        "## Frozen IS-only thresholds",
        "",
        f"- `pre_velocity_HIGH_Q3`: `{_fmt(thresholds['pre_velocity_HIGH_Q3'])}`",
        f"- `rel_vol_HIGH_Q3`: `{_fmt(thresholds['rel_vol_HIGH_Q3'])}`",
        "",
        "## Family results",
        "",
        *_render_result_table(results),
        "",
        "## Decision notes",
        "",
        "- This runner tests magnitude overlays only. It does not retest the null direction-alignment framing from PR #72.",
        "- Any hypothesis outside the prereg fire-rate operating band is killed even if the raw p-value is small.",
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
