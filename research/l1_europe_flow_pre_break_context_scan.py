"""Execute the frozen L1 EUROPE_FLOW pre-break-context prereg.

Pre-reg:
  docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml

# e2-lookahead-policy: tainted
# rel_vol_EUROPE_FLOW is used as a predictor of pnl_r on E2 entries. On E2, ~41% of trades
# have entry_ts < break_ts (range-touch fires before close-outside-ORB), making break-bar
# volume (rel_vol numerator) post-entry on that subset. All findings from this script for E2
# lanes are unreliable and must not be cited. Clean re-derivation with ovn_range_pct required.
# Registry: docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md

Scope:
  - MNQ
  - EUROPE_FLOW
  - O5 / E2 / CB1 / RR1.5
  - unfiltered baseline only
  - two admissible ORB-end features only:
      * orb_EUROPE_FLOW_pre_velocity
      * rel_vol_EUROPE_FLOW

No live or canonical-layer writes are permitted under any verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from research.lib import connect_db, write_csv
from research.result_doc_header import build_header
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml"
RESULT_PATH = Path("docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md")
ROW_CSV = "l1_europe_flow_pre_break_context_rows.csv"
SUMMARY_CSV = "l1_europe_flow_pre_break_context_summary.csv"

INSTRUMENT = "MNQ"
SESSION = "EUROPE_FLOW"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5


@dataclass(frozen=True)
class FeatureSpec:
    hypothesis_id: str
    feature_name: str
    column_name: str
    signal_name: str


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        hypothesis_id="H01_PRE_VELOCITY_HIGH_Q3",
        feature_name="pre_velocity_HIGH_Q3",
        column_name="orb_EUROPE_FLOW_pre_velocity",
        signal_name="pre_velocity_high_q3",
    ),
    FeatureSpec(
        hypothesis_id="H02_RELVOL_HIGH_Q3",
        feature_name="rel_vol_HIGH_Q3",
        column_name="rel_vol_EUROPE_FLOW",
        signal_name="rel_vol_high_q3",
    ),
)


@dataclass
class HypothesisResult:
    hypothesis_id: str
    feature_name: str
    threshold_is_p67: float
    n_total_is: int
    n_on_is: int
    fire_rate_is: float
    expr_on_is: float
    expr_off_is: float
    delta_is: float
    raw_p_is: float
    q_value_family: float
    bh_pass_family: bool
    years_positive_is: int
    n_on_oos: int
    fire_rate_oos: float
    expr_on_oos: float
    expr_off_oos: float
    delta_oos: float
    dir_match_oos: bool | None
    passes_is_gates: bool
    verdict: str


def _load_lane_rows() -> pd.DataFrame:
    with connect_db() as con:
        sql = f"""
        SELECT
            o.trading_day,
            o.pnl_r,
            d.orb_EUROPE_FLOW_pre_velocity,
            d.rel_vol_EUROPE_FLOW
        FROM orb_outcomes o
        JOIN daily_features d
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE o.symbol = '{INSTRUMENT}'
          AND o.orb_label = '{SESSION}'
          AND o.orb_minutes = {ORB_MINUTES}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.rr_target = {RR_TARGET}
          AND o.pnl_r IS NOT NULL
        ORDER BY o.trading_day
        """
        df = con.execute(sql).fetchdf()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    df["split"] = np.where(df["trading_day"] < holdout, "IS", "OOS")
    return df


def _bh_qvalues(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ordered_p = np.asarray(p_values, dtype=float)[order]
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = rank + 1
        val = ordered_p[rank] * n / idx
        prev = min(prev, val)
        adjusted[rank] = min(prev, 1.0)
    qvals = np.empty(n, dtype=float)
    qvals[order] = adjusted
    return qvals.tolist()


def _welch_p_and_delta(on: pd.Series, off: pd.Series) -> tuple[float, float]:
    onv = on.dropna().astype(float)
    offv = off.dropna().astype(float)
    delta = float(onv.mean() - offv.mean()) if len(onv) and len(offv) else float("nan")
    if len(onv) < 2 or len(offv) < 2:
        return float("nan"), delta
    _, p = stats.ttest_ind(onv, offv, equal_var=False)
    return float(p), delta


def _signal_summary(df: pd.DataFrame, signal_col: str) -> dict[str, float]:
    on = df.loc[df[signal_col], "pnl_r"]
    off = df.loc[~df[signal_col], "pnl_r"]
    _, delta = _welch_p_and_delta(on, off)
    return {
        "n_total": int(len(df)),
        "n_on": int(df[signal_col].sum()),
        "fire_rate": float(df[signal_col].mean()) if len(df) else float("nan"),
        "expr_on": float(on.mean()) if len(on) else float("nan"),
        "expr_off": float(off.mean()) if len(off) else float("nan"),
        "delta": delta,
    }


def _years_positive_is(is_df: pd.DataFrame, signal_col: str) -> int:
    years = 0
    for _, grp in is_df.loc[is_df[signal_col]].groupby("year"):
        if len(grp) and float(grp["pnl_r"].mean()) > 0:
            years += 1
    return years


def _dir_match_oos(delta_is: float, delta_oos: float, n_on_oos: int) -> bool | None:
    if n_on_oos < 5 or np.isnan(delta_oos):
        return None
    return bool(delta_oos > 0) if delta_is > 0 else bool(delta_oos < 0)


def evaluate_hypotheses(df: pd.DataFrame) -> tuple[pd.DataFrame, list[HypothesisResult]]:
    is_df = df.loc[df["split"] == "IS"].copy()
    oos_df = df.loc[df["split"] == "OOS"].copy()
    raw_ps: list[float] = []
    interim: list[dict[str, object]] = []

    for spec in FEATURES:
        threshold = float(is_df[spec.column_name].dropna().quantile(0.67))
        df[spec.signal_name] = df[spec.column_name] > threshold
        is_df[spec.signal_name] = is_df[spec.column_name] > threshold
        oos_df[spec.signal_name] = oos_df[spec.column_name] > threshold

        p_is, delta_is = _welch_p_and_delta(
            is_df.loc[is_df[spec.signal_name], "pnl_r"],
            is_df.loc[~is_df[spec.signal_name], "pnl_r"],
        )
        raw_ps.append(p_is)
        interim.append(
            {
                "spec": spec,
                "threshold": threshold,
                "is_summary": _signal_summary(is_df, spec.signal_name),
                "oos_summary": _signal_summary(oos_df, spec.signal_name),
                "raw_p_is": p_is,
                "delta_is": delta_is,
                "years_positive_is": _years_positive_is(is_df, spec.signal_name),
            }
        )

    qvals = _bh_qvalues(raw_ps)
    results: list[HypothesisResult] = []
    summary_rows: list[dict[str, object]] = []
    for idx, item in enumerate(interim):
        spec: FeatureSpec = item["spec"]  # type: ignore[assignment]
        is_summary = item["is_summary"]  # type: ignore[assignment]
        oos_summary = item["oos_summary"]  # type: ignore[assignment]
        raw_p_is = float(item["raw_p_is"])
        q_value = float(qvals[idx])
        bh_pass = bool(q_value < 0.05)
        years_positive = int(item["years_positive_is"])
        dir_match = _dir_match_oos(
            float(item["delta_is"]),
            float(oos_summary["delta"]),
            int(oos_summary["n_on"]),
        )
        passes_is_gates = bool(
            bh_pass
            and float(is_summary["delta"]) >= 0.05
            and float(is_summary["expr_on"]) > 0
            and years_positive >= 4
            and raw_p_is < 0.05
            and int(is_summary["n_on"]) >= 100
        )
        if passes_is_gates and dir_match is True:
            verdict = "CONTINUE"
        elif passes_is_gates:
            verdict = "PARK"
        else:
            verdict = "KILL"

        result = HypothesisResult(
            hypothesis_id=spec.hypothesis_id,
            feature_name=spec.feature_name,
            threshold_is_p67=float(item["threshold"]),
            n_total_is=int(is_summary["n_total"]),
            n_on_is=int(is_summary["n_on"]),
            fire_rate_is=float(is_summary["fire_rate"]),
            expr_on_is=float(is_summary["expr_on"]),
            expr_off_is=float(is_summary["expr_off"]),
            delta_is=float(is_summary["delta"]),
            raw_p_is=raw_p_is,
            q_value_family=q_value,
            bh_pass_family=bh_pass,
            years_positive_is=years_positive,
            n_on_oos=int(oos_summary["n_on"]),
            fire_rate_oos=float(oos_summary["fire_rate"]),
            expr_on_oos=float(oos_summary["expr_on"]),
            expr_off_oos=float(oos_summary["expr_off"]),
            delta_oos=float(oos_summary["delta"]),
            dir_match_oos=dir_match,
            passes_is_gates=passes_is_gates,
            verdict=verdict,
        )
        results.append(result)
        summary_rows.append(result.__dict__)
    return df, results


def overall_verdict(results: list[HypothesisResult]) -> str:
    if any(r.passes_is_gates and r.dir_match_oos is True for r in results):
        return "CONTINUE"
    if any(r.passes_is_gates for r in results):
        return "PARK"
    return "KILL"


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "NA"
    return "YES" if value else "NO"


def _render_summary_table(results: list[HypothesisResult]) -> list[str]:
    lines = [
        "| Hypothesis | Feature | P67 | N_on_IS | Fire_IS | ExpR_on_IS | ExpR_off_IS | Delta_IS | raw_p | q_family | BH | Years+ | N_on_OOS | Fire_OOS | Delta_OOS | dir_match | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.hypothesis_id} | {r.feature_name} | {r.threshold_is_p67:.4f} | "
            f"{r.n_on_is} | {r.fire_rate_is:.1%} | {r.expr_on_is:+.4f} | {r.expr_off_is:+.4f} | "
            f"{r.delta_is:+.4f} | {r.raw_p_is:.4f} | {r.q_value_family:.4f} | "
            f"{'PASS' if r.bh_pass_family else 'FAIL'} | {r.years_positive_is} | {r.n_on_oos} | "
            f"{r.fire_rate_oos:.1%} | {r.delta_oos:+.4f} | {_fmt_bool(r.dir_match_oos)} | **{r.verdict}** |"
        )
    return lines


def _render_yearly_table(df: pd.DataFrame, signal_col: str, label: str) -> list[str]:
    lines = [
        f"### {label} yearly on-signal breakdown",
        "",
        "| Year | N_on | ExpR_on | WinRate_on |",
        "|---|---:|---:|---:|",
    ]
    sub = df.loc[df[signal_col]].groupby("year", sort=True)
    for year, grp in sub:
        lines.append(
            f"| {int(year)} | {len(grp)} | {float(grp['pnl_r'].mean()):+.4f} | {(grp['pnl_r'] > 0).mean():.1%} |"
        )
    lines.append("")
    return lines


def build_markdown(df: pd.DataFrame, results: list[HypothesisResult]) -> str:
    ov = overall_verdict(results)
    header = build_header(
        prereg_path=PREREG_PATH,
        script_path=__file__,
        extra_lines=[
            "**Lane:** `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_UNFILTERED`",
            "**IS window:** `trading_day < 2026-01-01`",
            "**OOS window:** `trading_day >= 2026-01-01` (descriptive, no tuning)",
        ],
        observed_cell_count=2,
    )
    lines = list(header)
    lines.extend(
        [
            f"## Overall Verdict: **{ov}**",
            "",
            "Decision contract from the prereg:",
            "",
            "- `CONTINUE`: at least one hypothesis survives `K=2` family BH-FDR and all IS gates, with no OOS sign flip when OOS `N_on >= 5`.",
            "- `PARK`: an IS signal exists but OOS remains directional-only or does not confirm cleanly.",
            "- `KILL`: zero family survivors.",
            "",
            "## Family Summary",
            "",
            *_render_summary_table(results),
            "",
            "## Fire-Rate Sanity",
            "",
            "- Prereg target operating band: `5%-95%`.",
        ]
    )
    for r in results:
        lines.append(
            f"- `{r.feature_name}` fire rate: IS `{r.fire_rate_is:.1%}`, OOS `{r.fire_rate_oos:.1%}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if ov == "CONTINUE":
        lines.extend(
            [
                "- At least one admissible pre-break feature survived the frozen `K=2` family honestly.",
                "- This is still research output only. No production filter or live rewrite is implied by this result.",
            ]
        )
    elif ov == "PARK":
        lines.extend(
            [
                "- An IS signal exists under the frozen family, but OOS confirmation is not strong enough to upgrade it.",
                "- Honest next move is to keep the path bounded, not widen the feature family.",
            ]
        )
    else:
        lines.extend(
            [
                "- The prereg family closes cleanly under the frozen two-feature scan.",
                "- This does not justify reopening banned break-bar or ATR-normalized variants.",
            ]
        )
    lines.extend([""])
    for spec in FEATURES:
        lines.extend(_render_yearly_table(df[df["split"] == "IS"], spec.signal_name, spec.feature_name))
    lines.extend(
        [
            "## Outputs",
            "",
            f"- `research/output/{ROW_CSV}`",
            f"- `research/output/{SUMMARY_CSV}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df = _load_lane_rows()
    scored_df, results = evaluate_hypotheses(df)
    summary_df = pd.DataFrame([r.__dict__ for r in results])
    write_csv(scored_df, ROW_CSV)
    write_csv(summary_df, SUMMARY_CSV)
    RESULT_PATH.write_text(build_markdown(scored_df, results), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
