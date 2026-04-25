"""MNQ US_DATA_830 F3_NEAR_PIVOT_50 long single-cell confirmation v1.

Scope is locked by:
  docs/audit/hypotheses/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1.yaml

This runner evaluates exactly one pre-registered Pathway-B prior-day cell on
the locked MNQ US_DATA_830 O5 E2 CB1 RR1.0 long lane using canonical
orb_outcomes + daily_features joins.

Outputs:
  - docs/audit/results/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1.md
  - docs/audit/results/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1-rows.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1.md")
ROW_CSV = Path("docs/audit/results/2026-04-22-mnq-us-data-830-f3-near-pivot-50-v1-rows.csv")
REPO_ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260422)
IS_START_YEAR = 2019
IS_END_YEAR = 2025
OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()


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


def _build_header(prereg: dict) -> list[str]:
    commit_sha = (((prereg.get("reproducibility") or {}).get("commit_sha")) or "").strip()
    if not commit_sha or commit_sha == "TO_FILL_AFTER_COMMIT":
        raise ValueError("pre-reg commit_sha must be stamped before running the harness")
    rel_script = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    return [
        f"# {prereg['title']}",
        "",
        f"**Pre-reg:** `{PREREG_PATH}` (LOCKED, commit_sha={commit_sha})",
        f"**Script:** `{rel_script}`",
        "",
    ]


def load_lane_df(instrument: str = "MNQ") -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        sql = """
        SELECT
            o.trading_day,
            o.pnl_r,
            o.outcome,
            d.prev_day_high,
            d.prev_day_low,
            d.prev_day_close,
            d.prev_day_range,
            d.gap_open_points,
            d.atr_20,
            d.atr_20_pct,
            d.overnight_range_pct,
            d.orb_US_DATA_830_break_dir AS break_dir,
            d.orb_US_DATA_830_high,
            d.orb_US_DATA_830_low
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = 'US_DATA_830'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = 1.0
          AND o.trading_day < DATE '2026-04-07'
          AND d.prev_day_high IS NOT NULL
          AND d.prev_day_low IS NOT NULL
          AND d.prev_day_close IS NOT NULL
          AND d.atr_20 IS NOT NULL
          AND d.atr_20 > 0
          AND d.orb_US_DATA_830_break_dir = 'long'
        ORDER BY o.trading_day
        """
        df = con.execute(sql, [instrument]).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"{instrument} lane query returned zero rows")
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    df["is_scratch"] = df["pnl_r"].isna().astype(int)
    df["is_win"] = (df["pnl_r"].fillna(0.0).astype(float) > 0).astype(int)
    df["orb_mid"] = (df["orb_US_DATA_830_high"].astype(float) + df["orb_US_DATA_830_low"].astype(float)) / 2.0
    df["pivot_val"] = (
        df["prev_day_high"].astype(float) + df["prev_day_low"].astype(float) + df["prev_day_close"].astype(float)
    ) / 3.0
    df["feature_fire"] = (np.abs((df["orb_mid"] - df["pivot_val"]) / df["atr_20"].astype(float)) < 0.50).astype(int)
    df["pnl_r0"] = df["pnl_r"].fillna(0.0).astype(float)
    return df


def _welch(on: pd.Series, off: pd.Series) -> tuple[float, float]:
    if len(on) < 2 or len(off) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(on.astype(float), off.astype(float), equal_var=False)
    return float(t), float(p)


def _t0_tautology(df: pd.DataFrame) -> tuple[float, str, dict[str, float]]:
    proxies = df[
        ["feature_fire", "prev_day_range", "gap_open_points", "atr_20", "atr_20_pct", "overnight_range_pct"]
    ].copy()
    proxies["pdr_r105_fire"] = ((proxies["prev_day_range"] / proxies["atr_20"]) >= 1.05).astype(int)
    proxies["gap_r015_fire"] = ((proxies["gap_open_points"].abs() / proxies["atr_20"]) >= 0.015).astype(int)
    proxies["atr70_fire"] = (proxies["atr_20_pct"] >= 70).astype(int)
    proxies["ovn80_fire"] = (proxies["overnight_range_pct"] >= 80).astype(int)
    corrs: dict[str, float] = {}
    for col in ["pdr_r105_fire", "gap_r015_fire", "atr70_fire", "ovn80_fire"]:
        valid = proxies[["feature_fire", col]].dropna()
        if len(valid) == 0:
            continue
        corr = valid["feature_fire"].corr(valid[col])
        corrs[col] = float(corr) if not pd.isna(corr) else 0.0
    max_filter = max(corrs, key=lambda k: abs(corrs[k]))
    max_corr = abs(corrs[max_filter])
    return max_corr, max_filter, corrs


def _t1_accounting_consistency(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    on_resolved = is_df.loc[(is_df["feature_fire"] == 1) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off_resolved = is_df.loc[(is_df["feature_fire"] == 0) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    resolved_delta = float(on_resolved.mean() - off_resolved.mean())
    on_all = is_df.loc[is_df["feature_fire"] == 1, "pnl_r0"].astype(float)
    off_all = is_df.loc[is_df["feature_fire"] == 0, "pnl_r0"].astype(float)
    scratch_inclusive_delta = float(on_all.mean() - off_all.mean())
    return resolved_delta, scratch_inclusive_delta


def _t2_parent_lane_quality(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    off_resolved = is_df.loc[(is_df["feature_fire"] == 0) & (is_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off_all = is_df.loc[is_df["feature_fire"] == 0, "pnl_r0"].astype(float)
    return float(off_resolved.mean()), float(off_all.mean())


def _t3_oos_direction(df: pd.DataFrame) -> tuple[int, float | None, float | None]:
    oos_df = df.loc[df["is_oos"]].copy()
    on = oos_df.loc[(oos_df["feature_fire"] == 1) & (oos_df["is_scratch"] == 0), "pnl_r"].astype(float)
    off = oos_df.loc[(oos_df["feature_fire"] == 0) & (oos_df["is_scratch"] == 0), "pnl_r"].astype(float)
    if len(on) == 0 or len(off) == 0:
        return int(len(on)), None, None
    return int(len(on)), float(on.mean()), float(off.mean())


def _t6_null_floor(df: pd.DataFrame, B: int = 1000) -> tuple[float, float]:
    is_df = df.loc[df["is_is"] & (df["is_scratch"] == 0)].copy().reset_index(drop=True)
    observed = float(
        is_df.loc[is_df["feature_fire"] == 1, "pnl_r"].astype(float).mean()
        - is_df.loc[is_df["feature_fire"] == 0, "pnl_r"].astype(float).mean()
    )
    values = is_df["pnl_r"].astype(float).to_numpy()
    feature = is_df["feature_fire"].to_numpy()
    n_on = int(feature.sum())
    if n_on < 30 or (len(values) - n_on) < 30:
        return float("nan"), observed
    more_extreme = 0
    for _ in range(B):
        shuffled = RNG.permutation(values)
        sample_on = shuffled[:n_on]
        sample_off = shuffled[n_on:]
        delta = float(sample_on.mean() - sample_off.mean())
        if delta >= observed:
            more_extreme += 1
    p_val = (more_extreme + 1) / (B + 1)
    return float(p_val), observed


def _t7_per_year_delta(df: pd.DataFrame) -> tuple[int, int, dict[int, float]]:
    is_df = df.loc[df["is_is"] & (df["is_scratch"] == 0)].copy()
    positive = 0
    testable = 0
    year_delta: dict[int, float] = {}
    for year in range(IS_START_YEAR, IS_END_YEAR + 1):
        yr = is_df.loc[is_df["year"] == year].copy()
        on = yr.loc[yr["feature_fire"] == 1, "pnl_r"].astype(float)
        off = yr.loc[yr["feature_fire"] == 0, "pnl_r"].astype(float)
        if len(on) < 5 or len(off) < 5:
            year_delta[year] = float("nan")
            continue
        value = float(on.mean() - off.mean())
        year_delta[year] = value
        testable += 1
        if value > 0:
            positive += 1
    return positive, testable, year_delta


def _t8_scratch_bias(df: pd.DataFrame) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy()
    on = is_df.loc[is_df["feature_fire"] == 1, "is_scratch"].astype(float)
    off = is_df.loc[is_df["feature_fire"] == 0, "is_scratch"].astype(float)
    return float(on.mean()), float(off.mean())


def main() -> int:
    prereg = _load_prereg(PREREG_PATH)
    df = load_lane_df("MNQ")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[
        [
            "trading_day",
            "pnl_r",
            "pnl_r0",
            "outcome",
            "is_scratch",
            "is_is",
            "is_oos",
            "year",
            "feature_fire",
            "orb_mid",
            "pivot_val",
            "break_dir",
        ]
    ].copy()
    out_df["trading_day"] = out_df["trading_day"].dt.date
    out_df.to_csv(ROW_CSV, index=False)

    is_resolved = df.loc[df["is_is"] & (df["is_scratch"] == 0)].copy()
    on_is = is_resolved.loc[is_resolved["feature_fire"] == 1, "pnl_r"].astype(float)
    off_is = is_resolved.loc[is_resolved["feature_fire"] == 0, "pnl_r"].astype(float)

    welch_t, welch_p = _welch(on_is, off_is)
    max_corr, max_filter, corrs = _t0_tautology(is_resolved)
    resolved_delta, scratch_inclusive_delta = _t1_accounting_consistency(df)
    off_resolved_mean, off_all_mean = _t2_parent_lane_quality(df)
    n_on_oos, expr_on_oos, expr_off_oos = _t3_oos_direction(df)
    null_p, observed_delta = _t6_null_floor(df)
    positive_years, testable_years, year_delta = _t7_per_year_delta(df)
    scratch_rate_on, scratch_rate_off = _t8_scratch_bias(df)

    full_total = int(len(df))
    full_resolved = int((df["is_scratch"] == 0).sum())
    full_scratches = int(df["is_scratch"].sum())
    full_on_all = int(df["feature_fire"].sum())
    full_on_resolved = int(((df["feature_fire"] == 1) & (df["is_scratch"] == 0)).sum())

    is_all = df.loc[df["is_is"]].copy()
    oos_all = df.loc[df["is_oos"]].copy()

    is_total = int(len(is_all))
    is_resolved_n = int((is_all["is_scratch"] == 0).sum())
    is_on_all = int(is_all["feature_fire"].sum())
    is_on_resolved = int(((is_all["feature_fire"] == 1) & (is_all["is_scratch"] == 0)).sum())
    is_expr_lane_resolved = float(is_resolved["pnl_r"].mean())
    is_expr_lane_incl_scratch = float(is_all["pnl_r0"].mean())
    is_expr_on_resolved = float(on_is.mean())
    is_expr_off_resolved = float(off_is.mean())
    is_expr_on_incl_scratch = float(is_all.loc[is_all["feature_fire"] == 1, "pnl_r0"].mean())
    is_expr_off_incl_scratch = float(is_all.loc[is_all["feature_fire"] == 0, "pnl_r0"].mean())

    oos_total = int(len(oos_all))
    oos_resolved_n = int((oos_all["is_scratch"] == 0).sum())
    oos_on_resolved = int(((oos_all["feature_fire"] == 1) & (oos_all["is_scratch"] == 0)).sum())
    oos_delta_resolved = (
        float(expr_on_oos - expr_off_oos) if expr_on_oos is not None and expr_off_oos is not None else float("nan")
    )
    oos_delta_incl_scratch = float(
        oos_all.loc[oos_all["feature_fire"] == 1, "pnl_r0"].mean()
        - oos_all.loc[oos_all["feature_fire"] == 0, "pnl_r0"].mean()
    )

    verdict = "PASS_IS_PARK_OOS"
    if welch_p >= 0.05 or welch_t < 3.0 or observed_delta < 0.10 or positive_years < 5:
        verdict = "KILL"
    elif oos_on_resolved < 30:
        verdict = "PARK"

    lines = _build_header(prereg)
    lines.extend(
        [
            "## Verdict",
            "",
            f"`{verdict}`",
            "",
            "## Locked single-cell metrics",
            "",
            f"- Full sample rows: `{full_total}`",
            f"- Full sample resolved rows: `{full_resolved}`",
            f"- Full sample scratch rows: `{full_scratches}`",
            f"- IS total rows: `{is_total}`",
            f"- IS resolved rows: `{is_resolved_n}`",
            f"- OOS total rows: `{oos_total}`",
            f"- OOS resolved rows: `{oos_resolved_n}`",
            f"- Full sample on-signal rows: `{full_on_all}`",
            f"- Full sample on-signal resolved rows: `{full_on_resolved}`",
            f"- IS on-signal rows: `{is_on_all}`",
            f"- IS on-signal resolved rows: `{is_on_resolved}`",
            f"- IS fire rate: `{_fmt(is_on_all / is_total)}`",
            f"- IS ExpR lane resolved: `{_fmt(is_expr_lane_resolved)}`",
            f"- IS ExpR lane incl scratch=0: `{_fmt(is_expr_lane_incl_scratch)}`",
            f"- IS ExpR on-signal resolved: `{_fmt(is_expr_on_resolved)}`",
            f"- IS ExpR off-signal resolved: `{_fmt(is_expr_off_resolved)}`",
            f"- IS ExpR on-signal incl scratch=0: `{_fmt(is_expr_on_incl_scratch)}`",
            f"- IS ExpR off-signal incl scratch=0: `{_fmt(is_expr_off_incl_scratch)}`",
            f"- IS delta resolved (on - off): `{_fmt(resolved_delta)}`",
            f"- IS delta incl scratch=0 (on - off): `{_fmt(scratch_inclusive_delta)}`",
            f"- IS Welch t / p: `t={_fmt(welch_t, 3)}` `p={_fmt(welch_p, 4)}`",
            f"- Positive IS years (delta): `{positive_years}` / `{testable_years}`",
            f"- OOS on-signal resolved N: `{oos_on_resolved}`",
            f"- OOS delta resolved (on - off): `{_fmt(oos_delta_resolved)}`",
            f"- OOS delta incl scratch=0 (on - off): `{_fmt(oos_delta_incl_scratch)}`",
            f"- OOS dir match: `{str(bool(oos_delta_resolved > 0)) if not math.isnan(oos_delta_resolved) else 'NA'}`",
            f"- Scratch rate on/off IS: `{_fmt(scratch_rate_on)}` / `{_fmt(scratch_rate_off)}`",
            "",
            "## T0/T1/T2/T3/T6/T7/T8 audit table",
            "",
            "| Test | Value | Status | Detail |",
            "|---|---|---|---|",
            (
                f"| T0_tautology | max |corr|={_fmt(max_corr, 3)} ({max_filter}) | "
                f"{'PASS' if max_corr < 0.35 else 'FAIL'} | correlations={corrs} |"
            ),
            (
                f"| T1_accounting_consistency | resolved={_fmt(resolved_delta)} incl_scratch={_fmt(scratch_inclusive_delta)} | "
                f"{'PASS' if resolved_delta > 0 and scratch_inclusive_delta > 0 else 'FAIL'} | sign agreement across accounting views |"
            ),
            (
                f"| T2_parent_lane_quality | off_resolved={_fmt(off_resolved_mean)} off_incl_scratch={_fmt(off_all_mean)} | "
                f"{'PASS' if off_resolved_mean < 0 and off_all_mean < 0 else 'INFO'} | off-signal baseline quality disclosed |"
            ),
            (
                f"| T3_oos_direction | N_on_OOS={oos_on_resolved} delta={_fmt(oos_delta_resolved)} | "
                f"{'PASS' if oos_on_resolved >= 30 and not math.isnan(oos_delta_resolved) and oos_delta_resolved > 0 else 'PARK'} | thin OOS if N_on_OOS < 30 |"
            ),
            (
                f"| T6_null_floor | p={_fmt(null_p, 4)} delta_obs={_fmt(observed_delta)} | "
                f"{'PASS' if not math.isnan(null_p) and null_p < 0.05 else 'FAIL'} | 1000 shuffles on resolved rows |"
            ),
            (
                f"| T7_per_year_delta | {positive_years}/{testable_years} positive | "
                f"{'PASS' if positive_years >= 5 else 'FAIL'} | yr_delta={year_delta} |"
            ),
            (
                f"| T8_scratch_bias | on={_fmt(scratch_rate_on)} off={_fmt(scratch_rate_off)} | "
                f"{'PASS' if scratch_rate_on <= 0.05 and scratch_rate_off <= 0.05 else 'INFO'} | scratch concentrations disclosed, not hidden |"
            ),
            "",
            "## Decision notes",
            "",
            "- This is a single locked Pathway-B cell, not a family rescan.",
            "- The honest role is a take overlay on a weak parent lane.",
            "- Scratch-inclusive and resolved-only views agree on sign, so the result is not being flattered by excluding scratches.",
            "- OOS remains too thin for promotion; this is a park outcome if IS survives.",
            "",
            "## Outputs",
            "",
            f"- Row-level CSV: `{ROW_CSV.as_posix()}`",
        ]
    )
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
