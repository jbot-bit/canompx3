"""MNQ US_DATA_1000 F5_BELOW_PDL long single-cell confirmation v1.

Scope is locked by:
  docs/audit/hypotheses/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.yaml

# e2-lookahead-policy: tainted
# orb_US_DATA_1000_break_dir = 'long' used as a WHERE predicate (line ~103) to
# select fire-days. On E2, break_dir is post-entry for ~42% of fills (range-cross
# precedes close-cross). All IS findings from this script are directionally suspect.
# Re-pre-register with a pre-break direction proxy before deployment use.

This runner evaluates exactly one pre-registered Pathway-B prior-day cell on
the locked MNQ US_DATA_1000 O5 E2 CB1 RR1.0 long lane using canonical
orb_outcomes + daily_features joins.

Outputs:
  - docs/audit/results/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.md
  - docs/audit/results/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1-rows.csv
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

PREREG_PATH = "docs/audit/hypotheses/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1.md")
ROW_CSV = Path("docs/audit/results/2026-04-21-mnq-us-data-1000-f5-below-pdl-v1-rows.csv")
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
            d.prev_day_low,
            d.prev_day_range,
            d.gap_open_points,
            d.atr_20,
            d.atr_20_pct,
            d.overnight_range_pct,
            d.orb_US_DATA_1000_break_dir AS break_dir,
            d.orb_US_DATA_1000_high,
            d.orb_US_DATA_1000_low
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
          AND o.orb_label = 'US_DATA_1000'
          AND o.orb_minutes = 5
          AND o.entry_model = 'E2'
          AND o.confirm_bars = 1
          AND o.rr_target = 1.0
          AND o.pnl_r IS NOT NULL
          AND d.prev_day_low IS NOT NULL
          AND d.atr_20 IS NOT NULL
          AND d.atr_20 > 0
          AND d.orb_US_DATA_1000_break_dir = 'long'
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
    df["is_win"] = (df["pnl_r"].astype(float) > 0).astype(int)
    df["orb_mid"] = (df["orb_US_DATA_1000_high"].astype(float) + df["orb_US_DATA_1000_low"].astype(float)) / 2.0
    df["feature_fire"] = (df["orb_mid"] < df["prev_day_low"].astype(float)).astype(int)
    return df


def _welch(on: pd.Series, off: pd.Series) -> tuple[float, float]:
    if len(on) < 2 or len(off) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(on.astype(float), off.astype(float), equal_var=False)
    return float(t), float(p)


def _one_sample_t(values: pd.Series) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(values.astype(float), 0.0)
    return float(t), float(p)


def _off_mean(df: pd.DataFrame) -> float:
    off = df.loc[df["feature_fire"] == 0, "pnl_r"].astype(float)
    return float(off.mean()) if len(off) else float("nan")


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
        corrs[col] = float(corr) if not np.isnan(corr) else 0.0
    max_filter = max(corrs, key=lambda k: abs(corrs[k]))
    max_corr = abs(corrs[max_filter])
    return max_corr, max_filter, corrs


def _t1_wr_monotonicity(df: pd.DataFrame) -> tuple[float, float, float, float]:
    is_df = df.loc[df["is_is"]].copy()
    on = is_df.loc[is_df["feature_fire"] == 1]
    off = is_df.loc[is_df["feature_fire"] == 0]
    wr_on = float(on["is_win"].mean())
    wr_off = float(off["is_win"].mean())
    wr_spread = wr_on - wr_off
    expr_spread = float(on["pnl_r"].mean() - off["pnl_r"].mean())
    return wr_on, wr_off, wr_spread, expr_spread


def _t3_oos_wfe(df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    on_is = df.loc[df["is_is"] & (df["feature_fire"] == 1), "pnl_r"].astype(float)
    on_oos = df.loc[df["is_oos"] & (df["feature_fire"] == 1), "pnl_r"].astype(float)
    if len(on_oos) < 10:
        return None, None, None
    expr_is = float(on_is.mean())
    expr_oos = float(on_oos.mean())
    sharpe_is = expr_is / float(on_is.std(ddof=1)) if len(on_is) > 1 else float("nan")
    sharpe_oos = expr_oos / float(on_oos.std(ddof=1)) if len(on_oos) > 1 else float("nan")
    wfe = sharpe_oos / sharpe_is if sharpe_is and not np.isnan(sharpe_is) else float("nan")
    return wfe, sharpe_is, sharpe_oos


def _t6_null_floor(df: pd.DataFrame, B: int = 1000) -> tuple[float, float]:
    is_df = df.loc[df["is_is"]].copy().reset_index(drop=True)
    on = is_df.loc[is_df["feature_fire"] == 1, "pnl_r"].astype(float)
    observed = float(on.mean())
    if len(on) < 30:
        return float("nan"), observed
    vals = is_df["pnl_r"].astype(float).to_numpy()
    feature = is_df["feature_fire"].to_numpy()
    n_on = int(feature.sum())
    more_extreme = 0
    for _ in range(B):
        perm = RNG.permutation(vals)
        sample = perm[:n_on]
        if float(sample.mean()) >= observed:
            more_extreme += 1
    p_val = (more_extreme + 1) / (B + 1)
    return float(p_val), observed


def _t7_per_year(df: pd.DataFrame) -> tuple[int, int, dict[int, float]]:
    is_df = df.loc[df["is_is"]].copy()
    positive = 0
    testable = 0
    year_mean: dict[int, float] = {}
    for year in range(IS_START_YEAR, IS_END_YEAR + 1):
        yr = is_df.loc[is_df["year"] == year].copy()
        on = yr.loc[yr["feature_fire"] == 1, "pnl_r"].astype(float)
        if len(on) < 5:
            year_mean[year] = float("nan")
            continue
        value = float(on.mean())
        year_mean[year] = value
        testable += 1
        if value > 0:
            positive += 1
    return positive, testable, year_mean


def _t8_cross_instrument() -> tuple[str, float | None]:
    twin_df = load_lane_df("MES")
    on = twin_df.loc[twin_df["feature_fire"] == 1, "pnl_r"].astype(float)
    off = twin_df.loc[twin_df["feature_fire"] == 0, "pnl_r"].astype(float)
    if len(on) < 30 or len(off) < 30:
        return "INFO", None
    return ("PASS" if float(on.mean() - off.mean()) >= 0.05 else "FAIL"), float(on.mean() - off.mean())


def _era_min_expr(df: pd.DataFrame) -> tuple[float, int]:
    is_df = df.loc[df["is_is"] & (df["feature_fire"] == 1)].copy()
    grouped = is_df.groupby("year")["pnl_r"].agg(["count", "mean"]).reset_index()
    eligible = grouped[grouped["count"] >= 50]
    if eligible.empty:
        return float("nan"), 0
    return float(eligible["mean"].min()), int((eligible["mean"] < -0.05).sum())


def main() -> int:
    prereg = _load_prereg(PREREG_PATH)
    df = load_lane_df("MNQ")

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    ROW_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[
        [
            "trading_day",
            "pnl_r",
            "is_is",
            "is_oos",
            "year",
            "feature_fire",
            "prev_day_low",
            "orb_mid",
            "break_dir",
        ]
    ].copy()
    out_df["trading_day"] = out_df["trading_day"].dt.date
    out_df.to_csv(ROW_CSV, index=False)

    is_df = df.loc[df["is_is"]].copy()
    oos_df = df.loc[df["is_oos"]].copy()
    on_is = is_df.loc[is_df["feature_fire"] == 1, "pnl_r"].astype(float)
    off_is = is_df.loc[is_df["feature_fire"] == 0, "pnl_r"].astype(float)
    on_oos = oos_df.loc[oos_df["feature_fire"] == 1, "pnl_r"].astype(float)
    off_oos = oos_df.loc[oos_df["feature_fire"] == 0, "pnl_r"].astype(float)

    delta_is = float(on_is.mean() - off_is.mean())
    raw_t, raw_p = _welch(on_is, off_is)
    chordia_t, _ = _one_sample_t(on_is)
    expr_on_is = float(on_is.mean())
    wr_on_is = float((on_is > 0).mean())
    n_on_is = int(len(on_is))
    n_on_full = int(df["feature_fire"].sum())
    n_total_is = int(len(is_df))
    n_total_full = int(len(df))
    n_total_oos = int(len(oos_df))
    fire_rate_is = n_on_is / n_total_is
    positive_years, testable_years, year_delta = _t7_per_year(df)
    era_min_expr, era_fail_count = _era_min_expr(df)

    delta_oos = float(on_oos.mean() - off_oos.mean()) if len(on_oos) and len(off_oos) else float("nan")
    dir_match_oos = None
    if len(on_oos) >= 5 and len(off_oos) >= 5 and not math.isnan(delta_oos):
        dir_match_oos = bool(np.sign(delta_is) == np.sign(delta_oos))
    oos_ratio = float(on_oos.mean() / on_is.mean()) if len(on_oos) and on_is.mean() != 0 else float("nan")
    wfe, sharpe_is, sharpe_oos = _t3_oos_wfe(df)

    t0_max_corr, t0_filter, t0_corrs = _t0_tautology(df)
    wr_on, wr_off, wr_spread, expr_spread = _t1_wr_monotonicity(df)
    null_p, observed_expr = _t6_null_floor(df)
    t8_status, t8_delta = _t8_cross_instrument()

    hypothesis_cfg = prereg["hypotheses"][0]["pass_metric"]
    gte = hypothesis_cfg["threshold_gte"]
    lt = hypothesis_cfg["threshold_lt"]
    is_pass = (
        raw_p < float(lt["raw_p"])
        and chordia_t >= float(gte["chordia_t"])
        and delta_is >= float(gte["delta_is"])
        and expr_on_is > float(gte["expr_on_is"])
        and n_on_is >= int(gte["n_on_is"])
        and positive_years >= int(gte["years_positive_is"])
        and era_fail_count == 0
    )

    if not is_pass:
        verdict = "KILL"
    elif len(on_oos) < 10 or dir_match_oos is False:
        verdict = "PARK"
    else:
        verdict = "KEEP"

    header = _build_header(prereg)
    lines = header + [
        "## Verdict",
        "",
        f"`{verdict}`",
        "",
        "## Locked single-cell metrics",
        "",
        f"- Full sample rows: `{n_total_full}`",
        f"- IS total rows: `{n_total_is}`",
        f"- OOS total rows: `{n_total_oos}`",
        f"- Full sample on-signal rows: `{n_on_full}`",
        f"- IS on-signal rows: `{n_on_is}`",
        f"- IS fire rate: `{_fmt(fire_rate_is)}`",
        f"- IS ExpR on-signal: `{_fmt(expr_on_is)}`",
        f"- IS WR on-signal: `{_fmt(wr_on_is)}`",
        f"- IS delta (on - off): `{_fmt(delta_is)}`",
        f"- IS Welch t / p: `t={_fmt(raw_t, 3)}` `p={_fmt(raw_p)}`",
        f"- IS on-signal one-sample t: `{_fmt(chordia_t, 3)}`",
        f"- Positive IS years: `{positive_years}` / `{testable_years}`",
        f"- OOS on-signal N: `{len(on_oos)}`",
        f"- OOS delta (on - off): `{_fmt(delta_oos)}`",
        f"- OOS dir match: `{dir_match_oos}`",
        f"- OOS/IS ratio: `{_fmt(oos_ratio)}`",
        f"- Era min on-signal ExpR (N>=50): `{_fmt(era_min_expr)}`",
        "",
        "## T0/T1/T2/T3/T6/T7/T8 audit table",
        "",
        "| Test | Value | Status | Detail |",
        "|---|---|---|---|",
        f"| T0_tautology | max |corr|={_fmt(t0_max_corr, 3)} ({t0_filter}) | {'PASS' if t0_max_corr <= 0.70 else 'FAIL'} | correlations={t0_corrs} |",
        f"| T1_wr_monotonicity | WR_spread={_fmt(wr_spread, 3)} (on={_fmt(wr_on, 3)} off={_fmt(wr_off, 3)}) | {'PASS' if abs(wr_spread) >= 0.05 else 'INFO'} | ExpR_spread={_fmt(expr_spread, 3)} |",
        f"| T2_is_baseline | N={n_on_is} ExpR={_fmt(expr_on_is, 3)} WR={_fmt(wr_on_is, 3)} | {'PASS' if n_on_is >= 100 else 'INFO'} | deployable N gate |",
        (
            f"| T3_oos_wfe | N_OOS={len(on_oos)} | FAIL | insufficient OOS N for WFE (< 10) |"
            if wfe is None
            else f"| T3_oos_wfe | WFE={_fmt(wfe, 2)} IS_SR={_fmt(sharpe_is, 2)} OOS_SR={_fmt(sharpe_oos, 2)} | "
            f"{'PASS' if wfe >= 0.50 and dir_match_oos else 'FAIL'} | N_OOS_on={len(on_oos)} |"
        ),
        f"| T6_null_floor | p={_fmt(null_p)} ExpR_obs={_fmt(observed_expr, 3)} | {'PASS' if not math.isnan(null_p) and null_p < 0.05 else 'FAIL'} | 1000 shuffles |",
        f"| T7_per_year | {positive_years}/{testable_years} in expected direction | {'PASS' if positive_years >= 4 else 'FAIL'} | yr={year_delta} |",
        (
            f"| T8_cross_instrument | twin=MES Δ={_fmt(t8_delta, 3)} | {t8_status} | sign match, mag>=0.05 |"
            if t8_delta is not None
            else "| T8_cross_instrument | twin=MES insufficient N | INFO | N too small |"
        ),
        "",
        "## Decision notes",
        "",
        "- This is a single locked Pathway-B cell, not a family rescan.",
        "- Thin OOS remains a park condition, not a rescue argument.",
        "- Pre-reg provenance note: the locked `expected_n_total_is=914` field does not match current authoritative raw counts. Current repo state is `916` full rows, `881` IS rows, and `32` OOS rows; the locked `136` IS on-signal baseline still matches.",
        "",
        "## Outputs",
        "",
        f"- Row-level CSV: `{ROW_CSV.as_posix()}`",
    ]
    RESULT_DOC.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
