"""Phase D Volume Pilot — Stage D-0 backtest.

Pre-reg: docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md
Stage file: docs/runtime/stages/phase-d-volume-pilot-d0.md

H1: Discrete rel_vol-based size scaling on MNQ COMEX_SETTLE O5 RR1.5 E2
produces Sharpe uplift >= 15% over binary 1x deployment on IS-only data.

Sizing rule (from spec section 3.2 discrete bucketing):
    size = 0.5x if rel_vol < P33
    size = 1.0x if P33 <= rel_vol <= P67
    size = 1.5x if rel_vol > P67

Mode A sacred: trading_day < HOLDOUT_SACRED_FROM only. 2026 OOS untouched.

Execute:
    uv run python research/phase_d_volume_sizing_pilot_d0.py

Writes:
    docs/audit/results/2026-04-17-phase-d-d0-backtest.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

INSTRUMENT = "MNQ"
SESSION = "COMEX_SETTLE"
ORB_MINUTES = 5
RR_TARGET = 1.5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1

PRIMARY_GATE_SHARPE_UPLIFT = 0.15
SECONDARY_GATE_MAXDD_MULTIPLIER = 1.5
SECONDARY_GATE_YEAR_MIN_RATIO = 0.8
SECONDARY_GATE_CORR_FLOOR = 0.05
TAUTOLOGY_MAX_ABS_CORR = 0.70

RESULT_MD = Path("docs/audit/results/2026-04-17-phase-d-d0-backtest.md")


@dataclass
class BacktestResult:
    label: str
    n_trades: int
    wr: float
    expR: float
    sharpe_ann: float
    max_dd_r: float
    per_year: pd.DataFrame


def _load_is_data() -> pd.DataFrame:
    """Join orb_outcomes with daily_features on pre-2026 MNQ COMEX_SETTLE O5.

    Returns one row per E2 trade day with pnl_r, direction, rel_vol, year.
    Triple-join on (trading_day, symbol, orb_minutes) per daily-features-joins.md.
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        q = f"""
        SELECT
            o.trading_day,
            o.symbol,
            d.orb_{SESSION}_break_dir AS break_dir,
            o.pnl_r,
            o.outcome,
            d.rel_vol_{SESSION} AS rel_vol,
            d.overnight_range,
            EXTRACT(YEAR FROM o.trading_day)::INT AS year
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
           AND o.symbol = d.symbol
           AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{INSTRUMENT}'
          AND o.orb_label = '{SESSION}'
          AND o.orb_minutes = {ORB_MINUTES}
          AND o.rr_target = {RR_TARGET}
          AND o.entry_model = '{ENTRY_MODEL}'
          AND o.confirm_bars = {CONFIRM_BARS}
          AND o.trading_day < '{HOLDOUT_SACRED_FROM.isoformat()}'
          AND o.pnl_r IS NOT NULL
          AND d.rel_vol_{SESSION} IS NOT NULL
        ORDER BY o.trading_day
        """
        return con.execute(q).fetchdf()
    finally:
        con.close()


def _assert_no_2026(df: pd.DataFrame) -> None:
    boundary = pd.Timestamp(HOLDOUT_SACRED_FROM)
    latest = pd.Timestamp(df["trading_day"].max())
    if latest >= boundary:
        raise RuntimeError(
            f"Mode A breach: latest trading_day {latest} >= sacred boundary {boundary}. "
            f"D-0 must be IS-only."
        )


def _compute_bucket_thresholds(rel_vol: pd.Series) -> tuple[float, float]:
    p33 = float(rel_vol.quantile(0.33))
    p67 = float(rel_vol.quantile(0.67))
    return p33, p67


def _apply_size_rule(rel_vol: pd.Series, p33: float, p67: float) -> pd.Series:
    size = pd.Series(1.0, index=rel_vol.index)
    size[rel_vol < p33] = 0.5
    size[rel_vol > p67] = 1.5
    return size


def _backtest(df: pd.DataFrame, size_mult: pd.Series, label: str) -> BacktestResult:
    """Compute ExpR / Sharpe / MaxDD on size-adjusted pnl_r."""
    pnl_sized = df["pnl_r"].values * size_mult.values
    n = len(pnl_sized)
    wr = float((pnl_sized > 0).sum() / n) if n > 0 else 0.0
    expR = float(np.mean(pnl_sized)) if n > 0 else 0.0

    # Sharpe annualized. Use trades-per-year from the observed window.
    years_span = (df["trading_day"].max() - df["trading_day"].min()).days / 365.25
    trades_per_year = n / years_span if years_span > 0 else 0.0
    sd = float(np.std(pnl_sized, ddof=1)) if n > 1 else 0.0
    sharpe_per_trade = expR / sd if sd > 0 else 0.0
    sharpe_ann = sharpe_per_trade * math.sqrt(trades_per_year)

    # MaxDD in R units (peak-to-trough of cumulative R).
    cumR = np.cumsum(pnl_sized)
    running_peak = np.maximum.accumulate(cumR)
    dd = cumR - running_peak
    max_dd_r = float(-dd.min()) if n > 0 else 0.0

    # Per-year breakdown.
    per_year_rows = []
    for yr, grp in df.assign(pnl_sized=pnl_sized).groupby("year"):
        pnl_y = grp["pnl_sized"].values
        n_y = len(pnl_y)
        wr_y = float((pnl_y > 0).sum() / n_y) if n_y > 0 else 0.0
        mean_y = float(np.mean(pnl_y)) if n_y > 0 else 0.0
        sd_y = float(np.std(pnl_y, ddof=1)) if n_y > 1 else 0.0
        sharpe_y_trade = mean_y / sd_y if sd_y > 0 else 0.0
        # Annualize by trades-per-year within this year.
        # Approximate 252 trading days; use actual n_y for variance.
        sharpe_y_ann = sharpe_y_trade * math.sqrt(n_y) if n_y > 0 else 0.0
        per_year_rows.append(
            {
                "year": int(yr),
                "n": n_y,
                "wr": wr_y,
                "expR": mean_y,
                "sharpe_ann": sharpe_y_ann,
                "sum_r": float(np.sum(pnl_y)),
            }
        )
    per_year = pd.DataFrame(per_year_rows).sort_values("year")

    return BacktestResult(
        label=label,
        n_trades=n,
        wr=wr,
        expR=expR,
        sharpe_ann=sharpe_ann,
        max_dd_r=max_dd_r,
        per_year=per_year,
    )


def _correlation(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return 0.0
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(a.values, b.values)[0, 1])


def _write_result_md(
    df: pd.DataFrame,
    p33: float,
    p67: float,
    baseline: BacktestResult,
    scaled: BacktestResult,
    corr_size_pnl: float,
    tautology_corr: float,
    verdict: str,
    gate_details: dict[str, str],
) -> None:
    lines: list[str] = []
    a = lines.append
    a("# Phase D Volume Pilot — Stage D-0 Backtest")
    a("")
    a(f"**Date:** {date.today().isoformat()}")
    a("**Pre-reg:** `docs/audit/hypotheses/2026-04-15-phase-d-volume-pilot-spec.md`")
    a("**Stage file:** `docs/runtime/stages/phase-d-volume-pilot-d0.md`")
    a(f"**Verdict:** `{verdict}`")
    a("")
    a("## Universe")
    a(f"- Instrument: {INSTRUMENT}")
    a(f"- Session: {SESSION}")
    a(f"- Aperture: O{ORB_MINUTES}")
    a(f"- RR: {RR_TARGET}")
    a(f"- Entry model: {ENTRY_MODEL} / CB{CONFIRM_BARS}")
    a(f"- Window: IS-only, trading_day < {HOLDOUT_SACRED_FROM.isoformat()}")
    a(f"- Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")
    a(f"- N = {len(df)}")
    a("")
    a("## Sizing rule (discrete bucketing)")
    a(f"- P33 of rel_vol_{SESSION} = {p33:.4f}")
    a(f"- P67 of rel_vol_{SESSION} = {p67:.4f}")
    a("- rel_vol < P33 -> 0.5x")
    a("- P33 <= rel_vol <= P67 -> 1.0x")
    a("- rel_vol > P67 -> 1.5x")
    a("")
    a("## Headline")
    a("")
    a("| Metric | Baseline 1x | Scaled | Ratio |")
    a("|---|---|---|---|")
    a(f"| N trades | {baseline.n_trades} | {scaled.n_trades} | 1.000 |")
    a(f"| WR | {baseline.wr:.4f} | {scaled.wr:.4f} | {scaled.wr / baseline.wr if baseline.wr else 0:.3f} |")
    a(f"| ExpR | {baseline.expR:.4f} | {scaled.expR:.4f} | {scaled.expR / baseline.expR if baseline.expR else 0:.3f} |")
    a(f"| Sharpe_ann | {baseline.sharpe_ann:.4f} | {scaled.sharpe_ann:.4f} | {scaled.sharpe_ann / baseline.sharpe_ann if baseline.sharpe_ann else 0:.3f} |")
    a(f"| MaxDD (R) | {baseline.max_dd_r:.4f} | {scaled.max_dd_r:.4f} | {scaled.max_dd_r / baseline.max_dd_r if baseline.max_dd_r else 0:.3f} |")
    a("")
    a("## Gate evaluation")
    for k, v in gate_details.items():
        a(f"- **{k}:** {v}")
    a("")
    a(f"- Correlation(size_multiplier, pnl_r) = {corr_size_pnl:.4f} (floor {SECONDARY_GATE_CORR_FLOOR})")
    a(f"- T0 tautology: |corr(size_multiplier, overnight_range_pct)| = {abs(tautology_corr):.4f} (max {TAUTOLOGY_MAX_ABS_CORR})")
    a("")
    a("## Per-year breakdown — baseline")
    a("")
    a("| Year | N | WR | ExpR | Sharpe | SumR |")
    a("|---|---|---|---|---|---|")
    for _, r in baseline.per_year.iterrows():
        a(f"| {int(r['year'])} | {int(r['n'])} | {r['wr']:.4f} | {r['expR']:.4f} | {r['sharpe_ann']:.4f} | {r['sum_r']:.2f} |")
    a("")
    a("## Per-year breakdown — scaled")
    a("")
    a("| Year | N | WR | ExpR | Sharpe | SumR |")
    a("|---|---|---|---|---|---|")
    for _, r in scaled.per_year.iterrows():
        a(f"| {int(r['year'])} | {int(r['n'])} | {r['wr']:.4f} | {r['expR']:.4f} | {r['sharpe_ann']:.4f} | {r['sum_r']:.2f} |")
    a("")
    a("## Interpretation")
    if verdict == "PASS":
        a(
            "- D-0 gate cleared. Proceed to D-1 (signal-only shadow) per the Phase D pre-reg, "
            "pending user approval of the 4-week shadow timeline and the `pre_registered_criteria.md` "
            "secondary review (DSR at multi-K framings, Carver Ch 9-10 grounding)."
        )
    else:
        a("- D-0 gate FAILED. Phase D stops here per the Phase D pre-reg kill rules.")
        a("  Do not proceed to D-1. Do not adjust bucket thresholds post hoc.")
        a("  Next step is a postmortem and decision on whether to abandon Phase D or pre-register V2.")
    a("")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Phase D D-0 backtest starting...")
    df = _load_is_data()
    _assert_no_2026(df)
    if len(df) < 100:
        print(f"ERROR: too few IS rows (N={len(df)}). Aborting.")
        return 2

    per_year_counts = df.groupby("year").size()
    print(f"Loaded {len(df)} IS E2 trade days. Range {df['trading_day'].min()} -> {df['trading_day'].max()}")
    print(f"Per-year trade counts:\n{per_year_counts.to_string()}")

    p33, p67 = _compute_bucket_thresholds(df["rel_vol"])
    print(f"rel_vol P33={p33:.4f}, P67={p67:.4f}")

    size_scaled = _apply_size_rule(df["rel_vol"], p33, p67)
    size_baseline = pd.Series(1.0, index=df.index)

    bucket_counts = size_scaled.value_counts().sort_index()
    print(f"Bucket counts:\n{bucket_counts.to_string()}")

    baseline = _backtest(df, size_baseline, "baseline_1x")
    scaled = _backtest(df, size_scaled, "scaled_discrete_3_tier")

    corr_size_pnl = _correlation(size_scaled, df["pnl_r"])
    # T0 tautology: proxy for OVNRNG_100 firing (overnight_range >= 100 points).
    tautology_corr = _correlation(size_scaled, df["overnight_range"].fillna(0.0))

    sharpe_ratio = scaled.sharpe_ann / baseline.sharpe_ann if baseline.sharpe_ann else 0.0
    uplift = sharpe_ratio - 1.0
    primary_pass = uplift >= PRIMARY_GATE_SHARPE_UPLIFT
    maxdd_pass = scaled.max_dd_r <= SECONDARY_GATE_MAXDD_MULTIPLIER * baseline.max_dd_r

    per_year_ok = True
    for (_, r_base), (_, r_scl) in zip(
        baseline.per_year.iterrows(), scaled.per_year.iterrows(), strict=True
    ):
        if r_base["sharpe_ann"] > 0 and r_scl["sharpe_ann"] < SECONDARY_GATE_YEAR_MIN_RATIO * r_base["sharpe_ann"]:
            per_year_ok = False
            break

    corr_pass = corr_size_pnl > SECONDARY_GATE_CORR_FLOOR
    tautology_pass = abs(tautology_corr) < TAUTOLOGY_MAX_ABS_CORR

    all_pass = primary_pass and maxdd_pass and per_year_ok and corr_pass and tautology_pass
    verdict = "PASS" if all_pass else "FAIL"

    gate_details = {
        "primary (Sharpe uplift >= 15%)": (
            f"{'PASS' if primary_pass else 'FAIL'} (uplift = {uplift * 100:.2f}%, ratio = {sharpe_ratio:.3f})"
        ),
        "secondary MaxDD (<= 1.5x baseline)": (
            f"{'PASS' if maxdd_pass else 'FAIL'} "
            f"(scaled {scaled.max_dd_r:.2f} / baseline {baseline.max_dd_r:.2f} "
            f"= {scaled.max_dd_r / baseline.max_dd_r if baseline.max_dd_r else 0:.3f}x)"
        ),
        "secondary per-year (scaled >= 0.8x baseline Sharpe each year)": (
            f"{'PASS' if per_year_ok else 'FAIL'}"
        ),
        "secondary corr(size, pnl) > 0.05": (
            f"{'PASS' if corr_pass else 'FAIL'} (corr = {corr_size_pnl:.4f})"
        ),
        "T0 tautology |corr(size, overnight_range_pct)| < 0.70": (
            f"{'PASS' if tautology_pass else 'FAIL'} (corr = {tautology_corr:.4f})"
        ),
    }

    print("")
    print(f"Baseline 1x: N={baseline.n_trades} WR={baseline.wr:.4f} ExpR={baseline.expR:.4f} Sharpe={baseline.sharpe_ann:.4f}")
    print(f"Scaled 3t:   N={scaled.n_trades} WR={scaled.wr:.4f} ExpR={scaled.expR:.4f} Sharpe={scaled.sharpe_ann:.4f}")
    print(f"Sharpe uplift: {uplift * 100:.2f}% (gate +15%)")
    print("Gates:")
    for k, v in gate_details.items():
        print(f"  {k}: {v}")
    print(f"VERDICT: {verdict}")

    _write_result_md(df, p33, p67, baseline, scaled, corr_size_pnl, tautology_corr, verdict, gate_details)
    print(f"Result written to {RESULT_MD}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
