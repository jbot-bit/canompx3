"""Frozen rel-vol sizer-rule replay for PR48 MES/MGC follow-through.

Pre-reg:
  docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml

Purpose:
  Turn the PR48 monotonic participation result into a concrete frozen
  per-lane sizing rule and replay it on 2026 OOS without reopening search.

No capital action is permitted under any verdict.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

PREREG_PATH = "docs/audit/hypotheses/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.yaml"
RESULT_DOC = Path("docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md")
BREAKPOINTS_CSV = Path("research/output/pr48_mes_mgc_sizer_rule_breakpoints_v1.csv")
METRICS_CSV = Path("research/output/pr48_mes_mgc_sizer_rule_metrics_v1.csv")

INSTRUMENTS = ["MES", "MGC"]
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
SIZE_MAP = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}


@dataclass
class ReplayMetrics:
    n: int
    mean_multiplier: float
    baseline_expr: float
    full_expr: float
    normalized_expr: float
    full_delta_expr: float
    normalized_delta_expr: float
    baseline_sharpe: float
    full_sharpe: float
    active_trade_share: float


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _safe_sharpe(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return float("nan")
    return float(values.mean() / std)


def _list_sessions(con: duckdb.DuckDBPyConnection, symbol: str) -> list[str]:
    rows = con.execute(
        """
        SELECT DISTINCT orb_label
        FROM orb_outcomes
        WHERE symbol = ?
          AND orb_minutes = ?
          AND entry_model = ?
          AND confirm_bars = ?
          AND rr_target = ?
          AND pnl_r IS NOT NULL
        ORDER BY orb_label
        """,
        [symbol, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, RR_TARGET],
    ).fetchall()
    return [row[0] for row in rows]


def _load_session(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    session: str,
) -> pd.DataFrame:
    rel_col = f"rel_vol_{session}"
    sql = f"""
    WITH df AS (
      SELECT d.trading_day, d.symbol, d.{rel_col} AS rel_vol
      FROM daily_features d
      WHERE d.symbol = '{symbol}'
        AND d.orb_minutes = {ORB_MINUTES}
    )
    SELECT o.trading_day, o.pnl_r, o.entry_price, o.stop_price, df.rel_vol
    FROM orb_outcomes o
    JOIN df
      ON o.trading_day = df.trading_day
     AND o.symbol = df.symbol
    WHERE o.symbol = '{symbol}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {ORB_MINUTES}
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR_TARGET}
      AND o.pnl_r IS NOT NULL
    """
    df = con.sql(sql).to_df()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["session"] = session
    df["direction"] = np.where(df["entry_price"] > df["stop_price"], "long", "short")
    df["lane"] = df["session"] + "_" + df["direction"]
    df["instrument"] = symbol
    return df


def _load_instrument(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    frames = []
    for session in _list_sessions(con, symbol):
        sub = _load_session(con, symbol, session)
        if not sub.empty:
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _split_is_oos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = pd.Timestamp(HOLDOUT_SACRED_FROM)
    is_df = df.loc[df["trading_day"] < holdout].reset_index(drop=True)
    oos_df = df.loc[df["trading_day"] >= holdout].reset_index(drop=True)
    return is_df, oos_df


def _freeze_breakpoints(is_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lane, lane_df in is_df.groupby("lane", sort=True):
        clean = lane_df["rel_vol"].dropna().astype(float)
        session = str(lane_df["session"].iloc[0])
        direction = str(lane_df["direction"].iloc[0])
        instrument = str(lane_df["instrument"].iloc[0])
        if len(clean) < 50:
            raise ValueError(f"{instrument} {lane} has only {len(clean)} IS rel_vol rows")
        qs = np.nanquantile(clean.to_numpy(), [0.20, 0.40, 0.60, 0.80])
        rows.append(
            {
                "instrument": instrument,
                "lane": lane,
                "session": session,
                "direction": direction,
                "is_n": int(len(clean)),
                "p20": float(qs[0]),
                "p40": float(qs[1]),
                "p60": float(qs[2]),
                "p80": float(qs[3]),
            }
        )
    return pd.DataFrame(rows).sort_values(["instrument", "session", "direction"]).reset_index(drop=True)


def _assign_bucket(rel_vol: float, p20: float, p40: float, p60: float, p80: float) -> int | None:
    if pd.isna(rel_vol):
        return None
    if rel_vol < p20:
        return 1
    if rel_vol < p40:
        return 2
    if rel_vol < p60:
        return 3
    if rel_vol < p80:
        return 4
    return 5


def _apply_rule(df: pd.DataFrame, breakpoints: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        breakpoints[["instrument", "lane", "p20", "p40", "p60", "p80"]],
        on=["instrument", "lane"],
        how="left",
        validate="many_to_one",
    ).copy()
    merged["bucket"] = [
        _assign_bucket(rv, p20, p40, p60, p80)
        for rv, p20, p40, p60, p80 in zip(
            merged["rel_vol"],
            merged["p20"],
            merged["p40"],
            merged["p60"],
            merged["p80"],
            strict=False,
        )
    ]
    merged["size_multiplier"] = merged["bucket"].map(SIZE_MAP)
    merged["sized_pnl_r"] = merged["pnl_r"] * merged["size_multiplier"].fillna(0.0)
    return merged


def _bucket_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket in [1, 2, 3, 4, 5]:
        sub = df.loc[df["bucket"] == bucket]
        rows.append(
            {
                "bucket": bucket,
                "n": int(len(sub)),
                "expr": _safe_mean(sub["pnl_r"]) if len(sub) else float("nan"),
                "win_rate": float((sub["pnl_r"] > 0).mean()) if len(sub) else float("nan"),
                "mean_multiplier": _safe_mean(sub["size_multiplier"]) if len(sub) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _compute_metrics(df: pd.DataFrame) -> ReplayMetrics:
    baseline = df["pnl_r"].astype(float)
    sized = df["sized_pnl_r"].astype(float)
    total_size = float(df["size_multiplier"].fillna(0.0).sum())
    normalized_expr = float(sized.sum() / total_size) if total_size > 0 else float("nan")
    baseline_expr = _safe_mean(baseline)
    full_expr = _safe_mean(sized)
    return ReplayMetrics(
        n=int(len(df)),
        mean_multiplier=_safe_mean(df["size_multiplier"].fillna(0.0)),
        baseline_expr=baseline_expr,
        full_expr=full_expr,
        normalized_expr=normalized_expr,
        full_delta_expr=full_expr - baseline_expr,
        normalized_delta_expr=normalized_expr - baseline_expr if math.isfinite(normalized_expr) else float("nan"),
        baseline_sharpe=_safe_sharpe(baseline),
        full_sharpe=_safe_sharpe(sized),
        active_trade_share=float((df["size_multiplier"].fillna(0.0) > 0).mean()) if len(df) else float("nan"),
    )


def _instrument_verdict(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    is_metrics: ReplayMetrics,
    oos_metrics: ReplayMetrics,
    is_buckets: pd.DataFrame,
    oos_buckets: pd.DataFrame,
) -> tuple[str, str]:
    is_coverage = float(is_df["rel_vol"].notna().mean()) if len(is_df) else 0.0
    oos_coverage = float(oos_df["rel_vol"].notna().mean()) if len(oos_df) else 0.0
    if len(is_df) < 5000 or len(oos_df) < 500 or is_coverage < 0.90 or oos_coverage < 0.90:
        reason = (
            f"integrity fail: IS N={len(is_df)}, OOS N={len(oos_df)}, "
            f"IS rel_vol={is_coverage:.1%}, OOS rel_vol={oos_coverage:.1%}"
        )
        return "SCAN_ABORT", reason
    is_q1 = float(is_buckets.loc[is_buckets["bucket"] == 1, "expr"].iloc[0])
    is_q5 = float(is_buckets.loc[is_buckets["bucket"] == 5, "expr"].iloc[0])
    oos_q1 = float(oos_buckets.loc[oos_buckets["bucket"] == 1, "expr"].iloc[0])
    oos_q5 = float(oos_buckets.loc[oos_buckets["bucket"] == 5, "expr"].iloc[0])
    if not (math.isfinite(is_q1) and math.isfinite(is_q5) and math.isfinite(oos_q1) and math.isfinite(oos_q5)):
        return "SCAN_ABORT", "bucket summaries are incomplete"
    if (
        oos_metrics.full_expr > oos_metrics.baseline_expr
        and oos_metrics.normalized_expr > oos_metrics.baseline_expr
        and oos_metrics.normalized_expr > 0
        and oos_q5 > oos_q1
    ):
        reason = (
            f"OOS full ExpR {oos_metrics.full_expr:+.3f} > baseline {oos_metrics.baseline_expr:+.3f}, "
            f"normalized ExpR {oos_metrics.normalized_expr:+.3f} > baseline, and Q5 {oos_q5:+.3f} > Q1 {oos_q1:+.3f}."
        )
        return "SIZER_DEPLOY_CANDIDATE", reason
    if oos_metrics.full_expr > oos_metrics.baseline_expr and oos_q5 > oos_q1:
        reason = (
            f"OOS take-home improves ({oos_metrics.full_expr:+.3f} vs {oos_metrics.baseline_expr:+.3f}) "
            f"and bucket ordering stays constructive (Q5 {oos_q5:+.3f} > Q1 {oos_q1:+.3f}), "
            f"but normalized ExpR or absolute OOS sign is not yet strong enough."
        )
        return "SIZER_ALIVE_NOT_READY", reason
    reason = (
        f"OOS replay fails: full ExpR {oos_metrics.full_expr:+.3f} vs baseline {oos_metrics.baseline_expr:+.3f}, "
        f"Q5 {oos_q5:+.3f} vs Q1 {oos_q1:+.3f}."
    )
    return "SIZER_NO_GO", reason


def _render_bucket_table(df: pd.DataFrame) -> str:
    lines = ["| Bucket | N | ExpR | Win Rate | Mean Multiplier |", "|---|---:|---:|---:|---:|"]
    for row in df.itertuples(index=False):
        lines.append(
            f"| Q{row.bucket} | {row.n} | "
            f"{row.expr:+.3f} | {row.win_rate:.1%} | {row.mean_multiplier:.2f} |"
        )
    return "\n".join(lines)


def _render_metrics_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Instrument | Split | N | Mean Size | Base ExpR | Sized ExpR | Normalized ExpR | Full Delta | Norm Delta | Base Sharpe | Sized Sharpe | Active Share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instrument']} | {row['split']} | {row['n']} | {row['mean_multiplier']:.2f} | "
            f"{row['baseline_expr']:+.3f} | {row['full_expr']:+.3f} | {row['normalized_expr']:+.3f} | "
            f"{row['full_delta_expr']:+.3f} | {row['normalized_delta_expr']:+.3f} | "
            f"{row['baseline_sharpe']:+.3f} | {row['full_sharpe']:+.3f} | {row['active_trade_share']:.1%} |"
        )
    return "\n".join(lines)


def main() -> int:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        instrument_frames = {instrument: _load_instrument(con, instrument) for instrument in INSTRUMENTS}
    finally:
        con.close()

    breakpoints_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    verdict_rows: list[dict[str, str]] = []
    bucket_sections: list[str] = []

    for instrument in INSTRUMENTS:
        frame = instrument_frames[instrument]
        if frame.empty:
            verdict_rows.append(
                {"instrument": instrument, "verdict": "SCAN_ABORT", "reason": "no canonical rows loaded"}
            )
            continue

        is_df, oos_df = _split_is_oos(frame)
        breakpoints = _freeze_breakpoints(is_df)
        breakpoints_frames.append(breakpoints)
        is_scored = _apply_rule(is_df, breakpoints)
        oos_scored = _apply_rule(oos_df, breakpoints)

        is_buckets = _bucket_table(is_scored)
        oos_buckets = _bucket_table(oos_scored)
        is_metrics = _compute_metrics(is_scored)
        oos_metrics = _compute_metrics(oos_scored)
        verdict, reason = _instrument_verdict(is_df, oos_df, is_metrics, oos_metrics, is_buckets, oos_buckets)
        verdict_rows.append({"instrument": instrument, "verdict": verdict, "reason": reason})

        for split_name, metrics in [("IS", is_metrics), ("OOS", oos_metrics)]:
            metrics_rows.append(
                {
                    "instrument": instrument,
                    "split": split_name,
                    "n": metrics.n,
                    "mean_multiplier": metrics.mean_multiplier,
                    "baseline_expr": metrics.baseline_expr,
                    "full_expr": metrics.full_expr,
                    "normalized_expr": metrics.normalized_expr,
                    "full_delta_expr": metrics.full_delta_expr,
                    "normalized_delta_expr": metrics.normalized_delta_expr,
                    "baseline_sharpe": metrics.baseline_sharpe,
                    "full_sharpe": metrics.full_sharpe,
                    "active_trade_share": metrics.active_trade_share,
                }
            )

        bucket_sections.extend(
            [
                f"### {instrument} IS buckets",
                "",
                _render_bucket_table(is_buckets),
                "",
                f"### {instrument} OOS buckets",
                "",
                _render_bucket_table(oos_buckets),
                "",
            ]
        )

    all_breakpoints = (
        pd.concat(breakpoints_frames, ignore_index=True)
        if breakpoints_frames
        else pd.DataFrame(columns=["instrument", "lane", "session", "direction", "is_n", "p20", "p40", "p60", "p80"])
    )
    metrics_df = pd.DataFrame(metrics_rows)

    BREAKPOINTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_breakpoints.to_csv(BREAKPOINTS_CSV, index=False)
    metrics_df.to_csv(METRICS_CSV, index=False)

    summary_lines = [
        "# PR48 MES/MGC frozen rel-vol sizer-rule backtest v1",
        "",
        f"**Pre-reg:** `{PREREG_PATH}`",
        f"**Script:** `research/pr48_mes_mgc_sizer_rule_backtest_v1.py`",
        f"**Breakpoints:** `{BREAKPOINTS_CSV}`",
        f"**Metrics artifact:** `{METRICS_CSV}`",
        "",
        "## Verdict summary",
        "",
        "| Instrument | Verdict | Reason |",
        "|---|---|---|",
    ]
    for row in verdict_rows:
        summary_lines.append(f"| {row['instrument']} | **{row['verdict']}** | {row['reason']} |")
    summary_lines.extend(
        [
            "",
            "## Replay metrics",
            "",
            _render_metrics_table(metrics_rows),
            "",
            "## Frozen bucket surfaces",
            "",
            *bucket_sections,
            "## Notes",
            "",
            "- `Sized ExpR` is the actual take-home replay `mean(size * pnl_r)`.",
            "- `Normalized ExpR` is `sum(size * pnl_r) / sum(size)` so leverage drift does not masquerade as edge improvement.",
            "- `Mean Size` is reported for both IS and OOS because the frozen quintile map averages 1.0x only under stationary bucket frequencies.",
            "- This result is research-only. No live profile, lane allocation, or runtime sizing was modified.",
        ]
    )

    RESULT_DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DOC.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    for row in verdict_rows:
        print(f"{row['instrument']}: {row['verdict']} - {row['reason']}")
    print(f"RESULT_DOC: {RESULT_DOC}")
    print(f"BREAKPOINTS_CSV: {BREAKPOINTS_CSV}")
    print(f"METRICS_CSV: {METRICS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
