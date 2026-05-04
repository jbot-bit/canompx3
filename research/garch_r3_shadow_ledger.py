"""Frozen forward-shadow ledger for GARCH R3 sizing on an execution profile.

Purpose:
  Generate the first honest monitoring artifact for the frozen SESSION_CLIPPED
  GARCH R3 sizing policy without changing live execution or reopening search.

Source of truth:
  - current profile lane set from trading_app.prop_profiles
  - canonical forward outcomes from orb_outcomes via _load_strategy_outcomes
  - trade-day state from daily_features.garch_forecast_vol_pct

This script is intentionally research-only. It does not write to trading_app
tables and does not consume holdout data for tuning.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import pandas as pd

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.paths import GOLD_DB_PATH
from research import garch_profile_continuous_sizing_replay as cont
from research import garch_profile_production_replay as replay
from trading_app.config import apply_tight_stop
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.prop_profiles import get_profile
from trading_app.strategy_fitness import _load_strategy_outcomes

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
PRE_REG_REF = "draft PR #65 / research(garch): add R3 shadow prereg / commit b73914c9"
NORMALIZATION_FACTOR = 0.973
FROZEN_POLICY = "SESSION_CLIPPED"
FROZEN_SUPPORT = {
    "BRISBANE_1025": {"high_dir": True, "low_dir": False},
    "CME_PRECLOSE": {"high_dir": True, "low_dir": True},
    "CME_REOPEN": {"high_dir": False, "low_dir": False},
    "COMEX_SETTLE": {"high_dir": True, "low_dir": True},
    "EUROPE_FLOW": {"high_dir": True, "low_dir": True},
    "LONDON_METALS": {"high_dir": True, "low_dir": True},
    "NYSE_CLOSE": {"high_dir": False, "low_dir": False},
    "NYSE_OPEN": {"high_dir": False, "low_dir": False},
    "SINGAPORE_OPEN": {"high_dir": True, "low_dir": True},
    "TOKYO_OPEN": {"high_dir": True, "low_dir": True},
    "US_DATA_1000": {"high_dir": True, "low_dir": False},
    "US_DATA_830": {"high_dir": False, "low_dir": False},
}


def _ann_sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    s = pd.Series(values, dtype=float)
    sd = float(s.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((float(s.mean()) / sd) * math.sqrt(252.0))


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return float(worst)


def _worst_rolling_sum(values: list[float], window: int) -> float:
    s = pd.Series(values, dtype=float)
    roll = s.rolling(window).sum()
    if roll.notna().any():
        return float(roll.min())
    return 0.0


def _base_output_stem(profile_id: str) -> str:
    return f"garch-r3-session-clipped-shadow-{profile_id.replace('_', '-')}"


def _raw_weight(session: str, gp: float | None) -> tuple[float, str, bool]:
    if gp is None:
        return 1.0, "NEUTRAL_1X_MISSING", True
    support = FROZEN_SUPPORT.get(session, {"high_dir": False, "low_dir": False})
    if support.get("low_dir") and gp <= 30.0:
        return 0.5, "LOW_CUT", False
    if support.get("high_dir") and gp >= 70.0:
        return 1.5, "HIGH_BOOST", False
    return 1.0, "NEUTRAL_1X", False


def _effective_stop_by_strategy(profile_id: str) -> dict[str, float]:
    profile = get_profile(profile_id)
    lane_specs = profile.daily_lanes if profile.daily_lanes else replay.load_allocation_lanes(profile.profile_id)
    return {lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier) for lane in lane_specs}


def _translated_contracts(weight: float) -> int:
    return cont.translated_contracts(weight)


def _audit_feature_gaps(
    con: duckdb.DuckDBPyConnection,
    lane_defs: list[dict],
    start_date: date,
    as_of: date,
) -> pd.DataFrame:
    columns = ["trading_day", "instrument", "orb_minutes"]
    rows: list[dict[str, object]] = []
    for instrument, orb_minutes in sorted({(lane["instrument"], lane["orb_minutes"]) for lane in lane_defs}):
        gaps = con.execute(
            """
            SELECT trading_day
            FROM daily_features
            WHERE symbol = ?
              AND orb_minutes = ?
              AND trading_day >= ?
              AND trading_day <= ?
              AND garch_forecast_vol_pct IS NULL
            ORDER BY trading_day
            """,
            [instrument, orb_minutes, start_date, as_of],
        ).fetchall()
        for (trading_day,) in gaps:
            rows.append(
                {
                    "trading_day": trading_day,
                    "instrument": instrument,
                    "orb_minutes": orb_minutes,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(columns).reset_index(drop=True)


def build_trade_ledger(
    profile_id: str,
    start_date: date,
) -> tuple[pd.DataFrame, list[dict[str, object]], date, pd.DataFrame]:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    as_of = replay.latest_trading_day(con)
    lane_defs = replay.lane_definitions_for_profile(profile_id)
    feature_cache = replay.build_feature_cache(con, lane_defs)
    feature_gap_df = _audit_feature_gaps(con, lane_defs, start_date, as_of)
    stop_by_strategy = _effective_stop_by_strategy(profile_id)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for lane in lane_defs:
        outcomes = _load_strategy_outcomes(
            con,
            instrument=lane["instrument"],
            orb_label=lane["orb_label"],
            orb_minutes=lane["orb_minutes"],
            entry_model=lane["entry_model"],
            rr_target=lane["rr_target"],
            confirm_bars=lane["confirm_bars"],
            filter_type=lane["filter_type"],
            start_date=start_date,
            end_date=as_of,
        )
        stop_multiplier = stop_by_strategy.get(lane["strategy_id"], get_profile(profile_id).stop_multiplier)
        if stop_multiplier != 1.0:
            cost_spec = get_cost_spec(lane["instrument"])
            outcomes = apply_tight_stop(outcomes, stop_multiplier, cost_spec)

        kept = 0
        for outcome in outcomes:
            if outcome.get("outcome") not in ("win", "loss"):
                continue
            pnl_r = outcome.get("pnl_r")
            entry_price = outcome.get("entry_price")
            stop_price = outcome.get("stop_price")
            if pnl_r is None or entry_price is None or stop_price is None:
                continue
            kept += 1
            td = outcome["trading_day"]
            feat = feature_cache.get((lane["instrument"], lane["orb_minutes"], td), {})
            gp = feat.get("garch_forecast_vol_pct")
            raw_weight, bucket, used_fallback = _raw_weight(lane["orb_label"], gp)
            shadow_weight = raw_weight * NORMALIZATION_FACTOR
            translated_contracts = _translated_contracts(shadow_weight)
            cost_spec = get_cost_spec(lane["instrument"])
            risk_dollars = float(risk_in_dollars(cost_spec, float(entry_price), float(stop_price)))
            base_r = float(pnl_r)
            shadow_r = base_r * shadow_weight
            translated_r = base_r * translated_contracts
            base_dollars = base_r * risk_dollars
            shadow_dollars = base_dollars * shadow_weight
            translated_dollars = base_dollars * translated_contracts
            rows.append(
                {
                    "trading_day": td,
                    "strategy_id": lane["strategy_id"],
                    "instrument": lane["instrument"],
                    "orb_label": lane["orb_label"],
                    "orb_minutes": lane["orb_minutes"],
                    "entry_model": lane["entry_model"],
                    "rr_target": lane["rr_target"],
                    "confirm_bars": lane["confirm_bars"],
                    "filter_type": lane["filter_type"],
                    "garch_forecast_vol_pct": gp,
                    "raw_weight": raw_weight,
                    "shadow_weight": shadow_weight,
                    "shadow_size_bucket": bucket,
                    "used_missing_fallback": used_fallback,
                    "translated_contracts": translated_contracts,
                    "base_size": 1.0,
                    "pnl_r_base": base_r,
                    "pnl_r_shadow": shadow_r,
                    "pnl_r_delta": shadow_r - base_r,
                    "pnl_r_translated": translated_r,
                    "pnl_r_translated_delta": translated_r - base_r,
                    "risk_dollars_1x": risk_dollars,
                    "pnl_dollars_base": base_dollars,
                    "pnl_dollars_shadow": shadow_dollars,
                    "pnl_dollars_delta": shadow_dollars - base_dollars,
                    "pnl_dollars_translated": translated_dollars,
                    "pnl_dollars_translated_delta": translated_dollars - base_dollars,
                    "execution_source": "canonical_forward",
                }
            )

        if kept == 0:
            skipped.append(
                {
                    "strategy_id": lane["strategy_id"],
                    "instrument": lane["instrument"],
                    "orb_label": lane["orb_label"],
                    "reason": "no canonical forward trades after exact filter application",
                }
            )

    con.close()
    if not rows:
        raise ValueError(f"No replayable forward trades found for {profile_id!r} from {start_date}")
    trade_df = pd.DataFrame(rows).sort_values(["trading_day", "orb_label", "strategy_id"]).reset_index(drop=True)
    return trade_df, skipped, as_of, feature_gap_df


def build_daily_ledger(trade_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        trade_df.groupby("trading_day", as_index=False)
        .agg(
            base_r=("pnl_r_base", "sum"),
            shadow_r=("pnl_r_shadow", "sum"),
            delta_r=("pnl_r_delta", "sum"),
            translated_r=("pnl_r_translated", "sum"),
            translated_delta_r=("pnl_r_translated_delta", "sum"),
            base_dollars=("pnl_dollars_base", "sum"),
            shadow_dollars=("pnl_dollars_shadow", "sum"),
            delta_dollars=("pnl_dollars_delta", "sum"),
            translated_dollars=("pnl_dollars_translated", "sum"),
            translated_delta_dollars=("pnl_dollars_translated_delta", "sum"),
            trades=("strategy_id", "count"),
            changed_trades=("shadow_weight", lambda s: int((s != 1.0).sum())),
            missing_fallback_trades=("used_missing_fallback", "sum"),
        )
        .sort_values("trading_day")
        .reset_index(drop=True)
    )
    for col in [
        "base_r",
        "shadow_r",
        "delta_r",
        "translated_r",
        "translated_delta_r",
        "base_dollars",
        "shadow_dollars",
        "delta_dollars",
        "translated_dollars",
        "translated_delta_dollars",
    ]:
        daily[f"cum_{col}"] = daily[col].cumsum()
    return daily


def summarize(
    profile_id: str,
    trade_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    skipped: list[dict[str, object]],
    feature_gap_df: pd.DataFrame,
) -> dict[str, object]:
    profile = get_profile(profile_id)
    total_trades = int(len(trade_df))
    changed = int((trade_df["shadow_weight"] != 1.0).sum())
    signal_changed = int((trade_df["raw_weight"] != 1.0).sum())
    neutral = int((trade_df["shadow_size_bucket"].isin(["NEUTRAL_1X", "NEUTRAL_1X_MISSING"])).sum())
    missing = int(trade_df["used_missing_fallback"].sum())
    translated_changed = int((trade_df["translated_contracts"] != 1).sum())
    missing_rows = (
        trade_df.loc[trade_df["used_missing_fallback"], ["trading_day", "orb_label", "strategy_id"]]
        .sort_values(["trading_day", "orb_label", "strategy_id"])
        .reset_index(drop=True)
    )
    per_session = (
        trade_df.groupby("orb_label", as_index=False)
        .agg(
            delta_r=("pnl_r_delta", "sum"),
            delta_dollars=("pnl_dollars_delta", "sum"),
            trades=("strategy_id", "count"),
        )
        .sort_values("delta_dollars", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {
        "profile_id": profile_id,
        "copies": profile.copies,
        "total_trades": total_trades,
        "changed_trades": changed,
        "signal_changed_trades": signal_changed,
        "neutral_trades": neutral,
        "missing_fallback_trades": missing,
        "translated_changed_trades": translated_changed,
        "pct_trades_changed": float(changed / total_trades),
        "pct_signal_changed": float(signal_changed / total_trades),
        "pct_neutral_1x": float(neutral / total_trades),
        "pct_missing_state_fallback": float(missing / total_trades),
        "pct_translated_changed": float(translated_changed / total_trades),
        "base_total_r": float(trade_df["pnl_r_base"].sum()),
        "shadow_total_r": float(trade_df["pnl_r_shadow"].sum()),
        "delta_total_r": float(trade_df["pnl_r_delta"].sum()),
        "translated_total_r": float(trade_df["pnl_r_translated"].sum()),
        "translated_delta_total_r": float(trade_df["pnl_r_translated_delta"].sum()),
        "base_total_dollars": float(trade_df["pnl_dollars_base"].sum()),
        "shadow_total_dollars": float(trade_df["pnl_dollars_shadow"].sum()),
        "delta_total_dollars": float(trade_df["pnl_dollars_delta"].sum()),
        "translated_total_dollars": float(trade_df["pnl_dollars_translated"].sum()),
        "translated_delta_total_dollars": float(trade_df["pnl_dollars_translated_delta"].sum()),
        "base_sharpe_r": _ann_sharpe(daily_df["base_r"].tolist()),
        "shadow_sharpe_r": _ann_sharpe(daily_df["shadow_r"].tolist()),
        "translated_sharpe_r": _ann_sharpe(daily_df["translated_r"].tolist()),
        "base_sharpe_dollars": _ann_sharpe(daily_df["base_dollars"].tolist()),
        "shadow_sharpe_dollars": _ann_sharpe(daily_df["shadow_dollars"].tolist()),
        "translated_sharpe_dollars": _ann_sharpe(daily_df["translated_dollars"].tolist()),
        "base_max_dd_r": _max_drawdown(daily_df["base_r"].tolist()),
        "shadow_max_dd_r": _max_drawdown(daily_df["shadow_r"].tolist()),
        "translated_max_dd_r": _max_drawdown(daily_df["translated_r"].tolist()),
        "base_max_dd_dollars": _max_drawdown(daily_df["base_dollars"].tolist()),
        "shadow_max_dd_dollars": _max_drawdown(daily_df["shadow_dollars"].tolist()),
        "translated_max_dd_dollars": _max_drawdown(daily_df["translated_dollars"].tolist()),
        "worst_day_delta_dollars": float(daily_df["delta_dollars"].min()) if len(daily_df) else 0.0,
        "worst_5day_delta_dollars": _worst_rolling_sum(daily_df["delta_dollars"].tolist(), 5),
        "per_session": per_session,
        "missing_rows": missing_rows,
        "feature_gap_rows": feature_gap_df,
        "skipped_lanes": skipped,
    }
    metrics["shadow_sharpe_delta_r"] = metrics["shadow_sharpe_r"] - metrics["base_sharpe_r"]
    metrics["translated_sharpe_delta_r"] = metrics["translated_sharpe_r"] - metrics["base_sharpe_r"]
    metrics["shadow_sharpe_delta_dollars"] = metrics["shadow_sharpe_dollars"] - metrics["base_sharpe_dollars"]
    metrics["translated_sharpe_delta_dollars"] = metrics["translated_sharpe_dollars"] - metrics["base_sharpe_dollars"]
    metrics["shadow_max_dd_delta_r"] = metrics["shadow_max_dd_r"] - metrics["base_max_dd_r"]
    metrics["translated_max_dd_delta_r"] = metrics["translated_max_dd_r"] - metrics["base_max_dd_r"]
    metrics["shadow_max_dd_delta_dollars"] = metrics["shadow_max_dd_dollars"] - metrics["base_max_dd_dollars"]
    metrics["translated_max_dd_delta_dollars"] = metrics["translated_max_dd_dollars"] - metrics["base_max_dd_dollars"]
    if metrics["pct_missing_state_fallback"] > 0.05:
        metrics["launch_status"] = "BLOCKED"
        metrics["launch_reason"] = "missing-state fallback exceeds prereg 5% threshold"
    else:
        metrics["launch_status"] = "READY_FOR_FORWARD_MONITORING"
        metrics["launch_reason"] = "state coverage is within prereg launch tolerance"
    return metrics


def write_report(
    output_md: Path,
    as_of: date,
    start_date: date,
    profile_id: str,
    trade_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    metrics: dict[str, object],
    trade_csv: Path,
    daily_csv: Path,
) -> None:
    profile = get_profile(profile_id)
    per_session = metrics["per_session"]
    lines = [
        "# GARCH R3 Session-Clipped Shadow Ledger",
        "",
        f"**Date:** {as_of}",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, active={profile.active})",
        f"**Pre-reg:** `{PRE_REG_REF}`",
        f"**Policy:** `{FROZEN_POLICY}` with frozen normalization factor `{NORMALIZATION_FACTOR}`",
        f"**Forward window:** `{start_date}` to `{as_of}`",
        "**Source of truth:** current profile lane set + canonical forward outcomes + daily_features.garch_forecast_vol_pct",
        "**Execution status:** shadow-only; no live sizing change",
        "",
        "## Verdict",
        "",
        f"- Launch status: **{metrics['launch_status']}**",
        f"- Gate reason: `{metrics['launch_reason']}`",
        "- This is the next action from the stale-lock audit: operationalize the surviving GARCH `R3` path as a frozen monitoring ledger.",
        "- This is not a router, not a filter scan, and not a deployment claim.",
        "- The translated `0/1/2` contract column is diagnostic only; the primary shadow path is the normalized weight.",
        "",
        "## Coverage",
        "",
        f"- Trades covered: `{metrics['total_trades']}`",
        f"- Trading days covered: `{len(daily_df)}`",
        f"- Trades with non-neutral raw signal: `{metrics['signal_changed_trades']}` (`{metrics['pct_signal_changed']:.1%}`)",
        f"- Trades with non-1x normalized shadow weight: `{metrics['changed_trades']}` (`{metrics['pct_trades_changed']:.1%}`)",
        f"- Neutral 1x trades: `{metrics['neutral_trades']}` (`{metrics['pct_neutral_1x']:.1%}`)",
        f"- Missing-state fallback trades: `{metrics['missing_fallback_trades']}` (`{metrics['pct_missing_state_fallback']:.1%}`)",
        f"- Translated 0/1/2 contract changes: `{metrics['translated_changed_trades']}` (`{metrics['pct_translated_changed']:.1%}`)",
        f"- Raw feature-gap rows in `daily_features`: `{len(metrics['feature_gap_rows'])}`",
        "",
        "## Shadow vs Base",
        "",
        "| Path | Total R | Delta R | Total $ | Delta $ | Daily Sharpe Δ (R) | Daily Sharpe Δ ($) | MaxDD ΔR | MaxDD Δ$ |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Shadow weight | {metrics['shadow_total_r']:+.3f} | {metrics['delta_total_r']:+.3f} | {metrics['shadow_total_dollars']:+,.1f} | {metrics['delta_total_dollars']:+,.1f} | {metrics['shadow_sharpe_delta_r']:+.3f} | {metrics['shadow_sharpe_delta_dollars']:+.3f} | {metrics['shadow_max_dd_delta_r']:+.3f} | {metrics['shadow_max_dd_delta_dollars']:+,.1f} |",
        f"| Translated 0/1/2 | {metrics['translated_total_r']:+.3f} | {metrics['translated_delta_total_r']:+.3f} | {metrics['translated_total_dollars']:+,.1f} | {metrics['translated_delta_total_dollars']:+,.1f} | {metrics['translated_sharpe_delta_r']:+.3f} | {metrics['translated_sharpe_delta_dollars']:+.3f} | {metrics['translated_max_dd_delta_r']:+.3f} | {metrics['translated_max_dd_delta_dollars']:+,.1f} |",
        "",
        "## Risk diagnostics",
        "",
        f"- Worst day delta $: `{metrics['worst_day_delta_dollars']:+,.1f}`",
        f"- Worst 5-day delta $: `{metrics['worst_5day_delta_dollars']:+,.1f}`",
        "",
        "## Session contribution",
        "",
        "| Session | Trades | Delta R | Delta $ |",
        "|---|---|---|---|",
    ]
    for row in per_session.itertuples(index=False):
        lines.append(f"| {row.orb_label} | {int(row.trades)} | {row.delta_r:+.3f} | {row.delta_dollars:+,.1f} |")

    missing_rows = metrics["missing_rows"]
    if len(missing_rows):
        lines += [
            "",
            "## Missing-state rows",
            "",
            "| Trading day | Session | Strategy |",
            "|---|---|---|",
        ]
        for row in missing_rows.itertuples(index=False):
            lines.append(f"| {row.trading_day} | {row.orb_label} | {row.strategy_id} |")

    feature_gap_rows = metrics["feature_gap_rows"]
    if len(feature_gap_rows):
        lines += [
            "",
            "## Raw feature gaps",
            "",
            "| Trading day | Instrument | Orb minutes |",
            "|---|---|---|",
        ]
        for row in feature_gap_rows.itertuples(index=False):
            lines.append(f"| {row.trading_day} | {row.instrument} | {row.orb_minutes} |")

    if metrics["skipped_lanes"]:
        lines += [
            "",
            "## Skipped lanes",
            "",
            "| Strategy | Session | Reason |",
            "|---|---|---|",
        ]
        for row in metrics["skipped_lanes"]:
            lines.append(f"| {row['strategy_id']} | {row['orb_label']} | {row['reason']} |")

    lines += [
        "",
        "## Artifacts",
        "",
        f"- Trade ledger: `{trade_csv}`",
        f"- Daily ledger: `{daily_csv}`",
        "",
        "## Notes",
        "",
        "- Session support and normalization are frozen from the prior audit lineage; this script does not recompute them from forward data.",
        "- Missing `garch_forecast_vol_pct` falls back to neutral `1.0x` by construction.",
        "- Because the normalization factor is below 1.0, even neutral raw signals become `0.973x`; that is why normalized weight changes appear on every trade.",
        "- Current output uses canonical forward outcomes only. If a dedicated live shadow stream is added later, it should replace this source without changing the policy surface.",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen forward-shadow ledger for GARCH R3 sizing.")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--start-date", default=str(HOLDOUT_SACRED_FROM))
    parser.add_argument("--output-dir", default="data/forward_monitoring")
    parser.add_argument("--report", default="docs/audit/results/2026-04-21-garch-r3-session-clipped-shadow.md")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _base_output_stem(args.profile_id)
    trade_csv = output_dir / f"{stem}-trades.csv"
    daily_csv = output_dir / f"{stem}-daily.csv"
    report_md = Path(args.report)

    trade_df, skipped, as_of, feature_gap_df = build_trade_ledger(args.profile_id, start_date)
    daily_df = build_daily_ledger(trade_df)
    metrics = summarize(args.profile_id, trade_df, daily_df, skipped, feature_gap_df)

    trade_df.to_csv(trade_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)
    write_report(report_md, as_of, start_date, args.profile_id, trade_df, daily_df, metrics, trade_csv, daily_csv)

    print(f"[trade-ledger] {trade_csv}")
    print(f"[daily-ledger] {daily_csv}")
    print(f"[report] {report_md}")


if __name__ == "__main__":
    main()
