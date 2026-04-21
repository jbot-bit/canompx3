"""Profile-level lane-aware monotonic allocator replay on the deployed MNQ book."""

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
import yaml

from pipeline.paths import GOLD_DB_PATH
from pipeline.session_guard import validate_features_for_session
from research.garch_profile_production_replay import (
    build_trade_paths,
    calendar_days,
    lane_definitions_for_profile,
)
from trading_app.account_survival import _scenario_from_trade_paths, lots_for_position
from trading_app.meta_labeling.profile_monotonic import (
    FittedLaneAllocator,
    LaneAllocatorSpec,
    LaneFeatureSpec,
    apply_lane_allocator,
    fit_lane_allocator,
    translate_weight_to_contracts,
)
from trading_app.prop_profiles import get_profile
from trading_app.strategy_fitness import _load_strategy_outcomes

PROFILE_ID = "topstep_50k_mnq_auto"
HYPOTHESIS_PATH = Path("docs/audit/hypotheses/2026-04-21-profile-lane-aware-monotonic-allocator-v1.yaml")
RESULT_PATH = Path("docs/audit/results/2026-04-21-profile-lane-aware-monotonic-allocator-v1.md")
TRAIN_END = date(2025, 1, 1)
OOS_START = date(2025, 1, 1)
OOS_END = date(2025, 12, 31)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260421

BANNED_FEATURE_PATTERNS = (
    "rel_vol_",
    "break_bar_volume",
    "break_delay",
    "break_bar_continues",
    "_outcome",
    "_mae_",
    "_mfe_",
)


def _load_hypothesis(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _to_lane_specs(hypothesis: dict) -> dict[str, LaneAllocatorSpec]:
    lane_specs: dict[str, LaneAllocatorSpec] = {}
    for row in hypothesis["lane_feature_contract"]:
        lane_specs[row["strategy_id"]] = LaneAllocatorSpec(
            strategy_id=row["strategy_id"],
            orb_label=_strategy_lane_label(row["strategy_id"]),
            features=tuple(
                LaneFeatureSpec(name=feature["name"], direction=feature["direction"])
                for feature in row["features"]
            ),
        )
    return lane_specs


def _strategy_lane_label(strategy_id: str) -> str:
    parts = strategy_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected strategy id: {strategy_id}")
    if strategy_id.startswith("MNQ_US_DATA_1000"):
        return "US_DATA_1000"
    if strategy_id.startswith("MNQ_COMEX_SETTLE"):
        return "COMEX_SETTLE"
    if strategy_id.startswith("MNQ_TOKYO_OPEN"):
        return "TOKYO_OPEN"
    if strategy_id.startswith("MNQ_NYSE_OPEN"):
        return "NYSE_OPEN"
    if strategy_id.startswith("MNQ_SINGAPORE_OPEN"):
        return "SINGAPORE_OPEN"
    if strategy_id.startswith("MNQ_EUROPE_FLOW"):
        return "EUROPE_FLOW"
    raise ValueError(f"Unable to infer orb_label from strategy id: {strategy_id}")


def _assert_feature_contract_safe(lane: dict, spec: LaneAllocatorSpec) -> None:
    feature_names = [feature.name for feature in spec.features]
    safe, blocked = validate_features_for_session(feature_names, lane["orb_label"])
    if blocked:
        raise ValueError(f"{lane['strategy_id']} blocked by session_guard: {blocked}")
    if safe != feature_names:
        raise ValueError(f"{lane['strategy_id']} guard mismatch: expected {feature_names}, got {safe}")
    for feature_name in feature_names:
        if any(pattern in feature_name for pattern in BANNED_FEATURE_PATTERNS):
            raise ValueError(f"{lane['strategy_id']} feature banned for pre-entry sizing: {feature_name}")


def _load_daily_feature_rows(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    orb_minutes: int,
    feature_names: list[str],
    end_date: date,
) -> dict[date, dict[str, float | None]]:
    cols = ", ".join(["trading_day", *feature_names])
    query = f"""
        SELECT {cols}
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = ? AND trading_day <= ?
        ORDER BY trading_day
    """
    rows = con.execute(query, [instrument, orb_minutes, end_date]).fetchall()
    out: dict[date, dict[str, float | None]] = {}
    for row in rows:
        trading_day = row[0]
        out[trading_day] = {
            feature_name: None if value is None else float(value)
            for feature_name, value in zip(feature_names, row[1:], strict=True)
        }
    return out


def _build_lane_training_frame(
    con: duckdb.DuckDBPyConnection,
    lane: dict,
    spec: LaneAllocatorSpec,
    end_date: date,
) -> pd.DataFrame:
    feature_names = [feature.name for feature in spec.features]
    feature_rows = _load_daily_feature_rows(con, lane["instrument"], lane["orb_minutes"], feature_names, end_date)
    outcomes = _load_strategy_outcomes(
        con,
        instrument=lane["instrument"],
        orb_label=lane["orb_label"],
        orb_minutes=lane["orb_minutes"],
        entry_model=lane["entry_model"],
        rr_target=lane["rr_target"],
        confirm_bars=lane["confirm_bars"],
        filter_type=lane["filter_type"],
        end_date=end_date,
    )
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        if outcome.get("outcome") not in ("win", "loss"):
            continue
        trading_day = outcome["trading_day"]
        feat = feature_rows.get(trading_day)
        if feat is None:
            continue
        row = {
            "trading_day": trading_day,
            "pnl_r": float(outcome["pnl_r"]),
        }
        row.update(feat)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("trading_day").reset_index(drop=True)


def _ann_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252.0))


def _max_drawdown_pct(returns: np.ndarray) -> float:
    equity = 1.0 + np.cumsum(returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    return float(np.min(drawdown)) if len(drawdown) else 0.0


def _calmar_ratio(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    total_return = float(np.sum(returns))
    annual_return = total_return / max(len(returns) / 252.0, 1e-9)
    max_dd_pct = abs(_max_drawdown_pct(returns))
    if max_dd_pct <= 0:
        return 0.0
    return annual_return / max_dd_pct


def _bootstrap_sharpe_delta_ci(
    baseline_returns: np.ndarray,
    candidate_returns: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline_returns)
    if n != len(candidate_returns) or n < 2:
        return (float("nan"), float("nan"))
    indices = rng.integers(0, n, size=(resamples, n))
    base_samples = baseline_returns[indices]
    cand_samples = candidate_returns[indices]
    base_mean = base_samples.mean(axis=1)
    cand_mean = cand_samples.mean(axis=1)
    base_std = base_samples.std(axis=1, ddof=1)
    cand_std = cand_samples.std(axis=1, ddof=1)
    base_sharpe = np.divide(base_mean, base_std, out=np.zeros_like(base_mean), where=base_std > 0) * math.sqrt(252.0)
    cand_sharpe = np.divide(cand_mean, cand_std, out=np.zeros_like(cand_mean), where=cand_std > 0) * math.sqrt(252.0)
    delta = cand_sharpe - base_sharpe
    return (float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975)))


def _build_oos_contract_lookup(
    lane: dict,
    allocator: FittedLaneAllocator,
    scored_oos: pd.DataFrame,
) -> dict[tuple[str, date], int]:
    lookup: dict[tuple[str, date], int] = {}
    for row in scored_oos.itertuples(index=False):
        lookup[(lane["strategy_id"], row.trading_day)] = int(row.contracts)
    return lookup


def _make_scenarios(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list],
    cal_days: list[date],
    contract_lookup: dict[tuple[str, date], int] | None,
):
    by_day = {d: [] for d in cal_days}
    for lane in lane_defs:
        sid = lane["strategy_id"]
        for trade in trade_paths_by_lane[sid]:
            if trade.trading_day not in by_day:
                continue
            contracts = 1 if contract_lookup is None else contract_lookup.get((sid, trade.trading_day), 1)
            if contracts <= 0:
                continue
            scaled = replace(
                trade,
                pnl_dollars=round(trade.pnl_dollars * contracts, 2),
                mae_dollars=round(trade.mae_dollars * contracts, 2),
                mfe_dollars=round(trade.mfe_dollars * contracts, 2),
                contracts=contracts,
                lots=lots_for_position(trade.instrument, contracts),
            )
            by_day[trade.trading_day].append(scaled)
    return [_scenario_from_trade_paths(day, by_day[day]) for day in cal_days]


def _scenario_metrics(scenarios: list, account_size: float) -> dict[str, float]:
    daily_pnl = np.asarray([float(s.total_pnl_dollars) for s in scenarios], dtype=float)
    daily_returns = daily_pnl / account_size
    return {
        "total_pnl_dollars": float(daily_pnl.sum()),
        "annualized_sharpe": _ann_sharpe(daily_returns),
        "max_drawdown_dollars": float(np.min(np.cumsum(daily_pnl) - np.maximum.accumulate(np.cumsum(daily_pnl)))),
        "calmar_ratio": _calmar_ratio(daily_returns),
        "daily_returns": daily_returns,
    }


def _emit_report(
    output_path: Path,
    profile_id: str,
    lane_summaries: list[dict[str, object]],
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
    changed_pct: float,
    zero_pct: float,
    double_pct: float,
    turnover_per_day: float,
    sharpe_ci: tuple[float, float],
) -> None:
    delta_pnl = candidate_metrics["total_pnl_dollars"] - baseline_metrics["total_pnl_dollars"]
    delta_sharpe = candidate_metrics["annualized_sharpe"] - baseline_metrics["annualized_sharpe"]
    verdict = "PARK"
    if delta_sharpe > 0 and sharpe_ci[0] > 0 and candidate_metrics["max_drawdown_dollars"] >= baseline_metrics["max_drawdown_dollars"]:
        verdict = "PROMOTE_TO_SHADOW"
    elif delta_sharpe > 0:
        verdict = "CONDITIONAL"

    lines = [
        "# Profile Lane-Aware Monotonic Allocator Replay",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Pre-registration:** `{HYPOTHESIS_PATH}`",
        f"**Profile:** `{profile_id}`",
        f"**Evaluation window:** `2025-01-01` to `2025-12-31` (OOS-CV only)",
        "**Sacred holdout:** `2026-01-01+` untouched",
        "",
        "## Result",
        "",
        f"- Verdict: `{verdict}`",
        f"- Baseline total PnL: `{baseline_metrics['total_pnl_dollars']:+,.2f}`",
        f"- Candidate total PnL: `{candidate_metrics['total_pnl_dollars']:+,.2f}`",
        f"- Delta total PnL: `{delta_pnl:+,.2f}`",
        f"- Baseline Sharpe: `{baseline_metrics['annualized_sharpe']:+.4f}`",
        f"- Candidate Sharpe: `{candidate_metrics['annualized_sharpe']:+.4f}`",
        f"- Sharpe delta: `{delta_sharpe:+.4f}`",
        f"- Sharpe delta 95% bootstrap CI: `[{sharpe_ci[0]:+.4f}, {sharpe_ci[1]:+.4f}]`",
        f"- Baseline max drawdown: `{baseline_metrics['max_drawdown_dollars']:+,.2f}`",
        f"- Candidate max drawdown: `{candidate_metrics['max_drawdown_dollars']:+,.2f}`",
        f"- Baseline Calmar: `{baseline_metrics['calmar_ratio']:+.4f}`",
        f"- Candidate Calmar: `{candidate_metrics['calmar_ratio']:+.4f}`",
        f"- Trades with size change: `{changed_pct:.2%}`",
        f"- Skip rate: `{zero_pct:.2%}`",
        f"- Double-contract rate: `{double_pct:.2%}`",
        f"- Average absolute contract change per day: `{turnover_per_day:.4f}`",
        "",
        "## Lane Summaries",
        "",
        "| Strategy | Train N | OOS N | Train ExpR | Avg weight | Avg contracts | Changed % | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in lane_summaries:
        lines.append(
            f"| {row['strategy_id']} | {row['train_rows']} | {row['oos_rows']} | {row['train_expr_r']:+.6f} | "
            f"{row['avg_weight']:.4f} | {row['avg_contracts']:.4f} | {row['changed_pct']:.2%} | {row['fallback_reason'] or ''} |"
        )

    fallback_rows = [row for row in lane_summaries if row["fallback_reason"]]

    lines += [
        "",
        "## Notes",
        "",
        "- Calendar flags were excluded from this replay because `session_guard.py` does not explicitly whitelist them.",
        "- Break-bar, rel-vol, and post-break fields were excluded by contract.",
        "- Static baseline remains the hurdle. This replay is an overlay test, not a rescue search.",
        "",
    ]
    if fallback_rows:
        lines += [
            "### Fallback lanes",
            "",
        ]
        for row in fallback_rows:
            lines.append(f"- `{row['strategy_id']}` -> `{row['fallback_reason']}`")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the lane-aware monotonic allocator on the active profile")
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--output", default=str(RESULT_PATH))
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    hypothesis = _load_hypothesis(HYPOTHESIS_PATH)
    lane_specs = _to_lane_specs(hypothesis)

    con = duckdb.connect(str(Path(args.db_path)), read_only=True)
    lane_defs = lane_definitions_for_profile(args.profile_id)
    profile = get_profile(args.profile_id)

    lane_contract_lookup: dict[tuple[str, date], int] = {}
    lane_summaries: list[dict[str, object]] = []
    for lane in lane_defs:
        spec = lane_specs[lane["strategy_id"]]
        _assert_feature_contract_safe(lane, spec)
        full_frame = _build_lane_training_frame(con, lane, spec, OOS_END)
        train_frame = full_frame.loc[full_frame["trading_day"] < TRAIN_END].reset_index(drop=True)
        oos_frame = full_frame.loc[full_frame["trading_day"] >= OOS_START].reset_index(drop=True)

        allocator = fit_lane_allocator(train_frame, spec)
        scored_oos = apply_lane_allocator(allocator, oos_frame)
        lane_contract_lookup.update(_build_oos_contract_lookup(lane, allocator, scored_oos))
        lane_summaries.append(
            {
                "strategy_id": lane["strategy_id"],
                "train_rows": allocator.train_rows,
                "oos_rows": int(len(scored_oos)),
                "train_expr_r": float(train_frame["pnl_r"].mean()) if len(train_frame) else float("nan"),
                "avg_weight": float(scored_oos["desired_weight"].mean()) if len(scored_oos) else 1.0,
                "avg_contracts": float(scored_oos["contracts"].mean()) if len(scored_oos) else 1.0,
                "changed_pct": float((scored_oos["contracts"] != 1).mean()) if len(scored_oos) else 0.0,
                "fallback_reason": allocator.fallback_reason,
            }
        )

    trade_paths_by_lane, skipped_lanes = build_trade_paths(con, OOS_END, lane_defs, args.profile_id)
    if skipped_lanes:
        raise ValueError(f"Expected full replayable live book, got skipped lanes: {skipped_lanes}")

    cal_days = calendar_days(con, lane_defs, OOS_START, OOS_END)
    con.close()

    baseline_scenarios = _make_scenarios(lane_defs, trade_paths_by_lane, cal_days, None)
    candidate_scenarios = _make_scenarios(lane_defs, trade_paths_by_lane, cal_days, lane_contract_lookup)

    baseline_metrics = _scenario_metrics(baseline_scenarios, account_size=float(profile.account_size))
    candidate_metrics = _scenario_metrics(candidate_scenarios, account_size=float(profile.account_size))
    sharpe_ci = _bootstrap_sharpe_delta_ci(baseline_metrics["daily_returns"], candidate_metrics["daily_returns"])

    oos_contracts = np.asarray(list(lane_contract_lookup.values()), dtype=int)
    changed_pct = float(np.mean(oos_contracts != 1)) if len(oos_contracts) else 0.0
    zero_pct = float(np.mean(oos_contracts == 0)) if len(oos_contracts) else 0.0
    double_pct = float(np.mean(oos_contracts == 2)) if len(oos_contracts) else 0.0

    day_abs_changes = []
    for day in cal_days:
        delta = 0
        for lane in lane_defs:
            key = (lane["strategy_id"], day)
            if key in lane_contract_lookup:
                delta += abs(lane_contract_lookup[key] - 1)
        day_abs_changes.append(delta)
    turnover_per_day = float(np.mean(day_abs_changes)) if day_abs_changes else 0.0

    _emit_report(
        Path(args.output),
        args.profile_id,
        lane_summaries,
        baseline_metrics,
        candidate_metrics,
        changed_pct,
        zero_pct,
        double_pct,
        turnover_per_day,
        sharpe_ci,
    )

    print(Path(args.output))


if __name__ == "__main__":
    main()
