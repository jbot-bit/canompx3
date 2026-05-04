"""Production-style regime replay for a selected execution profile.

Purpose:
  Convert regime maps that already survived upstream research into discrete live
  actions on a selected execution profile using:
    - exact live lane set from docs/runtime/lane_allocation.json
    - canonical trade paths from account_survival.py
    - discrete contract actions (0 / 1 / 2)
    - profile-specific account rules and Criterion 11 survival logic

Pre-registration:
  docs/audit/hypotheses/2026-04-16-garch-profile-production-replay.yaml
  docs/audit/hypotheses/2026-04-16-garch-self-funded-production-replay.yaml

Output:
  docs/audit/results/2026-04-16-garch-profile-production-replay-<profile>.md
"""

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
import pandas as pd

from pipeline.cost_model import get_cost_spec, risk_in_dollars
from pipeline.paths import GOLD_DB_PATH
from research import garch_additive_sizing_audit as add
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_proxy_native_sizing_audit as native
from trading_app.account_survival import (
    _build_rules,
    _scenario_from_trade_paths,
    _with_consistency_rule,
    lots_for_position,
    simulate_survival,
    TradePath,
)
from trading_app.config import apply_tight_stop
from trading_app.prop_profiles import _LANE_NAMES, get_profile, load_allocation_lanes, parse_strategy_id
from trading_app.strategy_fitness import _load_strategy_outcomes

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
PROFILE_HYPOTHESIS = {
    "topstep_50k_mnq_auto": "docs/audit/hypotheses/2026-04-16-garch-profile-production-replay.yaml",
    "self_funded_tradovate": "docs/audit/hypotheses/2026-04-16-garch-self-funded-production-replay.yaml",
}
MAPS = [
    ("GARCH_NATIVE_DISCRETE", "GARCH_SESSION_CLIPPED"),
    ("OVN_NATIVE_DISCRETE", "OVN_SESSION_CLIPPED"),
    ("GARCH_OVN_NATIVE_DISCRETE", "GARCH_OVN_MEAN_CLIPPED"),
]
SURVIVAL_SEED = 20260416
SURVIVAL_PATHS = 5000
SURVIVAL_DAYS = 90


def latest_trading_day(con: duckdb.DuckDBPyConnection) -> date:
    return con.execute("SELECT MAX(trading_day) FROM daily_features").fetchone()[0]


def validated_native_profiles(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, dict[str, bool]]]:
    rows = add.load_scope_rows(con, "validated")
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for _, base_map in MAPS:
        cells = []
        for _, row in rows.iterrows():
            for direction in ["long", "short"]:
                df = native.load_score_trades(con, row, direction, base_map, is_oos=False)
                if len(df) < broad.MIN_TOTAL:
                    continue
                df_oos = native.load_score_trades(con, row, direction, base_map, is_oos=True)
                high = broad.test_spec(df, df_oos, broad.ThresholdSpec("high", 70))
                low = broad.test_spec(df, df_oos, broad.ThresholdSpec("low", 30))
                if high.get("skip") or low.get("skip"):
                    continue
                shape = broad.ntile_shape(df)
                cells.append(
                    native.ProxyCell(
                        orb_label=str(row["orb_label"]),
                        direction=direction,
                        high_sr_lift=float(high["sr_lift"]),
                        high_lift=float(high["lift"]),
                        high_p_sharpe=float(high["p_sharpe"]),
                        high_oos_lift=None if pd.isna(high["oos_lift"]) else float(high["oos_lift"]),
                        low_sr_lift=float(low["sr_lift"]),
                        low_lift=float(low["lift"]),
                        low_p_sharpe=float(low["p_sharpe"]),
                        low_oos_lift=None if pd.isna(low["oos_lift"]) else float(low["oos_lift"]),
                        shape_skip=bool(shape.get("skip", False)),
                        tail_bias=None if shape.get("skip") else float(shape["tail_bias"]),
                        best_bucket=None if shape.get("skip") else int(shape["best_bucket"]),
                    )
                )
        out[base_map] = add.session_profiles(cells)
    return out


def score_value(score_map: str, feature_row: dict[str, float | None]) -> float | None:
    gp = feature_row.get("garch_forecast_vol_pct")
    atr = feature_row.get("atr_20_pct")
    ovn = feature_row.get("overnight_range_pct")
    if score_map == "GARCH_SESSION_CLIPPED":
        return gp
    if score_map == "OVN_SESSION_CLIPPED":
        return ovn
    if score_map == "GARCH_OVN_MEAN_CLIPPED":
        if gp is None or ovn is None:
            return None
        return (float(gp) + float(ovn)) / 2.0
    raise ValueError(score_map)


def action_contracts(score_map: str, session: str, score: float | None, profiles: dict[str, dict[str, bool]]) -> int:
    if score is None:
        return 1
    p = profiles.get(session, {"high_dir": False, "low_dir": False, "high_mono": False})
    if p.get("low_dir") and score <= 30.0:
        return 0
    if p.get("high_dir") and score >= 70.0:
        return 2
    return 1


def max_drawdown(daily_pnl: list[float]) -> float:
    eq = 0.0
    peak = 0.0
    worst = 0.0
    for pnl in daily_pnl:
        eq += pnl
        peak = max(peak, eq)
        worst = min(worst, eq - peak)
    return worst


def ann_sharpe(daily_pnl: list[float]) -> float:
    if len(daily_pnl) < 2:
        return 0.0
    s = pd.Series(daily_pnl, dtype=float)
    sd = float(s.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float((float(s.mean()) / sd) * math.sqrt(252.0))


def build_feature_cache(
    con: duckdb.DuckDBPyConnection,
    lane_defs: list[dict],
) -> dict[tuple[str, int, date], dict[str, float | None]]:
    cache: dict[tuple[str, int, date], dict[str, float | None]] = {}
    pairs = sorted({(lane["instrument"], lane["orb_minutes"]) for lane in lane_defs})
    for instrument, orb_minutes in pairs:
        q = """
        SELECT trading_day, garch_forecast_vol_pct, atr_20_pct, overnight_range_pct
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = ?
        ORDER BY trading_day
        """
        for td, gp, atr, ovn in con.execute(q, [instrument, orb_minutes]).fetchall():
            cache[(instrument, orb_minutes, td)] = {
                "garch_forecast_vol_pct": None if gp is None else float(gp),
                "atr_20_pct": None if atr is None else float(atr),
                "overnight_range_pct": None if ovn is None else float(ovn),
            }
    return cache


def build_trade_paths(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    lane_defs: list[dict],
    profile_id: str,
) -> tuple[dict[str, list[TradePath]], list[dict[str, object]]]:
    profile = get_profile(profile_id)
    lane_specs = profile.daily_lanes if profile.daily_lanes else load_allocation_lanes(profile.profile_id)
    effective_stop_by_strategy = {
        lane.strategy_id: float(lane.planned_stop_multiplier or profile.stop_multiplier) for lane in lane_specs
    }
    out: dict[str, list[TradePath]] = {}
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
            end_date=as_of,
        )
        stop_multiplier = effective_stop_by_strategy.get(lane["strategy_id"], profile.stop_multiplier)
        if stop_multiplier != 1.0:
            cost_spec = get_cost_spec(lane["instrument"])
            outcomes = apply_tight_stop(outcomes, stop_multiplier, cost_spec)
        cost_spec = get_cost_spec(lane["instrument"])
        trades: list[TradePath] = []
        for outcome in outcomes:
            if outcome.get("outcome") not in ("win", "loss"):
                continue
            entry_price = outcome.get("entry_price")
            stop_price = outcome.get("stop_price")
            pnl_r = outcome.get("pnl_r")
            if entry_price is None or stop_price is None or pnl_r is None:
                continue
            risk_dollars = risk_in_dollars(cost_spec, float(entry_price), float(stop_price))
            mae_r = max(0.0, float(outcome.get("mae_r") or 0.0))
            mfe_r = max(0.0, float(outcome.get("mfe_r") or 0.0))
            trades.append(
                TradePath(
                    trading_day=outcome["trading_day"],
                    strategy_id=lane["strategy_id"],
                    entry_ts=outcome.get("entry_ts"),
                    exit_ts=outcome.get("exit_ts"),
                    pnl_dollars=float(float(pnl_r) * risk_dollars),
                    mae_dollars=float(mae_r * risk_dollars),
                    mfe_dollars=float(mfe_r * risk_dollars),
                    lots=lots_for_position(lane["instrument"], 1),
                    contracts=1,
                    instrument=lane["instrument"],
                )
            )
        if trades:
            out[lane["strategy_id"]] = trades
        else:
            skipped.append(
                {
                    "strategy_id": lane["strategy_id"],
                    "instrument": lane["instrument"],
                    "orb_label": lane["orb_label"],
                    "reason": "no canonical trade history after exact filter application",
                }
            )
    return out, skipped


def lane_definitions_for_profile(profile_id: str) -> list[dict]:
    profile = get_profile(profile_id)
    lane_specs = profile.daily_lanes if profile.daily_lanes else load_allocation_lanes(profile.profile_id)
    lane_defs: list[dict] = []
    for lane in lane_specs:
        parsed = parse_strategy_id(lane.strategy_id)
        is_half = lane.planned_stop_multiplier is not None or "0.5x" in lane.execution_notes
        lane_defs.append(
            {
                "profile_id": profile.profile_id,
                "strategy_id": lane.strategy_id,
                "instrument": lane.instrument,
                "orb_label": lane.orb_label,
                "entry_model": parsed["entry_model"],
                "rr_target": parsed["rr_target"],
                "confirm_bars": parsed["confirm_bars"],
                "filter_type": parsed["filter_type"],
                "orb_minutes": parsed["orb_minutes"],
                "lane_name": _LANE_NAMES.get(lane.orb_label, lane.orb_label),
                "stop_multiplier": profile.stop_multiplier,
                "is_half_size": is_half,
                "shadow_only": False,
                "execution_notes": lane.execution_notes,
                "max_orb_size_pts": lane.max_orb_size_pts,
            }
        )
    return lane_defs


def common_start(trade_paths_by_lane: dict[str, list]) -> date:
    return max(min(t.trading_day for t in trades) for trades in trade_paths_by_lane.values() if trades)


def calendar_days(
    con: duckdb.DuckDBPyConnection,
    lane_defs: list[dict],
    start: date,
    end: date,
) -> list[date]:
    instruments = sorted({lane["instrument"] for lane in lane_defs})
    placeholders = ", ".join("?" for _ in instruments)
    q = f"""
    SELECT DISTINCT trading_day
    FROM daily_features
    WHERE symbol IN ({placeholders})
      AND trading_day >= ?
      AND trading_day <= ?
    ORDER BY trading_day
    """
    return [r[0] for r in con.execute(q, [*instruments, start, end]).fetchall()]


def make_daily_scenarios(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list],
    feature_cache: dict[tuple[str, int, date], dict[str, float | None]],
    cal_days: list[date],
    score_map: str | None,
    profiles: dict[str, dict[str, bool]] | None,
):
    by_day = {d: [] for d in cal_days}
    session_contrib: dict[str, float] = {}
    for lane in lane_defs:
        sid = lane["strategy_id"]
        lane_session = lane["orb_label"]
        lane_instr = lane["instrument"]
        lane_orb = lane["orb_minutes"]
        for trade in trade_paths_by_lane[sid]:
            if trade.trading_day not in by_day:
                continue
            contracts = 1
            if score_map is not None and profiles is not None:
                feat = feature_cache.get((lane_instr, lane_orb, trade.trading_day), {})
                contracts = action_contracts(score_map, lane_session, score_value(score_map, feat), profiles)
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
            session_contrib[lane_session] = (
                session_contrib.get(lane_session, 0.0) + scaled.pnl_dollars - trade.pnl_dollars
            )
    scenarios = [_scenario_from_trade_paths(d, by_day[d]) for d in cal_days]
    return scenarios, session_contrib


def replay_metrics(scenarios) -> dict[str, float]:
    daily = [float(s.total_pnl_dollars) for s in scenarios]
    s = pd.Series(daily, dtype=float)
    roll5 = s.rolling(5).sum()
    return {
        "total_dollars": float(s.sum()),
        "ann_dollars": float(s.sum() / max(len(s) / 252.0, 1e-9)),
        "exp_day_dollars": float(s.mean()) if len(s) else 0.0,
        "sharpe_ann_dollars": ann_sharpe(daily),
        "max_dd_dollars": float(max_drawdown(daily)),
        "worst_day_dollars": float(s.min()) if len(s) else 0.0,
        "worst_5day_dollars": float(roll5.min()) if roll5.notna().any() else 0.0,
        "max_open_lots": float(max((s.max_open_lots for s in scenarios), default=0.0)),
        "days": float(len(s)),
    }


def emit(
    output_md: Path,
    profile_id: str,
    as_of: date,
    native_profiles_by_map: dict[str, dict[str, dict[str, bool]]],
    results: list[dict[str, object]],
    session_tables: dict[str, list[tuple[str, float]]],
    lane_defs: list[dict],
    skipped_lanes: list[dict[str, object]],
) -> None:
    profile = get_profile(profile_id)
    copy_label = f"{profile.copies}-copy total $" if profile.copies != 1 else "1-copy total $"
    lines = [
        "# Garch Profile Production Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{PROFILE_HYPOTHESIS.get(profile_id, 'profile-specific hypothesis file missing')}`",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, active={profile.active})",
        "**Purpose:** convert regime maps into discrete live actions on a selected profile under canonical account rules.",
        "**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.",
        "",
        "## Lane coverage",
        "",
        f"- Requested lanes: `{len(lane_defs)}`",
        f"- Replayed lanes: `{len(lane_defs) - len(skipped_lanes)}`",
        f"- Skipped lanes: `{len(skipped_lanes)}`",
        "",
        "## Native validated session scaffolds",
        "",
        "| Map | Session scaffold |",
        "|---|---|",
    ]
    for public_name, base_map in MAPS:
        lines.append(f"| {public_name} | {native.scaffold_label(native_profiles_by_map[base_map]) or 'none'} |")

    lines += [
        "",
        "## Replay results",
        "",
        f"| Map | Per-acct total $ | {copy_label} | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['map']} | {row['total_dollars']:+,.1f} | {row['copied_total_dollars']:+,.1f} | {row['sharpe_ann_dollars']:+.3f} | "
            f"{row['max_dd_dollars']:+,.1f} | {row['worst_day_dollars']:+,.1f} | {row['worst_5day_dollars']:+,.1f} | "
            f"{row['max_open_lots']:.0f} | {row['dd_survival_probability']:.3f} | {row['operational_pass_probability']:.3f} |"
        )

    base = next(r for r in results if r["map"] == "BASE_1X")
    lines += [
        "",
        "## Delta vs base",
        "",
        f"| Map | Δ per-acct $ | Δ {copy_label} | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        if row["map"] == "BASE_1X":
            continue
        lines.append(
            f"| {row['map']} | {row['total_dollars'] - base['total_dollars']:+,.1f} | "
            f"{row['copied_total_dollars'] - base['copied_total_dollars']:+,.1f} | "
            f"{row['sharpe_ann_dollars'] - base['sharpe_ann_dollars']:+.3f} | "
            f"{row['max_dd_dollars'] - base['max_dd_dollars']:+,.1f} | "
            f"{row['worst_day_dollars'] - base['worst_day_dollars']:+,.1f} | "
            f"{row['worst_5day_dollars'] - base['worst_5day_dollars']:+,.1f} | "
            f"{row['dd_survival_probability'] - base['dd_survival_probability']:+.3f} | "
            f"{row['operational_pass_probability'] - base['operational_pass_probability']:+.3f} |"
        )

    for name, rows in session_tables.items():
        if name == "BASE_1X":
            continue
        lines += ["", f"### Session delta: `{name}`", "", "| Session | Δ$ |", "|---|---|"]
        for sess, delta in rows[:12]:
            lines.append(f"| {sess} | {delta:+,.1f} |")

    if skipped_lanes:
        lines += ["", "## Skipped lanes", "", "| Strategy | Instrument | Session | Reason |", "|---|---|---|---|"]
        for row in skipped_lanes:
            lines.append(f"| {row['strategy_id']} | {row['instrument']} | {row['orb_label']} | {row['reason']} |")

    lines += [
        "",
        "## Reading the replay",
        "",
        "- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.",
        "- Native maps use validated-scope discovered session support, not the broad common-scaffold shortcut.",
        "- Discrete actions are `0` contracts in hostile low state, `1` in neutral, and `2` in favorable high state.",
        "- Survival metrics use canonical account-survival style replay on the resulting daily scenarios.",
        "- This is the first deployment-allocator slice, not the full continuous allocator architecture.",
        "- If skipped lanes are non-zero, the replay is only for the replayable subset and must not be over-read as a full-book result.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {output_md}")


def _default_output_for_profile(profile_id: str) -> Path:
    suffix = profile_id.replace("_", "-")
    return Path(f"docs/audit/results/2026-04-16-garch-profile-production-replay-{suffix}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile-specific production replay for regime maps.")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    as_of = latest_trading_day(con)
    native_profiles_by_map = validated_native_profiles(con)
    lane_defs = lane_definitions_for_profile(args.profile_id)
    trade_paths_by_lane, skipped_lanes = build_trade_paths(con, as_of, lane_defs, args.profile_id)
    if not trade_paths_by_lane:
        raise ValueError(f"No replayable lanes found for profile {args.profile_id!r}")
    start = common_start(trade_paths_by_lane)
    cal_days = calendar_days(con, lane_defs, start, as_of)
    feature_cache = build_feature_cache(con, lane_defs)
    con.close()

    profile = get_profile(args.profile_id)
    rules = _with_consistency_rule(_build_rules(profile), profile)

    results: list[dict[str, object]] = []
    session_tables: dict[str, list[tuple[str, float]]] = {}

    base_scenarios, _ = make_daily_scenarios(lane_defs, trade_paths_by_lane, feature_cache, cal_days, None, None)
    base_metrics = replay_metrics(base_scenarios)
    base_survival = simulate_survival(
        base_scenarios,
        rules,
        horizon_days=SURVIVAL_DAYS,
        n_paths=SURVIVAL_PATHS,
        seed=SURVIVAL_SEED,
    )
    results.append(
        {
            "map": "BASE_1X",
            **base_metrics,
            **{k: base_survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
            "copied_total_dollars": base_metrics["total_dollars"] * profile.copies,
        }
    )
    session_tables["BASE_1X"] = []

    for public_name, base_map in MAPS:
        scenarios, session_contrib = make_daily_scenarios(
            lane_defs,
            trade_paths_by_lane,
            feature_cache,
            cal_days,
            base_map,
            native_profiles_by_map[base_map],
        )
        metrics = replay_metrics(scenarios)
        survival = simulate_survival(
            scenarios,
            rules,
            horizon_days=SURVIVAL_DAYS,
            n_paths=SURVIVAL_PATHS,
            seed=SURVIVAL_SEED,
        )
        results.append(
            {
                "map": public_name,
                **metrics,
                **{k: survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
                "copied_total_dollars": metrics["total_dollars"] * profile.copies,
            }
        )
        session_tables[public_name] = sorted(session_contrib.items(), key=lambda x: x[1], reverse=True)

    emit(
        Path(args.output) if args.output else _default_output_for_profile(args.profile_id),
        args.profile_id,
        as_of,
        native_profiles_by_map,
        results,
        session_tables,
        lane_defs,
        skipped_lanes,
    )


if __name__ == "__main__":
    main()
