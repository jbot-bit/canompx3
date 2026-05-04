"""Profile replay for A4 portfolio-ranking allocator.

Purpose:
  Test whether a locked pre-entry composite state score improves profile
  utility by routing scarce daily slot budget toward better same-day
  opportunities, without changing position size or introducing hidden leverage.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from research import garch_profile_confluence_replay as confluence
from research import garch_profile_production_replay as replay

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-16-garch-a4-portfolio-ranking-allocator.yaml"


@dataclass(frozen=True)
class EligibleTrade:
    trading_day: date
    strategy_id: str
    instrument: str
    orb_label: str
    baseline_order: int
    score: float | None
    trade: replay.TradePath


def triple_mean_score(feature_row: dict[str, float | None]) -> float | None:
    gp = feature_row.get("garch_forecast_vol_pct")
    atr = feature_row.get("atr_20_pct")
    ovn = feature_row.get("overnight_range_pct")
    if gp is None or atr is None or ovn is None:
        return None
    return (float(gp) + float(atr) + float(ovn)) / 3.0


def build_eligible_trades(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list[replay.TradePath]],
    feature_cache: dict[tuple[str, int, date], dict[str, float | None]],
    cal_days: list[date],
) -> dict[date, list[EligibleTrade]]:
    by_day = {d: [] for d in cal_days}
    baseline_order = {lane["strategy_id"]: idx for idx, lane in enumerate(lane_defs)}
    lane_meta = {lane["strategy_id"]: lane for lane in lane_defs}
    for strategy_id, trades in trade_paths_by_lane.items():
        lane = lane_meta[strategy_id]
        for trade in trades:
            if trade.trading_day not in by_day:
                continue
            feat = feature_cache.get((lane["instrument"], lane["orb_minutes"], trade.trading_day), {})
            by_day[trade.trading_day].append(
                EligibleTrade(
                    trading_day=trade.trading_day,
                    strategy_id=strategy_id,
                    instrument=lane["instrument"],
                    orb_label=lane["orb_label"],
                    baseline_order=baseline_order[strategy_id],
                    score=triple_mean_score(feat),
                    trade=trade,
                )
            )
    return by_day


def select_base(rows: list[EligibleTrade], slot_budget: int) -> list[EligibleTrade]:
    return sorted(rows, key=lambda r: (r.baseline_order, r.strategy_id))[:slot_budget]


def select_candidate(rows: list[EligibleTrade], slot_budget: int) -> list[EligibleTrade]:
    if len(rows) <= slot_budget:
        return select_base(rows, slot_budget)
    return sorted(
        rows,
        key=lambda r: (
            r.score is None,
            -1e9 if r.score is None else -float(r.score),
            r.baseline_order,
            r.strategy_id,
        ),
    )[:slot_budget]


def build_scenarios(
    eligible_by_day: dict[date, list[EligibleTrade]],
    cal_days: list[date],
    slot_budget: int,
) -> tuple[list, list, dict[str, object]]:
    base_by_day: dict[date, list[replay.TradePath]] = {d: [] for d in cal_days}
    cand_by_day: dict[date, list[replay.TradePath]] = {d: [] for d in cal_days}
    lane_delta: dict[str, float] = {}
    session_delta: dict[str, float] = {}
    collision_day_count = 0
    rerouted_day_count = 0
    active_day_count = 0
    selected_slots_total = 0
    collision_day_delta = 0.0
    non_collision_identical = True

    for day in cal_days:
        rows = eligible_by_day.get(day, [])
        if rows:
            active_day_count += 1
        base_sel = select_base(rows, slot_budget)
        cand_sel = select_candidate(rows, slot_budget)

        if len(rows) <= slot_budget:
            if [r.strategy_id for r in base_sel] != [r.strategy_id for r in cand_sel]:
                non_collision_identical = False
        else:
            collision_day_count += 1

        base_ids = {r.strategy_id for r in base_sel}
        cand_ids = {r.strategy_id for r in cand_sel}
        if base_ids != cand_ids:
            rerouted_day_count += 1

        base_by_day[day] = [r.trade for r in base_sel]
        cand_by_day[day] = [r.trade for r in cand_sel]
        selected_slots_total += len(cand_sel)

        base_pnl = sum(r.trade.pnl_dollars for r in base_sel)
        cand_pnl = sum(r.trade.pnl_dollars for r in cand_sel)
        if len(rows) > slot_budget:
            collision_day_delta += cand_pnl - base_pnl

        by_strategy = {r.strategy_id: r for r in rows}
        for strategy_id in base_ids | cand_ids:
            row = by_strategy[strategy_id]
            delta = 0.0
            if strategy_id in cand_ids:
                delta += row.trade.pnl_dollars
            if strategy_id in base_ids:
                delta -= row.trade.pnl_dollars
            if abs(delta) < 1e-12:
                continue
            lane_delta[strategy_id] = lane_delta.get(strategy_id, 0.0) + delta
            session_delta[row.orb_label] = session_delta.get(row.orb_label, 0.0) + delta

    base_scenarios = [replay._scenario_from_trade_paths(d, base_by_day[d]) for d in cal_days]
    cand_scenarios = [replay._scenario_from_trade_paths(d, cand_by_day[d]) for d in cal_days]
    budget_utilization = 0.0
    if active_day_count > 0 and slot_budget > 0:
        budget_utilization = selected_slots_total / float(active_day_count * slot_budget)

    diagnostics = {
        "collision_day_count": collision_day_count,
        "rerouted_day_count": rerouted_day_count,
        "active_day_count": active_day_count,
        "pct_days_rerouted": float(rerouted_day_count / len(cal_days)) if cal_days else 0.0,
        "collision_day_only_delta_dollars": float(collision_day_delta),
        "budget_utilization_rate": float(budget_utilization),
        "identical_on_non_collision_days": bool(non_collision_identical),
        "lane_delta": lane_delta,
        "session_delta": session_delta,
    }
    return base_scenarios, cand_scenarios, diagnostics


def top_share(delta_map: dict[str, float], total_delta: float) -> float:
    if not delta_map or abs(total_delta) < 1e-9:
        return 0.0
    return max(abs(v) for v in delta_map.values()) / abs(total_delta)


def emit(
    output_md: Path,
    profile_id: str,
    as_of: date,
    lane_defs: list[dict],
    skipped_lanes: list[dict[str, object]],
    base_row: dict[str, object],
    cand_row: dict[str, object],
    diagnostics: dict[str, object],
) -> None:
    profile = replay.get_profile(profile_id)
    copy_label = f"{profile.copies}-copy total $" if profile.copies != 1 else "1-copy total $"
    lane_delta = diagnostics["lane_delta"]
    session_delta = diagnostics["session_delta"]
    total_delta = cand_row["total_dollars"] - base_row["total_dollars"]
    top_lane_rows = sorted(lane_delta.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    top_session_rows = sorted(session_delta.items(), key=lambda x: abs(x[1]), reverse=True)[:12]

    lines = [
        "# Garch A4 Portfolio-Ranking Allocator Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{HYPOTHESIS_FILE}`",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, max_slots={profile.max_slots}, active={profile.active})",
        "**Purpose:** test routing-only scarce-slot allocation using a locked triple-mean vol-state score.",
        "**Status:** operational stress test on the current research-provisional live book; not standalone edge proof and not a session-doctrine surface.",
        "",
        "## Lane coverage",
        "",
        f"- Requested lanes: `{len(lane_defs)}`",
        f"- Replayed lanes: `{len(lane_defs) - len(skipped_lanes)}`",
        f"- Skipped lanes: `{len(skipped_lanes)}`",
        "",
        "## Replay results",
        "",
        f"| Route | Per-acct total $ | {copy_label} | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in [base_row, cand_row]:
        lines.append(
            f"| {row['map']} | {row['total_dollars']:+,.1f} | {row['copied_total_dollars']:+,.1f} | {row['sharpe_ann_dollars']:+.3f} | "
            f"{row['max_dd_dollars']:+,.1f} | {row['worst_day_dollars']:+,.1f} | {row['worst_5day_dollars']:+,.1f} | "
            f"{row['max_open_lots']:.0f} | {row['dd_survival_probability']:.3f} | {row['operational_pass_probability']:.3f} |"
        )

    lines += [
        "",
        "## Delta vs base",
        "",
        f"| Candidate | Δ per-acct $ | Δ {copy_label} | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |",
        "|---|---|---|---|---|---|---|---|---|",
        (
            f"| {cand_row['map']} | {cand_row['total_dollars'] - base_row['total_dollars']:+,.1f} | "
            f"{cand_row['copied_total_dollars'] - base_row['copied_total_dollars']:+,.1f} | "
            f"{cand_row['sharpe_ann_dollars'] - base_row['sharpe_ann_dollars']:+.3f} | "
            f"{cand_row['max_dd_dollars'] - base_row['max_dd_dollars']:+,.1f} | "
            f"{cand_row['worst_day_dollars'] - base_row['worst_day_dollars']:+,.1f} | "
            f"{cand_row['worst_5day_dollars'] - base_row['worst_5day_dollars']:+,.1f} | "
            f"{cand_row['dd_survival_probability'] - base_row['dd_survival_probability']:+.3f} | "
            f"{cand_row['operational_pass_probability'] - base_row['operational_pass_probability']:+.3f} |"
        ),
        "",
        "## Routing diagnostics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Active days | {diagnostics['active_day_count']} |",
        f"| Collision days | {diagnostics['collision_day_count']} |",
        f"| Rerouted days | {diagnostics['rerouted_day_count']} |",
        f"| Pct days rerouted | {diagnostics['pct_days_rerouted']:.3f} |",
        f"| Collision-day-only delta $ | {diagnostics['collision_day_only_delta_dollars']:+,.1f} |",
        f"| Budget utilization rate | {diagnostics['budget_utilization_rate']:.3f} |",
        f"| Non-collision days identical | {diagnostics['identical_on_non_collision_days']} |",
        f"| Top abs lane-share of delta | {top_share(lane_delta, total_delta):.3f} |",
        f"| Top abs session-share of delta | {top_share(session_delta, total_delta):.3f} |",
        "",
        "## Top lane deltas",
        "",
        "| Strategy | Δ$ |",
        "|---|---|",
    ]
    for strategy_id, delta in top_lane_rows:
        lines.append(f"| {strategy_id} | {delta:+,.1f} |")

    lines += [
        "",
        "## Top session deltas",
        "",
        "| Session | Δ$ |",
        "|---|---|",
    ]
    for session, delta in top_session_rows:
        lines.append(f"| {session} | {delta:+,.1f} |")

    if skipped_lanes:
        lines += ["", "## Skipped lanes", "", "| Strategy | Instrument | Session | Reason |", "|---|---|---|---|"]
        for row in skipped_lanes:
            lines.append(f"| {row['strategy_id']} | {row['instrument']} | {row['orb_label']} | {row['reason']} |")

    lines += [
        "",
        "## Reading the replay",
        "",
        "- This is routing-only. Every selected lane stays at 1x. No upsizing, downsizing, or fractional sizing is allowed in this stage.",
        "- The candidate may differ from base only on collision days where eligible lanes exceed the profile slot budget.",
        "- Baseline order is fixed by the profile lane order. Candidate order is the locked triple-mean score with deterministic tie-breaks.",
        "- This stage tests deployment allocator utility, not standalone signal edge and not session doctrine.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {output_md}")


def _default_output_for_profile(profile_id: str) -> Path:
    suffix = profile_id.replace("_", "-")
    return Path(f"docs/audit/results/2026-04-16-garch-a4-portfolio-ranking-replay-{suffix}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile replay for A4 portfolio-ranking allocator.")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    con = replay.duckdb.connect(str(replay.GOLD_DB_PATH), read_only=True)
    as_of = replay.latest_trading_day(con)
    lane_defs = replay.lane_definitions_for_profile(args.profile_id)
    trade_paths_by_lane, skipped_lanes = replay.build_trade_paths(con, as_of, lane_defs, args.profile_id)
    if not trade_paths_by_lane:
        raise ValueError(f"No replayable lanes found for profile {args.profile_id!r}")
    start = replay.common_start(trade_paths_by_lane)
    cal_days = replay.calendar_days(con, lane_defs, start, as_of)
    feature_cache = confluence.build_score_feature_cache(con, lane_defs)
    con.close()

    profile = replay.get_profile(args.profile_id)
    rules = replay._with_consistency_rule(replay._build_rules(profile), profile)
    eligible_by_day = build_eligible_trades(lane_defs, trade_paths_by_lane, feature_cache, cal_days)
    base_scenarios, cand_scenarios, diagnostics = build_scenarios(eligible_by_day, cal_days, profile.max_slots)

    base_metrics = replay.replay_metrics(base_scenarios)
    base_survival = replay.simulate_survival(
        base_scenarios,
        rules,
        horizon_days=replay.SURVIVAL_DAYS,
        n_paths=replay.SURVIVAL_PATHS,
        seed=replay.SURVIVAL_SEED,
    )
    cand_metrics = replay.replay_metrics(cand_scenarios)
    cand_survival = replay.simulate_survival(
        cand_scenarios,
        rules,
        horizon_days=replay.SURVIVAL_DAYS,
        n_paths=replay.SURVIVAL_PATHS,
        seed=replay.SURVIVAL_SEED,
    )

    base_row = {
        "map": "BASE_1X_SLOT_ORDER",
        **base_metrics,
        **{k: base_survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
        "copied_total_dollars": base_metrics["total_dollars"] * profile.copies,
    }
    cand_row = {
        "map": "TRIPLE_MEAN_SLOT_RANK",
        **cand_metrics,
        **{k: cand_survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
        "copied_total_dollars": cand_metrics["total_dollars"] * profile.copies,
    }

    output_md = Path(args.output) if args.output else _default_output_for_profile(args.profile_id)
    emit(output_md, args.profile_id, as_of, lane_defs, skipped_lanes, base_row, cand_row, diagnostics)


if __name__ == "__main__":
    main()
