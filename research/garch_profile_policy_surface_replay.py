"""Profile replay for the expanded raw discrete garch policy surface.

Purpose:
  Take the discrete policies that were already verified on canonical raw trade
  rows and replay them through actual execution profiles.

This is still an operational stress test, not clean validation evidence.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from research import garch_normalized_sizing_audit as norm
from research import garch_profile_production_replay as replay
from research import garch_regime_family_audit as fam

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
PROFILE_HYPOTHESIS = {
    "topstep_50k_mnq_auto": "docs/audit/hypotheses/2026-04-16-garch-profile-policy-surface-replay.yaml",
    "self_funded_tradovate": "docs/audit/hypotheses/2026-04-16-garch-self-funded-policy-surface-replay.yaml",
}
POLICIES = [
    "SESSION_TAKE_HIGH_ONLY",
    "GLOBAL_TAKE_HIGH_ONLY",
    "SESSION_SKIP_LOW_ONLY",
    "GLOBAL_SKIP_LOW_ONLY",
    "SESSION_HIGH_2X_ONLY",
    "GLOBAL_HIGH_2X_ONLY",
    "SESSION_CLIPPED_0_1_2",
    "GLOBAL_CLIPPED_0_1_2",
]


def action_contracts(policy: str, gp: float | None, session: str, profiles: dict[str, dict[str, bool]]) -> int:
    if gp is None:
        return 1
    p = profiles.get(session, {"high_dir": False, "low_dir": False})
    session_high = bool(p.get("high_dir"))
    session_low = bool(p.get("low_dir"))

    if policy == "SESSION_TAKE_HIGH_ONLY":
        return 1 if (session_high and gp >= 70.0) or (not session_high) else 0
    if policy == "GLOBAL_TAKE_HIGH_ONLY":
        return 1 if gp >= 70.0 else 0
    if policy == "SESSION_SKIP_LOW_ONLY":
        return 0 if session_low and gp <= 30.0 else 1
    if policy == "GLOBAL_SKIP_LOW_ONLY":
        return 0 if gp <= 30.0 else 1
    if policy == "SESSION_HIGH_2X_ONLY":
        return 2 if session_high and gp >= 70.0 else 1
    if policy == "GLOBAL_HIGH_2X_ONLY":
        return 2 if gp >= 70.0 else 1
    if policy == "SESSION_CLIPPED_0_1_2":
        if session_high and gp >= 70.0:
            return 2
        if session_low and gp <= 30.0:
            return 0
        return 1
    if policy == "GLOBAL_CLIPPED_0_1_2":
        if gp >= 70.0:
            return 2
        if gp <= 30.0:
            return 0
        return 1
    raise ValueError(policy)


def make_policy_scenarios(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list],
    feature_cache: dict,
    cal_days: list,
    policy: str,
    profiles: dict[str, dict[str, bool]],
):
    by_day = {d: [] for d in cal_days}
    session_contrib: dict[str, float] = {}
    day_base_totals: dict[object, float] = {d: 0.0 for d in cal_days}
    day_alt_totals: dict[object, float] = {d: 0.0 for d in cal_days}
    day_session_deltas: dict[object, dict[str, float]] = {d: {} for d in cal_days}
    for lane in lane_defs:
        sid = lane["strategy_id"]
        lane_session = lane["orb_label"]
        lane_instr = lane["instrument"]
        lane_orb = lane["orb_minutes"]
        for trade in trade_paths_by_lane[sid]:
            if trade.trading_day not in by_day:
                continue
            feat = feature_cache.get((lane_instr, lane_orb, trade.trading_day), {})
            contracts = action_contracts(policy, replay.score_value("GARCH_SESSION_CLIPPED", feat), lane_session, profiles)
            scaled_pnl = round(trade.pnl_dollars * contracts, 2)
            delta_dollars = scaled_pnl - trade.pnl_dollars
            day_base_totals[trade.trading_day] += trade.pnl_dollars
            day_alt_totals[trade.trading_day] += scaled_pnl
            day_session_deltas[trade.trading_day][lane_session] = (
                day_session_deltas[trade.trading_day].get(lane_session, 0.0) + delta_dollars
            )
            if contracts <= 0:
                continue
            scaled = replace(
                trade,
                pnl_dollars=scaled_pnl,
                mae_dollars=round(trade.mae_dollars * contracts, 2),
                mfe_dollars=round(trade.mfe_dollars * contracts, 2),
                contracts=contracts,
                lots=replay.lots_for_position(trade.instrument, contracts),
            )
            by_day[trade.trading_day].append(scaled)
    for trading_day in cal_days:
        session_map = day_session_deltas[trading_day]
        rounded = {sess: round(delta, 2) for sess, delta in session_map.items()}
        daily_delta = round(round(day_alt_totals[trading_day], 2) - round(day_base_totals[trading_day], 2), 2)
        residual = round(daily_delta - sum(rounded.values()), 2)
        if abs(residual) > 0:
            target = max(rounded, key=lambda sess: abs(rounded[sess]), default=None)
            if target is None:
                rounded["ROUNDING_RESIDUAL"] = residual
            else:
                rounded[target] = round(rounded[target] + residual, 2)
        for sess, delta in rounded.items():
            session_contrib[sess] = round(session_contrib.get(sess, 0.0) + delta, 2)
    scenarios = [replay._scenario_from_trade_paths(d, by_day[d]) for d in cal_days]
    return scenarios, session_contrib


def emit(
    output_md: Path,
    profile_id: str,
    as_of,
    profiles: dict[str, dict[str, bool]],
    results: list[dict[str, object]],
    session_tables: dict[str, list[tuple[str, float]]],
    lane_defs: list[dict],
    skipped_lanes: list[dict[str, object]],
) -> None:
    profile = replay.get_profile(profile_id)
    copy_label = f"{profile.copies}-copy total $" if profile.copies != 1 else "1-copy total $"
    lines = [
        "# Garch Profile Policy Surface Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{PROFILE_HYPOTHESIS.get(profile_id, 'profile-specific hypothesis file missing')}`",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, active={profile.active})",
        "**Purpose:** replay the expanded raw discrete garch policy surface on a selected profile under canonical account rules.",
        "**Status:** operational stress test on the current research-provisional live book; not clean validation evidence until Mode-A shelf rebuild.",
        "",
        "## Lane coverage",
        "",
        f"- Requested lanes: `{len(lane_defs)}`",
        f"- Replayed lanes: `{len(lane_defs) - len(skipped_lanes)}`",
        f"- Skipped lanes: `{len(skipped_lanes)}`",
        "",
        "## Session directional support",
        "",
        "| Session | High directional support | Low directional support |",
        "|---|---|---|",
    ]
    for sess, p in sorted(profiles.items()):
        lines.append(f"| {sess} | {'Y' if p['high_dir'] else '.'} | {'Y' if p['low_dir'] else '.'} |")

    lines += [
        "",
        "## Replay results",
        "",
        f"| Policy | Per-acct total $ | {copy_label} | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['policy']} | {row['total_dollars']:+,.1f} | {row['copied_total_dollars']:+,.1f} | {row['sharpe_ann_dollars']:+.3f} | "
            f"{row['max_dd_dollars']:+,.1f} | {row['worst_day_dollars']:+,.1f} | {row['worst_5day_dollars']:+,.1f} | "
            f"{row['max_open_lots']:.0f} | {row['dd_survival_probability']:.3f} | {row['operational_pass_probability']:.3f} |"
        )

    base = next(r for r in results if r["policy"] == "BASE_1X")
    lines += [
        "",
        "## Delta vs base",
        "",
        f"| Policy | Δ per-acct $ | Δ {copy_label} | Sharpe Δ | MaxDD Δ$ | Worst day Δ$ | Worst 5d Δ$ | Survival Δ | Operational Δ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        if row["policy"] == "BASE_1X":
            continue
        lines.append(
            f"| {row['policy']} | {row['total_dollars'] - base['total_dollars']:+,.1f} | "
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
            lines.append(
                f"| {row['strategy_id']} | {row['instrument']} | {row['orb_label']} | {row['reason']} |"
            )

    lines += [
        "",
        "## Reading the replay",
        "",
        "- `BASE_1X` is the current live-like baseline: 1 contract per eligible lane trade.",
        "- Policies replay the raw verified discrete action counts (`0`, `1`, `2`) rather than fractional sizing.",
        "- Session-aware policies only act in sessions with raw directional support from the regime-family audit.",
        "- This remains an operational stress test; profile/account geometry can still reorder raw row-level policy winners.",
        "- If skipped lanes are non-zero, the replay is only for the replayable subset and must not be over-read as a full-book result.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {output_md}")


def _default_output_for_profile(profile_id: str) -> Path:
    suffix = profile_id.replace("_", "-")
    return Path(f"docs/audit/results/2026-04-16-garch-profile-policy-surface-replay-{suffix}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile replay for expanded raw discrete garch policies.")
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
    feature_cache = replay.build_feature_cache(con, lane_defs)
    cells, _ = fam.build_cells()
    profiles = norm.session_profiles(cells)
    con.close()

    profile = replay.get_profile(args.profile_id)
    rules = replay._with_consistency_rule(replay._build_rules(profile), profile)

    results: list[dict[str, object]] = []
    session_tables: dict[str, list[tuple[str, float]]] = {}

    base_scenarios, _ = replay.make_daily_scenarios(lane_defs, trade_paths_by_lane, feature_cache, cal_days, None, None)
    base_metrics = replay.replay_metrics(base_scenarios)
    base_survival = replay.simulate_survival(
        base_scenarios,
        rules,
        horizon_days=replay.SURVIVAL_DAYS,
        n_paths=replay.SURVIVAL_PATHS,
        seed=replay.SURVIVAL_SEED,
    )
    results.append(
        {
            "policy": "BASE_1X",
            **base_metrics,
            **{k: base_survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
            "copied_total_dollars": base_metrics["total_dollars"] * profile.copies,
        }
    )
    session_tables["BASE_1X"] = []

    for policy in POLICIES:
        scenarios, session_contrib = make_policy_scenarios(
            lane_defs,
            trade_paths_by_lane,
            feature_cache,
            cal_days,
            policy,
            profiles,
        )
        metrics = replay.replay_metrics(scenarios)
        survival = replay.simulate_survival(
            scenarios,
            rules,
            horizon_days=replay.SURVIVAL_DAYS,
            n_paths=replay.SURVIVAL_PATHS,
            seed=replay.SURVIVAL_SEED,
        )
        results.append(
            {
                "policy": policy,
                **metrics,
                **{k: survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
                "copied_total_dollars": metrics["total_dollars"] * profile.copies,
            }
        )
        session_tables[policy] = sorted(session_contrib.items(), key=lambda x: x[1], reverse=True)

    output_md = Path(args.output) if args.output else _default_output_for_profile(args.profile_id)
    emit(output_md, args.profile_id, as_of, profiles, results, session_tables, lane_defs, skipped_lanes)


if __name__ == "__main__":
    main()
