"""Profile replay for bounded continuous garch sizing translated to live contracts.

Purpose:
  Evaluate A2 bounded continuous sizing under the same replayable lane sets and
  profile/account rules as the A1 discrete replay. This stage intentionally
  avoids session-attribution claims until that layer is fully reconciled.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from research import garch_broad_exact_role_exhaustion as broad
from research import garch_normalized_sizing_audit as norm
from research import garch_profile_production_replay as replay
from research import garch_regime_family_audit as fam

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-16-garch-profile-continuous-sizing-replay.yaml"
MAPS = [
    "LOW_CUT_ONLY",
    "HIGH_BOOST_ONLY",
    "SESSION_CLIPPED",
    "SESSION_LINEAR",
    "GLOBAL_LINEAR",
]


def translated_contracts(weight: float) -> int:
    if weight < 0.75:
        return 0
    if weight < 1.50:
        return 1
    return 2


def build_trade_frame(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list],
    feature_cache: dict[tuple[str, int, date], dict[str, float | None]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lane in lane_defs:
        sid = lane["strategy_id"]
        lane_session = lane["orb_label"]
        lane_instr = lane["instrument"]
        lane_orb = lane["orb_minutes"]
        for trade in trade_paths_by_lane[sid]:
            feat = feature_cache.get((lane_instr, lane_orb, trade.trading_day), {})
            gp = feat.get("garch_forecast_vol_pct")
            if gp is None:
                continue
            rows.append(
                {
                    "strategy_id": sid,
                    "trading_day": pd.Timestamp(trade.trading_day),
                    "instrument": lane_instr,
                    "orb_label": lane_session,
                    "orb_minutes": lane_orb,
                    "gp": float(gp),
                    "trade": trade,
                    "is_oos": pd.Timestamp(trade.trading_day) >= pd.Timestamp(broad.IS_END),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No replayable trades with garch score coverage")
    return out.sort_values(["trading_day", "strategy_id"]).reset_index(drop=True)


def make_continuous_scenarios(
    trades_df: pd.DataFrame,
    cal_days: list[date],
    map_name: str,
    profiles: dict[str, dict[str, bool]],
):
    work = trades_df.copy()
    work["raw_weight"] = [
        norm.raw_weight(map_name, float(gp), str(sess), profiles) for gp, sess in zip(work["gp"], work["orb_label"])
    ]
    is_mean = float(work.loc[~work["is_oos"], "raw_weight"].mean())
    norm_factor = 1.0 / is_mean if is_mean > 0 else 1.0
    work["desired_weight"] = work["raw_weight"] * norm_factor
    work["contracts"] = work["desired_weight"].map(translated_contracts).astype(int)

    by_day = {d: [] for d in cal_days}
    for row in work.itertuples(index=False):
        if row.trading_day.date() not in by_day or row.contracts <= 0:
            continue
        scaled = replace(
            row.trade,
            pnl_dollars=round(row.trade.pnl_dollars * row.contracts, 2),
            mae_dollars=round(row.trade.mae_dollars * row.contracts, 2),
            mfe_dollars=round(row.trade.mfe_dollars * row.contracts, 2),
            contracts=row.contracts,
            lots=replay.lots_for_position(row.trade.instrument, row.contracts),
        )
        by_day[row.trading_day.date()].append(scaled)

    scenarios = [replay._scenario_from_trade_paths(d, by_day[d]) for d in cal_days]
    diagnostics = {
        "mean_desired_weight": float(work["desired_weight"].mean()),
        "mean_contracts": float(work["contracts"].mean()),
        "pct_trades_changed": float((work["contracts"] != 1).mean()),
        "pct_trades_zero": float((work["contracts"] == 0).mean()),
        "pct_trades_double": float((work["contracts"] == 2).mean()),
        "is_mean_desired_weight": float(work.loc[~work["is_oos"], "desired_weight"].mean()),
        "oos_mean_desired_weight": float(work.loc[work["is_oos"], "desired_weight"].mean())
        if bool(work["is_oos"].any())
        else 0.0,
    }
    return scenarios, diagnostics


def emit(
    output_md: Path,
    profile_id: str,
    as_of: date,
    lane_defs: list[dict],
    skipped_lanes: list[dict[str, object]],
    results: list[dict[str, object]],
) -> None:
    profile = replay.get_profile(profile_id)
    copy_label = f"{profile.copies}-copy total $" if profile.copies != 1 else "1-copy total $"
    lines = [
        "# Garch Profile Continuous Sizing Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{HYPOTHESIS_FILE}`",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, active={profile.active})",
        "**Purpose:** test A2 bounded continuous sizing translated into live-feasible integer contracts on the replayable lane set.",
        "**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-level doctrine surface.",
        "",
        "## Lane coverage",
        "",
        f"- Requested lanes: `{len(lane_defs)}`",
        f"- Replayed lanes: `{len(lane_defs) - len(skipped_lanes)}`",
        f"- Skipped lanes: `{len(skipped_lanes)}`",
        "",
        "## Replay results",
        "",
        f"| Map | Per-acct total $ | {copy_label} | Sharpe | MaxDD $ | Worst day $ | Worst 5d $ | Max open lots | 90d survival | Operational pass | Mean desired w | Mean contracts | Changed % | Zero % | Double % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['map']} | {row['total_dollars']:+,.1f} | {row['copied_total_dollars']:+,.1f} | {row['sharpe_ann_dollars']:+.3f} | "
            f"{row['max_dd_dollars']:+,.1f} | {row['worst_day_dollars']:+,.1f} | {row['worst_5day_dollars']:+,.1f} | "
            f"{row['max_open_lots']:.0f} | {row['dd_survival_probability']:.3f} | {row['operational_pass_probability']:.3f} | "
            f"{row.get('mean_desired_weight', 1.0):.3f} | {row.get('mean_contracts', 1.0):.3f} | "
            f"{row.get('pct_trades_changed', 0.0):.3%} | {row.get('pct_trades_zero', 0.0):.3%} | {row.get('pct_trades_double', 0.0):.3%} |"
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

    lines += [
        "",
        "## Reading the replay",
        "",
        "- This stage tests bounded continuous desired weights only after translating them into real integer contracts.",
        "- If a map improves the normalized desired-weight surface but collapses after translation, that is a valid negative result for A2.",
        "- Session-level attribution is intentionally omitted here because that layer is not yet authoritative.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {output_md}")


def _default_output_for_profile(profile_id: str) -> Path:
    suffix = profile_id.replace("_", "-")
    return Path(f"docs/audit/results/2026-04-16-garch-profile-continuous-sizing-replay-{suffix}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile replay for bounded continuous garch sizing.")
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
    trade_df = build_trade_frame(lane_defs, trade_paths_by_lane, feature_cache)
    con.close()

    profile = replay.get_profile(args.profile_id)
    rules = replay._with_consistency_rule(replay._build_rules(profile), profile)

    results: list[dict[str, object]] = []
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
            "map": "BASE_1X",
            **base_metrics,
            **{k: base_survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
            "copied_total_dollars": base_metrics["total_dollars"] * profile.copies,
            "mean_desired_weight": 1.0,
            "mean_contracts": 1.0,
            "pct_trades_changed": 0.0,
            "pct_trades_zero": 0.0,
            "pct_trades_double": 0.0,
        }
    )

    for map_name in MAPS:
        scenarios, diagnostics = make_continuous_scenarios(trade_df, cal_days, map_name, profiles)
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
                "map": map_name,
                **metrics,
                **{k: survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
                "copied_total_dollars": metrics["total_dollars"] * profile.copies,
                **diagnostics,
            }
        )

    output_md = Path(args.output) if args.output else _default_output_for_profile(args.profile_id)
    emit(output_md, args.profile_id, as_of, lane_defs, skipped_lanes, results)


if __name__ == "__main__":
    main()
