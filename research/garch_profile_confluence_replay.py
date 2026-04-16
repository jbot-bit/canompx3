"""Profile replay for simple confluence allocator maps.

Purpose:
  Evaluate A3 simple confluence maps built from garch, overnight range, and ATR
  under the same replayable lane sets and profile/account rules as prior
  deployment replays.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from research import garch_additive_sizing_audit as add
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_profile_production_replay as replay

DEFAULT_PROFILE_ID = "topstep_50k_mnq_auto"
HYPOTHESIS_FILE = "docs/audit/hypotheses/2026-04-16-garch-a3-confluence-allocator-replay.yaml"
MAPS = [
    ("GARCH_NATIVE_DISCRETE", "GARCH_SESSION_CLIPPED"),
    ("OVN_NATIVE_DISCRETE", "OVN_SESSION_CLIPPED"),
    ("GARCH_OVN_NATIVE_DISCRETE", "GARCH_OVN_MEAN_CLIPPED"),
    ("GARCH_ATR_NATIVE_DISCRETE", "GARCH_ATR_MEAN_CLIPPED"),
    ("TRIPLE_MEAN_NATIVE_DISCRETE", "TRIPLE_MEAN_CLIPPED"),
]


@dataclass
class ConfluenceCell:
    orb_label: str
    direction: str
    high_sr_lift: float
    high_lift: float
    high_p_sharpe: float
    high_oos_lift: float | None
    low_sr_lift: float
    low_lift: float
    low_p_sharpe: float
    low_oos_lift: float | None
    shape_skip: bool
    tail_bias: float | None
    best_bucket: int | None


def score_sql(map_name: str) -> str:
    if map_name == "GARCH_SESSION_CLIPPED":
        return "d.garch_forecast_vol_pct"
    if map_name == "OVN_SESSION_CLIPPED":
        return "d.overnight_range_pct"
    if map_name == "GARCH_OVN_MEAN_CLIPPED":
        return "((d.garch_forecast_vol_pct + d.overnight_range_pct) / 2.0)"
    if map_name == "GARCH_ATR_MEAN_CLIPPED":
        return "((d.garch_forecast_vol_pct + d.atr_20_pct) / 2.0)"
    if map_name == "TRIPLE_MEAN_CLIPPED":
        return "((d.garch_forecast_vol_pct + d.overnight_range_pct + d.atr_20_pct) / 3.0)"
    raise ValueError(map_name)


def load_score_trades(
    con,
    row: pd.Series,
    direction: str,
    map_name: str,
    *,
    is_oos: bool,
) -> pd.DataFrame:
    filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
    if filter_sql is None:
        return pd.DataFrame()
    date_clause = ">=" if is_oos else "<"
    score_expr = score_sql(map_name)
    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      {score_expr} AS gp
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    {join_sql}
    WHERE o.symbol = '{row["instrument"]}'
      AND o.orb_minutes = {row["orb_minutes"]}
      AND o.orb_label = '{row["orb_label"]}'
      AND o.entry_model = '{row["entry_model"]}'
      AND o.rr_target = {row["rr_target"]}
      AND o.pnl_r IS NOT NULL
      AND {score_expr} IS NOT NULL
      AND d.orb_{row["orb_label"]}_break_dir = '{direction}'
      AND {filter_sql}
      AND o.trading_day {date_clause} DATE '{broad.IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["gp"] = df["gp"].astype(float)
    return df


def build_cells_for_map(con, map_name: str) -> list[ConfluenceCell]:
    rows = broad.load_rows(con)
    rows = rows[rows["filter_type"].map(broad.in_scope)].copy()
    rows = rows[rows["src"] == "validated"].copy()
    cells: list[ConfluenceCell] = []
    for _, row in rows.iterrows():
        for direction in ["long", "short"]:
            df = load_score_trades(con, row, direction, map_name, is_oos=False)
            if len(df) < broad.MIN_TOTAL:
                continue
            df_oos = load_score_trades(con, row, direction, map_name, is_oos=True)
            high = broad.test_spec(df, df_oos, broad.ThresholdSpec("high", 70))
            low = broad.test_spec(df, df_oos, broad.ThresholdSpec("low", 30))
            if high.get("skip") or low.get("skip"):
                continue
            shape = broad.ntile_shape(df)
            cells.append(
                ConfluenceCell(
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
    return cells


def native_profiles(con) -> dict[str, dict[str, dict[str, bool]]]:
    out: dict[str, dict[str, dict[str, bool]]] = {}
    for _public_name, map_name in MAPS:
        cells = build_cells_for_map(con, map_name)
        out[map_name] = add.session_profiles(cells)
    return out


def scaffold_label(profiles: dict[str, dict[str, bool]]) -> str:
    sessions = []
    for sess, p in sorted(profiles.items()):
        bits = []
        if p.get("high_dir"):
            bits.append("H")
        if p.get("low_dir"):
            bits.append("L")
        if p.get("high_mono"):
            bits.append("M")
        if bits:
            sessions.append(f"{sess}({''.join(bits)})")
    return ", ".join(sessions)


def action_contracts(score_map: str, session: str, score: float | None, profiles: dict[str, dict[str, bool]]) -> int:
    if score is None:
        return 1
    p = profiles.get(session, {"high_dir": False, "low_dir": False})
    if p.get("low_dir") and score <= 30.0:
        return 0
    if p.get("high_dir") and score >= 70.0:
        return 2
    return 1


def build_score_feature_cache(con, lane_defs: list[dict]) -> dict[tuple[str, int, object], dict[str, float | None]]:
    cache: dict[tuple[str, int, object], dict[str, float | None]] = {}
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
    if score_map == "GARCH_ATR_MEAN_CLIPPED":
        if gp is None or atr is None:
            return None
        return (float(gp) + float(atr)) / 2.0
    if score_map == "TRIPLE_MEAN_CLIPPED":
        if gp is None or atr is None or ovn is None:
            return None
        return (float(gp) + float(atr) + float(ovn)) / 3.0
    raise ValueError(score_map)


def make_daily_scenarios(
    lane_defs: list[dict],
    trade_paths_by_lane: dict[str, list],
    feature_cache: dict[tuple[str, int, object], dict[str, float | None]],
    cal_days: list,
    score_map: str | None,
    profiles: dict[str, dict[str, bool]] | None,
):
    by_day = {d: [] for d in cal_days}
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
                lots=replay.lots_for_position(trade.instrument, contracts),
            )
            by_day[trade.trading_day].append(scaled)
    scenarios = [replay._scenario_from_trade_paths(d, by_day[d]) for d in cal_days]
    return scenarios


def emit(output_md: Path, profile_id: str, as_of, lane_defs: list[dict], skipped_lanes: list[dict[str, object]], native_profiles_by_map, results):
    profile = replay.get_profile(profile_id)
    copy_label = f"{profile.copies}-copy total $" if profile.copies != 1 else "1-copy total $"
    lines = [
        "# Garch A3 Confluence Allocator Replay",
        "",
        f"**Date:** {as_of}",
        f"**Pre-registration:** `{HYPOTHESIS_FILE}`",
        f"**Profile:** `{profile_id}` (`{profile.firm}`, `{profile.account_size:,}`, copies={profile.copies}, stop={profile.stop_multiplier}x, active={profile.active})",
        "**Purpose:** compare simple confluence allocator maps built from garch, overnight, and ATR under canonical profile replay rules.",
        "**Status:** operational stress test on the current research-provisional live book; not edge proof and not a session-attribution surface.",
        "",
        "## Lane coverage",
        "",
        f"- Requested lanes: `{len(lane_defs)}`",
        f"- Replayed lanes: `{len(lane_defs) - len(skipped_lanes)}`",
        f"- Skipped lanes: `{len(skipped_lanes)}`",
        "",
        "## Native validated scaffolds",
        "",
        "| Map | Session scaffold |",
        "|---|---|",
    ]
    for public_name, base_map in MAPS:
        lines.append(f"| {public_name} | {scaffold_label(native_profiles_by_map[base_map]) or 'none'} |")

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

    lines += [
        "",
        "## Reading the replay",
        "",
        "- This stage compares simple auditable confluence maps only. No tree model, no session-attribution claim, no forward tuning.",
        "- Native scaffolds are earned separately by each map on validated populations before profile replay translation.",
        "- Portfolio-ranking allocator work is explicitly deferred; this stage only answers whether simple confluence translation beats solo maps.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {output_md}")


def _default_output_for_profile(profile_id: str) -> Path:
    suffix = profile_id.replace("_", "-")
    return Path(f"docs/audit/results/2026-04-16-garch-a3-confluence-allocator-replay-{suffix}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile replay for simple confluence allocator maps.")
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
    feature_cache = build_score_feature_cache(con, lane_defs)
    native_profiles_by_map = native_profiles(con)
    con.close()

    profile = replay.get_profile(args.profile_id)
    rules = replay._with_consistency_rule(replay._build_rules(profile), profile)

    results: list[dict[str, object]] = []
    base_scenarios = make_daily_scenarios(lane_defs, trade_paths_by_lane, feature_cache, cal_days, None, None)
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
        }
    )

    for public_name, base_map in MAPS:
        scenarios = make_daily_scenarios(
            lane_defs,
            trade_paths_by_lane,
            feature_cache,
            cal_days,
            base_map,
            native_profiles_by_map[base_map],
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
                "map": public_name,
                **metrics,
                **{k: survival[k] for k in ["dd_survival_probability", "operational_pass_probability"]},
                "copied_total_dollars": metrics["total_dollars"] * profile.copies,
            }
        )

    output_md = Path(args.output) if args.output else _default_output_for_profile(args.profile_id)
    emit(output_md, args.profile_id, as_of, lane_defs, skipped_lanes, native_profiles_by_map, results)


if __name__ == "__main__":
    main()
