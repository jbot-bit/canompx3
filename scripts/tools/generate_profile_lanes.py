"""Audit and rebuild dormant profile lanes against the current deployable shelf.

This is a research / operations support tool. It does not mutate profile
configuration. It proves which inactive profiles are stale, then emits the
current allocator-backed DailyLaneSpec suggestions for review.

Usage:
    python scripts/tools/generate_profile_lanes.py
    python scripts/tools/generate_profile_lanes.py --profile tradeify_50k
    python scripts/tools/generate_profile_lanes.py --include-active
    python scripts/tools/generate_profile_lanes.py --date 2026-04-19
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_allocator import (
    build_allocation,
    compute_lane_scores,
    compute_orb_size_stats,
    compute_pairwise_correlation,
    enrich_scores_with_liveness,
)
from trading_app.prop_profiles import (
    _P90_ORB_PTS,
    ACCOUNT_PROFILES,
    ACCOUNT_TIERS,
    AccountProfile,
    DailyLaneSpec,
    effective_daily_lanes,
)
from trading_app.validated_shelf import deployable_validated_relation


def select_profile_ids(
    *,
    profile_id: str | None = None,
    include_active: bool = False,
) -> list[str]:
    """Return profile ids to audit.

    Default behavior is intentionally narrow: audit inactive profiles only.
    Active profiles are already protected by the live allocator / pre-session
    surfaces and should not be rewritten casually.
    """
    if profile_id is not None:
        if profile_id not in ACCOUNT_PROFILES:
            raise KeyError(f"Unknown profile_id: {profile_id}")
        return [profile_id]
    return [
        pid
        for pid, profile in ACCOUNT_PROFILES.items()
        if include_active or not profile.active
    ]


def load_deployable_strategy_ids(db_path: str | Path = GOLD_DB_PATH) -> set[str]:
    """Return the current deployable shelf strategy ids."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return {
            row[0]
            for row in con.execute(
                f"SELECT strategy_id FROM {deployable_validated_relation(con)}"
            ).fetchall()
        }
    finally:
        con.close()


def split_current_lanes(
    lane_specs: tuple[DailyLaneSpec, ...],
    deployable_ids: set[str],
) -> tuple[list[DailyLaneSpec], list[DailyLaneSpec]]:
    """Split lane specs into valid vs ghost against current deployable ids."""
    valid = [lane for lane in lane_specs if lane.strategy_id in deployable_ids]
    ghosts = [lane for lane in lane_specs if lane.strategy_id not in deployable_ids]
    return valid, ghosts


def lane_cap_for(
    instrument: str,
    orb_label: str,
    orb_size_stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    """Return session-specific P90 ORB cap, with instrument fallback."""
    stats = orb_size_stats.get((instrument, orb_label))
    if stats is not None:
        _avg_pts, p90_pts = stats
        return p90_pts
    return _P90_ORB_PTS.get(instrument, 100.0)


def summarize_lane_delta(
    current_valid: list[DailyLaneSpec],
    allocation,
) -> tuple[list[str], list[str], list[str]]:
    """Return kept, dropped, and added strategy ids for the report."""
    current_ids = [lane.strategy_id for lane in current_valid]
    allocation_ids = [lane.strategy_id for lane in allocation]
    kept = [sid for sid in current_ids if sid in allocation_ids]
    dropped = [sid for sid in current_ids if sid not in allocation_ids]
    added = [sid for sid in allocation_ids if sid not in current_ids]
    return kept, dropped, added


def eligible_candidates_for_profile(
    profile: AccountProfile,
    scores,
):
    """Filter scored lanes to the profile's allowed universe."""
    candidates = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
    if profile.allowed_instruments:
        candidates = [s for s in candidates if s.instrument in profile.allowed_instruments]
    if profile.allowed_sessions:
        candidates = [s for s in candidates if s.orb_label in profile.allowed_sessions]
    return candidates


def compute_profile_allocation(
    profile: AccountProfile,
    scores,
    orb_size_stats: dict[tuple[str, str], tuple[float, float]],
):
    """Build the current allocator-backed recommendation for one profile."""
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0
    candidates = eligible_candidates_for_profile(profile, scores)
    correlation_matrix = compute_pairwise_correlation(candidates) if candidates else {}
    return build_allocation(
        scores,
        max_slots=profile.max_slots,
        max_dd=max_dd,
        allowed_instruments=profile.allowed_instruments,
        allowed_sessions=profile.allowed_sessions,
        stop_multiplier=profile.stop_multiplier,
        orb_size_stats=orb_size_stats,
        correlation_matrix=correlation_matrix,
    )


def print_profile_report(
    profile_id: str,
    profile: AccountProfile,
    deployable_ids: set[str],
    scores,
    orb_size_stats: dict[tuple[str, str], tuple[float, float]],
) -> None:
    """Emit a human-readable dormant-profile rebuild report."""
    tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
    max_dd = tier.max_dd if tier else 3000.0
    current_lanes = effective_daily_lanes(profile)
    current_valid, current_ghosts = split_current_lanes(current_lanes, deployable_ids)
    score_by_id = {score.strategy_id: score for score in scores}
    candidates = eligible_candidates_for_profile(profile, scores)
    allocation = compute_profile_allocation(profile, scores, orb_size_stats)
    kept_valid, dropped_valid, added = summarize_lane_delta(current_valid, allocation)

    print("=" * 80)
    print(f"Profile: {profile_id}")
    print(f"  Active: {profile.active}")
    print(f"  Firm: {profile.firm}, Size: ${profile.account_size:,}, Copies: {profile.copies}")
    print(f"  Max slots: {profile.max_slots}, Max DD: ${max_dd:,.0f}")
    print(f"  Sessions: {sorted(profile.allowed_sessions) if profile.allowed_sessions else 'ANY'}")
    print(f"  Instruments: {sorted(profile.allowed_instruments) if profile.allowed_instruments else 'ANY'}")
    print(f"  Stop multiplier: {profile.stop_multiplier}")
    print(f"  Current lanes: {len(current_lanes)} ({len(current_valid)} valid, {len(current_ghosts)} ghosts)")
    print(f"  Eligible deployable candidates: {len(candidates)}")
    print(f"  Allocator-backed recommendation: {len(allocation)} lanes")
    print()

    if current_ghosts:
        print("  Ghost lanes:")
        for lane in current_ghosts:
            print(f"    - {lane.strategy_id}")
        print()

    if dropped_valid:
        print("  Current valid lanes omitted by recommendation:")
        for strategy_id in dropped_valid:
            lane_score = score_by_id.get(strategy_id)
            if lane_score is None:
                print(f"    - {strategy_id}")
                continue
            print(
                "    - "
                f"{strategy_id} "
                f"[{lane_score.status}: {lane_score.status_reason}; "
                f"annual_r={lane_score.annual_r_estimate:.1f}; "
                f"sr={lane_score.sr_status}]"
            )
        print()

    if not allocation:
        print("  Verdict: NO CURRENT DEPLOYABLE REBUILD")
        print()
        return

    print(f"  # --- Generated DailyLaneSpec entries for {profile_id} ---")
    print("  daily_lanes=(")
    for lane in allocation:
        max_orb = lane_cap_for(lane.instrument, lane.orb_label, orb_size_stats)
        if lane.strategy_id not in deployable_ids:
            raise RuntimeError(f"Allocator returned non-deployable lane: {lane.strategy_id}")
        print(
            f'    DailyLaneSpec("{lane.strategy_id}", '
            f'"{lane.instrument}", "{lane.orb_label}", '
            f"max_orb_size_pts={max_orb:.1f}),"
        )
    print("  ),")
    print()

    total_annual_r = sum(lane.annual_r_estimate for lane in allocation)
    instruments_used = sorted({lane.instrument for lane in allocation})
    sessions_used = sorted({lane.orb_label for lane in allocation})
    print(f"  Kept current valid lanes: {len(kept_valid)}")
    print(f"  Dropped current valid lanes: {len(dropped_valid)}")
    print(f"  Added new lanes: {len(added)}")
    print(f"  Rebuilt instruments: {instruments_used}")
    print(f"  Rebuilt sessions: {sessions_used}")
    print(f"  Estimated annual R: {total_annual_r:.1f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dormant profile lanes and emit allocator-backed rebuild suggestions."
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Single profile id to audit. Default: all inactive profiles.",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Also include active profiles in the audit output.",
    )
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Rebuild date (YYYY-MM-DD). Default: today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_ids = select_profile_ids(profile_id=args.profile, include_active=args.include_active)
    deployable_ids = load_deployable_strategy_ids()
    orb_size_stats = compute_orb_size_stats(args.date)
    scores = compute_lane_scores(rebalance_date=args.date)
    enrich_scores_with_liveness(scores)

    print(f"Dormant profile inventory audit as of {args.date}")
    print(f"Deployable shelf ids: {len(deployable_ids)}")
    print(f"Scored lanes: {len(scores)}")
    print(f"Profiles audited: {len(profile_ids)}")
    print()

    for profile_id in profile_ids:
        print_profile_report(
            profile_id,
            ACCOUNT_PROFILES[profile_id],
            deployable_ids,
            scores,
            orb_size_stats,
        )


if __name__ == "__main__":
    main()
