"""One-shot: Generate validated lane assignments for inactive profiles with ghosts.

Runs the canonical allocator for each broken profile and outputs
the recommended DailyLaneSpec entries. Review output before applying
to prop_profiles.py.

Usage:
    python scripts/tools/generate_profile_lanes.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_allocator import build_allocation, compute_lane_scores
from trading_app.prop_profiles import (
    _P90_ORB_PTS,
    ACCOUNT_PROFILES,
    ACCOUNT_TIERS,
)

# Profiles to regenerate (inactive with ghosts, excluding topstep_50k which is
# intentionally conditional with 0 validated alternatives for MGC+TOKYO_OPEN)
PROFILES_TO_FIX = [
    "tradeify_50k",
    "topstep_50k_type_a",
    "topstep_100k_type_a",
    "tradeify_50k_type_b",
    "tradeify_100k_type_b",
]

# Validate all lane strategy_ids against validated_setups
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
validated_ids = set(
    r[0] for r in con.execute("SELECT strategy_id FROM validated_setups WHERE status = 'active'").fetchall()
)
con.close()


def main() -> None:
    today = date.today()
    print(f"Generating profile lanes as of {today}")
    print(f"Validated strategies: {len(validated_ids)}")
    print()

    # Score all strategies
    scores = compute_lane_scores(rebalance_date=today)
    print(f"Scored {len(scores)} strategies")
    print()

    for pid in PROFILES_TO_FIX:
        profile = ACCOUNT_PROFILES[pid]
        tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
        max_dd = tier.max_dd if tier else 3000.0

        print("=" * 80)
        print(f"Profile: {pid}")
        print(f"  Firm: {profile.firm}, Size: ${profile.account_size:,}")
        print(f"  Max slots: {profile.max_slots}, Max DD: ${max_dd:,.0f}")
        print(f"  Sessions: {sorted(profile.allowed_sessions) if profile.allowed_sessions else 'ANY'}")
        print(f"  Instruments: {sorted(profile.allowed_instruments) if profile.allowed_instruments else 'ANY'}")
        print(f"  Stop multiplier: {profile.stop_multiplier}")
        print()

        # Count current ghosts
        current_ghosts = [ln for ln in profile.daily_lanes if ln.strategy_id not in validated_ids]
        current_valid = [ln for ln in profile.daily_lanes if ln.strategy_id in validated_ids]
        print(f"  Current: {len(profile.daily_lanes)} lanes ({len(current_valid)} valid, {len(current_ghosts)} ghosts)")

        allocation = build_allocation(
            scores,
            max_slots=profile.max_slots,
            max_dd=max_dd,
            allowed_instruments=profile.allowed_instruments,
            allowed_sessions=profile.allowed_sessions,
            stop_multiplier=profile.stop_multiplier,
        )

        print(f"  Allocator output: {len(allocation)} lanes")
        print()

        # Generate DailyLaneSpec code
        print(f"  # --- Generated DailyLaneSpec entries for {pid} ---")
        print("  daily_lanes=(")
        for lane in allocation:
            p90 = _P90_ORB_PTS.get(lane.instrument, 100.0)
            # Verify it's actually validated
            assert lane.strategy_id in validated_ids, f"Allocator returned non-validated: {lane.strategy_id}"
            print(
                f'    DailyLaneSpec("{lane.strategy_id}", '
                f'"{lane.instrument}", "{lane.orb_label}", '
                f"max_orb_size_pts={p90}),"
            )
        print("  ),")
        print()

        # Summary stats
        if allocation:
            instruments_used = set(ln.instrument for ln in allocation)
            sessions_used = set(ln.orb_label for ln in allocation)
            total_annual_r = sum(ln.annual_r_estimate for ln in allocation)
            print(f"  Instruments: {sorted(instruments_used)}")
            print(f"  Sessions: {sorted(sessions_used)}")
            print(f"  Est annual R: {total_annual_r:.1f}")
        print()


if __name__ == "__main__":
    main()
