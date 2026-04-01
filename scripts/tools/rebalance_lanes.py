"""
Monthly lane rebalancer — CLI entry point.

Usage:
    python scripts/tools/rebalance_lanes.py --date 2026-04-01
    python scripts/tools/rebalance_lanes.py --date 2026-04-01 --profile apex_100k_manual
    python scripts/tools/rebalance_lanes.py --date 2026-04-01 --all-profiles
"""

import argparse
import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading_app.lane_allocator import (
    build_allocation,
    compute_lane_scores,
    generate_report,
    save_allocation,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly lane rebalancer")
    parser.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date.today(),
        help="Rebalance date (YYYY-MM-DD). Trailing window ends here. Default: today.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile ID (e.g., apex_100k_manual). Default: first active profile.",
    )
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Run allocation for ALL active profiles.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: docs/runtime/lane_allocation.json",
    )
    args = parser.parse_args()

    print(f"Rebalance date: {args.date}")
    print("Computing lane scores...")

    scores = compute_lane_scores(rebalance_date=args.date)
    print(f"Scored {len(scores)} validated strategies")

    # Determine profiles to process
    if args.all_profiles:
        profiles = {pid: p for pid, p in ACCOUNT_PROFILES.items() if p.active}
    elif args.profile:
        if args.profile not in ACCOUNT_PROFILES:
            print(f"ERROR: Profile '{args.profile}' not found")
            sys.exit(1)
        profiles = {args.profile: ACCOUNT_PROFILES[args.profile]}
    else:
        # First active profile
        for pid, p in ACCOUNT_PROFILES.items():
            if p.active:
                profiles = {pid: p}
                break
        else:
            print("ERROR: No active profiles found")
            sys.exit(1)

    for pid, profile in profiles.items():
        print(f"\n{'=' * 60}")
        print(f"Profile: {pid} ({profile.firm})")
        print(f"{'=' * 60}")

        allocation = build_allocation(
            scores,
            max_slots=profile.max_slots,
            max_dd=2000.0,  # TODO: read from ACCOUNT_TIERS
            allowed_instruments=profile.allowed_instruments,
            allowed_sessions=profile.allowed_sessions,
            stop_multiplier=profile.stop_multiplier,
        )

        report = generate_report(scores, allocation, args.date, pid)
        print(report)

        # Save allocation
        out_path = save_allocation(scores, allocation, args.date, pid, args.output)
        print(f"\nAllocation saved to: {out_path}")


if __name__ == "__main__":
    main()
