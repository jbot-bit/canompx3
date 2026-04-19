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
    compute_orb_size_stats,
    build_allocation,
    compute_lane_scores,
    compute_pairwise_correlation,
    enrich_scores_with_dsr_diagnostics,
    enrich_scores_with_liveness,
    generate_report,
    save_allocation,
)
from trading_app.prop_profiles import ACCOUNT_PROFILES, ACCOUNT_TIERS


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

    # Enrich with SR liveness state
    enrich_scores_with_liveness(scores)
    sr_counts = {}
    for s in scores:
        sr_counts[s.sr_status] = sr_counts.get(s.sr_status, 0) + 1
    print(f"SR liveness: {sr_counts}")

    # Compute ORB size stats (per-session avg/P90 from trailing window)
    orb_stats = compute_orb_size_stats(args.date)
    print(f"ORB size stats: {len(orb_stats)} instrument×session combos")

    # Show 3mo decay warnings
    decaying = [s for s in scores if s.recent_3mo_expr is not None and s.recent_3mo_expr < 0 and s.trailing_expr > 0]
    if decaying:
        print(f"3mo decay warnings: {len(decaying)} strategies")
        for s in decaying[:5]:
            print(f"  {s.strategy_id}: 12mo={s.trailing_expr:+.4f} 3mo={s.recent_3mo_expr:+.4f}")

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

        # DD limit from canonical ACCOUNT_TIERS (not hardcoded)
        tier = ACCOUNT_TIERS.get((profile.firm, profile.account_size))
        max_dd = tier.max_dd if tier else 3000.0

        # Compute pairwise correlation matrix for deployable candidates
        deployable = [s for s in scores if s.status in ("DEPLOY", "RESUME", "PROVISIONAL")]
        if profile.allowed_instruments:
            deployable = [s for s in deployable if s.instrument in profile.allowed_instruments]
        if profile.allowed_sessions:
            deployable = [s for s in deployable if s.orb_label in profile.allowed_sessions]
        print(f"Computing pairwise correlation for {len(deployable)} candidates...")
        corr_matrix = compute_pairwise_correlation(deployable)
        print(f"  {len(corr_matrix)} pairs computed")

        # A2b-2 Shape E: populate per-lane DSR diagnostic fields + per-rebalance globals.
        # Diagnostic only; does NOT affect ranking or selection.
        dsr_globals = enrich_scores_with_dsr_diagnostics(scores, corr_matrix)
        print(
            f"  DSR diagnostics: n_eff_raw={dsr_globals['n_eff_raw']}, "
            f"n_hat_eq9={dsr_globals['n_hat_eq9']}, avg_rho={dsr_globals['avg_rho_hat']}"
        )

        allocation = build_allocation(
            scores,
            max_slots=profile.max_slots,
            max_dd=max_dd,
            allowed_instruments=profile.allowed_instruments,
            allowed_sessions=profile.allowed_sessions,
            stop_multiplier=profile.stop_multiplier,
            orb_size_stats=orb_stats,
            correlation_matrix=corr_matrix,
        )

        report = generate_report(scores, allocation, args.date, pid)
        print(report)

        # Save allocation
        out_path = save_allocation(
            scores, allocation, args.date, pid, args.output,
            orb_size_stats=orb_stats, dsr_globals=dsr_globals,
        )
        print(f"Allocation saved to: {out_path}")


if __name__ == "__main__":
    main()
