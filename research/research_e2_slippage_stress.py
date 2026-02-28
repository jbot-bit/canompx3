#!/usr/bin/env python3
"""
E2 Slippage Stress Test — PASS2 Audit Phase D

Stress-tests E2 (stop-market) outcomes at 1x, 1.5x, 2x, 3x slippage
multiples. Reports Sharpe/winrate degradation per instrument/session.

E2 entry price = ORB high/low + slippage ticks. This script answers:
"How badly does E2 performance degrade if real slippage is worse than assumed?"

Uses cost_model.stress_test_costs() for consistent friction scaling.
Reads directly from gold.db (read-only) — no schema changes.

Usage:
    python research/research_e2_slippage_stress.py
    python research/research_e2_slippage_stress.py --instrument MGC
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import COST_SPECS, stress_test_costs
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS


SLIPPAGE_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]


def run_stress_test(instrument: str | None = None):
    """Run slippage stress test for one or all instruments."""
    instruments = [instrument] if instrument else list(ACTIVE_ORB_INSTRUMENTS)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        print("=" * 80)
        print("E2 SLIPPAGE STRESS TEST")
        print("=" * 80)
        print()

        for inst in instruments:
            cost_spec = COST_SPECS[inst]
            base_friction = cost_spec.total_friction

            # Get E2 validated strategies for this instrument
            strategies = con.execute(
                """SELECT strategy_id, orb_label, sample_size, win_rate,
                          expectancy_r, sharpe_ann, avg_risk_dollars
                   FROM validated_setups
                   WHERE instrument = ? AND entry_model = 'E2'
                   AND status = 'active'
                   ORDER BY sharpe_ann DESC NULLS LAST""",
                [inst],
            ).fetchall()

            if not strategies:
                print(f"{inst}: No active E2 strategies — skipping")
                print()
                continue

            print(f"{inst}: {len(strategies)} active E2 strategies")
            print(f"  Base friction: ${base_friction:.2f} RT")
            print(f"  Point value: ${cost_spec.point_value:.0f}")
            print()

            # Header
            header = f"  {'Multiplier':>10} | {'Friction':>10} | {'Avg ExpR':>10} | {'Avg WR':>8} | {'Pct Positive':>12}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for mult in SLIPPAGE_MULTIPLIERS:
                stressed = stress_test_costs(cost_spec, mult)
                extra_friction = stressed.total_friction - base_friction
                # extra_friction is in dollars; convert to R using avg_risk_dollars

                adjusted_exp_r_list = []
                for sid, orb_label, n, wr, exp_r, sharpe, avg_risk_d in strategies:
                    if exp_r is None or avg_risk_d is None or avg_risk_d <= 0:
                        continue
                    extra_r = extra_friction / avg_risk_d
                    adj_exp_r = exp_r - extra_r
                    adjusted_exp_r_list.append(adj_exp_r)

                if not adjusted_exp_r_list:
                    print(f"  {mult:>10.1f}x | ${stressed.total_friction:>8.2f} | {'N/A':>10} | {'N/A':>8} | {'N/A':>12}")
                    continue

                avg_exp_r = sum(adjusted_exp_r_list) / len(adjusted_exp_r_list)
                pct_positive = sum(1 for e in adjusted_exp_r_list if e > 0) / len(adjusted_exp_r_list)

                # Winrate doesn't change with slippage (same entry/exit, different accounting)
                avg_wr = sum(s[3] for s in strategies if s[3]) / len(strategies)

                print(
                    f"  {mult:>10.1f}x | ${stressed.total_friction:>8.2f} | "
                    f"{avg_exp_r:>10.4f} | {avg_wr:>7.1%} | {pct_positive:>11.1%}"
                )

            print()

            # Show worst-hit strategies at 2x
            stressed_2x = stress_test_costs(cost_spec, 2.0)
            extra_2x = stressed_2x.total_friction - base_friction
            print(f"  Strategies that go NEGATIVE at 2x slippage:")
            negative_at_2x = []
            for sid, orb_label, n, wr, exp_r, sharpe, avg_risk_d in strategies:
                if exp_r is None or avg_risk_d is None or avg_risk_d <= 0:
                    continue
                adj = exp_r - (extra_2x / avg_risk_d)
                if adj <= 0:
                    negative_at_2x.append((sid, orb_label, exp_r, adj))

            if negative_at_2x:
                for sid, sess, base_exp, adj_exp in negative_at_2x:
                    print(f"    {sid}: {sess} base={base_exp:.4f} -> 2x={adj_exp:.4f}")
            else:
                print("    None — all survive 2x slippage")

            print()

    finally:
        con.close()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="E2 slippage stress test")
    parser.add_argument("--instrument", type=str, default=None)
    args = parser.parse_args()
    run_stress_test(instrument=args.instrument)
