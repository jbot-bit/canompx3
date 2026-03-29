#!/usr/bin/env python3
"""Independent outcome replay verifier — bar-by-bar from first principles.

CRITICAL: This script does NOT import outcome_builder. The whole point is to
independently verify outcome_builder's logic by replaying 1m bars.

For each sampled outcome:
1. Load entry_price, stop_price, target_price from orb_outcomes
2. Load 1m bars from entry_ts to session end
3. Walk bars forward: check target hit, stop hit, ambiguous bar, session end
4. Compare replayed outcome/pnl to stored values

Usage:
    python scripts/verification/verify_outcome_replay.py
    python scripts/verification/verify_outcome_replay.py --sample 1000
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH

UTC = ZoneInfo("UTC")
BRISBANE = ZoneInfo("Australia/Brisbane")

SEED = 42
random.seed(SEED)


def connect_ro():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def replay_outcome(bars, entry_price, stop_price, target_price, direction, cost_spec):
    """Replay bars forward to determine outcome independently.

    Args:
        bars: list of (ts_utc, open, high, low, close, volume) tuples, chronological
        entry_price: trade entry price
        stop_price: stop loss price
        target_price: profit target price
        direction: 'long' or 'short'
        cost_spec: CostSpec for friction

    Returns:
        dict with: outcome, exit_price, exit_ts, pnl_r, mae_r, mfe_r, ambiguous_bar
    """
    if not bars:
        return {
            "outcome": "no_bars",
            "exit_price": None,
            "exit_ts": None,
            "pnl_r": None,
            "mae_r": None,
            "mfe_r": None,
            "ambiguous_bar": False,
        }

    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        return {
            "outcome": "zero_risk",
            "exit_price": None,
            "exit_ts": None,
            "pnl_r": None,
            "mae_r": None,
            "mfe_r": None,
            "ambiguous_bar": False,
        }

    risk_dollars = risk_pts * cost_spec.point_value + cost_spec.total_friction
    max_adverse_pts = 0.0
    max_favorable_pts = 0.0

    for ts, _bar_open, bar_high, bar_low, _bar_close, _bar_vol in bars:
        # Track MAE/MFE from bar extremes
        if direction == "long":
            adverse = entry_price - bar_low
            favorable = bar_high - entry_price
        else:
            adverse = bar_high - entry_price
            favorable = entry_price - bar_low

        max_adverse_pts = max(max_adverse_pts, adverse)
        max_favorable_pts = max(max_favorable_pts, favorable)

        # Check target and stop
        if direction == "long":
            target_hit = bar_high >= target_price
            stop_hit = bar_low <= stop_price
        else:
            target_hit = bar_low <= target_price
            stop_hit = bar_high >= stop_price

        # Ambiguous bar: both hit in same bar → conservative = LOSS
        if target_hit and stop_hit:
            exit_price = stop_price
            pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
            pnl_dollars = pnl_pts * cost_spec.point_value - cost_spec.total_friction
            return {
                "outcome": "loss",
                "exit_price": exit_price,
                "exit_ts": ts,
                "pnl_r": round(pnl_dollars / risk_dollars, 4),
                "mae_r": round(max_adverse_pts * cost_spec.point_value / risk_dollars, 4),
                "mfe_r": round(max_favorable_pts * cost_spec.point_value / risk_dollars, 4),
                "ambiguous_bar": True,
            }

        if target_hit:
            exit_price = target_price
            pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
            pnl_dollars = pnl_pts * cost_spec.point_value - cost_spec.total_friction
            return {
                "outcome": "win",
                "exit_price": exit_price,
                "exit_ts": ts,
                "pnl_r": round(pnl_dollars / risk_dollars, 4),
                "mae_r": round(max_adverse_pts * cost_spec.point_value / risk_dollars, 4),
                "mfe_r": round(max_favorable_pts * cost_spec.point_value / risk_dollars, 4),
                "ambiguous_bar": False,
            }

        if stop_hit:
            exit_price = stop_price
            pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
            pnl_dollars = pnl_pts * cost_spec.point_value - cost_spec.total_friction
            return {
                "outcome": "loss",
                "exit_price": exit_price,
                "exit_ts": ts,
                "pnl_r": round(pnl_dollars / risk_dollars, 4),
                "mae_r": round(max_adverse_pts * cost_spec.point_value / risk_dollars, 4),
                "mfe_r": round(max_favorable_pts * cost_spec.point_value / risk_dollars, 4),
                "ambiguous_bar": False,
            }

    # Session end — exit at last bar close (scratch)
    last_ts, _, _, _, last_close, _ = bars[-1]
    exit_price = last_close
    pnl_pts = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    pnl_dollars = pnl_pts * cost_spec.point_value - cost_spec.total_friction
    return {
        "outcome": "scratch",
        "exit_price": exit_price,
        "exit_ts": last_ts,
        "pnl_r": round(pnl_dollars / risk_dollars, 4),
        "mae_r": round(max_adverse_pts * cost_spec.point_value / risk_dollars, 4),
        "mfe_r": round(max_favorable_pts * cost_spec.point_value / risk_dollars, 4),
        "ambiguous_bar": False,
    }


def get_stratified_sample(con, total_n):
    """Get a stratified sample of outcomes to replay."""
    # Stratify by instrument × entry_model × outcome × orb_minutes
    # Only replay win/loss — scratches have NULL pnl_r, exit_price, exit_ts
    # (scratch = neither target nor stop hit by session end, no exit to verify)
    strata = con.execute("""
        SELECT symbol, entry_model, outcome, orb_minutes, COUNT(*) as n
        FROM orb_outcomes
        WHERE entry_price IS NOT NULL AND symbol IN ('MNQ', 'MGC', 'MES')
          AND outcome IN ('win', 'loss')
          AND exit_ts IS NOT NULL AND exit_price IS NOT NULL
        GROUP BY symbol, entry_model, outcome, orb_minutes
        ORDER BY n DESC
    """).fetchall()

    # Allocate samples proportionally, minimum 2 per stratum
    total_pop = sum(n for _, _, _, _, n in strata)
    samples = []

    for sym, em, outcome, om, n in strata:
        alloc = max(2, int(total_n * n / total_pop))
        rows = con.execute(
            """
            SELECT trading_day, orb_label, orb_minutes, rr_target, confirm_bars,
                   entry_model, entry_price, stop_price, target_price, exit_price,
                   outcome, pnl_r, mae_r, mfe_r, entry_ts, exit_ts,
                   ambiguous_bar, risk_dollars, symbol
            FROM orb_outcomes
            WHERE symbol = ? AND entry_model = ? AND outcome = ? AND orb_minutes = ?
              AND entry_price IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """,
            [sym, em, outcome, om, alloc],
        ).fetchall()
        samples.extend(rows)

    random.shuffle(samples)
    return samples[:total_n]


def main():
    parser = argparse.ArgumentParser(description="Independent outcome replay verifier")
    parser.add_argument("--sample", type=int, default=600, help="Number of outcomes to replay")
    args = parser.parse_args()

    print("=" * 60)
    print("  OUTCOME REPLAY VERIFIER — independent bar-by-bar")
    print(f"  Sample size: {args.sample}")
    print("=" * 60)

    con = connect_ro()

    samples = get_stratified_sample(con, args.sample)
    print(f"\n  Loaded {len(samples)} stratified outcomes")

    # Count by stratum
    by_stratum = defaultdict(int)
    for s in samples:
        by_stratum[(s[18], s[5], s[10], s[2])] += 1  # sym, em, outcome, om

    cols = [
        "trading_day",
        "orb_label",
        "orb_minutes",
        "rr_target",
        "confirm_bars",
        "entry_model",
        "entry_price",
        "stop_price",
        "target_price",
        "exit_price",
        "outcome",
        "pnl_r",
        "mae_r",
        "mfe_r",
        "entry_ts",
        "exit_ts",
        "ambiguous_bar",
        "risk_dollars",
        "symbol",
    ]

    total = 0
    matched = 0
    mismatched = 0
    mismatch_details = []
    outcome_mismatches = 0
    pnl_mismatches = 0
    ambig_mismatches = 0

    for row in samples:
        d = dict(zip(cols, row, strict=False))
        instrument = d["symbol"]
        cost_spec = get_cost_spec(instrument)

        entry_ts = d["entry_ts"]
        exit_ts = d["exit_ts"]
        if entry_ts is None:
            continue

        # Determine direction from entry vs stop
        if d["entry_price"] > d["stop_price"]:
            direction = "long"
        else:
            direction = "short"

        # Load 1m bars from entry_ts to reasonable session end
        # Use exit_ts + buffer to ensure we have enough bars
        if exit_ts is not None:
            end_ts = exit_ts + timedelta(minutes=5)
        else:
            end_ts = entry_ts + timedelta(hours=24)

        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars_1m
            WHERE symbol = ? AND ts_utc >= ? AND ts_utc <= ?
            ORDER BY ts_utc
        """,
            [instrument, entry_ts, end_ts],
        ).fetchall()

        if not bars:
            continue

        # Replay
        result = replay_outcome(bars, d["entry_price"], d["stop_price"], d["target_price"], direction, cost_spec)

        total += 1
        stored_outcome = d["outcome"]
        replayed_outcome = result["outcome"]

        # Compare outcome type
        outcome_match = stored_outcome == replayed_outcome
        if not outcome_match:
            # Allow scratch vs win/loss if exit_ts is at session boundary
            outcome_mismatches += 1

        # Compare pnl_r
        pnl_match = True
        if d["pnl_r"] is not None and result["pnl_r"] is not None:
            pnl_diff = abs(d["pnl_r"] - result["pnl_r"])
            if pnl_diff > 0.005:
                pnl_match = False
                pnl_mismatches += 1
        elif d["pnl_r"] is not None or result["pnl_r"] is not None:
            pnl_match = False

        # Compare ambiguous bar
        ambig_match = d["ambiguous_bar"] == result["ambiguous_bar"]
        if not ambig_match:
            ambig_mismatches += 1

        if outcome_match and pnl_match:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_details) < 30:
                mismatch_details.append(
                    {
                        "sym": instrument,
                        "day": d["trading_day"],
                        "session": d["orb_label"],
                        "em": d["entry_model"],
                        "om": d["orb_minutes"],
                        "rr": d["rr_target"],
                        "stored_outcome": stored_outcome,
                        "replay_outcome": replayed_outcome,
                        "stored_pnl_r": d["pnl_r"],
                        "replay_pnl_r": result["pnl_r"],
                        "pnl_diff": abs((d["pnl_r"] or 0) - (result["pnl_r"] or 0)),
                        "stored_ambig": d["ambiguous_bar"],
                        "replay_ambig": result["ambiguous_bar"],
                    }
                )

    con.close()

    # Report
    print(f"\n  Replayed: {total}")
    print(f"  MATCHED:  {matched} ({100 * matched / total:.1f}%)")
    print(f"  MISMATCH: {mismatched} ({100 * mismatched / total:.1f}%)")
    print(f"    Outcome type: {outcome_mismatches}")
    print(f"    pnl_r (>0.005R): {pnl_mismatches}")
    print(f"    Ambiguous bar: {ambig_mismatches}")

    if mismatch_details:
        print(f"\n  MISMATCH DETAILS (top {min(len(mismatch_details), 20)}):")
        print(
            f"  {'Sym':>4s} {'Day':>12s} {'Session':>20s} {'EM':>3s} {'OM':>3s} "
            f"{'Stored':>8s} {'Replay':>8s} {'St_R':>7s} {'Re_R':>7s} {'Diff':>6s}"
        )
        for m in sorted(mismatch_details, key=lambda x: -x["pnl_diff"])[:20]:
            print(
                f"  {m['sym']:>4s} {str(m['day']):>12s} {m['session']:>20s} "
                f"{m['em']:>3s} {m['om']:>3d} {m['stored_outcome']:>8s} "
                f"{m['replay_outcome']:>8s} {m['stored_pnl_r']:>+7.4f} "
                f"{m['replay_pnl_r']:>+7.4f} {m['pnl_diff']:>6.4f}"
            )

        # Categorize mismatches
        by_type = defaultdict(int)
        for m in mismatch_details:
            if m["stored_outcome"] != m["replay_outcome"]:
                by_type[f"{m['stored_outcome']}->{m['replay_outcome']}"] += 1
            else:
                by_type["pnl_only"] += 1
        print(f"\n  Mismatch categories: {dict(by_type)}")

    print(f"\n  {'=' * 55}")
    verdict = "PASS" if mismatched == 0 else "FAIL"
    print(f"  VERDICT: {verdict}")
    print(f"  {'=' * 55}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
