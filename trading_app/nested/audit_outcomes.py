"""
Spot-check audit for nested_outcomes: independently reconstructs outcomes
for random days and compares against stored values.

Catches:
  1. Resampling errors (5m bars don't match manual reconstruction)
  2. Outcome computation drift (stored outcome != recomputed outcome)
  3. E3 sub-bar fill verification misses (5m says fill, 1m says no)
  4. Suspiciously low/high E3 rejection rates

Usage:
    python -m trading_app.nested.audit_outcomes --instrument MGC --n-days 10
    python -m trading_app.nested.audit_outcomes --instrument MGC --n-days 20 --seed 42
"""

import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from pipeline.init_db import ORB_LABELS
from pipeline.build_daily_features import compute_trading_day_utc_range
from trading_app.outcome_builder import compute_single_outcome, RR_TARGETS, CONFIRM_BARS_OPTIONS
from trading_app.config import ENTRY_MODELS
from trading_app.nested.builder import resample_to_5m, _verify_e3_sub_bar_fill


def audit_nested_outcomes(
    db_path: Path | None = None,
    instrument: str = "MGC",
    n_days: int = 10,
    seed: int | None = None,
    orb_minutes_list: list[int] | None = None,
) -> dict:
    """Spot-check audit: reconstruct outcomes independently and compare.

    Returns dict with audit results:
      - n_checked: total outcome rows checked
      - n_match: outcomes that match exactly
      - n_mismatch: outcomes that differ
      - mismatches: list of mismatch details
      - e3_stats: E3 sub-bar rejection statistics
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if orb_minutes_list is None:
        orb_minutes_list = [15, 30]

    cost_spec = get_cost_spec(instrument)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        results = {
            "n_checked": 0,
            "n_match": 0,
            "n_mismatch": 0,
            "mismatches": [],
            "e3_stats": {
                "e3_total": 0,
                "e3_5m_fill": 0,
                "e3_1m_confirmed": 0,
                "e3_1m_rejected": 0,
            },
            "resampling_checks": 0,
            "resampling_errors": 0,
        }

        for orb_minutes in orb_minutes_list:
            print(f"\n--- Auditing orb_minutes={orb_minutes} ---")

            # Get all distinct trading days that have nested_outcomes
            days_rows = con.execute(
                """SELECT DISTINCT trading_day
                   FROM nested_outcomes
                   WHERE symbol = ? AND orb_minutes = ?
                   ORDER BY trading_day""",
                [instrument, orb_minutes],
            ).fetchall()

            all_days = [r[0] for r in days_rows]
            if not all_days:
                print(f"  No nested_outcomes for orb_minutes={orb_minutes}. Skipping.")
                continue

            print(f"  {len(all_days)} days with nested outcomes")

            # Sample random days
            rng = random.Random(seed)
            sample_days = rng.sample(all_days, min(n_days, len(all_days)))
            print(f"  Auditing {len(sample_days)} random days: {[str(d) for d in sample_days[:5]]}...")

            for trading_day in sample_days:
                _audit_single_day(
                    con, trading_day, instrument, orb_minutes,
                    cost_spec, results,
                )

        # Print summary
        _print_audit_summary(results)
        return results

    finally:
        con.close()


def _audit_single_day(con, trading_day, instrument, orb_minutes, cost_spec, results):
    """Audit all nested outcomes for a single trading day."""

    # 1. Load daily_features for this day + orb_minutes
    df_row = con.execute(
        """SELECT *
           FROM daily_features
           WHERE symbol = ? AND trading_day = ? AND orb_minutes = ?""",
        [instrument, trading_day, orb_minutes],
    ).fetchall()
    col_names = [desc[0] for desc in con.description]

    if not df_row:
        print(f"    WARN: No daily_features for {trading_day} orb_minutes={orb_minutes}")
        return

    features = dict(zip(col_names, df_row[0]))

    # 2. Load bars_1m for this trading day
    td_start, td_end = compute_trading_day_utc_range(trading_day)
    bars_1m_df = con.execute(
        """SELECT ts_utc, open, high, low, close, volume
           FROM bars_1m
           WHERE symbol = ?
           AND ts_utc >= ?::TIMESTAMPTZ
           AND ts_utc < ?::TIMESTAMPTZ
           ORDER BY ts_utc ASC""",
        [instrument, td_start.isoformat(), td_end.isoformat()],
    ).fetchdf()
    if not bars_1m_df.empty:
        bars_1m_df["ts_utc"] = pd.to_datetime(bars_1m_df["ts_utc"], utc=True)

    if bars_1m_df.empty:
        return

    # 3. Load stored nested outcomes for this day
    stored_rows = con.execute(
        """SELECT orb_label, rr_target, confirm_bars, entry_model,
                  outcome, pnl_r, entry_ts, entry_price, stop_price, target_price,
                  mae_r, mfe_r
           FROM nested_outcomes
           WHERE symbol = ? AND trading_day = ? AND orb_minutes = ?
           ORDER BY orb_label, entry_model, rr_target, confirm_bars""",
        [instrument, trading_day, orb_minutes],
    ).fetchall()
    stored_cols = [desc[0] for desc in con.description]
    stored_index = {}
    for r in stored_rows:
        d = dict(zip(stored_cols, r))
        key = (d["orb_label"], d["entry_model"], d["rr_target"], d["confirm_bars"])
        stored_index[key] = d

    # 4. Reconstruct outcomes independently
    day_checked = 0
    day_matched = 0

    for orb_label in ORB_LABELS:
        break_dir = features.get(f"orb_{orb_label}_break_dir")
        break_ts = features.get(f"orb_{orb_label}_break_ts")
        orb_high = features.get(f"orb_{orb_label}_high")
        orb_low = features.get(f"orb_{orb_label}_low")

        if break_dir is None or break_ts is None:
            continue
        if orb_high is None or orb_low is None:
            continue

        # Resample to 5m
        bars_5m_df = resample_to_5m(bars_1m_df, break_ts)

        # Quick resampling sanity check: 5m bar count should be ~1/5 of 1m count
        post_orb_1m = bars_1m_df[bars_1m_df["ts_utc"] > pd.Timestamp(break_ts)]
        results["resampling_checks"] += 1
        if not bars_5m_df.empty and not post_orb_1m.empty:
            ratio = len(post_orb_1m) / len(bars_5m_df)
            if ratio < 1.0 or ratio > 6.0:
                results["resampling_errors"] += 1
                print(f"    RESAMPLE WARN: {trading_day} {orb_label}: "
                      f"{len(post_orb_1m)} 1m bars -> {len(bars_5m_df)} 5m bars "
                      f"(ratio={ratio:.1f}, expected ~5)")

        if bars_5m_df.empty:
            continue

        for rr_target in RR_TARGETS:
            for cb in CONFIRM_BARS_OPTIONS:
                for em in ENTRY_MODELS:
                    key = (orb_label, em, rr_target, cb)
                    stored = stored_index.get(key)
                    if stored is None:
                        continue

                    # Recompute outcome
                    recomputed = compute_single_outcome(
                        bars_df=bars_5m_df,
                        break_ts=break_ts,
                        orb_high=orb_high,
                        orb_low=orb_low,
                        break_dir=break_dir,
                        rr_target=rr_target,
                        confirm_bars=cb,
                        trading_day_end=td_end,
                        cost_spec=cost_spec,
                        entry_model=em,
                    )

                    # E3 sub-bar fill verification (independent check)
                    if em == "E3":
                        results["e3_stats"]["e3_total"] += 1
                        if recomputed.get("entry_ts") is not None:
                            results["e3_stats"]["e3_5m_fill"] += 1
                            fill_ok = _verify_e3_sub_bar_fill(
                                bars_1m_df,
                                recomputed["entry_ts"],
                                recomputed["entry_price"],
                                break_dir,
                            )
                            if fill_ok:
                                results["e3_stats"]["e3_1m_confirmed"] += 1
                            else:
                                results["e3_stats"]["e3_1m_rejected"] += 1
                                # After rejection, outcome should be None
                                recomputed = {
                                    "entry_ts": None, "entry_price": None,
                                    "stop_price": None, "target_price": None,
                                    "outcome": None, "exit_ts": None,
                                    "exit_price": None, "pnl_r": None,
                                    "mae_r": None, "mfe_r": None,
                                }

                    # Compare stored vs recomputed
                    results["n_checked"] += 1
                    day_checked += 1

                    match = _outcomes_match(stored, recomputed)
                    if match:
                        results["n_match"] += 1
                        day_matched += 1
                    else:
                        results["n_mismatch"] += 1
                        mismatch_detail = {
                            "trading_day": str(trading_day),
                            "orb_label": orb_label,
                            "orb_minutes": orb_minutes,
                            "entry_model": em,
                            "rr_target": rr_target,
                            "confirm_bars": cb,
                            "stored_outcome": stored["outcome"],
                            "stored_pnl_r": stored["pnl_r"],
                            "recomputed_outcome": recomputed["outcome"],
                            "recomputed_pnl_r": recomputed["pnl_r"],
                        }
                        results["mismatches"].append(mismatch_detail)
                        if len(results["mismatches"]) <= 10:
                            print(f"    MISMATCH: {trading_day} {orb_label} {em} "
                                  f"RR{rr_target} CB{cb}: "
                                  f"stored={stored['outcome']}/{stored['pnl_r']} "
                                  f"vs recomputed={recomputed['outcome']}/{recomputed['pnl_r']}")

    if day_checked > 0:
        print(f"  {trading_day}: {day_matched}/{day_checked} match "
              f"({day_matched/day_checked*100:.1f}%)")


def _outcomes_match(stored: dict, recomputed: dict) -> bool:
    """Compare stored vs recomputed outcome, allowing for float tolerance."""
    # Outcome string must match exactly
    if stored["outcome"] != recomputed["outcome"]:
        return False

    # If both None, that's a match
    if stored["outcome"] is None and recomputed["outcome"] is None:
        return True

    # pnl_r must match within tolerance
    s_pnl = stored.get("pnl_r")
    r_pnl = recomputed.get("pnl_r")
    if s_pnl is None and r_pnl is None:
        pass  # both None = match
    elif s_pnl is not None and r_pnl is not None:
        if abs(s_pnl - r_pnl) > 0.01:
            return False
    else:
        return False  # one None, other not

    # entry_price must match
    s_entry = stored.get("entry_price")
    r_entry = recomputed.get("entry_price")
    if s_entry is None and r_entry is None:
        pass  # both None = match
    elif s_entry is not None and r_entry is not None:
        if abs(s_entry - r_entry) > 0.01:
            return False
    else:
        return False  # one None, other not

    return True


def _print_audit_summary(results: dict):
    """Print formatted audit summary."""
    print("\n" + "=" * 70)
    print("NESTED OUTCOMES AUDIT SUMMARY")
    print("=" * 70)

    total = results["n_checked"]
    if total == 0:
        print("  No outcomes checked. Is the nested builder done?")
        return

    match_pct = results["n_match"] / total * 100
    print(f"  Total checked:  {total}")
    print(f"  Matches:        {results['n_match']} ({match_pct:.1f}%)")
    print(f"  Mismatches:     {results['n_mismatch']}")

    print(f"\n  Resampling checks: {results['resampling_checks']}")
    print(f"  Resampling errors: {results['resampling_errors']}")

    e3 = results["e3_stats"]
    print("\n  E3 Sub-bar Fill Verification:")
    print(f"    E3 outcomes total:    {e3['e3_total']}")
    print(f"    5m showed fill:       {e3['e3_5m_fill']}")
    if e3["e3_5m_fill"] > 0:
        confirm_pct = e3["e3_1m_confirmed"] / e3["e3_5m_fill"] * 100
        reject_pct = e3["e3_1m_rejected"] / e3["e3_5m_fill"] * 100
        print(f"    1m confirmed fill:    {e3['e3_1m_confirmed']} ({confirm_pct:.1f}%)")
        print(f"    1m rejected (phantom):{e3['e3_1m_rejected']} ({reject_pct:.1f}%)")

        if reject_pct > 30:
            print("    ** WARNING: >30% E3 phantom fill rate is suspiciously high **")
        elif reject_pct < 1:
            print("    ** NOTE: <1% rejection rate — sub-bar check may not be filtering much **")

    if results["mismatches"]:
        print(f"\n  First {min(10, len(results['mismatches']))} mismatches:")
        for m in results["mismatches"][:10]:
            print(f"    {m['trading_day']} {m['orb_label']} {m['entry_model']} "
                  f"RR{m['rr_target']} CB{m['confirm_bars']}: "
                  f"stored={m['stored_outcome']}/{m['stored_pnl_r']} "
                  f"vs recomputed={m['recomputed_outcome']}/{m['recomputed_pnl_r']}")

    if results["n_mismatch"] == 0:
        print("\n  ALL CHECKS PASSED [OK]")
    else:
        print(f"\n  ** {results['n_mismatch']} MISMATCHES FOUND — INVESTIGATE **")

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Spot-check audit of nested_outcomes against independent reconstruction"
    )
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--n-days", type=int, default=10,
                        help="Number of random days to audit (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--orb-minutes", type=int, nargs="+", default=[15, 30])
    args = parser.parse_args()

    audit_nested_outcomes(
        instrument=args.instrument,
        n_days=args.n_days,
        seed=args.seed,
        orb_minutes_list=args.orb_minutes,
    )


if __name__ == "__main__":
    main()
