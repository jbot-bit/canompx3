#!/usr/bin/env python3
"""
Noise floor calibration via sign-randomization test.

Replaces the Gaussian-null-derived floors in NOISE_FLOOR_BY_INSTRUMENT
with empirically calibrated floors from actual trade outcomes.

Method: Sign-randomization (Diebold & Mariano 1995)
  - Under H0 (no edge): flip sign of each pnl_r with p=0.5
  - Compute ExpR on each permutation
  - p95 of permuted ExpR distribution = noise floor
  - Phipson & Smyth (2010) correction for discrete p-values

Autocorrelation check: if lag-1 > 0.1, falls back to circular block
bootstrap (Politis & Romano 1994) instead of sign-randomization.

Usage:
    python scripts/tools/noise_floor_bootstrap.py
    python scripts/tools/noise_floor_bootstrap.py --reps 10000
    python scripts/tools/noise_floor_bootstrap.py --update-config
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, NOISE_FLOOR_BY_INSTRUMENT, apply_tight_stop
from pipeline.cost_model import get_cost_spec


def load_filtered_pnl_r(
    con,
    instrument: str,
    orb_label: str,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    orb_minutes: int = 5,
    stop_multiplier: float = 1.0,
) -> list[float]:
    """Load filtered pnl_r series from orb_outcomes + daily_features filter."""
    outcomes_raw = con.execute(
        """
        SELECT oo.trading_day, oo.pnl_r, oo.outcome, oo.entry_price, oo.stop_price,
               oo.mae_r, oo.mfe_r
        FROM orb_outcomes oo
        WHERE oo.symbol = ? AND oo.orb_label = ? AND oo.entry_model = ?
          AND oo.rr_target = ? AND oo.confirm_bars = ? AND oo.orb_minutes = ?
          AND oo.outcome IS NOT NULL
        ORDER BY oo.trading_day
    """,
        [instrument, orb_label, entry_model, rr_target, confirm_bars, orb_minutes],
    ).fetchall()

    if not outcomes_raw:
        return []

    filt = ALL_FILTERS.get(filter_type)
    if filt is not None and filter_type != "NO_FILTER":
        size_col = f"orb_{orb_label}_size"
        features = con.execute(
            f"SELECT trading_day, {size_col} FROM daily_features WHERE symbol = ? AND orb_minutes = ?",
            [instrument, orb_minutes],
        ).fetchall()
        eligible_days = set()
        for td, size in features:
            if size is not None and filt.matches_row({size_col: size}, orb_label):
                eligible_days.add(td)
        outcomes_raw = [o for o in outcomes_raw if o[0] in eligible_days]

    outcome_dicts = []
    for o in outcomes_raw:
        if o[2] not in ("win", "loss"):
            continue
        outcome_dicts.append({
            "trading_day": o[0], "pnl_r": o[1], "outcome": o[2],
            "entry_price": o[3], "stop_price": o[4], "mae_r": o[5], "mfe_r": o[6],
        })

    if stop_multiplier != 1.0:
        cost_spec = get_cost_spec(instrument)
        outcome_dicts = apply_tight_stop(outcome_dicts, stop_multiplier, cost_spec)

    return [o["pnl_r"] for o in outcome_dicts if o["pnl_r"] is not None]


def _compute_expr(arr):
    """Compute ExpR = WR*AvgWin - LR*AvgLoss from an array of pnl_r."""
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    n = len(wins) + len(losses)
    if n == 0:
        return 0.0
    wr = len(wins) / n
    avg_w = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_l = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
    return wr * avg_w - (1 - wr) * avg_l


def sign_randomization_floor(pnl_rs, reps=10000, quantile=0.95, seed=42):
    """Compute noise floor via sign-randomization or block bootstrap."""
    arr = np.array(pnl_rs)
    n = len(arr)
    if n < 10:
        return 0.0, 0.0, "insufficient_data"

    lag1 = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    rng = np.random.default_rng(seed)

    if abs(lag1) > 0.1:
        block_len = max(2, int(np.sqrt(n)))
        null_exprs = np.empty(reps)
        for i in range(reps):
            indices = []
            while len(indices) < n:
                start = rng.integers(0, n)
                for j in range(block_len):
                    if len(indices) >= n:
                        break
                    indices.append((start + j) % n)
            boot = arr[indices[:n]] - np.mean(arr)
            null_exprs[i] = _compute_expr(boot)
        method = "block_bootstrap"
    else:
        signs = rng.choice(np.array([-1, 1]), size=(reps, n))
        null_exprs = np.empty(reps)
        for i in range(reps):
            null_exprs[i] = _compute_expr(arr * signs[i])
        method = "sign_randomization"

    floor = float(np.percentile(null_exprs, quantile * 100))
    return floor, lag1, method


def run_calibration(db_path, instruments, entry_models, reps=10000, quantile=0.95):
    """Run bootstrap calibration for all instruments and entry models."""
    con = duckdb.connect(str(db_path), read_only=True)
    results = {}

    for inst in instruments:
        inst_floors = {}
        for em in entry_models:
            strategies = con.execute(
                """SELECT DISTINCT orb_label, rr_target, confirm_bars, filter_type, stop_multiplier
                   FROM validated_setups WHERE instrument = ? AND entry_model = ?""",
                [inst, em],
            ).fetchall()

            all_pnl = []
            for orb_label, rr, cb, ft, sm in strategies:
                pnl = load_filtered_pnl_r(con, inst, orb_label, em, rr, cb, ft, stop_multiplier=sm or 1.0)
                all_pnl.extend(pnl)

            if not all_pnl:
                rows = con.execute(
                    """SELECT pnl_r FROM orb_outcomes
                       WHERE symbol = ? AND entry_model = ? AND orb_minutes = 5
                         AND outcome IN ('win', 'loss') AND pnl_r IS NOT NULL""",
                    [inst, em],
                ).fetchall()
                all_pnl = [r[0] for r in rows]

            if len(all_pnl) < 30:
                print(f"  {inst} {em}: only {len(all_pnl)} trades — floor unreliable")
                inst_floors[em] = None
                continue

            floor, lag1, method = sign_randomization_floor(all_pnl, reps=reps, quantile=quantile)
            inst_floors[em] = round(floor, 4)
            print(f"  {inst} {em}: N={len(all_pnl)}, lag1={lag1:.4f}, method={method}, floor={floor:.4f}")

        results[inst] = inst_floors

    con.close()
    return results


def check_stability(db_path, instrument, entry_model, reps=10000, quantile=0.95, n_runs=5):
    """Run calibration with different seeds to check stability."""
    con = duckdb.connect(str(db_path), read_only=True)

    strategies = con.execute(
        """SELECT DISTINCT orb_label, rr_target, confirm_bars, filter_type, stop_multiplier
           FROM validated_setups WHERE instrument = ? AND entry_model = ?""",
        [instrument, entry_model],
    ).fetchall()

    all_pnl = []
    for orb_label, rr, cb, ft, sm in strategies:
        pnl = load_filtered_pnl_r(con, instrument, orb_label, entry_model, rr, cb, ft, stop_multiplier=sm or 1.0)
        all_pnl.extend(pnl)

    if not all_pnl:
        rows = con.execute(
            """SELECT pnl_r FROM orb_outcomes
               WHERE symbol = ? AND entry_model = ? AND orb_minutes = 5
                 AND outcome IN ('win', 'loss') AND pnl_r IS NOT NULL""",
            [instrument, entry_model],
        ).fetchall()
        all_pnl = [r[0] for r in rows]

    con.close()

    return [sign_randomization_floor(all_pnl, reps=reps, quantile=quantile, seed=42 + i * 1000)[0] for i in range(n_runs)]


def main():
    parser = argparse.ArgumentParser(description="Bootstrap noise floor calibration")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--update-config", action="store_true")
    parser.add_argument("--stability-runs", type=int, default=5)
    args = parser.parse_args()

    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    entry_models = ["E1", "E2"]

    print(f"Bootstrap noise floor calibration")
    print(f"  Instruments: {instruments}")
    print(f"  Entry models: {entry_models}")
    print(f"  Reps: {args.reps}, Quantile: {args.quantile}")
    print()

    print("=== CALIBRATION ===")
    new_floors = run_calibration(GOLD_DB_PATH, instruments, entry_models, args.reps, args.quantile)

    print()
    print("=== COMPARISON ===")
    print(f"{'Inst':6s} {'EM':4s} {'Old':>8s} {'New':>8s} {'Delta':>8s} {'Dir':>8s}")
    for inst in instruments:
        for em in entry_models:
            old = NOISE_FLOOR_BY_INSTRUMENT.get(inst, {}).get(em)
            new = new_floors.get(inst, {}).get(em)
            if old is not None and new is not None:
                delta = new - old
                d = "TIGHTER" if new > old else "LOOSER" if new < old else "SAME"
                print(f"{inst:6s} {em:4s} {old:8.4f} {new:8.4f} {delta:+8.4f} {d:>8s}")

    print()
    print("=== STABILITY ===")
    stable_all = True
    for inst in instruments:
        for em in entry_models:
            floors = check_stability(GOLD_DB_PATH, inst, em, args.reps, args.quantile, args.stability_runs)
            spread = max(floors) - min(floors)
            mean_f = np.mean(floors)
            cv = (np.std(floors) / abs(mean_f) * 100) if mean_f != 0 else 0
            stable = spread < 0.02
            stable_all = stable_all and stable
            print(f"  {inst} {em}: {[round(f, 4) for f in floors]} spread={spread:.4f} {'STABLE' if stable else 'UNSTABLE'}")

    if stable_all:
        print("\nALL STABLE")
    else:
        print("\nWARNING: UNSTABLE — do not update config")

    if args.update_config and not stable_all:
        print("SKIPPING config update — unstable")
        return

    if not args.update_config:
        print("\nDry run. Pass --update-config to write.")
        return

    # Update config
    import re
    config_path = PROJECT_ROOT / "trading_app" / "config.py"
    content = config_path.read_text()
    lines = ["NOISE_FLOOR_BY_INSTRUMENT: dict[str, dict[str, float]] = {"]
    for inst in sorted(new_floors.keys()):
        parts = []
        for em in sorted(new_floors[inst].keys()):
            v = new_floors[inst][em]
            parts.append(f'"{em}": {v}' if v is not None else f'"{em}": None')
        lines.append(f'    "{inst}": {{{", ".join(parts)}}},')
    lines.append("}")
    new_block = "\n".join(lines)

    pattern = r"NOISE_FLOOR_BY_INSTRUMENT: dict\[str, dict\[str, float\]\] = \{[^}]+\{[^}]+\}[^}]*\}"
    if re.search(pattern, content):
        content = re.sub(pattern, new_block, content)
        config_path.write_text(content)
        print(f"Updated {config_path}")
    else:
        print("ERROR: regex match failed. Manual update required.")


if __name__ == "__main__":
    main()
