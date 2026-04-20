"""
Phase 2-9 X_MES_ATR60 block-bootstrap re-audit (Target 1 from vol-regime sprint).

Applies moving-block bootstrap null (from vol_regime_gates_g_h_i_j) to the
2 X_MES_ATR60-filtered MNQ COMEX_SETTLE Mode-A lanes and 2 OVNRNG_100
reference lanes that were CONTINUE×4'd by the 2026-04-20 framing audit
without a block-bootstrap gate.

Pre-reg: docs/audit/hypotheses/2026-04-20-phase-2-9-xmes-block-bootstrap-reaudit.yaml
Result:  docs/audit/results/2026-04-20-phase-2-9-xmes-block-bootstrap-reaudit.md (written after)

READ-ONLY. No DB writes. No validated_setups/experimental_strategies changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from research.vol_regime_gates_g_h_i_j import load_lane, moving_block_bootstrap
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM)
SEED = 20260420
N_PERMS = 10000

# (cell_id, filter_key, rr_target, priority)
CELLS = [
    ("MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60", "X_MES_ATR60", 1.0, "primary"),
    ("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60", "X_MES_ATR60", 1.5, "primary"),
    ("MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100", "OVNRNG_100", 1.0, "reference"),
    ("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100", "OVNRNG_100", 1.5, "reference"),
]

ORB_LABEL = "COMEX_SETTLE"
ORB_MINUTES = 5


def run_cell(con, cell_id, filter_key, rr, priority):
    df = load_lane(con, ORB_LABEL, ORB_MINUTES, rr)
    variant_mask = filter_signal(df, filter_key, ORB_LABEL)

    is_mask = (df["trading_day"] < HOLDOUT).to_numpy()
    pnl_base = df.loc[is_mask, "pnl_r"].to_numpy()
    variant_in_base = (variant_mask[is_mask] == 1).astype(bool)

    n_base = int(len(pnl_base))
    n_fire = int(variant_in_base.sum())
    fire_rate = float(n_fire / n_base) if n_base else 0.0

    if n_fire < 30:
        return {
            "cell_id": cell_id,
            "priority": priority,
            "status": "UNDERPOWERED",
            "n_base": n_base,
            "n_fire": n_fire,
            "fire_rate": round(fire_rate, 4),
            "note": "N<30 fire days; block-bootstrap not meaningful (RULE 3.2).",
        }

    result = moving_block_bootstrap(pnl_base, variant_in_base, n_perms=N_PERMS, seed=SEED)

    # Parametric t-stat on the actual variant-fire distribution (reporting only)
    pnl_fire = pnl_base[variant_in_base]
    t_is = float(pnl_fire.mean() / pnl_fire.std(ddof=1) * np.sqrt(len(pnl_fire)))

    # Decision per pre-reg decision tree
    p_boot = result["p_boot"]
    if priority == "primary":
        if p_boot < 0.05:
            verdict = "CONFIRM"
        elif p_boot < 0.10:
            verdict = "DOWNGRADE"
        else:
            verdict = "RETIRE_CANDIDATE"
    else:  # reference
        if p_boot < 0.05:
            verdict = "REFERENCE_PASS"
        else:
            verdict = "REFERENCE_FAIL_FLAG_CALIBRATION"

    return {
        "cell_id": cell_id,
        "priority": priority,
        "filter_key": filter_key,
        "rr_target": rr,
        "status": "TESTED",
        "n_base": n_base,
        "n_fire": n_fire,
        "fire_rate": round(fire_rate, 4),
        "t_is": round(t_is, 2),
        "observed_expr": round(result["observed_variant_expr"], 4),
        "null_mean": round(result["null_mean"], 4),
        "null_p95": round(result["null_p95"], 4),
        "null_p05": round(result["null_p05"], 4),
        "p_boot": round(p_boot, 4),
        "n_perms": result["n_perms"],
        "block_size": result["block_size"],
        "verdict": verdict,
    }


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print(f"Phase 2-9 X_MES_ATR60 block-bootstrap re-audit")
    print(f"  K = {len(CELLS)} cells  n_perms = {N_PERMS}  seed = {SEED}")
    print(f"  IS window: trading_day < {HOLDOUT.date()}")
    print()

    results = []
    for cell_id, filter_key, rr, priority in CELLS:
        r = run_cell(con, cell_id, filter_key, rr, priority)
        results.append(r)

        if r["status"] == "UNDERPOWERED":
            print(f"  [{r['priority'].upper():>9}] {r['cell_id']}  N_fire={r['n_fire']}  UNDERPOWERED")
            continue

        print(f"  [{r['priority'].upper():>9}] {r['cell_id']}")
        print(f"    n_base={r['n_base']}  n_fire={r['n_fire']}  fire_rate={r['fire_rate']:.3f}")
        print(f"    observed_expr={r['observed_expr']:+.4f}  t_is={r['t_is']:+.2f}")
        print(f"    null_mean={r['null_mean']:+.4f}  null_p95={r['null_p95']:+.4f}  p05={r['null_p05']:+.4f}")
        print(f"    p_boot={r['p_boot']:.4f}  block={r['block_size']}  VERDICT={r['verdict']}")
        print()

    # Aggregate verdict per pre-reg decision tree
    primary = [r for r in results if r.get("priority") == "primary" and r.get("status") == "TESTED"]
    reference = [r for r in results if r.get("priority") == "reference" and r.get("status") == "TESTED"]

    primary_pass = all(r["p_boot"] < 0.05 for r in primary)
    primary_retire = [r["cell_id"] for r in primary if r["p_boot"] >= 0.10]
    primary_downgrade = [r["cell_id"] for r in primary if 0.05 <= r["p_boot"] < 0.10]
    reference_fail = [r["cell_id"] for r in reference if r["p_boot"] >= 0.05]

    print("=" * 72)
    print("AGGREGATE VERDICT (per locked pre-reg decision tree)")
    print("=" * 72)
    if primary_pass and not reference_fail:
        aggregate = "CONFIRM_PHASE_2_9"
    elif primary_retire:
        aggregate = f"RETIRE_XMES_CANDIDATE ({', '.join(primary_retire)})"
    elif primary_downgrade:
        aggregate = f"DOWNGRADE_XMES_LANES ({', '.join(primary_downgrade)})"
    else:
        aggregate = "FLAG_NULL_CALIBRATION (reference unexpectedly failed)"

    print(f"  {aggregate}")
    print()

    # Persist
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "phase_2_9_xmes_block_bootstrap_reaudit.json"
    payload = {
        "seed": SEED,
        "n_perms": N_PERMS,
        "holdout": str(HOLDOUT.date()),
        "aggregate_verdict": aggregate,
        "primary_pass": primary_pass,
        "primary_retire": primary_retire,
        "primary_downgrade": primary_downgrade,
        "reference_fail": reference_fail,
        "cells": results,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Output JSON: {out}")


if __name__ == "__main__":
    main()
