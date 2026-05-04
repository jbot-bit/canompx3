#!/usr/bin/env python3
"""H0 — MGC real-slippage sensitivity on native_low_r_v1 killed cells.

Pre-reg: docs/audit/hypotheses/2026-04-20-mgc-real-slippage-sensitivity.yaml
Parent audit: docs/audit/results/2026-04-20-mgc-adversarial-reexamination.md §9 H0

Confirmatory audit (K=0 MinBTL trials). Question: does the 2026-04-19 MGC
thread closure hold under a range of friction assumptions, or is it an
artifact of one under-modeled slippage choice?

Design per research-truth-protocol.md § Canonical filter delegation and
institutional-rigor.md Rule 4 (never re-encode canonical logic):

- Cell identity, filter application, and LR-target rewrite are inherited
  directly from the canonical upstream `research_mgc_payoff_compression_audit`
  (the same module `research_mgc_native_low_r_v1.py` uses). Re-implementing
  here would re-enter the failure class of the first H0 attempt (2026-04-20
  HALT — filter_type re-encoding, direction inference, date-window mismatch).
- Slippage sensitivity is applied by replacing the MGC CostSpec's
  `slippage` field with each grid value and recomputing per-row
  `lower_X_pnl_r` using the canonical `pipeline.cost_model` math.

Grid: [2, 4, 6.75, 10] ticks round-trip. MGC tick = $1 so dollars = ticks.
Modeled = 2 ticks. Pilot mean = 6.75 ticks. High-stress = 10 ticks.

Halt condition (pre-reg §baseline_cross_check):
  Baseline reproduction at slippage=2 must match native_low_r_v1 reported
  ExpR values within ±0.002R for all 5 cells. Otherwise HALT.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import CostSpec, get_cost_spec, to_r_multiple
from research.research_mgc_payoff_compression_audit import (
    FAMILIES,
    build_family_trade_matrix,
    load_rows,
)

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01")

# Slippage grid (MGC tick_size=0.10, point_value=10.0 → $1/tick, so ticks == dollars)
SLIPPAGE_GRID_DOLLARS = [2.00, 4.00, 6.75, 10.00]

# Five cells killed in path_accurate_subr_v1; source = native_low_r_v1 BH survivors
# (family_id, variant_col, reported_native_low_r_v1_exp_r)
CELLS: list[tuple[str, str, float]] = [
    ("NYSE_OPEN_OVNRNG_50_RR1", "lower_0_75_pnl_r", 0.2226),
    ("US_DATA_1000_ATR_P70_RR1", "lower_0_5_pnl_r", 0.0710),
    ("US_DATA_1000_OVNRNG_10_RR1", "lower_0_5_pnl_r", 0.0685),
    ("US_DATA_1000_BROAD_RR1", "lower_0_5_pnl_r", 0.0488),
    ("NYSE_OPEN_BROAD_RR1", "lower_0_5_pnl_r", 0.0380),
]

# Family → canonical FamilySpec (for filter look-up); built once from FAMILIES
FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

# Variant col → conservative target fraction of R (matches
# `research_mgc_payoff_compression_audit.conservative_lower_target_pnl`)
VARIANT_TARGET_R = {
    "lower_0_5_pnl_r": 0.5,
    "lower_0_75_pnl_r": 0.75,
}

MGC_MODELED_SPEC = get_cost_spec("MGC")
MGC_MODELED_SLIPPAGE = MGC_MODELED_SPEC.slippage  # $2.00 — canonical modeled


def spec_at_slippage(slippage_dollars: float) -> CostSpec:
    """MGC CostSpec with slippage field replaced; commission + spread unchanged."""
    return replace(MGC_MODELED_SPEC, slippage=float(slippage_dollars))


def target_net_r_at(spec: CostSpec, entry: float, stop: float, target_r: float) -> float | None:
    """Port of `research_mgc_payoff_compression_audit.target_net_r`, spec-parameterized."""
    risk_points = abs(float(entry) - float(stop))
    if risk_points <= 0:
        return None
    return round(to_r_multiple(spec, float(entry), float(stop), risk_points * target_r), 4)


def recompute_lower_target_pnl(
    row: pd.Series,
    spec: CostSpec,
    target_r: float,
) -> float | None:
    """Spec-parameterized port of `conservative_lower_target_pnl`.

    Matches the canonical logic:
      - if ambiguous_bar AND outcome='loss': return recomputed pnl_r (not rescued)
      - elif mfe_r (recomputed at new spec) >= net_target (at new spec): exit at net_target
      - else: return recomputed pnl_r

    mfe points are invariant to friction; we back them out from the stored mfe_r
    (which was computed at modeled spec via `pnl_points_to_r` — friction in denom only):
        mfe_points = stored_mfe_r × risk_modeled / point_value
    Then re-project into the new spec's R-frame (friction in denom only):
        mfe_r_new = mfe_points × point_value / risk_new
    """
    entry = row["entry_price"]
    stop = row["stop_price"]
    stored_pnl_r = row["pnl_r"]
    stored_mfe_r = row["mfe_r"]
    outcome = row["outcome"]
    ambiguous = bool(row.get("ambiguous_bar", False))

    if pd.isna(stored_pnl_r) or pd.isna(entry) or pd.isna(stop):
        return None

    raw_stop_dist = abs(float(entry) - float(stop))
    if raw_stop_dist <= 0:
        return None

    pv = spec.point_value
    friction_modeled = MGC_MODELED_SPEC.total_friction
    friction_new = spec.total_friction
    risk_modeled = raw_stop_dist * pv + friction_modeled
    risk_new = raw_stop_dist * pv + friction_new

    # Recompute stored pnl_r at new spec (invariant: back out gross points, re-apply)
    pnl_points = (float(stored_pnl_r) * risk_modeled + friction_modeled) / pv
    pnl_r_new = (pnl_points * pv - friction_new) / risk_new

    if ambiguous and outcome == "loss":
        return float(pnl_r_new)

    net_target_new = target_net_r_at(spec, entry, stop, target_r)
    if net_target_new is None:
        return float(pnl_r_new)

    if pd.notna(stored_mfe_r):
        mfe_points = float(stored_mfe_r) * risk_modeled / pv
        mfe_r_new = mfe_points * pv / risk_new
        if mfe_r_new >= net_target_new:
            return float(net_target_new)

    return float(pnl_r_new)


def build_sensitivity() -> dict:
    """Load canonical trade matrix, recompute LR variants across slippage grid."""
    rows = load_rows(end_exclusive=str(HOLDOUT_START.date()))
    trades = build_family_trade_matrix(rows)
    if trades.empty:
        raise SystemExit("Empty trade matrix — canonical upstream failed to produce rows.")
    trades["trading_day"] = pd.to_datetime(trades["trading_day"])
    is_trades = trades[trades["trading_day"] < HOLDOUT_START].copy()

    results: list[dict] = []
    for family_id, variant_col, reported in CELLS:
        cell_rows = is_trades[is_trades["family_id"] == family_id].copy()
        target_r = VARIANT_TARGET_R[variant_col]
        n = len(cell_rows)
        baseline_pipeline = float(cell_rows[variant_col].mean()) if n else float("nan")

        per_slippage: dict[float, dict] = {}
        for slip in SLIPPAGE_GRID_DOLLARS:
            spec = spec_at_slippage(slip)
            recomputed_values = [
                recompute_lower_target_pnl(row, spec, target_r)
                for _, row in cell_rows.iterrows()
            ]
            recomputed = pd.Series(recomputed_values, dtype=float)
            exp_r = float(recomputed.mean())
            wr = float((recomputed > 0).mean())
            avg_win = float(recomputed[recomputed > 0].mean()) if (recomputed > 0).any() else 0.0
            avg_loss = float(recomputed[recomputed <= 0].mean()) if (recomputed <= 0).any() else 0.0
            year_breakdown: dict[int, dict] = {}
            if n:
                years = cell_rows["trading_day"].dt.year
                for y in sorted(years.unique()):
                    mask = (years == y).to_numpy()
                    if int(mask.sum()) >= 10:
                        year_breakdown[int(y)] = {
                            "n": int(mask.sum()),
                            "exp_r": float(recomputed[mask].mean()),
                        }
            per_slippage[float(slip)] = {
                "exp_r": exp_r,
                "win_rate": wr,
                "avg_win_r": avg_win,
                "avg_loss_r": avg_loss,
                "per_year": year_breakdown,
            }

        results.append({
            "cell_slug": f"{family_id}_{variant_col.replace('lower_', 'LR').replace('_pnl_r', '').replace('_', '')}",
            "family_id": family_id,
            "variant_col": variant_col,
            "target_r": target_r,
            "n_is_trades": n,
            "baseline_pipeline": baseline_pipeline,
            "baseline_native_low_r_v1_reported": reported,
            "by_slippage_dollars": per_slippage,
        })
    return {"cells": results}


def evaluate_baseline_cross_check(payload: dict) -> bool:
    """Return True if every cell matches its native_low_r_v1 reported ExpR
    within ±0.002R at slippage=2 (modeled). Per pre-reg §baseline_cross_check."""
    ok = True
    print("\n=== BASELINE CROSS-CHECK (modeled slippage = $2 = 2 ticks) ===")
    for cell in payload["cells"]:
        reported = cell["baseline_native_low_r_v1_reported"]
        pipeline_val = cell["baseline_pipeline"]
        recomputed_2 = cell["by_slippage_dollars"][2.0]["exp_r"]
        diff_pipeline = abs(pipeline_val - reported)
        diff_recomp = abs(recomputed_2 - reported)
        status_pipe = "OK" if diff_pipeline <= 0.002 else "MISMATCH"
        status_recomp = "OK" if diff_recomp <= 0.002 else "MISMATCH"
        print(
            f"  {cell['cell_slug']:<45} "
            f"reported={reported:+.4f} "
            f"pipeline={pipeline_val:+.4f} [{status_pipe}] "
            f"recomp@2={recomputed_2:+.4f} [{status_recomp}]"
        )
        if diff_pipeline > 0.002 or diff_recomp > 0.002:
            ok = False
    return ok


def print_sensitivity_summary(payload: dict) -> None:
    print("\n=== SENSITIVITY CURVE (IS ExpR per cell per slippage-ticks) ===")
    header = f"{'Cell':<45} " + " ".join(f"{s:>8}" for s in SLIPPAGE_GRID_DOLLARS)
    print(header)
    for cell in payload["cells"]:
        vals = [cell["by_slippage_dollars"][s]["exp_r"] for s in SLIPPAGE_GRID_DOLLARS]
        print(f"  {cell['cell_slug']:<43} " + " ".join(f"{v:+.4f}" for v in vals))


def print_kill_criteria(payload: dict) -> None:
    print("\n=== KILL CRITERIA EVALUATION (per pre-reg §kill_criteria) ===")
    for cell in payload["cells"]:
        by_slip = cell["by_slippage_dollars"]
        exp_at_4 = by_slip[4.0]["exp_r"]
        exp_at_675 = by_slip[6.75]["exp_r"]
        exp_at_10 = by_slip[10.0]["exp_r"]
        if exp_at_10 > 0.05:
            verdict = "A_pilot_not_binding: survives even at 10-tick — mechanism question, opens H3"
        elif exp_at_675 > 0.05:
            verdict = "A_closure_soft_at_pilot_mean: shadow-track candidate"
        elif exp_at_4 < 0:
            verdict = "A_closure_confirmed_strong: below zero by 4 ticks"
        else:
            verdict = "SOFT decay: positive at modest friction, below +0.05R by pilot mean"
        print(f"  {cell['cell_slug']:<45} {verdict}")


def main() -> None:
    payload = build_sensitivity()
    cross_check_ok = evaluate_baseline_cross_check(payload)
    out_path = OUTPUT_DIR / "mgc_real_slippage_sensitivity_v1.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")

    if not cross_check_ok:
        print("\nHALT — baseline cross-check failed. Do not draw sensitivity conclusions.")
        raise SystemExit(1)

    print_sensitivity_summary(payload)
    print_kill_criteria(payload)


if __name__ == "__main__":
    main()
