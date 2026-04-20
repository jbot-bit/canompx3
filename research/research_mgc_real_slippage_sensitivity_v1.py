"""H0 — MGC Real-Slippage Sensitivity on Native Low-R v1 Killed Cells.

Pre-reg: docs/audit/hypotheses/2026-04-20-mgc-real-slippage-sensitivity.yaml

Question: do any of the 5 cells that survived native_low_r_v1 (at K=16 BH)
and were subsequently killed in path_accurate_subr_v1 STILL show
ExpR > +0.05R under a plausible range of slippage assumptions?

Not discovery — confirmatory audit. Trial cost from MinBTL budget: 0.

Slippage grid (ticks round-trip): [2 (modeled), 4, 6.75 (pilot mean), 10].

For each trade in each cell, recompute pnl_r with adjusted friction:

    new_pnl_r = (pnl_points * point_value - friction_adjusted) / risk_adjusted
    where friction_adjusted = commission + spread_doubled + slippage_adjusted
          risk_adjusted     = |entry - stop| * point_value + friction_adjusted

Slippage_adjusted in DOLLARS = ticks_rt * tick_value = ticks_rt * $1 for MGC.

For LR05/LR075 targets (low-R exits at 0.5R or 0.75R), if mfe reached
target before stop, exit at target; else use stored full-RR outcome.
Because mfe_r is friction-inclusive at the ORIGINAL modeled slippage,
we adjust the target-hit R-value by the slippage delta too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IS_START = "2022-06-13"
IS_END = "2026-01-01"

# MGC cost spec — canonical from pipeline.cost_model
MGC_COMMISSION_RT = 1.74
MGC_SPREAD_DOUBLED = 2.00
MGC_MODELED_SLIPPAGE = 2.00  # = 2 ticks * $1
MGC_POINT_VALUE = 10.0
MGC_TICK_SIZE = 0.10
MGC_MIN_TICKS_FLOOR = 10
MGC_MIN_RISK_FLOOR_POINTS = MGC_MIN_TICKS_FLOOR * MGC_TICK_SIZE

# Slippage grid in TICKS ROUND-TRIP (MGC has $1/tick so $ = ticks)
SLIPPAGE_GRID_TICKS = [2, 4, 6.75, 10]

# The 5 native_low_r_v1 BH survivors. Columns: (slug, session, filter_type,
# orb_minutes, entry_model, confirm_bars, rr_source, direction, lr_target)
# rr_source is the orb_outcomes RR the audit used (always 1.0 for low-R rewrites)
# direction is inferred from native_low_r_v1 — all long
CELLS = [
    dict(slug="NYSE_OPEN_OVNRNG_50_RR1_LR075", session="NYSE_OPEN",
         filter_type="OVNRNG_50", orb_minutes=5, entry_model="E2",
         confirm_bars=1, rr_source=1.0, direction="long", lr_target=0.75),
    dict(slug="US_DATA_1000_ATR_P70_RR1_LR05", session="US_DATA_1000",
         filter_type="ATR_P70", orb_minutes=5, entry_model="E2",
         confirm_bars=1, rr_source=1.0, direction="long", lr_target=0.5),
    dict(slug="US_DATA_1000_OVNRNG_10_RR1_LR05", session="US_DATA_1000",
         filter_type="OVNRNG_10", orb_minutes=5, entry_model="E2",
         confirm_bars=1, rr_source=1.0, direction="long", lr_target=0.5),
    dict(slug="US_DATA_1000_BROAD_RR1_LR05", session="US_DATA_1000",
         filter_type=None, orb_minutes=5, entry_model="E2",
         confirm_bars=1, rr_source=1.0, direction="long", lr_target=0.5),
    dict(slug="NYSE_OPEN_BROAD_RR1_LR05", session="NYSE_OPEN",
         filter_type=None, orb_minutes=5, entry_model="E2",
         confirm_bars=1, rr_source=1.0, direction="long", lr_target=0.5),
]

# Reported native_low_r_v1 IS ExpR (for baseline cross-check reproduction)
NATIVE_LOW_R_V1_BASELINE = {
    "NYSE_OPEN_OVNRNG_50_RR1_LR075": 0.2226,
    "US_DATA_1000_ATR_P70_RR1_LR05": 0.0710,
    "US_DATA_1000_OVNRNG_10_RR1_LR05": 0.0685,
    "US_DATA_1000_BROAD_RR1_LR05": 0.0488,
    "NYSE_OPEN_BROAD_RR1_LR05": 0.0380,
}


def connect():
    return duckdb.connect(str(GOLD_DB_PATH), read_only=True)


def load_cell_trades(con, cell: dict) -> pd.DataFrame:
    """Load orb_outcomes + daily_features for one cell, long direction, IS window.

    Returns per-trade records with entry_price, stop_price, pnl_r (stored),
    mfe_r (stored), outcome, trading_day.
    """
    # Direction is inferred: long = entry_price > stop_price.
    dir_sql = "(o.entry_price > o.stop_price)" if cell["direction"] == "long" else "(o.entry_price < o.stop_price)"
    filter_sql = ""
    params = [
        cell["session"],
        cell["orb_minutes"],
        cell["entry_model"],
        cell["confirm_bars"],
        cell["rr_source"],
        IS_START,
        IS_END,
    ]

    df_sql = f"""
    WITH ox AS (
      SELECT
        o.trading_day,
        o.entry_price,
        o.stop_price,
        o.pnl_r,
        o.mfe_r,
        o.outcome
      FROM orb_outcomes o
      {"JOIN daily_features d ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes" if cell["filter_type"] else ""}
      WHERE o.symbol='MGC'
        AND o.orb_label = ?
        AND o.orb_minutes = ?
        AND o.entry_model = ?
        AND o.confirm_bars = ?
        AND o.rr_target = ?
        AND {dir_sql}
        AND o.trading_day >= ?::DATE
        AND o.trading_day <  ?::DATE
        AND o.pnl_r IS NOT NULL
        {filter_sql}
    )
    SELECT * FROM ox ORDER BY trading_day
    """
    # Apply filter_type if present. Canonical filter IDs:
    #  OVNRNG_50:   daily_features.overnight_range >= 50.0
    #  OVNRNG_10:   daily_features.overnight_range >= 10.0
    #  ATR_P70:     daily_features.atr_20_pct >= 0.70
    #  (None = broad, no filter)
    if cell["filter_type"] == "OVNRNG_50":
        df_sql = df_sql.replace("{filter_sql}", "AND d.overnight_range >= 50.0")
    elif cell["filter_type"] == "OVNRNG_10":
        df_sql = df_sql.replace("{filter_sql}", "AND d.overnight_range >= 10.0")
    elif cell["filter_type"] == "ATR_P70":
        df_sql = df_sql.replace("{filter_sql}", "AND d.atr_20_pct >= 0.70")
    else:
        df_sql = df_sql.replace("{filter_sql}", "")

    return con.execute(df_sql, params).df()


def compute_friction_dollars(slippage_ticks_rt: float) -> float:
    """Total MGC friction in dollars at given slippage_ticks_rt."""
    slippage_dollars = slippage_ticks_rt * MGC_TICK_SIZE * MGC_POINT_VALUE  # ticks * $/tick
    return MGC_COMMISSION_RT + MGC_SPREAD_DOUBLED + slippage_dollars


def recompute_pnl_r(row, slippage_ticks_rt: float, lr_target: float) -> float:
    """Recompute a single trade's R at adjusted slippage, applying LR target rewrite.

    Low-R target rewrite rule (from native_low_r_v1):
      if mfe_r >= lr_target → exit at lr_target R (winner at lr_target)
      else → use stored pnl_r behavior (loser at -1R, or small winner below target)

    But both mfe_r and stored pnl_r were computed at modeled slippage. We adjust
    both via the friction delta.
    """
    entry = float(row["entry_price"])
    stop = float(row["stop_price"])
    raw_stop_dist_points = abs(entry - stop)
    # Enforce min-risk-floor per cost_model.risk_in_dollars semantics
    stop_dist_points = max(raw_stop_dist_points, MGC_MIN_RISK_FLOOR_POINTS)
    raw_risk_dollars = stop_dist_points * MGC_POINT_VALUE

    friction_new = compute_friction_dollars(slippage_ticks_rt)
    risk_new = raw_risk_dollars + friction_new

    # Convert stored pnl_r (at modeled friction) back to pnl_points, then reapply
    # at new friction.
    friction_modeled = compute_friction_dollars(MGC_MODELED_SLIPPAGE / (MGC_TICK_SIZE * MGC_POINT_VALUE))
    # Actually MGC_MODELED_SLIPPAGE=2.00 dollars; ticks_rt = 2.00 / $1 per tick = 2 ticks
    # So: friction_modeled = 1.74 + 2.00 + 2.00 = 5.74
    risk_modeled = raw_risk_dollars + friction_modeled

    stored_pnl_r = float(row["pnl_r"])
    stored_mfe_r = float(row["mfe_r"])
    outcome = row["outcome"]

    # Reconstruct pnl_points from stored pnl_r at modeled friction:
    # pnl_r = (pnl_points * point_value - friction_modeled) / risk_modeled  [if winner at target]
    # For losers (pnl_r = -1.0 by convention), pnl_points = -stop_dist_points
    if outcome == "loss" or stored_pnl_r <= -0.9:
        # Loss: exits at stop, pnl_points = -stop_dist_points
        pnl_points = -stop_dist_points
    else:
        # Winner: back out pnl_points
        pnl_points = (stored_pnl_r * risk_modeled + friction_modeled) / MGC_POINT_VALUE

    # mfe_r is computed without friction deducted from numerator (pnl_points_to_r):
    # mfe_r = (mfe_points * point_value) / risk_modeled  (no friction subtraction)
    # Back out mfe_points:
    mfe_points = stored_mfe_r * risk_modeled / MGC_POINT_VALUE

    # Now apply LR target rewrite: did MFE reach the lr_target in R-terms
    # under the NEW friction model?
    # Target in points such that to_r_multiple(target_points) = lr_target:
    #   lr_target = (target_points * point_value - friction_new) / risk_new
    #   target_points = (lr_target * risk_new + friction_new) / point_value
    target_points = (lr_target * risk_new + friction_new) / MGC_POINT_VALUE

    if mfe_points >= target_points:
        # Winner at LR target under new friction
        new_pnl_r = (target_points * MGC_POINT_VALUE - friction_new) / risk_new
    else:
        # Did NOT reach target; apply new friction to original outcome
        if outcome == "loss":
            # Loser at full stop
            new_pnl_r = (-stop_dist_points * MGC_POINT_VALUE - friction_new) / risk_new
        else:
            # Winner below target (partial win at stored pnl_points)
            new_pnl_r = (pnl_points * MGC_POINT_VALUE - friction_new) / risk_new

    return new_pnl_r


def analyze_cell(con, cell: dict) -> dict:
    df = load_cell_trades(con, cell)
    result = {
        "slug": cell["slug"],
        "n_is_trades": len(df),
        "baseline_reported": NATIVE_LOW_R_V1_BASELINE.get(cell["slug"]),
        "by_slippage": {},
    }

    if len(df) == 0:
        result["error"] = "NO TRADES LOADED — check filter logic"
        return result

    for sl_ticks in SLIPPAGE_GRID_TICKS:
        pnl_r_new = df.apply(
            lambda row: recompute_pnl_r(row, sl_ticks, cell["lr_target"]),
            axis=1,
        )
        exp_r = float(pnl_r_new.mean())
        win_rate = float((pnl_r_new > 0).mean())
        avg_win_r = float(pnl_r_new[pnl_r_new > 0].mean()) if (pnl_r_new > 0).any() else 0.0
        avg_loss_r = float(pnl_r_new[pnl_r_new <= 0].mean()) if (pnl_r_new <= 0).any() else 0.0

        # Per-year breakdown
        per_year = {}
        if "trading_day" in df.columns:
            years = pd.to_datetime(df["trading_day"]).dt.year
            for y in sorted(years.unique()):
                mask = (years == y).to_numpy()
                if mask.sum() >= 10:
                    per_year[int(y)] = {
                        "n": int(mask.sum()),
                        "exp_r": float(pnl_r_new[mask].mean()),
                    }

        result["by_slippage"][sl_ticks] = {
            "exp_r": exp_r,
            "win_rate": win_rate,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "per_year": per_year,
        }

    return result


def main() -> None:
    con = connect()
    try:
        all_results = []
        for cell in CELLS:
            print(f"\n--- {cell['slug']} ---")
            r = analyze_cell(con, cell)
            print(f"  N IS trades: {r['n_is_trades']}")
            print(f"  Baseline reported (native_low_r_v1): {r.get('baseline_reported')}")
            for sl, s in r.get("by_slippage", {}).items():
                marker = "  <---" if s["exp_r"] < 0.05 else "  PASS"
                print(f"    slippage={sl} ticks: ExpR={s['exp_r']:+.4f} WR={s['win_rate']:.3f}{marker}")
            all_results.append(r)

        # Cross-check at slippage=2 (modeled) vs baseline reported
        print("\n=== BASELINE CROSS-CHECK (slippage=2 ticks vs native_low_r_v1 reported) ===")
        halt = False
        for r in all_results:
            rep = r.get("baseline_reported")
            if rep is None:
                continue
            recomp = r["by_slippage"].get(2, {}).get("exp_r")
            if recomp is None:
                continue
            diff = abs(recomp - rep)
            status = "OK" if diff <= 0.005 else "MISMATCH"
            print(f"  {r['slug']}: reported={rep:+.4f}, recomputed={recomp:+.4f}, |diff|={diff:.4f} [{status}]")
            if diff > 0.010:
                halt = True
        if halt:
            print("  HALT — recomputation path disagrees with native_low_r_v1 by >0.01R.")
            print("  Investigate before drawing audit conclusions.")

        out = OUTPUT_DIR / "mgc_real_slippage_sensitivity_v1.json"
        out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {out}")

        print("\n=== SENSITIVITY SUMMARY ===")
        print(f"{'Cell':<45} {'2':>8} {'4':>8} {'6.75':>8} {'10':>8}")
        for r in all_results:
            if "by_slippage" not in r:
                continue
            vals = [r["by_slippage"].get(s, {}).get("exp_r", float("nan")) for s in SLIPPAGE_GRID_TICKS]
            print(f"  {r['slug']:<45} " + " ".join(f"{v:+.4f}" for v in vals))

        # Kill criteria evaluation
        print("\n=== KILL CRITERIA EVALUATION (pre-reg) ===")
        for r in all_results:
            if "by_slippage" not in r:
                continue
            exp_at_675 = r["by_slippage"].get(6.75, {}).get("exp_r", float("nan"))
            exp_at_10 = r["by_slippage"].get(10, {}).get("exp_r", float("nan"))
            if exp_at_10 > 0.05:
                verdict = "SURVIVES at all slippage — mechanism-driven, opens H3 for investigation"
            elif exp_at_675 > 0.05:
                verdict = "SURVIVES at pilot mean — add to shadow-track registry"
            elif exp_at_675 > 0 and exp_at_10 < 0:
                verdict = "SOFT closure — declines monotonically with slippage but stays above zero at pilot"
            else:
                verdict = "ROBUST closure — already below +0.05R at modest friction increase"
            print(f"  {r['slug']}: {verdict}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
