"""research/research_partial_profit.py

Phase 3 Research: Partial Profit Taking Simulation

Hypothesis: Scaling out 50% of position at an intermediate R-level and
trailing the stop on the remainder improves net R across all outcomes.

Methodology:
- Bar-by-bar replay of ALL E1+E2 outcomes (CB1, O5) — NOT filtered to validated
- Parameter grid: partial_exit_r × trail_stop_r (sensitivity analysis)
- Conservative ambiguous-bar rule (stop before target, matching outcome_builder)
- Cost model: 1.5× friction when partial fires (1 entry + 2 exits)
- Paired t-test on daily P&L aggregates (baseline vs partial)
- BH FDR across all (instrument × session × rr_target × partial × trail) combos
- Year-by-year stability (must be positive in 4+ years)
- Geometric growth comparison (Carver's concern)

Lookahead guards:
- NO use of stored mfe_r as a FILTER (only as computation skip — mathematical certainty)
- NO pre-filtering to validated strategies (runs on ALL break-day outcomes)
- Parameter grid with BH FDR (not hand-picked single config)
- Results labeled "promising hypothesis" until walk-forward holdout

@research-source: docs/plans/2026-03-04-m25-audit-improvements-plan.md (Phase 3)
@academic-grounding: Carver (geometric growth warning), Chan (MAE/MFE optimal levels),
    Murray (partial profits reduce variance more than expected return)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.cost_model import COST_SPECS, CostSpec, risk_in_dollars
from research.lib.stats import bh_fdr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIVE_INSTRUMENTS = ["MGC", "MNQ", "MES", "M2K"]
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]  # Skip 1.0 (0.1% rescuable), skip 4.0 (sparse)
PARTIAL_LEVELS = [0.5, 1.0, 1.5]     # Grid for sensitivity analysis
TRAIL_LEVELS = [0.0, 0.25, 0.5]      # 0=breakeven, 0.5=plan value
DB_PATH = Path(__file__).resolve().parents[1] / "gold.db"

# Minimum trades per cell for statistical testing
MIN_TRADES = 30


# ---------------------------------------------------------------------------
# Bar-by-bar partial profit simulation
# ---------------------------------------------------------------------------
def simulate_partial_trade(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    direction: int,        # 1=long, -1=short
    entry_price: float,
    stop_price: float,
    target_price: float,
    risk_points: float,
    partial_r: float,
    trail_r: float,
) -> dict:
    """Simulate a single trade with partial profit taking.

    Returns dict with:
        partial_fired: bool — did the partial exit trigger?
        leg1_pnl_points: float — P&L in points for the scaled-out half
        leg2_pnl_points: float — P&L in points for the remaining half
        composite_pnl_points: float — 0.5 * leg1 + 0.5 * leg2
        exit_type: str — what ended the trade
    """
    n_bars = len(highs)
    if n_bars == 0:
        return {
            "partial_fired": False,
            "leg1_pnl_points": 0.0,
            "leg2_pnl_points": 0.0,
            "composite_pnl_points": 0.0,
            "exit_type": "no_bars",
        }

    # Compute partial and trail levels
    if direction == 1:  # long
        partial_price = entry_price + partial_r * risk_points
        trail_stop = entry_price + trail_r * risk_points
    else:  # short
        partial_price = entry_price - partial_r * risk_points
        trail_stop = entry_price - trail_r * risk_points

    # ── Phase 1: find first bar where partial or stop is hit ──
    if direction == 1:
        hit_partial = highs >= partial_price
        hit_stop = lows <= stop_price
    else:
        hit_partial = lows <= partial_price
        hit_stop = highs >= stop_price

    any_phase1 = hit_partial | hit_stop
    if not any_phase1.any():
        # Neither hit — EOD exit, no partial
        eod_pnl = (closes[-1] - entry_price) * direction
        return {
            "partial_fired": False,
            "leg1_pnl_points": 0.0,
            "leg2_pnl_points": eod_pnl,
            "composite_pnl_points": eod_pnl,
            "exit_type": "eod_no_partial",
        }

    idx1 = int(np.argmax(any_phase1))

    # Ambiguous: both stop and partial in same bar → stop wins (conservative)
    if hit_stop[idx1] and hit_partial[idx1]:
        stop_pnl = (stop_price - entry_price) * direction
        return {
            "partial_fired": False,
            "leg1_pnl_points": 0.0,
            "leg2_pnl_points": stop_pnl,
            "composite_pnl_points": stop_pnl,
            "exit_type": "stop_ambiguous_with_partial",
        }

    if hit_stop[idx1]:
        # Stop hit before partial — normal loss, no partial
        stop_pnl = (stop_price - entry_price) * direction
        return {
            "partial_fired": False,
            "leg1_pnl_points": 0.0,
            "leg2_pnl_points": stop_pnl,
            "composite_pnl_points": stop_pnl,
            "exit_type": "stop_before_partial",
        }

    # ── Partial fires at idx1 ──
    leg1_pnl = partial_r * risk_points  # direction-normalized (always positive)

    # Check if target is ALSO hit in the same bar as partial
    if direction == 1:
        target_also = highs[idx1] >= target_price
    else:
        target_also = lows[idx1] <= target_price

    if target_also:
        # Both partial and target in same bar. For long: partial_price < target_price,
        # so partial fires first (lower level reached first). Remaining half hits target.
        rr = (target_price - entry_price) * direction / risk_points
        leg2_pnl = rr * risk_points
        return {
            "partial_fired": True,
            "leg1_pnl_points": leg1_pnl,
            "leg2_pnl_points": leg2_pnl,
            "composite_pnl_points": 0.5 * leg1_pnl + 0.5 * leg2_pnl,
            "exit_type": "partial_and_target_same_bar",
        }

    # ── Phase 2: remaining half with new trail stop ──
    remaining_highs = highs[idx1 + 1:]
    remaining_lows = lows[idx1 + 1:]
    remaining_closes = closes[idx1 + 1:]

    if len(remaining_highs) == 0:
        # Partial fired on last bar — use that bar's close for remaining half
        leg2_pnl = (closes[idx1] - entry_price) * direction
        return {
            "partial_fired": True,
            "leg1_pnl_points": leg1_pnl,
            "leg2_pnl_points": leg2_pnl,
            "composite_pnl_points": 0.5 * leg1_pnl + 0.5 * leg2_pnl,
            "exit_type": "partial_eod_same_bar",
        }

    if direction == 1:
        hit_target2 = remaining_highs >= target_price
        hit_trail2 = remaining_lows <= trail_stop
    else:
        hit_target2 = remaining_lows <= target_price
        hit_trail2 = remaining_highs >= trail_stop

    any_phase2 = hit_target2 | hit_trail2
    if not any_phase2.any():
        # EOD exit for remaining half
        leg2_pnl = (remaining_closes[-1] - entry_price) * direction
        return {
            "partial_fired": True,
            "leg1_pnl_points": leg1_pnl,
            "leg2_pnl_points": leg2_pnl,
            "composite_pnl_points": 0.5 * leg1_pnl + 0.5 * leg2_pnl,
            "exit_type": "partial_then_eod",
        }

    idx2 = int(np.argmax(any_phase2))

    # Ambiguous: trail and target in same bar → trail wins (conservative)
    if hit_trail2[idx2] and hit_target2[idx2]:
        leg2_pnl = (trail_stop - entry_price) * direction
        exit_type = "partial_then_trail_ambiguous"
    elif hit_trail2[idx2]:
        leg2_pnl = (trail_stop - entry_price) * direction
        exit_type = "partial_then_trail"
    else:
        leg2_pnl = (target_price - entry_price) * direction
        exit_type = "partial_then_target"

    return {
        "partial_fired": True,
        "leg1_pnl_points": leg1_pnl,
        "leg2_pnl_points": leg2_pnl,
        "composite_pnl_points": 0.5 * leg1_pnl + 0.5 * leg2_pnl,
        "exit_type": exit_type,
    }


def compute_partial_r(
    result: dict,
    risk_points: float,
    cost_spec: CostSpec,
    entry_price: float,
    stop_price: float,
) -> float:
    """Convert simulation result to R-multiple with correct friction.

    Friction = 1.5× total_friction when partial fires (1 entry + 2 exits),
    1.0× when it doesn't (normal trade).
    """
    risk_dollars = risk_in_dollars(cost_spec, entry_price, stop_price)
    if risk_dollars <= 0:
        return 0.0

    friction_multiplier = 1.5 if result["partial_fired"] else 1.0
    total_friction = cost_spec.total_friction * friction_multiplier

    pnl_dollars = result["composite_pnl_points"] * cost_spec.point_value - total_friction

    return pnl_dollars / risk_dollars


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_bars(con: duckdb.DuckDBPyConnection, instrument: str) -> pd.DataFrame:
    """Load all 1m bars for instrument, sorted by timestamp."""
    print(f"  Loading bars_1m for {instrument}...")
    df = con.execute("""
        SELECT ts_utc, high, low, close
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """, [instrument]).fetchdf()
    # Ensure ts_utc is timezone-naive UTC for fast comparison
    df["ts_utc"] = pd.to_datetime(df["ts_utc"]).dt.tz_localize(None)
    print(f"  Loaded {len(df):,} bars")
    return df


def load_outcomes(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    rr_targets: list[float],
) -> pd.DataFrame:
    """Load ALL E1+E2 outcomes (CB1, O5) for bar-by-bar simulation."""
    placeholders = ",".join(["?"] * len(rr_targets))
    df = con.execute(f"""
        SELECT
            trading_day, symbol, orb_label, rr_target,
            entry_ts, entry_price, stop_price, target_price,
            outcome, pnl_r, mfe_r,
            -- Infer direction from target vs entry
            CASE WHEN target_price > entry_price THEN 1 ELSE -1 END AS direction
        FROM orb_outcomes
        WHERE symbol = ?
          AND entry_model IN ('E1', 'E2')
          AND confirm_bars = 1
          AND orb_minutes = 5
          AND rr_target IN ({placeholders})
          AND entry_ts IS NOT NULL
          AND entry_price IS NOT NULL
          AND stop_price IS NOT NULL
          AND target_price IS NOT NULL
    """, [instrument] + rr_targets).fetchdf()

    # Timezone-naive UTC for fast comparison
    df["entry_ts"] = pd.to_datetime(df["entry_ts"]).dt.tz_localize(None)
    df["risk_points"] = (df["entry_price"] - df["stop_price"]).abs()

    # Compute trading_day_end (23:00 UTC = 09:00 next Brisbane day)
    df["td_end"] = pd.to_datetime(df["trading_day"]) + pd.Timedelta(hours=23)

    print(f"  Loaded {len(df):,} outcomes ({df['outcome'].value_counts().to_dict()})")
    return df


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_simulation(
    instrument: str,
    con: duckdb.DuckDBPyConnection,
    partial_levels: list[float],
    trail_levels: list[float],
    rr_targets: list[float],
) -> pd.DataFrame:
    """Run partial profit simulation for one instrument across all grid combos."""

    bars_df = load_bars(con, instrument)
    outcomes_df = load_outcomes(con, instrument, rr_targets)

    if outcomes_df.empty:
        return pd.DataFrame()

    cost_spec = COST_SPECS[instrument]

    # Pre-convert bars to numpy for fast indexing
    bars_ts = bars_df["ts_utc"].values.astype("datetime64[us]")
    bars_high = bars_df["high"].values
    bars_low = bars_df["low"].values
    bars_close = bars_df["close"].values

    results = []
    n_outcomes = len(outcomes_df)
    t0 = time.time()

    for i, row in enumerate(outcomes_df.itertuples(index=False)):
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_outcomes - i - 1) / rate
            print(f"  {i+1:,}/{n_outcomes:,} outcomes "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        entry_ts = np.datetime64(row.entry_ts, "us")
        td_end = np.datetime64(row.td_end, "us")

        # Slice bars from entry to trading_day_end
        start_idx = int(np.searchsorted(bars_ts, entry_ts))
        end_idx = int(np.searchsorted(bars_ts, td_end))

        if start_idx >= end_idx:
            continue

        h = bars_high[start_idx:end_idx]
        l = bars_low[start_idx:end_idx]
        c = bars_close[start_idx:end_idx]

        direction = int(row.direction)
        entry_price = float(row.entry_price)
        stop_price = float(row.stop_price)
        target_price = float(row.target_price)
        risk_points = float(row.risk_points)
        baseline_r = float(row.pnl_r) if row.pnl_r is not None else 0.0
        # None MFE = unknown → must simulate (don't skip)
        mfe_r = float(row.mfe_r) if row.mfe_r is not None else float("inf")

        if risk_points <= 0:
            continue

        for partial_r in partial_levels:
            # Skip RR targets where partial >= RR (meaningless)
            if partial_r >= row.rr_target:
                continue

            for trail_r in trail_levels:
                # Skip if trail > partial (trail can't be above the partial exit level)
                if trail_r > partial_r:
                    continue

                # Optimization: if MFE < partial_r, partial can NEVER fire.
                # This is a mathematical certainty, not a lookahead filter.
                # The trade's best favorable excursion never reached partial_r.
                if mfe_r < partial_r and baseline_r < partial_r:
                    partial_pnl_r = baseline_r  # Identical to baseline
                else:
                    sim = simulate_partial_trade(
                        h, l, c, direction,
                        entry_price, stop_price, target_price, risk_points,
                        partial_r, trail_r,
                    )
                    partial_pnl_r = compute_partial_r(
                        sim, risk_points, cost_spec, entry_price, stop_price,
                    )

                results.append({
                    "trading_day": row.trading_day,
                    "instrument": instrument,
                    "session": row.orb_label,
                    "rr_target": row.rr_target,
                    "partial_r": partial_r,
                    "trail_r": trail_r,
                    "baseline_r": baseline_r,
                    "partial_pnl_r": partial_pnl_r,
                    "delta_r": partial_pnl_r - baseline_r,
                })

    elapsed = time.time() - t0
    print(f"  Completed {instrument}: {len(results):,} simulation results in {elapsed:.0f}s")
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------
def analyze_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results, run paired t-tests, BH FDR, year-by-year stability."""

    if df.empty:
        return pd.DataFrame()

    # Group by (instrument, session, rr_target, partial_r, trail_r)
    group_cols = ["instrument", "session", "rr_target", "partial_r", "trail_r"]

    rows = []
    for key, grp in df.groupby(group_cols):
        instrument, session, rr_target, partial_r, trail_r = key
        n_trades = len(grp)

        if n_trades < MIN_TRADES:
            continue

        # ── Aggregate daily P&L for paired test ──
        daily = grp.groupby("trading_day").agg(
            baseline_daily=("baseline_r", "sum"),
            partial_daily=("partial_pnl_r", "sum"),
        ).reset_index()
        daily["diff"] = daily["partial_daily"] - daily["baseline_daily"]

        n_days = len(daily)
        if n_days < 10:
            continue

        # Paired t-test (one-tailed: is partial BETTER?)
        diff_arr = daily["diff"].values
        non_zero = diff_arr[diff_arr != 0]
        if len(non_zero) < 10:
            continue

        t_stat, p_two = stats.ttest_1samp(non_zero, 0.0)
        p_val = float(p_two / 2) if t_stat > 0 else 1.0 - float(p_two / 2)

        # Totals
        total_baseline = grp["baseline_r"].sum()
        total_partial = grp["partial_pnl_r"].sum()
        total_delta = total_partial - total_baseline
        per_trade_delta = total_delta / n_trades

        # ── Year-by-year stability ──
        grp_copy = grp.copy()
        grp_copy["year"] = pd.to_datetime(grp_copy["trading_day"]).dt.year
        yearly = grp_copy.groupby("year")["delta_r"].sum()
        n_years = len(yearly)
        years_positive = int((yearly > 0).sum())

        # ── Geometric growth comparison ──
        # Use f=0.02 (2% risk per trade) as reference fraction
        f = 0.02
        baseline_geo = float(np.sum(np.log1p(f * grp["baseline_r"].values)))
        partial_geo = float(np.sum(np.log1p(
            np.clip(f * grp["partial_pnl_r"].values, -0.99, None)
        )))
        geo_delta_pct = ((np.exp(partial_geo) / np.exp(baseline_geo)) - 1) * 100 \
            if baseline_geo != 0 else 0.0

        rows.append({
            "instrument": instrument,
            "session": session,
            "rr_target": rr_target,
            "partial_r": partial_r,
            "trail_r": trail_r,
            "n_trades": n_trades,
            "n_days": n_days,
            "total_baseline_r": round(total_baseline, 2),
            "total_partial_r": round(total_partial, 2),
            "total_delta_r": round(total_delta, 2),
            "per_trade_delta_r": round(per_trade_delta, 4),
            "t_stat": round(t_stat, 3),
            "p_val": round(p_val, 6),
            "n_years": n_years,
            "years_positive": years_positive,
            "geo_delta_pct": round(geo_delta_pct, 2),
        })

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    # ── BH FDR correction across ALL tested combinations ──
    p_values = stats_df["p_val"].values
    n_tests = len(p_values)
    rejected = bh_fdr(p_values, q=0.05)
    stats_df["bh_rejected"] = [i in rejected for i in range(n_tests)]
    stats_df["p_bh"] = stats_df["p_val"] * n_tests / (
        stats_df.index + 1
    )  # approximate adjusted p

    # Proper BH adjusted p-values
    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(n_tests)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = p_values[idx] * n_tests / (rank + 1)
    # Ensure monotonicity
    for rank in range(n_tests - 2, -1, -1):
        idx = sorted_indices[rank]
        idx_next = sorted_indices[rank + 1]
        adjusted[idx] = min(adjusted[idx], adjusted[idx_next])
    adjusted = np.clip(adjusted, 0, 1)
    stats_df["p_bh"] = np.round(adjusted, 6)

    return stats_df


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------
def apply_decision_gate(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Apply GO/NO-GO criteria per the plan.

    GO criteria (ALL must hold):
    1. BH FDR rejected (p_bh < 0.05)
    2. Improvement in 4+ of 5 years tested (years_positive >= 0.8 * n_years)
    3. Geometric growth impact < 10% reduction (geo_delta_pct > -10)
    4. Net R improvement (total_delta_r > 0)
    """
    if stats_df.empty:
        return stats_df

    stats_df["gate_fdr"] = stats_df["bh_rejected"]
    stats_df["gate_years"] = stats_df["years_positive"] >= (0.8 * stats_df["n_years"])
    stats_df["gate_geo"] = stats_df["geo_delta_pct"] > -10.0
    stats_df["gate_net_r"] = stats_df["total_delta_r"] > 0

    stats_df["GO"] = (
        stats_df["gate_fdr"]
        & stats_df["gate_years"]
        & stats_df["gate_geo"]
        & stats_df["gate_net_r"]
    )

    return stats_df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_report(stats_df: pd.DataFrame) -> None:
    """Print structured research output per RESEARCH_RULES.md format."""

    print("\n" + "=" * 80)
    print("PARTIAL PROFIT TAKING RESEARCH — RESULTS")
    print("=" * 80)

    n_tests = len(stats_df)
    n_go = stats_df["GO"].sum() if "GO" in stats_df.columns else 0
    n_fdr = stats_df["bh_rejected"].sum() if "bh_rejected" in stats_df.columns else 0

    print(f"\nTotal combinations tested: {n_tests}")
    print(f"BH FDR survivors (q=0.05): {n_fdr}")
    print(f"Full GO (all 4 gates): {n_go}")

    if n_go > 0:
        go_df = stats_df[stats_df["GO"]].sort_values("total_delta_r", ascending=False)
        print(f"\n{'─' * 80}")
        print("SURVIVED SCRUTINY:")
        print(f"{'─' * 80}")
        for _, r in go_df.iterrows():
            print(f"  {r['instrument']} {r['session']} RR{r['rr_target']} | "
                  f"partial@{r['partial_r']}R trail@{r['trail_r']}R | "
                  f"N={r['n_trades']:,} | delta={r['total_delta_r']:+,.1f}R "
                  f"({r['per_trade_delta_r']:+.4f}R/trade) | "
                  f"p_bh={r['p_bh']:.4f} | years+={r['years_positive']}/{r['n_years']} | "
                  f"geo={r['geo_delta_pct']:+.1f}%")

    # Show top FDR survivors that failed other gates
    fdr_only = stats_df[stats_df["bh_rejected"] & ~stats_df.get("GO", False)]
    if len(fdr_only) > 0:
        print(f"\n{'─' * 80}")
        print("FDR SURVIVORS that FAILED other gates:")
        print(f"{'─' * 80}")
        for _, r in fdr_only.head(20).iterrows():
            gates = []
            if not r.get("gate_years", True):
                gates.append(f"years={r['years_positive']}/{r['n_years']}")
            if not r.get("gate_geo", True):
                gates.append(f"geo={r['geo_delta_pct']:+.1f}%")
            if not r.get("gate_net_r", True):
                gates.append(f"netR={r['total_delta_r']:+.1f}")
            print(f"  {r['instrument']} {r['session']} RR{r['rr_target']} | "
                  f"partial@{r['partial_r']}R trail@{r['trail_r']}R | "
                  f"FAILED: {', '.join(gates)}")

    # Summary by parameter combo
    print(f"\n{'─' * 80}")
    print("SUMMARY BY PARAMETER COMBO (all instruments aggregated):")
    print(f"{'─' * 80}")
    for pr in sorted(stats_df["partial_r"].unique()):
        for tr in sorted(stats_df["trail_r"].unique()):
            sub = stats_df[(stats_df["partial_r"] == pr) & (stats_df["trail_r"] == tr)]
            if sub.empty:
                continue
            total_delta = sub["total_delta_r"].sum()
            n_go_sub = sub["GO"].sum() if "GO" in sub.columns else 0
            n_fdr_sub = sub["bh_rejected"].sum()
            print(f"  partial={pr}R trail={tr}R: "
                  f"delta={total_delta:+,.0f}R | "
                  f"FDR={n_fdr_sub}/{len(sub)} | GO={n_go_sub}/{len(sub)}")

    print(f"\n{'─' * 80}")
    print("CAVEATS:")
    print(f"{'─' * 80}")
    print("  - In-sample analysis only. Results are 'promising hypothesis' until WF holdout.")
    print("  - Bar resolution is 1-minute. Intra-bar order unknown for ambiguous bars.")
    print("  - Cost model uses 1.5× friction (conservative). Real impact depends on ORB size.")
    print("  - Partial exit requires contracts >= 2. Vol-sizing may produce contracts=1.")
    print(f"  - {n_tests} total comparisons tested. BH FDR at q=0.05 applied.")

    print(f"\n{'─' * 80}")
    print("NEXT STEPS:")
    print(f"{'─' * 80}")
    if n_go > 0:
        print("  - Walk-forward holdout on most recent 12 months (label -> 'validated finding')")
        print("  - Implement partial exit in outcome_builder (scale_out_at_r parameter)")
        print("  - Add partial_pnl_r column to orb_outcomes schema")
        print("  - Paper trade with partial exits enabled")
    else:
        print("  - NO-GO: Document as research finding and close.")
        print("  - Partial profit taking does not survive statistical scrutiny.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Partial profit taking research")
    parser.add_argument("--instrument", type=str, default=None,
                        help="Single instrument (default: all 4)")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--partial-levels", type=str, default="0.5,1.0,1.5",
                        help="Comma-separated partial exit R levels")
    parser.add_argument("--trail-levels", type=str, default="0.0,0.25,0.5",
                        help="Comma-separated trail stop R levels")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ACTIVE_INSTRUMENTS
    partial_levels = [float(x) for x in args.partial_levels.split(",")]
    trail_levels = [float(x) for x in args.trail_levels.split(",")]

    print("=" * 80)
    print("PARTIAL PROFIT TAKING RESEARCH")
    print(f"Instruments: {instruments}")
    print(f"RR targets: {RR_TARGETS}")
    print(f"Partial levels: {partial_levels}")
    print(f"Trail levels: {trail_levels}")
    print(f"Grid size: {len(partial_levels)} × {len(trail_levels)} = "
          f"{len(partial_levels) * len(trail_levels)} combos per cell")
    print(f"DB: {args.db_path}")
    print("=" * 80)

    con = duckdb.connect(args.db_path, read_only=True)

    all_results = []
    for instrument in instruments:
        print(f"\n{'─' * 40}")
        print(f"Processing {instrument}...")
        print(f"{'─' * 40}")
        result_df = run_simulation(
            instrument, con, partial_levels, trail_levels, RR_TARGETS,
        )
        if not result_df.empty:
            all_results.append(result_df)

    con.close()

    if not all_results:
        print("No results. Exiting.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal simulation results: {len(combined):,}")

    # Statistical analysis
    print("\nRunning statistical analysis...")
    stats_df = analyze_results(combined)

    if stats_df.empty:
        print("No cells with enough trades for analysis.")
        return

    # Decision gate
    stats_df = apply_decision_gate(stats_df)

    # Output
    print_report(stats_df)

    # Save to CSV
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    stats_path = output_dir / "partial_profit_results.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nDetailed results saved to: {stats_path}")

    # Save raw results for further analysis
    raw_path = output_dir / "partial_profit_raw.parquet"
    combined.to_parquet(raw_path, index=False)
    print(f"Raw simulation data saved to: {raw_path}")


if __name__ == "__main__":
    main()
