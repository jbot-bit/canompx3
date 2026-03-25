"""Slippage scenario runner for Apex MNQ lanes.

Computes ExpR at various slippage levels (1-5 ticks) for each lane.
Identifies break-even tick count where ExpR goes to zero.
Read-only — does not modify any production data.

Usage:
    python -m scripts.tools.slippage_scenario
"""

import sys
from pathlib import Path

import duckdb
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.paths import GOLD_DB_PATH
from trading_app.strategy_fitness import _load_strategy_outcomes

# 4 Apex MNQ lanes (from prop_profiles.py apex_50k_manual)
# filter_type must match exactly — orb_outcomes is joined with daily_features
# via _load_strategy_outcomes to apply the correct filter.
LANES = [
    {
        "name": "NYSE_CLOSE VOL_RV12_N20 RR1.0 O15",
        "symbol": "MNQ",
        "orb_label": "NYSE_CLOSE",
        "entry_model": "E2",
        "rr_target": 1.0,
        "confirm_bars": 1,
        "orb_minutes": 15,
        "filter_type": "VOL_RV12_N20",
    },
    {
        "name": "SINGAPORE_OPEN ORB_G8 RR4.0 O15",
        "symbol": "MNQ",
        "orb_label": "SINGAPORE_OPEN",
        "entry_model": "E2",
        "rr_target": 4.0,
        "confirm_bars": 1,
        "orb_minutes": 15,
        "filter_type": "ORB_G8",
    },
    {
        "name": "COMEX_SETTLE ORB_G8 RR1.0 O5",
        "symbol": "MNQ",
        "orb_label": "COMEX_SETTLE",
        "entry_model": "E2",
        "rr_target": 1.0,
        "confirm_bars": 1,
        "orb_minutes": 5,
        "filter_type": "ORB_G8",
    },
    {
        "name": "NYSE_OPEN X_MES_ATR60 RR1.0 O15",
        "symbol": "MNQ",
        "orb_label": "NYSE_OPEN",
        "entry_model": "E2",
        "rr_target": 1.0,
        "confirm_bars": 1,
        "orb_minutes": 15,
        "filter_type": "X_MES_ATR60",
    },
]

# MNQ cost parameters
TICK_SIZE_PTS = 0.25
POINT_VALUE = 2.0
TICK_DOLLAR = TICK_SIZE_PTS * POINT_VALUE  # $0.50


def run_scenario():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    print("=" * 90)
    print("SLIPPAGE SCENARIO ANALYSIS — 4 Apex MNQ Lanes")
    print(f"DB: {GOLD_DB_PATH}")
    print(f"MNQ tick = {TICK_SIZE_PTS} pts = ${TICK_DOLLAR:.2f}")
    print("Backtest uses 1-tick E2 entry slippage. Exit slippage NOT modeled.")
    print("=" * 90)

    for lane in LANES:
        # Use _load_strategy_outcomes to correctly apply filter_type via daily_features join.
        # Raw orb_outcomes queries would include trades on days the filter excludes.
        outcomes = _load_strategy_outcomes(
            con,
            instrument=lane["symbol"],
            orb_label=lane["orb_label"],
            orb_minutes=lane["orb_minutes"],
            entry_model=lane["entry_model"],
            rr_target=lane["rr_target"],
            confirm_bars=lane["confirm_bars"],
            filter_type=lane["filter_type"],
        )

        # Filter to entered trades only (exclude scratches/no-fills)
        rows = [o for o in outcomes if o.get("pnl_r") is not None and o.get("outcome") not in (None, "scratch")]

        if not rows:
            print(f"\n{lane['name']}: NO DATA")
            continue

        pnl_r = np.array([r["pnl_r"] for r in rows])

        # risk_dollars not returned by _load_strategy_outcomes — query separately
        risk_q = """
        SELECT AVG(risk_dollars) FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ? AND entry_model = ?
          AND rr_target = ? AND confirm_bars = ? AND orb_minutes = ?
          AND risk_dollars > 0
        """
        avg_risk_row = con.execute(
            risk_q,
            [
                lane["symbol"],
                lane["orb_label"],
                lane["entry_model"],
                lane["rr_target"],
                lane["confirm_bars"],
                lane["orb_minutes"],
            ],
        ).fetchone()
        risk_d_val = avg_risk_row[0] if avg_risk_row and avg_risk_row[0] else 0
        risk_d = np.array([risk_d_val]) if risk_d_val > 0 else np.array([])
        n = len(pnl_r)
        wr = np.mean(pnl_r > 0)
        base_expr = np.mean(pnl_r)
        avg_risk = np.mean(risk_d) if len(risk_d) > 0 else 0
        avg_risk_pts = avg_risk / POINT_VALUE if POINT_VALUE > 0 else 0

        print(f"\n{'-' * 90}")
        print(f"Lane: {lane['name']}")
        print(f"N={n}, WR={wr:.1%}, AvgRisk=${avg_risk:.2f} ({avg_risk_pts:.1f} pts), Backtest ExpR={base_expr:+.4f}R")
        print(f"{'-' * 90}")
        print(f"{'Extra Ticks':<13} {'Total':<7} {'Entry-Only ExpR':<18} {'$/trade':<12} {'Positive?':<10}")
        print("-" * 60)

        breakeven_tick = None
        for extra in range(0, 11):
            total_ticks = 1 + extra
            # Extra slippage cost on entry side only (stop exits assumed exact)
            extra_cost_dollars = extra * TICK_DOLLAR
            # Convert to R impact: extra_cost / avg_risk_dollars
            if avg_risk > 0:
                r_impact = extra_cost_dollars / avg_risk
            else:
                r_impact = 0
            adjusted_expr = base_expr - r_impact
            dollar_per_trade = adjusted_expr * avg_risk
            positive = adjusted_expr > 0

            if not positive and breakeven_tick is None:
                # Interpolate exact break-even
                prev_expr = base_expr - ((extra - 1) * TICK_DOLLAR / avg_risk) if extra > 0 else base_expr
                if prev_expr > 0 and avg_risk > 0:
                    breakeven_tick = (extra - 1) + prev_expr / (TICK_DOLLAR / avg_risk)
                else:
                    breakeven_tick = extra

            print(
                f"{extra:<13} {total_ticks:<7} {adjusted_expr:+.4f}R{'':<10} "
                f"${dollar_per_trade:+.2f}{'':<5} {'YES' if positive else 'NO'}"
            )

        if breakeven_tick is None:
            breakeven_tick = base_expr / (TICK_DOLLAR / avg_risk) if avg_risk > 0 else float("inf")
        print(
            f"\nBreak-even: {breakeven_tick:.1f} extra ticks ({breakeven_tick + 1:.1f} total) = ${breakeven_tick * TICK_DOLLAR:.2f}/side"
        )

    con.close()

    print(f"\n{'=' * 90}")
    print("NOTE: This models entry-only slippage. Stops also slip in practice.")
    print("Run MNQ tbbo pilot (research/research_mnq_e2_slippage_pilot.py) for empirical data.")
    print("=" * 90)


if __name__ == "__main__":
    run_scenario()
