"""Forward performance monitor with BH correction at m=5.

For each active lane, maintains a running t-test of forward mean R vs zero.
Applies Benjamini-Hochberg correction across all monitored lanes.
Reports power thresholds derived from backtest effect sizes.

This monitors FORWARD data only. Backtest noise floor p-values measure a
different thing (permutation null) and should NOT be mixed with forward
t-test p-values.

Usage:
    python -m scripts.tools.forward_monitor
"""

import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.prop_profiles import get_lane_registry
from trading_app.strategy_fitness import _load_strategy_outcomes

FORWARD_START = date(2026, 1, 1)

# Backtest stats per lane (from validated_setups — these are reference values for power analysis).
# These are the ONLY hardcoded values — lane structure comes from prop_profiles.
_BACKTEST_STATS: dict[str, dict] = {
    "NYSE_CLOSE": {"backtest_expr": 0.2078, "backtest_std": 0.891},
    "SINGAPORE_OPEN": {"backtest_expr": 0.1557, "backtest_std": 1.844},
    "COMEX_SETTLE": {"backtest_expr": 0.1094, "backtest_std": 0.864},
    "NYSE_OPEN": {"backtest_expr": 0.0933, "backtest_std": 0.956},
    "TOKYO_OPEN": {"backtest_expr": 0.2832, "backtest_std": 1.42},
}


def _build_lanes() -> list[dict]:
    """Build lane list from canonical registry + backtest stats overlay."""
    registry = get_lane_registry()
    lanes = []
    for i, (label, lane) in enumerate(sorted(registry.items()), 1):
        bt = _BACKTEST_STATS.get(label, {"backtest_expr": 0.10, "backtest_std": 1.0})
        lanes.append(
            {
                "name": f"L{i} {label} {lane['filter_type']} RR{lane['rr_target']} O{lane['orb_minutes']}",
                "symbol": lane["instrument"],
                "orb_label": label,
                "entry_model": lane["entry_model"],
                "rr_target": lane["rr_target"],
                "confirm_bars": lane["confirm_bars"],
                "orb_minutes": lane["orb_minutes"],
                "filter_type": lane["filter_type"],
                **bt,
            }
        )
    return lanes


LANES = _build_lanes()


def compute_power_n(effect: float, std: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Sample size needed to detect effect at given power and alpha (two-sided t-test)."""
    if effect <= 0 or std <= 0:
        return 999999
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) * std / effect) ** 2
    return int(np.ceil(n))


def benjamini_hochberg(p_values: list[tuple[str, float]], alpha: float = 0.05) -> list[tuple[str, float, bool]]:
    """BH FDR correction. Returns (name, adjusted_p, significant)."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    adjusted = []
    prev_adj = 1.0
    for i in range(m - 1, -1, -1):
        name, raw_p = sorted_pv[i]
        rank = i + 1
        adj_p = min(prev_adj, raw_p * m / rank)
        adj_p = min(adj_p, 1.0)
        prev_adj = adj_p
        adjusted.append((name, adj_p))
    adjusted.reverse()
    return [(name, adj_p, adj_p <= alpha) for name, adj_p in adjusted]


def run_monitor():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)

    print("=" * 100)
    print(f"FORWARD PERFORMANCE MONITOR — BH correction at m={len(LANES)}")
    print(f"Forward start: {FORWARD_START} | DB: {GOLD_DB_PATH}")
    print("NOTE: These are forward t-test p-values, NOT backtest noise floor p-values.")
    print("=" * 100)

    results = []

    for lane in LANES:
        # Use _load_strategy_outcomes to correctly apply filter_type.
        outcomes = _load_strategy_outcomes(
            con,
            instrument=lane["symbol"],
            orb_label=lane["orb_label"],
            orb_minutes=lane["orb_minutes"],
            entry_model=lane["entry_model"],
            rr_target=lane["rr_target"],
            confirm_bars=lane["confirm_bars"],
            filter_type=lane["filter_type"],
            start_date=FORWARD_START,
        )
        # Filter to entered trades only
        entered = [o for o in outcomes if o.get("pnl_r") is not None and o.get("outcome") not in (None, "scratch")]
        pnl = np.array([o["pnl_r"] for o in entered]) if entered else np.array([])
        n = len(pnl)

        # Power threshold: N needed at 80% power to detect backtest effect
        n_needed = compute_power_n(lane["backtest_expr"], lane["backtest_std"])
        pct_power = (n / n_needed * 100) if n_needed > 0 else 0

        if n < 5:
            status = "INSUFFICIENT"
            raw_p = 1.0
            mean_r = 0.0
            cum_r = 0.0
        else:
            mean_r = float(np.mean(pnl))
            cum_r = float(np.sum(pnl))
            t_stat, raw_p = stats.ttest_1samp(pnl, 0.0)
            # One-sided: we care about mean > 0
            if mean_r <= 0:
                raw_p = 1.0
            else:
                raw_p = raw_p / 2  # convert two-sided to one-sided

            if n < n_needed * 0.10:  # less than 10% of needed sample
                status = "INSUFFICIENT"
            else:
                status = "PENDING"

        results.append(
            {
                "name": lane["name"],
                "n": n,
                "mean_r": mean_r,
                "cum_r": cum_r,
                "raw_p": raw_p,
                "n_needed": n_needed,
                "pct_power": pct_power,
                "status": status,
            }
        )

    # Apply BH correction
    p_values = [(r["name"], r["raw_p"]) for r in results]
    bh_results = benjamini_hochberg(p_values)
    bh_map = {name: (adj_p, sig) for name, adj_p, sig in bh_results}

    # Print results
    print(
        f"\n{'Lane':<45} {'N':>5} {'MeanR':>8} {'CumR':>8} {'Raw p':>8} {'BH p':>8} {'Status':<14} {'N needed':>9} {'% power':>8}"
    )
    print("-" * 155)

    for r in results:
        adj_p, sig = bh_map.get(r["name"], (1.0, False))
        if sig:
            final_status = "SIGNAL"
        elif r["status"] == "INSUFFICIENT":
            final_status = "INSUFFICIENT"
        elif adj_p <= 0.10:
            final_status = "MARGINAL"
        else:
            final_status = "NOISE"

        print(
            f"{r['name']:<45} {r['n']:>5} {r['mean_r']:>+8.4f} {r['cum_r']:>+8.2f} "
            f"{r['raw_p']:>8.4f} {adj_p:>8.4f} {final_status:<14} {r['n_needed']:>9,} {r['pct_power']:>7.1f}%"
        )

    print(f"\n{'=' * 100}")
    print("Status key:")
    print("  SIGNAL      = BH-adjusted p <= 0.05 (statistically significant forward evidence)")
    print("  MARGINAL    = BH-adjusted p <= 0.10 (suggestive but not significant)")
    print("  NOISE       = BH-adjusted p > 0.10 (no forward evidence)")
    print("  INSUFFICIENT = Forward N < 10% of needed sample size (cannot test yet)")
    print("=" * 100)

    con.close()


if __name__ == "__main__":
    run_monitor()
