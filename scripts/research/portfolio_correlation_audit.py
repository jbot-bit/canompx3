"""Portfolio reconstruction correlation audit.

Builds the full pairwise correlation matrix for all active validated MNQ
strategies, clusters into independent families, and selects the optimal
6-lane portfolio for topstep_50k_mnq_auto.

Usage:
    python scripts/research/portfolio_correlation_audit.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Repo bootstrap
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.lane_correlation import _load_lane_daily_pnl, _pearson

# ── Phase 1: Load all validated MNQ strategies ───────────────────────


def load_validated_strategies(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Load all active MNQ validated_setups with their key metrics."""
    rows = con.execute("""
        SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
               vs.entry_model, vs.confirm_bars, vs.rr_target, vs.filter_type,
               es.expectancy_r, es.sharpe_ratio, es.sample_size, es.win_rate,
               es.trades_per_year, es.p_value
        FROM validated_setups vs
        JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
        WHERE vs.instrument = 'MNQ' AND vs.status = 'active'
        ORDER BY es.expectancy_r DESC
    """).fetchall()
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r, strict=True)) for r in rows]


# ── Phase 2: Build daily P&L series for each strategy ────────────────


def build_pnl_series(con: duckdb.DuckDBPyConnection, strategies: list[dict]) -> dict[str, dict[date, float]]:
    """Build {strategy_id: {trading_day: pnl_r}} for each strategy."""
    result = {}
    for s in strategies:
        lane = {
            "instrument": s["instrument"],
            "orb_label": s["orb_label"],
            "orb_minutes": s["orb_minutes"],
            "entry_model": s["entry_model"],
            "rr_target": s["rr_target"],
            "confirm_bars": s["confirm_bars"],
            "filter_type": s["filter_type"],
        }
        result[s["strategy_id"]] = _load_lane_daily_pnl(con, lane)
    return result


# ── Phase 3: Pairwise correlation matrix ─────────────────────────────


@dataclass
class PairMetrics:
    rho: float
    jaccard: float
    shared_days: int
    a_days: int
    b_days: int


def compute_pairwise(
    pnl_series: dict[str, dict[date, float]],
    strategy_ids: list[str],
) -> dict[tuple[str, str], PairMetrics]:
    """Compute Pearson rho and Jaccard overlap for all pairs."""
    pairs = {}
    for i, a in enumerate(strategy_ids):
        for j, b in enumerate(strategy_ids):
            if j <= i:
                continue
            days_a = set(pnl_series[a].keys())
            days_b = set(pnl_series[b].keys())
            shared = sorted(days_a & days_b)
            union = days_a | days_b

            n_shared = len(shared)
            jaccard = n_shared / len(union) if union else 0.0

            if n_shared >= 5:
                xs = [pnl_series[a][d] for d in shared]
                ys = [pnl_series[b][d] for d in shared]
                rho = _pearson(xs, ys)
            else:
                rho = 0.0

            pairs[(a, b)] = PairMetrics(
                rho=rho,
                jaccard=jaccard,
                shared_days=n_shared,
                a_days=len(days_a),
                b_days=len(days_b),
            )
    return pairs


# ── Phase 4: Hierarchical clustering ─────────────────────────────────


def _canonical_key(a: str, b: str) -> tuple[str, str]:
    """Canonical pair key: alphabetical order."""
    return (a, b) if a < b else (b, a)


def cluster_strategies(
    strategy_ids: list[str],
    pairs: dict[tuple[str, str], PairMetrics],
    rho_threshold: float = 0.70,
) -> list[list[str]]:
    """Single-linkage clustering: merge if any member has rho > threshold."""
    clusters: list[set[str]] = [{s} for s in strategy_ids]

    # Build O(1) lookup keyed by canonical (alphabetical) pair
    rho_lookup: dict[tuple[str, str], float] = {}
    for (a, b), pm in pairs.items():
        rho_lookup[_canonical_key(a, b)] = pm.rho

    def get_rho(a: str, b: str) -> float:
        return rho_lookup.get(_canonical_key(a, b), 0.0)

    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                should_merge = any(get_rho(a, b) > rho_threshold for a in clusters[i] for b in clusters[j])
                if should_merge:
                    clusters[i] = clusters[i] | clusters[j]
                    clusters.pop(j)
                    changed = True
                    break
            if changed:
                break

    return [sorted(c) for c in clusters]


# ── Phase 5: Greedy 6-lane selection ─────────────────────────────────


def greedy_select(
    strategies: list[dict],
    pairs: dict[tuple[str, str], PairMetrics],
    max_lanes: int = 6,
    rho_limit: float = 0.70,
) -> list[dict]:
    """Greedy select: best ExpR first, each must have rho < limit with all selected."""
    ranked = sorted(strategies, key=lambda s: s["expectancy_r"], reverse=True)
    selected: list[dict] = []

    for candidate in ranked:
        cid = candidate["strategy_id"]
        if len(selected) >= max_lanes:
            break

        passes = True
        for sel in selected:
            sid = sel["strategy_id"]
            pm = pairs.get(_canonical_key(cid, sid))
            if pm and pm.rho > rho_limit:
                passes = False
                break

        if passes:
            selected.append(candidate)

    return selected


# ── Phase 6: Criterion 11 Monte Carlo ────────────────────────────────


def criterion_11_monte_carlo(
    pnl_series: dict[str, dict[date, float]],
    selected_ids: list[str],
    dd_limit_dollars: float = 2000.0,
    risk_per_trade_dollars: float = 200.0,
    stop_mult: float = 0.75,
    n_sims: int = 2000,
    n_days: int = 90,
) -> dict:
    """Monte Carlo: sample from aggregate daily P&L, check 90-day survival."""
    import random

    all_days: dict[date, float] = defaultdict(float)
    for sid in selected_ids:
        for day, pnl_r in pnl_series[sid].items():
            all_days[day] += pnl_r * risk_per_trade_dollars * stop_mult

    daily_pnl_list = sorted(all_days.values())
    if not daily_pnl_list:
        return {"survival_rate": 0.0, "n_sims": n_sims, "avg_final_pnl": 0.0}

    random.seed(42)
    survivals = 0
    final_pnls = []

    for _ in range(n_sims):
        equity = 0.0
        max_equity = 0.0
        alive = True
        for _ in range(n_days):
            daily = random.choice(daily_pnl_list)
            equity += daily
            max_equity = max(max_equity, equity)
            dd = max_equity - equity
            if dd >= dd_limit_dollars:
                alive = False
                break
        if alive:
            survivals += 1
        final_pnls.append(equity)

    return {
        "survival_rate": survivals / n_sims,
        "n_sims": n_sims,
        "avg_final_pnl": sum(final_pnls) / len(final_pnls),
        "median_final_pnl": sorted(final_pnls)[len(final_pnls) // 2],
        "worst_5pct": sorted(final_pnls)[int(n_sims * 0.05)],
        "n_daily_samples": len(daily_pnl_list),
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    configure_connection(con)

    # Phase 1: Load strategies
    strategies = load_validated_strategies(con)
    print(f"Loaded {len(strategies)} active MNQ validated strategies\n")

    # Phase 2: Build daily P&L series
    print("Building daily P&L series (with canonical filter application)...")
    pnl_series = build_pnl_series(con, strategies)
    for s in strategies:
        sid = s["strategy_id"]
        n_days = len(pnl_series[sid])
        s["_trade_days"] = n_days
    print(f"  Done: {len(pnl_series)} series built\n")

    # Phase 3: Correlation matrix
    sids = [s["strategy_id"] for s in strategies]
    print("Computing pairwise correlations...")
    pairs = compute_pairwise(pnl_series, sids)
    print(f"  {len(pairs)} pairs computed\n")

    # Report high-correlation pairs
    print("=" * 80)
    print("HIGH CORRELATION PAIRS (rho > 0.50)")
    print("=" * 80)
    high_pairs = sorted(
        [(k, v) for k, v in pairs.items() if v.rho > 0.50],
        key=lambda x: x[1].rho,
        reverse=True,
    )
    for (a, b), pm in high_pairs[:30]:
        a_short = a.replace("MNQ_", "").replace("_E2_", " ").replace("_CB1_", " ")
        b_short = b.replace("MNQ_", "").replace("_E2_", " ").replace("_CB1_", " ")
        print(f"  rho={pm.rho:+.3f} J={pm.jaccard:.2f} shared={pm.shared_days:>4}  {a_short}  vs  {b_short}")

    # Phase 4: Clustering
    print(f"\n{'=' * 80}")
    print("INDEPENDENT FAMILIES (rho > 0.70 clustering)")
    print("=" * 80)
    clusters = cluster_strategies(sids, pairs, rho_threshold=0.70)
    for i, cluster in enumerate(clusters):
        short = [s.replace("MNQ_", "").replace("_E2_", " ").replace("_CB1_", " ") for s in cluster]
        print(f"  Family {i + 1} ({len(cluster)} members): {', '.join(short)}")
    print(f"\n  INDEPENDENT BET COUNT: {len(clusters)}")

    # Phase 5: Greedy selection
    print(f"\n{'=' * 80}")
    print("GREEDY 6-LANE SELECTION (rho < 0.70)")
    print("=" * 80)
    selected = greedy_select(strategies, pairs, max_lanes=6)
    for i, s in enumerate(selected):
        sid = s["strategy_id"]
        print(f"  L{i + 1}: {sid}")
        print(
            f"      ExpR={s['expectancy_r']:+.3f} Sh={s['sharpe_ratio']:.2f} "
            f"N={s['sample_size']} WR={s['win_rate']:.1%} "
            f"TradeDays={s['_trade_days']} p={s['p_value']:.6f}"
        )

    # Verify pairwise rho among selected
    print("\n  Selected pairwise correlations:")
    sel_ids = [s["strategy_id"] for s in selected]
    for i, a in enumerate(sel_ids):
        for j, b in enumerate(sel_ids):
            if j <= i:
                continue
            pm = pairs.get(_canonical_key(a, b))
            rho = pm.rho if pm else 0.0
            marker = " *** OVER 0.70 ***" if rho > 0.70 else ""
            a_s = a.split("_CB1_")[1] if "_CB1_" in a else a[-15:]
            b_s = b.split("_CB1_")[1] if "_CB1_" in b else b[-15:]
            sess_a = a.split("_E2_")[0].replace("MNQ_", "")
            sess_b = b.split("_E2_")[0].replace("MNQ_", "")
            print(f"    {sess_a} {a_s} vs {sess_b} {b_s}: rho={rho:+.3f}{marker}")

    # Compare with current deployed
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH CURRENT DEPLOYED")
    print("=" * 80)
    deployed = [
        "MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5",
        "MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5",
        "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5",
        "MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5",
        "MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5",
        "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100",
    ]
    deployed_strats = [s for s in strategies if s["strategy_id"] in deployed]
    deployed_expr = sum(s["expectancy_r"] for s in deployed_strats)
    selected_expr = sum(s["expectancy_r"] for s in selected)
    print(f"  Current deployed total ExpR: {deployed_expr:+.3f} ({len(deployed_strats)} lanes)")
    print(f"  Recommended total ExpR:      {selected_expr:+.3f} ({len(selected)} lanes)")
    print(f"  Delta:                       {selected_expr - deployed_expr:+.3f}")

    # Show current deployed pairwise correlations too
    if len(deployed_strats) >= 2:
        print("\n  Current deployed pairwise correlations:")
        for i, a in enumerate(deployed):
            for j, b in enumerate(deployed):
                if j <= i:
                    continue
                if a not in [s["strategy_id"] for s in strategies]:
                    continue
                if b not in [s["strategy_id"] for s in strategies]:
                    continue
                pm = pairs.get(_canonical_key(a, b))
                rho = pm.rho if pm else 0.0
                marker = " *** OVER 0.70 ***" if rho > 0.70 else ""
                sess_a = a.split("_E2_")[0].replace("MNQ_", "")
                sess_b = b.split("_E2_")[0].replace("MNQ_", "")
                print(f"    {sess_a} vs {sess_b}: rho={rho:+.3f}{marker}")

    # Phase 6: Criterion 11
    print(f"\n{'=' * 80}")
    print("CRITERION 11 -- MONTE CARLO (90-day, $2K DD, S0.75)")
    print("=" * 80)

    risk_rows = con.execute("""
        SELECT AVG(es.avg_risk_dollars) FROM experimental_strategies es
        WHERE es.instrument = 'MNQ' AND es.orb_minutes = 5
        AND es.avg_risk_dollars IS NOT NULL AND es.avg_risk_dollars > 0
        LIMIT 100
    """).fetchone()
    avg_risk = risk_rows[0] if risk_rows and risk_rows[0] else 200.0
    print(f"  Using avg risk_dollars: ${avg_risk:.0f}")

    c11_rec = criterion_11_monte_carlo(
        pnl_series,
        sel_ids,
        dd_limit_dollars=2000.0,
        risk_per_trade_dollars=avg_risk,
        stop_mult=0.75,
    )
    print("\n  RECOMMENDED portfolio:")
    print(f"    Survival rate: {c11_rec['survival_rate']:.1%}")
    print(f"    Avg final P&L: ${c11_rec['avg_final_pnl']:.0f}")
    print(f"    Median final:  ${c11_rec['median_final_pnl']:.0f}")
    print(f"    Worst 5%:      ${c11_rec['worst_5pct']:.0f}")
    print(f"    Daily samples: {c11_rec['n_daily_samples']}")

    deployed_ids = [s["strategy_id"] for s in deployed_strats]
    if deployed_ids:
        c11_cur = criterion_11_monte_carlo(
            pnl_series,
            deployed_ids,
            dd_limit_dollars=2000.0,
            risk_per_trade_dollars=avg_risk,
            stop_mult=0.75,
        )
        print("\n  CURRENT deployed portfolio:")
        print(f"    Survival rate: {c11_cur['survival_rate']:.1%}")
        print(f"    Avg final P&L: ${c11_cur['avg_final_pnl']:.0f}")
        print(f"    Median final:  ${c11_cur['median_final_pnl']:.0f}")
        print(f"    Worst 5%:      ${c11_cur['worst_5pct']:.0f}")

    verdict = "PASS" if c11_rec["survival_rate"] >= 0.70 else "FAIL"
    print(f"\n  Criterion 11 verdict: {verdict} ({c11_rec['survival_rate']:.1%} >= 70%)")

    con.close()


if __name__ == "__main__":
    main()
